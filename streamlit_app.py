import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# --- PAGE SETUP ---
st.set_page_config(page_title="AQUARAIN Weather Engine", page_icon="🌦️", layout="wide")

st.title("🌦️ AQUARAIN: Ensemble Weather Forecast")
st.markdown("""
This engine combines **Logistic Regression (Sigmoid)**, **XGBoost**, and **Time-Series models** to predict rainfall probability and volume.
""")

# --- AUTHENTICATION HELPER ---
def get_gspread_client():
    # Looks for 'gcp_service_account' in Streamlit Secrets
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_dict = dict(st.secrets["gcp_service_account"])
    # Standardize private key formatting
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_and_clean_data():
    try:
        gc = get_gspread_client()
        ws = gc.open('Weather_Data_Sheet').sheet1
        rows = ws.get_all_values()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        
        df.columns = df.columns.str.strip()
        features = ['Station Pressure', 'Mean Dew Point', 'Mean Humidity (%)', 'Vapor Pressure']
        target = 'Rainfall (mm)'
        
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna(subset=['Date'])
        df[features + [target]] = df[features + [target]].ffill().fillna(0)
        df['Rain_Binary'] = (df[target] > 0.1).astype(int)
        
        return df, features, target
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

df, features, target = load_and_clean_data()

if df is not None:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 1095, 180)
    train_button = st.sidebar.button("Retrain Engine")

    # --- ENGINE LOGIC ---
    @st.cache_resource
    def train_models(_df, _features, _target):
        X = _df[_features]
        y = _df['Rain_Binary']
        ts_data = _df.set_index('Date')[_target].resample('D').mean().fillna(0)
        
        # Classifiers
        log_m = LogisticRegression(solver='liblinear').fit(X, y)
        xgb_m = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5).fit(X, y)
        
        # Time-Series
        hw_m = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=1095).fit()
        sa_m = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        
        return log_m, xgb_m, hw_m, sa_m, ts_data

    log_model, xgb_model, hw_model, sarima_model, ts_data = train_models(df, features, target)

    # --- GENERATE FORECAST ---
    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    monthly_lookup = df.groupby(df['Date'].dt.month)[features].mean()
    
    # Pre-calculate TS Volumes
    sarima_forecast = sarima_model.get_forecast(steps=forecast_days).summary_frame()['mean']
    hw_forecast = hw_model.forecast(forecast_days)

    forecast_results = []
    for i, d in enumerate(future_dates):
        m_avg = monthly_lookup.loc[d.month].values.reshape(1, -1)
        p_sig = float(log_model.predict_proba(m_avg)[0][1])
        p_xgb = float(xgb_model.predict_proba(m_avg)[0][1])
        v_sa = float(max(0, sarima_forecast.iloc[i]))
        v_hw = float(max(0, hw_forecast.iloc[i]))
        
        ensemble_p = (p_sig + p_xgb) / 2
        res = "Rain" if (ensemble_p > 0.5 or v_sa > 1.0) else "No Rain"
        forecast_results.append([d, p_sig, p_xgb, v_sa, v_hw, res])

    f_df = pd.DataFrame(forecast_results, columns=['Date', 'Sig_P', 'XGB_P', 'SA_mm', 'HW_mm', 'Res'])

    # --- TABS FOR VISUALIZATION ---
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Charts", "🔍 Validation Metrics", "📋 Raw Results"])

    with tab1:
        st.subheader(f"{forecast_days}-Day Probability Comparison")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(f_df['Date'], f_df['Sig_P'], label='AQUARAIN (Sigmoid)', color='blue')
        ax1.plot(f_df['Date'], f_df['XGB_P'], label='XGBoost', color='orange', linestyle='--')
        ax1.axhline(0.5, color='red', alpha=0.3)
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("Predicted Rainfall Volumes (mm)")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.fill_between(f_df['Date'], f_df['SA_mm'], color='green', alpha=0.3, label='SARIMA')
        ax2.fill_between(f_df['Date'], f_df['HW_mm'], color='red', alpha=0.2, label='Holt-Winters')
        ax2.legend()
        st.pyplot(fig2)

    with tab2:
        # Backtesting logic for the last 31 days
        val_df = df.iloc[-31:]
        val_probs = log_model.predict_proba(val_df[features])[:, 1]
        
        st.subheader("Backtesting: Predicted vs Actual (Last 31 Days)")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.bar(val_df['Date'], val_df['Rain_Binary'], color='red', alpha=0.2, label='Actual')
        ax3.plot(val_df['Date'], val_probs, color='blue', marker='o', label='Engine Prob')
        ax3.legend()
        st.pyplot(fig3)

    with tab3:
        st.dataframe(f_df, use_container_width=True)
        
        # Download button for CSV
        csv = f_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", data=csv, file_name="forecast_results.csv", mime='text/csv')

else:
    st.warning("Please check your Google Sheets connection and Secrets configuration.")