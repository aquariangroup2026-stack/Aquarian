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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Weather Engine", layout="wide")
st.title("🌦️ Weather Forecast & Validation Engine")

# --- AUTHENTICATION ---
def get_gspread_client():
    # Streamlit Secrets should contain the JSON key content in 'gcp_service_account'
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)

try:
    gc = get_gspread_client()
    st.toast("Connected to Google Sheets!", icon="✅")
except Exception as e:
    st.error(f"Authentication Failed: {e}")
    st.stop()

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Replace 'Weather_Data_Sheet' with your actual Google Sheet name
        ws = gc.open('Weather_Data_Sheet').sheet1
        rows = ws.get_all_values()
        
        # Create DataFrame
        df = pd.DataFrame(rows[1:], columns=rows[0])
        
        # Clean Column Names
        df.columns = df.columns.str.strip()
        
        # Define Features and Target
        features = ['Station Pressure', 'Mean Dew Point', 'Mean Humidity (%)', 'Vapor Pressure']
        target = 'Rainfall (mm)'
        
        # Convert to Numeric
        for col in features + [target]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Date Handling
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna(subset=['Date'])
        
        # Fill Missing Values (Forward fill then fill remaining with 0)
        df[features + [target]] = df[features + [target]].ffill().fillna(0)
        
        # Create Binary Target (1 if Rain > 0.1mm, else 0)
        df['Rain_Binary'] = (df[target] > 0.1).astype(int)
        
        return df, features, target
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), [], ""

df, features, target = load_data()

if df.empty:
    st.stop()

# --- APP LAYOUT ---
tabs = st.tabs(["📈 180-Day Forecast", "🔍 Validation (Last 31 Days)", "📊 Raw Data"])

# ==========================================
# TAB 1: FUTURE PROJECTIONS (180 DAYS)
# ==========================================
with tabs[0]:
    st.header("Future Projections (Next 6 Months)")
    
    # 1. Train Models on ALL Data
    X = df[features]
    y = df['Rain_Binary']
    ts_data = df.set_index('Date')[target].resample('D').mean().fillna(0)
    
    # Classification Models
    log_model = LogisticRegression(solver='liblinear').fit(X, y)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05).fit(X, y)
    
    # Time Series Model (Holt-Winters)
    # Seasonal period: 1095 days. If data is short, fall back to 7 or 30
    seasonal_p = 1095 if len(ts_data) > 730 else 30
    try:
        hw_model = ExponentialSmoothing(
            ts_data, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=seasonal_p
        ).fit()
    except:
        # Fallback if Holt-Winters fails (e.g., too little data)
        hw_model = ExponentialSmoothing(ts_data, trend='add').fit()
    
    # 2. Forecasting Loop
    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=180)
    
    # Create lookup table for monthly averages to use as proxy features
    monthly_lookup = df.groupby(df['Date'].dt.month)[features].mean()
    
    forecast_results = []
    
    # Forecast Amount using Holt-Winters
    hw_forecast_values = hw_model.forecast(180)
    
    for idx, d in enumerate(future_dates):
        # Get average features for this month
        m_avg = monthly_lookup.loc[d.month].values.reshape(1, -1)
        
        # Predict Probabilities
        p_sig = log_model.predict_proba(m_avg)[0][1]
        p_xgb = xgb_model.predict_proba(m_avg)[0][1]
        
        # Get Amount from HW Forecast
        pred_amount = max(0, hw_forecast_values.iloc[idx]) # Ensure no negative rain
        
        forecast_results.append([d, p_sig, p_xgb, pred_amount])

    f_df = pd.DataFrame(forecast_results, columns=['Date', 'Sig_P', 'XGB_P', 'Pred_Amount'])
    
    # 3. Plotting Future
    st.subheader("Rain Probability Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f_df['Date'], f_df['Sig_P'], label='Logistic Regression', alpha=0.7)
    ax.plot(f_df['Date'], f_df['XGB_P'], label='XGBoost', linestyle='--', alpha=0.8)
    ax.axhline(0.5, color='red', linestyle=':', label='Threshold')
    ax.set_ylabel("Probability (0-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ==========================================
# TAB 2: VALIDATION (BACKTESTING)
# ==========================================
with tabs[1]:
    st.header("Backtesting Accuracy (Last 31 Days)")

    # 1. Split Data (Hide last 31 days)
    cutoff_date = df['Date'].max() - pd.Timedelta(days=31)
    train_split = df[df['Date'] <= cutoff_date]
    val_split = df[df['Date'] > cutoff_date].copy()

    if len(val_split) == 0:
        st.warning("Not enough data to perform validation (Need > 31 days).")
    else:
        # 2. Retrain Models on PAST Data Only
        X_train = train_split[features]
        y_train = train_split['Rain_Binary']
        ts_train = train_split.set_index('Date')[target].resample('D').mean().fillna(0)

        # Retrain Classifiers
        log_val = LogisticRegression(solver='liblinear').fit(X_train, y_train)
        xgb_val = XGBClassifier(n_estimators=100, learning_rate=0.05).fit(X_train, y_train)

        # Retrain Time Series
        seasonal_p = 1095 if len(ts_train) > 730 else 7
        try:
            hw_val = ExponentialSmoothing(
                ts_train, trend='add', seasonal='add', seasonal_periods=seasonal_p
            ).fit()
        except:
             hw_val = ExponentialSmoothing(ts_train, trend='add').fit()

        # 3. Predict on Validation Set
        # Probabilities
        val_split['Prob_Sig'] = log_val.predict_proba(val_split[features])[:, 1]
        val_split['Prob_XGB'] = xgb_val.predict_proba(val_split[features])[:, 1]
        
        # Amounts
        forecast_steps = len(val_split)
        hw_preds = hw_val.forecast(forecast_steps)
        # Align index just in case
        val_split['Pred_Amount'] = hw_preds.values

        # 4. Calculate Metrics
        from sklearn.metrics import accuracy_score, mean_absolute_error
        
        # Accuracy: How often did we correctly say "It will rain" vs "It won't"?
        acc_xgb = accuracy_score(val_split['Rain_Binary'], (val_split['Prob_XGB'] > 0.5).astype(int))
        
        # MAE: How far off was the amount in mm?
        mae_hw = mean_absolute_error(val_split[target], val_split['Pred_Amount'])

        col1, col2 = st.columns(2)
        col1.metric("Prediction Accuracy (XGB)", f"{acc_xgb:.1%}", help="Did it rain? (Yes/No)")
        col2.metric("Rain Amount Error (MAE)", f"{mae_hw:.2f} mm", help="Avg difference in mm")

        st.divider()

        # --- PLOT B: RAINFALL AMOUNTS (BARS for Actual) ---
        st.subheader("💧 Rainfall Amount: Actual vs. Forecast")
        fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
        
        # ACTUAL = BARS (Blue)
        ax_ts.bar(val_split['Date'], val_split[target], label='Actual Rainfall (mm)', color='#1f77b4', alpha=0.6)
        
        # FORECAST = LINE (Red/Orange)
        ax_ts.plot(val_split['Date'], val_split['Pred_Amount'], label='Forecast (Holt-Winters)', color='#d62728', linestyle='--', linewidth=2)
        
        ax_ts.set_ylabel("Rainfall (mm)")
        ax_ts.legend()
        ax_ts.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_ts)

        # --- PLOT A: PROBABILITIES ---
        st.subheader("🌧️ Rain Probability vs. Events")
        fig_val, ax_val = plt.subplots(figsize=(10, 4))
        
        # Forecast Probability = Line
        ax_val.plot(val_split['Date'], val_split['Prob_XGB'], label='Forecast Prob (XGB)', color='#ff7f0e', linewidth=2)
        
        # Actual Events = Gray Background Bars
        # We multiply by 1 to make the bar height 1.0 (full height) wherever rain occurred
        ax_val.bar(val_split['Date'], val_split['Rain_Binary'], color='gray', alpha=0.2, label='Actual Rain Day')

        ax_val.set_ylim(0, 1.05)
        ax_val.axhline(0.5, color='black', linestyle=':', linewidth=1)
        ax_val.set_ylabel("Probability")
        ax_val.legend(loc='upper left')
        st.pyplot(fig_val)

# ==========================================
# TAB 3: RAW DATA
# ==========================================
with tabs[2]:
    st.header("Dataset Overview")
    st.dataframe(df.sort_values('Date', ascending=False))