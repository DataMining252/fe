from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline

from service import get_historical_data, get_forecast

# ==========================================
# CONFIG UI
# ==========================================
st.set_page_config(
    page_title="GoldInsight Dashboard",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 GoldInsight: Price Forecast")

# ==========================================
# FILTER
# ==========================================
st.markdown("### 🔍 Filter")

gold_cols = ['open','high','low','close']
feature_cols = ['dxy','sp500','oil','interest_rate','cpi']

col1, col2, col3 = st.columns(3)

with col1:
    period = st.selectbox(
        "Period",
        ["week", "month", "3months", "custom"]
    )

start_date, end_date = None, None
with col2:
    if period == "custom":
        start_date = st.date_input("Start date", value=date.today())
with col3:
    if period == "custom":
        end_date = st.date_input("End date", value=date.today())

selected_cols = st.multiselect(
    "Columns",
    gold_cols + feature_cols,
    default=["close"]
)

n_days = st.slider("Forecast days", 1, 7, 5)

# ==========================================
# LOAD CONTROL
# ==========================================
if "loaded" not in st.session_state:
    st.session_state.loaded = True

if st.button("🔄️ Refresh"):
    st.session_state.loaded = True

# ==========================================
# LOAD DATA
# ==========================================
if st.session_state.loaded:

    st.header("📊 Gold Prices and Predict Chart")

    with st.spinner("⏳ Loading..."):

        hist_res = get_historical_data(
            period=period,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
            columns=selected_cols
        )

        pred_res = get_forecast(n_days)

        # ===== HISTORICAL =====
        hist_df = pd.DataFrame([
            {"date": row["date"], **row["data"]}
            for row in hist_res["historical"]
        ])

        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df.set_index("date", inplace=True)
        hist_df = hist_df.sort_index()

        # ===== FORECAST =====
        pred_df = pd.DataFrame(pred_res["forecast"])
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df.set_index("date", inplace=True)

        pred_df.rename(columns={
            "prediction_value": "Forecast_Value",
            "prediction_lstm": "Forecast_LSTM_raw"
        }, inplace=True)

        # ===== SMOOTH LSTM =====
        lstm_series = pred_df["Forecast_LSTM_raw"]

        x = np.arange(len(lstm_series))
        y = lstm_series.values

        x_new = np.linspace(x.min(), x.max(), 50)
        y_smooth = make_interp_spline(x, y, k=3)(x_new)

        date_range = pd.date_range(
            start=lstm_series.index.min(),
            end=lstm_series.index.max(),
            periods=len(x_new)
        )

        lstm_smooth_df = pd.DataFrame({
            "date": date_range,
            "Forecast_LSTM": y_smooth
        }).set_index("date")

    # ==========================================
    # OVERVIEW
    # ==========================================
    st.subheader("📌 Overview")

    if "close" not in hist_df.columns:
        st.error("❌ CLOSE price is required for forecasting")
        st.stop()

    last_price = hist_df["close"].iloc[-1]
    prev_price = hist_df["close"].iloc[-2] if len(hist_df) > 1 else last_price
    pct_change = (last_price - prev_price) / prev_price * 100

    volatility = hist_df["close"].pct_change().std()

    col1, col2, col3 = st.columns(3)

    col1.metric("Gold Price", f"{last_price:.2f} USD", f"{pct_change:.2f}%")
    col2.metric("Data Points", len(hist_df))
    col3.metric("Volatility", f"{volatility:.4f}")

    # ==========================================
    # CHART
    # ==========================================
    fig = go.Figure()

    # plot selected columns
    for col in selected_cols:
        if col in hist_df.columns:
            fig.add_trace(go.Scatter(
                x=hist_df.tail(30).index,
                y=hist_df.tail(30)[col],
                mode='lines',
                name=col.upper()
            ))

    # forecast (only for close)
    if "close" in selected_cols:
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df["Forecast_Value"],
            mode='lines+markers',
            name='Forecast (XGB)',
            line=dict(color='#d62728', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=lstm_smooth_df.index,
            y=lstm_smooth_df["Forecast_LSTM"],
            mode='lines',
            name='Forecast (LSTM)',
            line=dict(color='#ff7f0e', width=4, dash='dash')
        ))

        fig.add_vline(
            x=hist_df.index[-1],
            line_width=2,
            line_dash="dot",
            line_color="black"
        )

    fig.update_layout(
        title="Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

    # ==========================================
    # MODEL COMPARISON
    # ==========================================
    st.subheader("🤖 Model Comparison")

    compare_df = pred_df[["Forecast_Value", "Forecast_LSTM_raw"]].rename(
        columns={"Forecast_Value": "XGBoost", "Forecast_LSTM_raw": "LSTM"}
    )

    st.line_chart(compare_df)

    # ==========================================
    # DATA DISTRIBUTION (FIX LỖI ALTair)
    # ==========================================
    st.subheader("💫 Data Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Price Distribution")

        prices = hist_df["close"].dropna()

        hist, bins = np.histogram(prices, bins=20)

        dist_df = pd.DataFrame({
            "range": [f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(len(bins)-1)],
            "count": hist
        })

        st.bar_chart(dist_df.set_index("range"))

    with col2:
        st.write("Return Distribution")

        returns = hist_df["close"].pct_change().dropna()

        hist, bins = np.histogram(returns, bins=20)

        dist_df = pd.DataFrame({
            "range": [f"{bins[i]:.4f} - {bins[i+1]:.4f}" for i in range(len(bins)-1)],
            "count": hist
        })

        st.bar_chart(dist_df.set_index("range"))

    # ==========================================
    # FEATURE CORRELATION
    # ==========================================
    if any(col in selected_cols for col in feature_cols):

        st.subheader("🔗 Feature Correlation")

        full_df = pd.DataFrame([
            {"date": row["date"], **row["data"]}
            for row in hist_res["historical"]
        ])

        full_df["date"] = pd.to_datetime(full_df["date"])
        full_df.set_index("date", inplace=True)

        st.dataframe(full_df.corr().style.background_gradient(cmap="coolwarm"))

    # ==========================================
    # TREND SIGNAL
    # ==========================================
    st.subheader("📈 Trend Signal")

    xgb = pred_df["Forecast_Value"].iloc[0]
    lstm = pred_df["Forecast_LSTM_raw"].iloc[0]

    if xgb > last_price and lstm > last_price:
        st.success("Strong Uptrend 📈 (Both models agree)")
    elif xgb < last_price and lstm < last_price:
        st.error("Strong Downtrend 📉 (Both models agree)")
    else:
        st.warning("⚠️ Mixed Signal: Short-term vs Long-term conflict")