import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import datetime
from streamlit_lightweight_charts import renderLightweightCharts

st.set_page_config(page_title="📊 Forex Live Signals", layout="wide")

# --------------------
# Fetch Data Function
# --------------------
@st.cache_data(ttl=60)
def load_data(symbol, period, interval):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

# --------------------
# Generate Trading Signals
# --------------------
def generate_signals(df):
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    signals = []
    for i in range(len(df)):
        if df["EMA20"].iloc[i] > df["EMA50"].iloc[i] and df["RSI"].iloc[i] < 70:
            signals.append("BUY")
        elif df["EMA20"].iloc[i] < df["EMA50"].iloc[i] and df["RSI"].iloc[i] > 30:
            signals.append("SELL")
        else:
            signals.append("WAIT")
    df["Signal"] = signals
    return df

# --------------------
# App Layout
# --------------------
st.title("📢 Forex Live Signal System")
symbol = st.text_input("Enter Forex Pair (Yahoo format, e.g. EURUSD=X):", "EURUSD=X")
period = st.selectbox("Select Period:", ["1d","5d","1mo","3mo"])
interval = st.selectbox("Select Interval:", ["1m","5m","15m","30m","1h"])

if st.button("Get Live Signals"):
    df = load_data(symbol, period, interval)
    df = generate_signals(df)

    # Latest signal
    latest = df.iloc[-1]
    st.subheader("📍 Latest Signal")
    st.write(f"**{latest['Signal']}** at {latest['Datetime']} | Price: {latest['Close']:.5f}")

    # --------------------
    # TradingView Chart
    # --------------------
    st.subheader("📊 Live Price Chart")
    chart_data = [
        {
            "time": row["Datetime"].strftime("%Y-%m-%dT%H:%M:%S"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }
        for _, row in df.iterrows()
    ]

    chart_options = {
        "height": 500,
        "rightPriceScale": {"visible": True},
        "timeScale": {"timeVisible": True, "secondsVisible": True}, # ⏰ exact time
        "grid": {"vertLines": {"visible": False}, "horzLines": {"visible": False}},
    }

    series = [
        {"type": "Candlestick", "data": chart_data},
    ]

    renderLightweightCharts([{"chart": chart_options, "series": series}], key="chart")

    # Show full table
    st.subheader("📑 Recent Data & Signals")
    st.dataframe(df.tail(20))
