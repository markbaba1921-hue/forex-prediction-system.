import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

st.set_page_config(page_title="Forex Pro Signals", layout="wide")

# ----------------------
# Fetch data safely
# ----------------------
def fetch_data(symbol, period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            return None
        # flatten MultiIndex if exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# ----------------------
# Compute indicators
# ----------------------
def add_indicators(df):
    df["EMA20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    bb = BollingerBands(close=df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    return df

# ----------------------
# Generate signals
# ----------------------
def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        if df["EMA20"].iloc[i] > df["EMA50"].iloc[i] and df["RSI"].iloc[i] < 70 and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]:
            signals.append("BUY")
        elif df["EMA20"].iloc[i] < df["EMA50"].iloc[i] and df["RSI"].iloc[i] > 30 and df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]:
            signals.append("SELL")
        else:
            signals.append("")
    signals.insert(0, "")
    df["Signal"] = signals
    return df

# ----------------------
# Plot chart with signals
# ----------------------
def plot_chart(df, symbol):
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Candles"
    ))

    # EMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color="blue", width=1), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="orange", width=1), name="EMA50"))

    # Buy/Sell markers
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"), name="BUY"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"), name="SELL"))

    fig.update_layout(
        title=f"{symbol} Price Chart with Signals",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )
    return fig

# ----------------------
# Streamlit UI
# ----------------------
st.title("📈 Forex Pro Signals — Live & Precise")

symbol = st.text_input("Enter Forex Pair (Yahoo format)", "EURUSD=X")
period = st.selectbox("Time Period", ["1d","5d","1mo"])
interval = st.selectbox("Interval", ["1m","5m","15m","30m","1h"])

mode = st.radio("Signal Mode", ["Technical", "ML (stub)", "Composite"], horizontal=True)

if st.button("Get Live Signals"):
    df = fetch_data(symbol, period, interval)
    if df is None or df.empty:
        st.error("No data returned. Try different timeframe.")
    else:
        df = add_indicators(df)
        df = generate_signals(df)

        # Latest Signal
        latest = df.iloc[-1]
        st.subheader("📢 Live Signal (Latest Candle)")
        st.write(f"{latest.name} | Price: {latest['Close']:.5f} | Signal: {latest['Signal']}")

        # Plot Chart
        fig = plot_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        # Show last few rows
        st.subheader("📊 Data Preview")
        st.dataframe(df.tail(10))
