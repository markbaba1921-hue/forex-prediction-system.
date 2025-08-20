import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go

# ----------------------------
# Compute indicators
# ----------------------------
def compute_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    rsi = ta.momentum.RSIIndicator(df["Close"], window=14)
    df["RSI14"] = rsi.rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()

    return df


# ----------------------------
# Generate trading signals
# ----------------------------
def generate_signals(df):
    latest = df.iloc[-1]

    signal = "🔍 Neutral"
    color = "white"

    if latest["EMA20"] > latest["EMA50"] and latest["RSI14"] < 70 and latest["MACD"] > latest["Signal_Line"]:
        signal = "🟢 BUY"
        color = "green"
    elif latest["EMA20"] < latest["EMA50"] and latest["RSI14"] > 30 and latest["MACD"] < latest["Signal_Line"]:
        signal = "🔴 SELL"
        color = "red"

    return signal, color


# ----------------------------
# Plot candlestick + indicators
# ----------------------------
def plot_candles(df, pair):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candles"
    ))

    # EMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='blue', width=1), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1), name="EMA50"))

    # Last signal arrow
    sig, color = generate_signals(df)
    last_price = df["Close"].iloc[-1]
    last_time = df.index[-1]

    if "BUY" in sig:
        fig.add_annotation(x=last_time, y=last_price,
                           text="BUY ⬆️", showarrow=True, arrowhead=2, font=dict(color="green"))
    elif "SELL" in sig:
        fig.add_annotation(x=last_time, y=last_price,
                           text="SELL ⬇️", showarrow=True, arrowhead=2, font=dict(color="red"))

    fig.update_layout(title=f"{pair} Price & Signals", xaxis_rangeslider_visible=False)
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Forex Signals", layout="wide")
st.title("💹 Forex Signals — EMA/RSI/MACD")

pair = st.text_input("Enter Forex Pair (Yahoo Finance format, e.g., EURUSD=X):", "EURUSD=X")
period = st.selectbox("Select Time Period:", ["5d", "1mo", "3mo", "6mo", "1y"])
interval = st.selectbox("Select Interval:", ["15m", "30m", "1h", "4h", "1d"])

if st.button("Get Signals"):
    try:
        data = yf.download(pair, period=period, interval=interval)

        if data.empty:
            st.error("⚠️ No data found. Try another pair or interval.")
        else:
            # FIX: flatten columns if MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            df = compute_indicators(data)
            sig, color = generate_signals(df)

            st.subheader("📊 Latest Signal")
            st.markdown(f"<h2 style='color:{color}'>{sig}</h2>", unsafe_allow_html=True)

            st.subheader("📈 Candlestick Chart")
            st.plotly_chart(plot_candles(df, pair), use_container_width=True)

            st.subheader("📑 Data Preview")
            st.dataframe(df.tail())

    except Exception as e:
        st.error(f"Error: {e}")
