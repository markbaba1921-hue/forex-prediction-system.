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
# Generate trading signals (for every row)
# ----------------------------
def generate_signals(df):
    df["Signal"] = "Neutral"
    df["Signal_Color"] = "white"

    for i in range(1, len(df)):
        if (df["EMA20"].iloc[i] > df["EMA50"].iloc[i]) and (df["RSI14"].iloc[i] < 70) and (df["MACD"].iloc[i] > df["Signal_Line"].iloc[i]):
            df.loc[df.index[i], "Signal"] = "BUY"
            df.loc[df.index[i], "Signal_Color"] = "green"
        elif (df["EMA20"].iloc[i] < df["EMA50"].iloc[i]) and (df["RSI14"].iloc[i] > 30) and (df["MACD"].iloc[i] < df["Signal_Line"].iloc[i]):
            df.loc[df.index[i], "Signal"] = "SELL"
            df.loc[df.index[i], "Signal_Color"] = "red"

    return df


# ----------------------------
# Plot candlestick + signals
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

    # BUY/SELL arrows
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"], mode="markers+text",
        marker=dict(color="green", size=10, symbol="triangle-up"),
        text=["BUY"]*len(buys), textposition="top center", name="BUY Signals"
    ))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"], mode="markers+text",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        text=["SELL"]*len(sells), textposition="bottom center", name="SELL Signals"
    ))

    fig.update_layout(title=f"{pair} Price & Signals", xaxis_rangeslider_visible=False)
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Forex Signals", layout="wide")
st.title("💹 Forex Signals — EMA/RSI/MACD (with Time & History)")

# Auto-refresh every 60 seconds
st.experimental_autorefresh(interval=60*1000, key="refresh")

pair = st.text_input("Enter Forex Pair (Yahoo Finance format, e.g., EURUSD=X):", "EURUSD=X")
period = st.selectbox("Select Time Period:", ["5d", "1mo", "3mo", "6mo"])
interval = st.selectbox("Select Interval:", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])

if st.button("Get Signals"):
    try:
        data = yf.download(pair, period=period, interval=interval)

        if data.empty:
            st.error("⚠️ No data found. Try another pair or interval.")
        else:
            # Flatten MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            df = compute_indicators(data)
            df = generate_signals(df)

            # Latest signal
            latest = df.iloc[-1]
            st.subheader("📊 Latest Signal")
            st.markdown(f"<h2 style='color:{latest['Signal_Color']}'>{latest['Signal']} ({latest.name})</h2>", unsafe_allow_html=True)

            # Chart
            st.subheader("📈 Candlestick Chart with Signals")
            st.plotly_chart(plot_candles(df, pair), use_container_width=True)

            # History of signals
            st.subheader("📑 Signal History")
            signal_history = df[df["Signal"] != "Neutral"][["Close", "Signal"]]
            st.dataframe(signal_history.tail(20))  # show last 20 signals

    except Exception as e:
        st.error(f"Error: {e}")
