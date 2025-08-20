import streamlit as st
import yfinance as yf
import pandas as pd
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
    df["Signal"] = "Neutral"
    for i in range(1, len(df)):
        if (df["EMA20"].iloc[i] > df["EMA50"].iloc[i]) and (df["RSI14"].iloc[i] < 70) and (df["MACD"].iloc[i] > df["Signal_Line"].iloc[i]):
            df.loc[df.index[i], "Signal"] = "BUY"
        elif (df["EMA20"].iloc[i] < df["EMA50"].iloc[i]) and (df["RSI14"].iloc[i] > 30) and (df["MACD"].iloc[i] < df["Signal_Line"].iloc[i]):
            df.loc[df.index[i], "Signal"] = "SELL"
    return df


# ----------------------------
# Plot candlestick chart
# ----------------------------
def plot_candles(df, pair):
    fig = go.Figure()

    # Candlesticks
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

    # Buy signals
    buys = df[df["Signal"] == "BUY"]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        mode="markers+text", name="BUY",
        marker=dict(color="green", size=10, symbol="triangle-up"),
        text=["BUY"]*len(buys), textposition="top center"
    ))

    # Sell signals
    sells = df[df["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        mode="markers+text", name="SELL",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        text=["SELL"]*len(sells), textposition="bottom center"
    ))

    fig.update_layout(title=f"{pair} Price & Signals", xaxis_rangeslider_visible=False)
    return fig


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Forex Signals", layout="wide")
st.title("💹 Forex Signals — EMA/RSI/MACD (with Time & History)")

pair = st.text_input("Enter Forex Pair (e.g., EURUSD=X):", "EURUSD=X")

# ✅ Only use safe intervals
valid_intervals = ["30m", "1h", "4h", "1d"]
interval = st.selectbox("Select Interval:", valid_intervals)
period = st.selectbox("Select Time Period:", ["5d", "1mo", "3mo"])

if st.button("Get Signals"):
    try:
        data = yf.download(pair, period=period, interval=interval)

        if data.empty:
            st.error("⚠️ No data found. Try another pair or interval.")
        else:
            # Fix multiindex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            # Keep timestamp
            data.index = pd.to_datetime(data.index)

            df = compute_indicators(data)
            df = generate_signals(df)

            # Latest signal
            latest = df.iloc[-1]
            st.subheader("📊 Latest Signal")
            st.write(f"**{latest['Signal']}** at {latest.name} (Price: {latest['Close']:.5f})")

            # Chart
            st.plotly_chart(plot_candles(df, pair), use_container_width=True)

            # Signal history
            st.subheader("📑 Recent Signals")
            signal_history = df[df["Signal"] != "Neutral"][["Close", "Signal"]].tail(20)
            st.dataframe(signal_history)

    except Exception as e:
        st.error(f"Error: {e}")
