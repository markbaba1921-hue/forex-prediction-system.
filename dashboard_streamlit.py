import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objs as go
import time

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
    df["MACD_Hist"] = macd.macd_diff()

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
# Plot advanced chart
# ----------------------------
def plot_chart(df, pair):
    fig = go.Figure()

    # 🕯️ Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Candles"
    ))

    # EMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color="blue", width=1), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="orange", width=1), name="EMA50"))

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

    fig.update_layout(title=f"{pair} Price & Signals", xaxis_rangeslider_visible=False, height=600)
    return fig


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Forex Signals Live", layout="wide")
st.title("⚡ Live Forex Signals — EMA/RSI/MACD")

pair = st.text_input("Enter Forex Pair (e.g., EURUSD=X):", "EURUSD=X")
interval = st.selectbox("Interval:", ["5m", "15m", "30m", "1h", "4h", "1d"])
period = st.selectbox("Period:", ["5d", "1mo", "3mo"])

refresh_rate = st.slider("Auto-refresh every X seconds:", 10, 300, 60)

placeholder = st.empty()

while True:
    try:
        data = yf.download(pair, period=period, interval=interval)

        if data.empty:
            placeholder.error("⚠️ No data found. Try another pair/interval.")
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            df = compute_indicators(data)
            df = generate_signals(df)

            latest = df.iloc[-1]

            with placeholder.container():
                st.subheader("📊 Latest Signal")
                st.write(f"**{latest['Signal']}** at {latest.name} (Price: {latest['Close']:.5f})")

                st.plotly_chart(plot_chart(df, pair), use_container_width=True)

                col1, col2 = st.columns(2)

                # RSI chart
                with col1:
                    st.subheader("RSI (14)")
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], line=dict(color="purple"), name="RSI14"))
                    rsi_fig.add_hline(y=70, line=dict(color="red", dash="dash"))
                    rsi_fig.add_hline(y=30, line=dict(color="green", dash="dash"))
                    rsi_fig.update_layout(height=250)
                    st.plotly_chart(rsi_fig, use_container_width=True)

                # MACD chart
                with col2:
                    st.subheader("MACD")
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="blue"), name="MACD"))
                    macd_fig.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], line=dict(color="orange"), name="Signal"))
                    macd_fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color="gray"))
                    macd_fig.update_layout(height=250)
                    st.plotly_chart(macd_fig, use_container_width=True)

                # Table of recent signals
                st.subheader("📑 Recent Signals")
                st.dataframe(df[df["Signal"] != "Neutral"][["Close", "Signal"]].tail(20))

        time.sleep(refresh_rate)

    except Exception as e:
        placeholder.error(f"Error: {e}")
        time.sleep(refresh_rate)
