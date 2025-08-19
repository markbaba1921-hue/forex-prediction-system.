import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

st.title("📈 Advanced Forex Prediction Dashboard")

# Select forex pair
pair = st.selectbox("Choose a Forex pair:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])

# Download last 90 days data
data = yf.download(pair, period="90d", interval="1h")

# Technical indicators
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()
data["RSI"] = 100 - (100 / (1 + (data["Close"].diff().clip(lower=0).rolling(14).mean() /
                                 data["Close"].diff().clip(upper=0).abs().rolling(14).mean())))

# Generate signals
data["Buy_Signal"] = (data["RSI"] < 30) & (data["SMA_20"] > data["SMA_50"])
data["Sell_Signal"] = (data["RSI"] > 70) & (data["SMA_20"] < data["SMA_50"])

st.write("### Recent Data with Signals", data.tail())

# Plot chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data["Close"], label="Close Price", color="blue")
ax.plot(data.index, data["SMA_20"], label="SMA 20", color="red")
ax.plot(data.index, data["SMA_50"], label="SMA 50", color="green")

# Plot Buy/Sell markers
ax.scatter(data.index[data["Buy_Signal"]], data["Close"][data["Buy_Signal"]],
           label="Buy Signal", marker="^", color="green", alpha=1)
ax.scatter(data.index[data["Sell_Signal"]], data["Close"][data["Sell_Signal"]],
           label="Sell Signal", marker="v", color="red", alpha=1)

ax.set_title(f"{pair} Price + Buy/Sell Signals")
ax.legend()
st.pyplot(fig)

# RSI Plot
fig2, ax2 = plt.subplots(figsize=(10, 2))
ax2.plot(data.index, data["RSI"], label="RSI", color="purple")
ax2.axhline(70, linestyle="--", color="red")
ax2.axhline(30, linestyle="--", color="green")
ax2.set_title("Relative Strength Index (RSI)")
st.pyplot(fig2)
