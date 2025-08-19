import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title("📈 Forex Prediction System (Lite Version)")

# Select forex pair
pair = st.selectbox("Choose a Forex pair:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])

# Download last 60 days data
data = yf.download(pair, period="60d", interval="1h")

st.write("### Recent Data", data.tail())

# Plot price chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data["Close"], label="Close Price")
ax.set_title(f"{pair} Price Chart (Last 60 Days)")
ax.legend()
st.pyplot(fig)
