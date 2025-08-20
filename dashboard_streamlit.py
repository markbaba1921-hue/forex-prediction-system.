import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# -------------------------------
# Technical Indicator Functions
# -------------------------------
def compute_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI (clean 1D version)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df

# -------------------------------
# Signal Generation
# -------------------------------
def generate_signals(df):
    df["signal"] = "HOLD"
    df.loc[
        (df["EMA20"] > df["EMA50"]) & (df["RSI14"] < 70) & (df["MACD_hist"] > 0),
        "signal"
    ] = "BUY"

    df.loc[
        (df["EMA20"] < df["EMA50"]) & (df["RSI14"] > 30) & (df["MACD_hist"] < 0),
        "signal"
    ] = "SELL"

    df["conf_score"] = 0.0
    df.loc[df["signal"] == "BUY", "conf_score"] = 0.7
    df.loc[df["signal"] == "SELL", "conf_score"] = 0.7
    return df

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Forex Signals", layout="wide")
st.title("💹 Forex Signals — EMA/RSI/MACD")

pair = st.text_input("Enter Forex Pair (Yahoo Finance format, e.g. EURUSD=X):", "EURUSD=X")
period = st.selectbox("Select Time Period:", ["1mo", "3mo", "6mo", "1y"])
interval = st.selectbox("Select Interval:", ["15m", "30m", "1h", "1d"])

if st.button("Get Signals"):
    try:
        data = yf.download(pair, period=period, interval=interval)
        if data.empty:
            st.error("⚠️ No data found. Try another pair or interval.")
        else:
            df = compute_indicators(data)
            sig = generate_signals(df)

            st.subheader("📊 Price & Indicators (latest)")
            st.line_chart(df[["Close", "EMA20", "EMA50"]])

            # -------------------------------
            # Now: latest recommendation (safe version)
            # -------------------------------
            if sig.empty:
                st.warning("No signal data available yet. Try longer lookback.")
            else:
                latest = sig.iloc[-1]
                price_now = float(latest["Close"])
                rsi_now = float(latest["RSI14"])
                macd_hist_now = float(latest["MACD_hist"])
                signal_now = str(latest.get("signal", "HOLD"))
                conf_now = float(latest.get("conf_score", 0))

                st.subheader("📢 Recommendation (now)")

                if signal_now == "BUY":
                    st.success(
                        f"✅ BUY at {price_now:.5f}  | RSI={rsi_now:.1f}, "
                        f"MACD_hist={macd_hist_now:.5f}, Confidence={conf_now:.2f}"
                    )
                elif signal_now == "SELL":
                    st.error(
                        f"❌ SELL at {price_now:.5f} | RSI={rsi_now:.1f}, "
                        f"MACD_hist={macd_hist_now:.5f}, Confidence={conf_now:.2f}"
                    )
                else:
                    st.info(
                        f"⏳ WAIT  | Price={price_now:.5f}, RSI={rsi_now:.1f}, "
                        f"MACD_hist={macd_hist_now:.5f}"
                    )

            # Show signals table
            st.subheader("📑 Signals Table")
            st.dataframe(sig.tail(20))

    except Exception as e:
        st.error(f"Error: {str(e)}")
