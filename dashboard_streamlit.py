import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forex Signals (Lite & Robust)", layout="wide")
st.title("💹 Forex Signals — EMA/RSI/MACD")

# -------------------------------
# Sidebar controls
# -------------------------------
pair = st.sidebar.selectbox("Forex pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"], index=0)
interval = st.sidebar.selectbox("Interval", ["15m", "30m", "1h", "1d"], index=2)
lookback = st.sidebar.selectbox("Lookback", ["30d", "60d", "90d", "180d"], index=2)

# Tunable thresholds (make them easier so you see signals)
rsi_buy_max = st.sidebar.slider("RSI max for BUY", 20, 60, 45, 1)
rsi_sell_min = st.sidebar.slider("RSI min for SELL", 40, 80, 55, 1)

st.sidebar.info("Tip: If no signals appear, widen lookback or relax RSI thresholds.")

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(ttl=300)
def load_data(_pair: str, _period: str, _interval: str) -> pd.DataFrame:
    df = yf.download(_pair, period=_period, interval=_interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    return df[["Close"]].dropna()

df = load_data(pair, lookback, interval)

if df.empty:
    st.error("No data returned. Try a different pair/interval or longer lookback.")
    st.stop()

# -------------------------------
# Indicators (robust implementations)
# -------------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # EMAs
    out["EMA_fast"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_slow"] = out["Close"].ewm(span=26, adjust=False).mean()
    # SMA (for context)
    out["SMA20"] = out["Close"].rolling(20).mean()
    # RSI & MACD
    out["RSI14"] = rsi(out["Close"], 14)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    # Clean
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

ind = add_indicators(df)
if ind.empty or len(ind) < 60:
    st.warning("Not enough clean data after indicators. Increase lookback.")
    st.stop()

# -------------------------------
# Signal engine (clear & explainable)
# -------------------------------
def generate_signals(ind: pd.DataFrame, rsi_buy_max: int, rsi_sell_min: int) -> pd.DataFrame:
    s = ind.copy()

    # Crossovers
    s["cross_up"] = (s["EMA_fast"] > s["EMA_slow"]) & (s["EMA_fast"].shift(1) <= s["EMA_slow"].shift(1))
    s["cross_dn"] = (s["EMA_fast"] < s["EMA_slow"]) & (s["EMA_fast"].shift(1) >= s["EMA_slow"].shift(1))

    # Conditions
    buy_cond  = s["cross_up"] & (s["RSI14"] <= rsi_buy_max) & (s["MACD_hist"] > 0)
    sell_cond = s["cross_dn"] & (s["RSI14"] >= rsi_sell_min) & (s["MACD_hist"] < 0)

    s["Buy_Signal"]  = buy_cond
    s["Sell_Signal"] = sell_cond
    s["signal"] = np.where(s["Buy_Signal"], "BUY", np.where(s["Sell_Signal"], "SELL", "HOLD"))

    # Confidence: how many conditions are met (0..3) / 3
    s["conf_score"] = 0.0
    s.loc[s["Buy_Signal"],  "conf_score"] = (
        (s["cross_up"].astype(int) + (s["RSI14"] <= rsi_buy_max).astype(int) + (s["MACD_hist"] > 0).astype(int)) / 3
    )
    s.loc[s["Sell_Signal"], "conf_score"] = (
        (s["cross_dn"].astype(int) + (s["RSI14"] >= rsi_sell_min).astype(int) + (s["MACD_hist"] < 0).astype(int)) / 3
    )
    return s

sig = generate_signals(ind, rsi_buy_max, rsi_sell_min)

# -------------------------------
# Now: latest recommendation
# -------------------------------
latest = sig.iloc[-1]
price_now = float(latest["Close"])
rsi_now = float(latest["RSI14"])
macd_hist_now = float(latest["MACD_hist"])
signal_now = latest["signal"]
conf_now = float(latest["conf_score"])

st.subheader("📢 Recommendation (now)")
if signal_now == "BUY":
    st.success(f"✅ BUY at {price_now:.5f}  | RSI={rsi_now:.1f}, MACD_hist={macd_hist_now:.5f}, Confidence={conf_now:.2f}")
elif signal_now == "SELL":
    st.error(f"❌ SELL at {price_now:.5f} | RSI={rsi_now:.1f}, MACD_hist={macd_hist_now:.5f}, Confidence={conf_now:.2f}")
else:
    st.info(f"⏳ WAIT  | Price={price_now:.5f}, RSI={rsi_now:.1f}, MACD_hist={macd_hist_now:.5f}")

# -------------------------------
# Last 5 signals (history)
# -------------------------------
hist = sig.loc[sig["signal"].isin(["BUY","SELL"]), ["Close","RSI14","MACD_hist","signal","conf_score"]].tail(5)
hist = hist.rename(columns={"Close":"Price","RSI14":"RSI","MACD_hist":"MACD_hist","conf_score":"Confidence"})
st.subheader("🧾 Last 5 signals")
if len(hist) == 0:
    st.write("No signals yet in this period. Try a longer lookback or relax thresholds.")
else:
    st.dataframe(hist)

# -------------------------------
# Chart with markers
# -------------------------------
st.subheader("📊 Chart")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sig.index, sig["Close"], label="Close")
ax.plot(sig.index, sig["EMA_fast"], label="EMA 12")
ax.plot(sig.index, sig["EMA_slow"], label="EMA 26")
# Markers
ax.scatter(sig.index[sig["Buy_Signal"]],  sig["Close"][sig["Buy_Signal"]],  marker="^", s=80, label="BUY",  color="green")
ax.scatter(sig.index[sig["Sell_Signal"]], sig["Close"][sig["Sell_Signal"]], marker="v", s=80, label="SELL", color="red")
ax.legend()
ax.set_title(f"{pair} — Price & Signals")
st.pyplot(fig)

# RSI panel
fig2, ax2 = plt.subplots(figsize=(10, 2))
ax2.plot(sig.index, sig["RSI14"], label="RSI14")
ax2.axhline(rsi_buy_max, linestyle="--")
ax2.axhline(rsi_sell_min, linestyle="--")
ax2.set_title("RSI")
st.pyplot(fig2)
