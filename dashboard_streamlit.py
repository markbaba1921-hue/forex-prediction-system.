import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
import pytz
import ta
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------
# Page
# --------------------------
st.set_page_config(page_title="Forex Live Signals (Stable)", layout="wide")
st.title("⚡ Forex Live Signals — Stable Chart, Precise Time, Real-Time Alerts")

# Your display timezone (change if you like)
APP_TZ = "Africa/Addis_Ababa"

# --------------------------
# Data
# --------------------------
@st.cache_data(ttl=30)
def fetch_prices(pair: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch from Yahoo Finance; flatten columns; localize timezone."""
    df = yf.download(pair, interval=interval, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    # Index to tz-aware, then convert to desired TZ
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(APP_TZ)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    rsi = ta.momentum.RSIIndicator(x["Close"], window=14)
    x["RSI14"] = rsi.rsi()
    macd = ta.trend.MACD(x["Close"])
    x["MACD"] = macd.macd()
    x["MACD_signal"] = macd.macd_signal()
    x["MACD_hist"] = macd.macd_diff()
    atr = ta.volatility.AverageTrueRange(x["High"], x["Low"], x["Close"], window=14)
    x["ATR14"] = atr.average_true_range()
    return x.replace([np.inf, -np.inf], np.nan).dropna()

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    bull = (s["EMA20"] > s["EMA50"]) & (s["RSI14"] < 70) & (s["MACD"] > s["MACD_signal"])
    bear = (s["EMA20"] < s["EMA50"]) & (s["RSI14"] > 30) & (s["MACD"] < s["MACD_signal"])
    cross_up = (s["EMA20"] > s["EMA50"]) & (s["EMA20"].shift(1) <= s["EMA50"].shift(1))
    cross_dn = (s["EMA20"] < s["EMA50"]) & (s["EMA20"].shift(1) >= s["EMA50"].shift(1))
    vol_ok = s["ATR14"] / s["Close"] > 0.0005

    s["Signal"] = "WAIT"
    s.loc[(bull | cross_up) & vol_ok, "Signal"] = "BUY"
    s.loc[(bear | cross_dn) & vol_ok, "Signal"] = "SELL"

    s["PrevSignal"] = s["Signal"].shift(1).fillna("WAIT")
    s["NewSignal"] = (s["Signal"] != "WAIT") & (s["Signal"] != s["PrevSignal"])

    # confidence score 0..1 (how many techs agree)
    s["tech_score"] = (
        (s["EMA20"] > s["EMA50"]).astype(int)
        + (s["RSI14"] < 70).astype(int)
        + (s["RSI14"] > 30).astype(int)
        + (s["MACD"] > s["MACD_signal"]).astype(int)
        + (s["ATR14"] / s["Close"] > 0.0005).astype(int)
    ) / 5.0

    return s

# --------------------------
# Sidebar / Controls
# --------------------------
pair = st.text_input("Forex Pair (Yahoo format)", "EURUSD=X")
colA, colB, colC, colD = st.columns([1,1,1,1.2])
with colA:
    interval = st.selectbox("Interval", ["1m","5m","15m","30m","60m","1d"], index=1)
with colB:
    period = st.selectbox("History", ["1d","5d","1mo","3mo","6mo","1y"], index=1)
with colC:
    refresh_sec = st.slider("Auto-refresh (sec)", 5, 120, 15, step=5)
with colD:
    shown = st.slider("Show last N candles", 150, 3000, 600, step=50)

# soft auto-refresh (preserves state)
st_autorefresh(interval=refresh_sec*1000, key="auto_refresh")

# --------------------------
# Pipeline
# --------------------------
raw = fetch_prices(pair, interval, period)
if raw.empty or len(raw) < 60:
    st.error("No/insufficient data. Try different pair, longer history, or slower interval.")
    st.stop()

ind = add_indicators(raw)
df = add_signals(ind)
view = df.tail(shown)

# Latest bar (exact time)
latest = df.iloc[-1]
ts = latest.name.strftime("%Y-%m-%d %H:%M:%S %Z")

# --------------------------
# Live Signal box
# --------------------------
st.subheader("📢 Live Signal (latest candle)")
if latest["Signal"] == "BUY":
    msg = "✅ **BUY NOW**" if latest["NewSignal"] else "✅ BUY (still valid)"
    st.success(f"{msg} — {ts} | Price: {latest['Close']:.5f} | Tech={latest['tech_score']:.2f}")
elif latest["Signal"] == "SELL":
    msg = "❌ **SELL NOW**" if latest["NewSignal"] else "❌ SELL (still valid)"
    st.error(f"{msg} — {ts} | Price: {latest['Close']:.5f} | Tech={latest['tech_score']:.2f}")
else:
    st.info(f"⏳ WAIT — {ts} | Price: {latest['Close']:.5f} | Tech={latest['tech_score']:.2f}")

# --------------------------
# Stable Chart (Plotly)
# --------------------------
st.subheader("📈 Price Chart (stable, zoom preserved)")

# Plotly doesn't like tz-aware index on x, make a naive-local copy for plotting
plot_df = view.copy()
plot_df["t_plot"] = plot_df.index.tz_convert(APP_TZ).tz_localize(None)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.62, 0.18, 0.20], vertical_spacing=0.03)

# Candles
fig.add_trace(
    go.Candlestick(
        x=plot_df["t_plot"], open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="Candles"
    ),
    row=1, col=1
)

# EMAs
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["EMA20"], name="EMA20", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["EMA50"], name="EMA50", mode="lines"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["RSI14"], name="RSI14"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Bar(x=plot_df["t_plot"], y=plot_df["MACD_hist"], name="MACD hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["MACD"], name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["MACD_signal"], name="Signal"), row=3, col=1)

# Signal markers (arrows)
sig_pts = plot_df[plot_df["Signal"].isin(["BUY","SELL"])].copy()
if not sig_pts.empty:
    fig.add_trace(go.Scatter(
        x=sig_pts["t_plot"],
        y=sig_pts["Close"],
        mode="markers+text",
        text=sig_pts["Signal"],
        textposition="top center",
        marker=dict(symbol=["triangle-up" if s=="BUY" else "triangle-down" for s in sig_pts["Signal"]],
                    size=12),
        name="Signals"
    ), row=1, col=1)

# Layout: keep zoom/pan stable between refreshes
fig.update_layout(
    height=760,
    xaxis_rangeslider_visible=False,
    uirevision=f"{pair}_{interval}_v1",
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=10, r=10, t=30, b=10)
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Recent Signals table
# --------------------------
st.subheader("🧾 Recent Signals (exact time)")
hist = df[df["Signal"].isin(["BUY","SELL"])][["Signal","Close"]].tail(40).copy()
if not hist.empty:
    hist["Time"] = hist.index.strftime("%Y-%m-%d %H:%M:%S %Z")
    hist = hist[["Time","Signal","Close"]].rename(columns={"Close":"Price"})
st.dataframe(hist if not hist.empty else pd.DataFrame({"Info":["No signals yet in the shown window."]}))
