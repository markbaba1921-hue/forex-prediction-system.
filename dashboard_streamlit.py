import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------- Settings --------
APP_TZ = "Africa/Addis_Ababa"   # show times in your local timezone

# -------- Helpers --------
def clamp_period_for_interval(interval: str, period: str) -> tuple[str, str | None]:
    """
    Yahoo limits how much history you can fetch for small intervals.
    We gently correct the period if it would return no data.
    """
    note = None
    if interval == "1m" and period not in {"1d", "5d", "7d"}:
        note = "Interval 1m supports up to ~7 days. Period auto-set to 5d."
        return "5d", note
    if interval in {"5m", "15m", "30m"} and period not in {"5d", "1mo", "3mo"}:
        note = f"Interval {interval} works best up to ~3 months. Period auto-set to 1mo."
        return "1mo", note
    if interval in {"60m"} and period not in {"5d", "1mo", "3mo"}:
        note = "Interval 60m works best up to ~3 months. Period auto-set to 3mo."
        return "3mo", note
    return period, note

def localize_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(tz)
    return df

# -------- Indicators --------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()

    rsi = ta.momentum.RSIIndicator(out["Close"], window=14)
    out["RSI14"] = rsi.rsi()

    macd = ta.trend.MACD(out["Close"])
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_hist"] = macd.macd_diff()

    # Clean NaNs produced by rolling/ewm warmup
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

# -------- Signals --------
def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    buy_now  = (s["EMA20"] > s["EMA50"]) & (s["RSI14"] < 70) & (s["MACD"] > s["MACD_signal"])
    sell_now = (s["EMA20"] < s["EMA50"]) & (s["RSI14"] > 30) & (s["MACD"] < s["MACD_signal"])

    s["Signal"] = "Neutral"
    s.loc[buy_now,  "Signal"] = "BUY"
    s.loc[sell_now, "Signal"] = "SELL"

    s["PrevSignal"] = s["Signal"].shift(1).fillna("Neutral")
    s["NewSignal"] = (s["Signal"] != "Neutral") & (s["Signal"] != s["PrevSignal"])
    return s

# -------- Chart --------
def build_chart(view: pd.DataFrame, pair: str, uirevision_key: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2]
    )

    # Row 1: Candles + EMAs + markers
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        name="Candles"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=view.index, y=view["EMA20"], name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["EMA50"], name="EMA50"), row=1, col=1)

    buys = view[view["Signal"] == "BUY"]
    sells = view[view["Signal"] == "SELL"]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        mode="markers", name="BUY",
        marker=dict(symbol="triangle-up", size=12, color="green")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        mode="markers", name="SELL",
        marker=dict(symbol="triangle-down", size=12, color="red")
    ), row=1, col=1)

    # Row 2: RSI
    fig.add_trace(go.Scatter(x=view.index, y=view["RSI14"], name="RSI14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Row 3: MACD
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_hist"], name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD_signal"], name="Signal"), row=3, col=1)

    fig.update_layout(
        title=f"{pair} — Live Signals",
        xaxis_rangeslider_visible=False,
        height=700,
        uirevision=uirevision_key,   # <- keeps your zoom/pan across refreshes
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -------- UI --------
st.set_page_config(page_title="Forex Live Signals (Stable)", layout="wide")
st.title("⚡ Forex Live Signals — stable zoom + real-time ‘NOW’ signal")

pair = st.text_input("Forex Pair (Yahoo format)", "EURUSD=X")
interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m", "1d"], index=2)

# Period options adapt to interval
default_periods = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["5d", "1mo", "3mo"],
    "15m": ["5d", "1mo", "3mo"],
    "30m": ["5d", "1mo", "3mo"],
    "60m": ["5d", "1mo", "3mo"],
    "1d": ["1mo", "3mo", "6mo", "1y"]
}
period_choices = default_periods.get(interval, ["5d", "1mo", "3mo"])
period = st.selectbox("History", period_choices, index=0)

colA, colB, colC = st.columns([1,1,1])
with colA:
    last_n = st.slider("Show last N candles", 100, 2000, 400, step=50)
with colB:
    live_mode = st.toggle("Live Mode (auto-refresh)", value=True, help="Turn off to keep zoom 100% stable")
with colC:
    refresh_sec = st.slider("Refresh seconds", 10, 120, 30, step=5)

# Optional auto-refresh (without flicker). If missing, just skip.
if live_mode:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_key")
    except Exception:
        st.info("Install 'streamlit-autorefresh' for auto-refresh. It’s already listed in requirements.txt.")

# Fetch data (with safe period clamp)
safe_period, note = clamp_period_for_interval(interval, period)
if note:
    st.caption(f"ℹ️ {note}")

data = yf.download(pair, period=safe_period, interval=interval, auto_adjust=True, progress=False)

if data is None or data.empty:
    st.error("No data returned. Try a different pair/interval or shorter history.")
    st.stop()

# Flatten MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [c[0] for c in data.columns]

# Localize to your timezone for exact time display
data = localize_index(data, APP_TZ)

# Compute indicators & signals
df = compute_indicators(data)
if df.empty or len(df) < 30:
    st.warning("Not enough data after indicators. Increase history.")
    st.stop()

df = add_signals(df)

# "NOW" signal on the latest candle
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) >= 2 else latest
now_sig = latest["Signal"]
new_now = bool(latest["NewSignal"])

st.subheader("📢 Live signal (latest candle)")
if now_sig == "BUY":
    if new_now:
        st.success(f"✅ BUY **NOW**  — {latest.name} | Price: {latest['Close']:.5f}")
    else:
        st.success(f"✅ BUY (conditions still valid) — {latest.name} | Price: {latest['Close']:.5f}")
elif now_sig == "SELL":
    if new_now:
        st.error(f"❌ SELL **NOW** — {latest.name} | Price: {latest['Close']:.5f}")
    else:
        st.error(f"❌ SELL (conditions still valid) — {latest.name} | Price: {latest['Close']:.5f}")
else:
    st.info(f"⏳ WAIT — {latest.name} | Price: {latest['Close']:.5f}")

# Chart (stable zoom thanks to uirevision)
view = df.tail(last_n)
fig = build_chart(view, pair, uirevision_key=f"{pair}-{interval}-{last_n}")
st.plotly_chart(fig, use_container_width=True)

# Recent signals table with exact timestamps
st.subheader("🧾 Recent signals (timestamped)")
sig_hist = df[df["Signal"] != "Neutral"][["Signal", "Close"]].tail(30)
sig_hist = sig_hist.rename(columns={"Close":"Price"})
if len(sig_hist) == 0:
    st.write("No signals in the visible period. Try more history or a different interval.")
else:
    st.dataframe(sig_hist)

# Small stats
st.caption(
    "Signals are generated from EMA20/EMA50 + RSI(14) + MACD cross rules. "
    "‘NOW’ means the latest candle currently meets conditions. "
    "Zoom is preserved across refreshes via Plotly’s uirevision."
)
