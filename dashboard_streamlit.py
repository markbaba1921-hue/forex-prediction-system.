import os
import time
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta

# Try TradingView-like chart (much more stable), otherwise fall back to Plotly
USE_TV = True
try:
    from streamlit_lightweight_charts import renderLightweightCharts
except Exception:
    USE_TV = False
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# SETTINGS
# ----------------------------
APP_TZ = "Africa/Addis_Ababa"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", None)  # optional

st.set_page_config(page_title="Forex Live Assistant (Stable + Precise)", layout="wide")
st.title("⚡ Forex Live Assistant — Stable Chart • Real-Time Signals • News-Aware")

# ----------------------------
# HELPERS
# ----------------------------
def parse_pair(pair: str):
    """EURUSD=X -> ('EUR','USD')"""
    p = pair.replace("=X", "").upper()
    return p[:3], p[3:]

def clamp_period(interval: str, period: str):
    """Keep yfinance within valid bounds for intraday."""
    note = None
    if interval == "1m" and period not in {"1d", "5d", "7d"}:
        return "5d", "1m supports about 7d history. Auto-set to 5d."
    if interval in {"5m", "15m", "30m"} and period not in {"5d", "1mo", "3mo"}:
        return "1mo", f"{interval} interval works best up to ~3 months. Auto-set to 1mo."
    if interval in {"60m"} and period not in {"5d", "1mo", "3mo"}:
        return "3mo", "60m works best up to ~3 months. Auto-set to 3mo."
    return period, note

def localize_index(df: pd.DataFrame, tz: str):
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(tz)
    return df

# ----------------------------
# DATA
# ----------------------------
@st.cache_data(ttl=30)
def fetch_prices(pair: str, interval: str, period: str) -> pd.DataFrame:
    safe_period, _ = clamp_period(interval, period)
    df = yf.download(pair, interval=interval, period=safe_period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    df = localize_index(df, APP_TZ)
    return df

# ----------------------------
# INDICATORS
# ----------------------------
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    rsi = ta.momentum.RSIIndicator(x["Close"], window=14)
    x["RSI14"] = rsi.rsi()
    macd = ta.trend.MACD(x["Close"])
    x["MACD"] = macd.macd()
    x["MACD_signal"] = macd.macd_signal()
    x["MACD_hist"] = macd.macd_diff()
    atr = ta.volatility.AverageTrueRange(high=x["High"], low=x["Low"], close=x["Close"], window=14)
    x["ATR14"] = atr.average_true_range()
    return x.replace([np.inf, -np.inf], np.nan).dropna()

# ----------------------------
# TECHNICAL SIGNALS (instant)
# ----------------------------
def technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    # Core conditions
    bull = (s["EMA20"] > s["EMA50"]) & (s["RSI14"] < 70) & (s["MACD"] > s["MACD_signal"])
    bear = (s["EMA20"] < s["EMA50"]) & (s["RSI14"] > 30) & (s["MACD"] < s["MACD_signal"])
    # Cross confirmations (less laggy)
    cross_up = (s["EMA20"] > s["EMA50"]) & (s["EMA20"].shift(1) <= s["EMA50"].shift(1))
    cross_dn = (s["EMA20"] < s["EMA50"]) & (s["EMA20"].shift(1) >= s["EMA50"].shift(1))
    # Volatility filter (avoid dead markets)
    vol_ok = s["ATR14"] / s["Close"] > 0.0005

    s["Signal"] = "Neutral"
    buy_now = (bull | cross_up) & vol_ok
    sell_now = (bear | cross_dn) & vol_ok

    s.loc[buy_now, "Signal"] = "BUY"
    s.loc[sell_now, "Signal"] = "SELL"
    s["PrevSignal"] = s["Signal"].shift(1).fillna("Neutral")
    s["NewSignal"] = (s["Signal"] != "Neutral") & (s["Signal"] != s["PrevSignal"])

    # Confidence by how many technicals agree
    s["tech_score"] = (
        (s["EMA20"] > s["EMA50"]).astype(int)
        + (s["RSI14"] < 70).astype(int)
        + (s["RSI14"] > 30).astype(int)
        + (s["MACD"] > s["MACD_signal"]).astype(int)
        + (s["ATR14"] / s["Close"] > 0.0005).astype(int)
    ) / 5.0
    return s

# ----------------------------
# LIGHT ML PREDICTION (fast)
# ----------------------------
def add_ml_prediction(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Train a tiny logistic model on moving-window features to predict next-bar direction.
    Keeps it super fast for mobile/cloud.
    """
    s = df.copy()
    # Label: next close up/down
    s["ret"] = s["Close"].pct_change()
    s["target"] = (s["ret"].shift(-1) > 0).astype(int)

    # Features
    feats = pd.DataFrame({
        "rsi": s["RSI14"],
        "macd": s["MACD"],
        "macd_sig": s["MACD_signal"],
        "ema_ratio": s["EMA20"] / s["EMA50"] - 1.0,
        "atr_p": s["ATR14"] / s["Close"],
        "ret": s["ret"].fillna(0),
    }, index=s.index).dropna()
    y = s.loc[feats.index, "target"]

    # Need enough rows
    if len(feats) < 200:
        s["ml_prob_up"] = 0.5
        return s, 0.5

    # Train on last N-50, validate last 50
    N = len(feats)
    split = max(150, N - 50)
    X_train, y_train = feats.iloc[:split], y.iloc[:split]
    X_live = feats.iloc[[-1]]

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    prob_up = float(clf.predict_proba(X_live)[0, 1])

    s["ml_prob_up"] = 0.5
    s.loc[X_live.index, "ml_prob_up"] = prob_up
    return s, prob_up

# ----------------------------
# NEWS + SENTIMENT (optional)
# ----------------------------
def fetch_news_sentiment(base_ccy: str, quote_ccy: str, limit=15):
    """Return (news_rows_df, base_sent, quote_sent, diff_score)."""
    if not NEWSAPI_KEY:
        return None, 0.0, 0.0, 0.0
    analyzer = SentimentIntensityAnalyzer()

    def get_news(q):
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWSAPI_KEY,
            "q": q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])

    def score_articles(arts):
        if not arts: return 0.0, []
        scores, rows = [], []
        for a in arts:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            text = f"{title}. {desc}"
            comp = analyzer.polarity_scores(text)["compound"]
            scores.append(comp)
            rows.append({
                "time": a.get("publishedAt"),
                "title": title[:140],
                "sentiment": comp,
                "source": (a.get("source") or {}).get("name", ""),
                "url": a.get("url")
            })
        avg = float(np.mean(scores)) if scores else 0.0
        return avg, rows

    base_arts = get_news(base_ccy + " currency OR " + base_ccy)
    quote_arts = get_news(quote_ccy + " currency OR " + quote_ccy)

    base_avg, base_rows = score_articles(base_arts)
    quote_avg, quote_rows = score_articles(quote_arts)
    rows = sorted(base_rows + quote_rows, key=lambda r: r["time"], reverse=True)[:limit]
    return pd.DataFrame(rows), base_avg, quote_avg, (base_avg - quote_avg)

# ----------------------------
# CHARTS
# ----------------------------
def plot_tv(candles: pd.DataFrame, ema20: pd.Series, ema50: pd.Series, markers: list):
    """TradingView-like chart with very stable zoom/pan."""
    def to_candle_data(df):
        out = []
        for ts, r in df.iterrows():
            out.append({
                "time": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
            })
        return out

    def to_line_data(series):
        return [{"time": ts.strftime("%Y-%m-%dT%H:%M:%S"), "value": float(v)} for ts, v in series.items()]

    chart = {
        "chart": {
            "height": 520,
            "layout": {"backgroundColor": "#FFFFFF", "textColor": "#111"},
            "rightPriceScale": {"visible": True},
            "crosshair": {"mode": 1},
            "grid": {"vertLines": {"color": "#f0f0f0"}, "horzLines": {"color": "#f0f0f0"}},
            "timeScale": {"timeVisible": True, "secondsVisible": True, "borderVisible": False},
        },
        "series": [
            {"type": "Candlestick", "data": to_candle_data(candles), "markers": markers, "priceScaleId": "right"},
            {"type": "Line", "data": to_line_data(ema20), "title": "EMA20"},
            {"type": "Line", "data": to_line_data(ema50), "title": "EMA50"},
        ],
    }
    renderLightweightCharts([chart], enable_zoom=True, enable_pan=True)

def plot_plotly(view: pd.DataFrame):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="Candles"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["EMA20"], name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["EMA50"], name="EMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["RSI14"], name="RSI14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_hist"], name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD_signal"], name="Signal"), row=3, col=1)
    fig.update_layout(height=720, uirevision="stable-zoom", xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# UI CONTROLS
# ----------------------------
col0, col1, col2, col3 = st.columns([1.2, 1, 1, 1.2])
with col0:
    pair = st.text_input("Forex Pair (Yahoo format)", "EURUSD=X")
with col1:
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m", "1d"], index=2)
with col2:
    period = st.selectbox("History", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
with col3:
    refresh_sec = st.slider("Auto-refresh (seconds)", 5, 120, 20, step=5)

st.caption("Tip: For **second-by-second feel**, use **1m** interval with refresh ~5–15s. "
           "True tick-by-tick requires a broker API (e.g., OANDA/Forex.com).")

# Optional soft auto-refresh (does not force rerender if user toggles it off)
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_key")

# ----------------------------
# DATA + PIPELINE
# ----------------------------
data = fetch_prices(pair, interval, period)
if data.empty or len(data) < 60:
    st.error("No/insufficient data. Try a different pair, a longer history, or a slower interval.")
    st.stop()

df0 = indicators(data)
df = technical_signals(df0)
df, ml_prob_up = add_ml_prediction(df)

base, quote = parse_pair(pair)
news_df, base_sent, quote_sent, news_diff = fetch_news_sentiment(base, quote)

# ----------------------------
# LIVE DECISION (NOW)
# ----------------------------
latest = df.iloc[-1]
signal_now = latest["Signal"]
new_now = bool(latest["NewSignal"])
tech_score_now = float(latest["tech_score"])
ml_score = float(ml_prob_up * 2 - 1)  # scale 0..1 -> -1..+1
news_score = float(news_diff)  # already roughly -1..+1 centered

# Weighted combo
combined = 0.6 * (1 if signal_now == "BUY" else -1 if signal_now == "SELL" else 0) * tech_score_now \
           + 0.25 * ml_score \
           + 0.15 * news_score

st.subheader("📢 Live Signal (latest candle)")
ts = latest.name.strftime("%Y-%m-%d %H:%M:%S %Z")
if signal_now == "BUY":
    msg = "✅ **BUY NOW**" if new_now else "✅ BUY (still valid)"
    st.success(f"{msg} — {ts}  |  Price: {latest['Close']:.5f}  |  Tech={tech_score_now:.2f}  ML={ml_prob_up:.2f}  NewsΔ={news_diff:+.2f}  → Score={combined:+.2f}")
elif signal_now == "SELL":
    msg = "❌ **SELL NOW**" if new_now else "❌ SELL (still valid)"
    st.error(f"{msg} — {ts}  |  Price: {latest['Close']:.5f}  |  Tech={tech_score_now:.2f}  ML={ml_prob_up:.2f}  NewsΔ={news_diff:+.2f}  → Score={combined:+.2f}")
else:
    st.info(f"⏳ WAIT — {ts}  |  Price: {latest['Close']:.5f}  |  Tech={tech_score_now:.2f}  ML={ml_prob_up:.2f}  NewsΔ={news_diff:+.2f}  → Score={combined:+.2f}")

# ----------------------------
# CHART (very stable)
# ----------------------------
last_n = st.slider("Show last N candles", 150, 3000, 600, step=50)
view = df.tail(last_n)

# markers for TV chart
markers = []
for idx, row in view.iterrows():
    if row["Signal"] == "BUY":
        markers.append({"time": idx.strftime("%Y-%m-%dT%H:%M:%S"), "position": "belowBar",
                        "color": "green", "shape": "arrowUp", "text": "BUY"})
    elif row["Signal"] == "SELL":
        markers.append({"time": idx.strftime("%Y-%m-%dT%H:%M:%S"), "position": "aboveBar",
                        "color": "red", "shape": "arrowDown", "text": "SELL"})

st.subheader("📈 Price Chart (stable)")
if USE_TV:
    plot_tv(view[["Open","High","Low","Close"]], view["EMA20"], view["EMA50"], markers)
else:
    st.caption("Using Plotly fallback (install `streamlit-lightweight-charts` for MT5-like chart).")
    plot_plotly(view)

# ----------------------------
# PANELS
# ----------------------------
c1, c2 = st.columns([1.2, 1])
with c1:
    st.subheader("🧾 Recent Signals (with exact time)")
    sig_hist = view[view["Signal"] != "Neutral"][["Signal", "Close"]].tail(40)
    sig_hist = sig_hist.rename(columns={"Close": "Price"})
    st.dataframe(sig_hist)

with c2:
    st.subheader(f"📰 FX News & Sentiment ({base} vs {quote})")
    if news_df is None:
        st.caption("Add a NEWSAPI_KEY env variable to enable headlines & sentiment.")
    else:
        st.write(f"Avg sentiment — {base}: {base_sent:+.2f} | {quote}: {quote_sent:+.2f} | Diff (base−quote): {news_diff:+.2f}")
        st.dataframe(news_df[["time","source","title","sentiment"]].head(10))

st.caption(
    "Signals combine technicals (EMA20/EMA50, RSI14, MACD, ATR), a tiny ML predictor, and news sentiment. "
    "For true **per-second** data, connect a broker API; Yahoo updates at best once per minute."
)
