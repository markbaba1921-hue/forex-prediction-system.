import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------- App setup -------------
st.set_page_config(page_title="Forex Pro — Live Signals", layout="wide")
st.title("⚡ Forex Pro — Live Signals (Stable + ML + News)")

APP_TZ = "Africa/Addis_Ababa"  # change if you prefer

# ------------- Helpers -------------
@st.cache_data(ttl=30)
def fetch_prices(pair: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(pair, interval=interval, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    # timezones
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

def tech_signals(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    bull = (s["EMA20"] > s["EMA50"]) & (s["RSI14"] < 70) & (s["MACD"] > s["MACD_signal"])
    bear = (s["EMA20"] < s["EMA50"]) & (s["RSI14"] > 30) & (s["MACD"] < s["MACD_signal"])
    cross_up = (s["EMA20"] > s["EMA50"]) & (s["EMA20"].shift(1) <= s["EMA50"].shift(1))
    cross_dn = (s["EMA20"] < s["EMA50"]) & (s["EMA20"].shift(1) >= s["EMA50"].shift(1))
    vol_ok = s["ATR14"] / s["Close"] > 0.0005

    s["TechSignal"] = "WAIT"
    s.loc[(bull | cross_up) & vol_ok, "TechSignal"] = "BUY"
    s.loc[(bear | cross_dn) & vol_ok, "TechSignal"] = "SELL"

    # confidence score 0..1 (simple voting)
    s["tech_score"] = (
        (s["EMA20"] > s["EMA50"]).astype(int)
        + (s["RSI14"] < 70).astype(int)
        + (s["RSI14"] > 30).astype(int)
        + (s["MACD"] > s["MACD_signal"]).astype(int)
        + (s["ATR14"] / s["Close"] > 0.0005).astype(int)
    ) / 5.0
    return s

# -------- News sentiment ----------
analyzer = SentimentIntensityAnalyzer()

@st.cache_data(ttl=300)
def fetch_news_sentiment(pair: str):
    """
    Get Yahoo Finance RSS for the base currency (e.g., 'EUR', 'USD'),
    score headlines with VADER, return mean sentiment (-1..1) and last headlines.
    """
    # crude mapping from 'EURUSD=X' -> ['EUR','USD']
    bases = []
    if "=" in pair:
        base = pair.split("=")[0].upper()
        if len(base) >= 6:
            b1, b2 = base[:3], base[3:6]
            bases = [b1, b2]
    if not bases:
        bases = ["USD"]

    headlines = []
    scores = []

    for code in bases:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={code}%3DX&region=US&lang=en-US"
        try:
            d = feedparser.parse(url)
            for e in d.entries[:10]:
                title = e.title
                if title:
                    s = analyzer.polarity_scores(title)["compound"]
                    scores.append(s)
                    headlines.append((title, s))
        except Exception:
            continue

    if len(scores) == 0:
        return 0.0, []

    avg = float(np.mean(scores))
    # normalize to 0..1
    norm = (avg + 1) / 2.0
    return norm, headlines[:15]

# -------- ML model (light & fast) --------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret1"] = x["Close"].pct_change()
    x["ret3"] = x["Close"].pct_change(3)
    x["ret6"] = x["Close"].pct_change(6)
    x["ema_slope"] = x["EMA20"] - x["EMA20"].shift(3)
    x["macd_hist_3"] = x["MACD_hist"].rolling(3).mean()
    x["rsi_slope"] = x["RSI14"] - x["RSI14"].shift(3)
    x["vol"] = x["ATR14"] / x["Close"]
    x["future_ret"] = x["Close"].shift(-1) / x["Close"] - 1.0
    x["y"] = (x["future_ret"] > 0).astype(int)  # 1 up, 0 down
    feats = ["ret1","ret3","ret6","ema_slope","macd_hist_3","rsi_slope","vol",
             "EMA20","EMA50","RSI14","MACD","MACD_signal","ATR14"]
    x = x.dropna()
    return x, feats

def train_and_predict(xdf: pd.DataFrame, feats):
    if len(xdf) < 200:
        return 0.5, None  # not enough data, neutral
    train = xdf.iloc[:-1]
    test_last = xdf.iloc[-1:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42))
    ])
    model.fit(train[feats], train["y"])
    proba_up = float(model.predict_proba(test_last[feats])[:,1][0])  # up probability
    return proba_up, model

# ------------- UI controls -------------
pair = st.text_input("Forex Pair (Yahoo format)", "EURUSD=X")
c1, c2, c3, c4 = st.columns([1,1,1,1.2])
with c1:
    interval = st.selectbox("Interval", ["1m","5m","15m","30m","60m","1d"], index=1)
with c2:
    period = st.selectbox("History", ["1d","5d","1mo","3mo","6mo","1y"], index=1)
with c3:
    refresh_sec = st.slider("Auto-refresh (sec)", 5, 120, 15, step=5)
with c4:
    shown = st.slider("Show last N candles", 150, 4000, 700, step=50)

st_autorefresh(interval=refresh_sec*1000, key="auto_refresh_v2")

# ------------- Data pipeline -------------
raw = fetch_prices(pair, interval, period)
if raw.empty or len(raw) < 80:
    st.error("No or insufficient data. Try different pair, longer history, or slower interval.")
    st.stop()

ind = add_indicators(raw)
tech = tech_signals(ind)
feat_df, feat_cols = build_features(tech)
proba_up, model = train_and_predict(feat_df, feat_cols)

# News sentiment (0..1)
news_score, headlines = fetch_news_sentiment(pair)

# Latest
latest = tech.iloc[-1]
ts = latest.name.strftime("%Y-%m-%d %H:%M:%S %Z")

# Composite score: tech (weight 0.4) + ML (0.4; mapped 0..1) + news (0.2)
tech_component = float(latest["tech_score"])
ml_component = float(proba_up)            # 0..1 up probability
news_component = float(news_score)        # 0..1 positive

composite = 0.4*tech_component + 0.4*ml_component + 0.2*news_component

if composite >= 0.58:
    action = "BUY"
elif composite <= 0.42:
    action = "SELL"
else:
    action = "WAIT"

# ------------- Live Signal box -------------
st.subheader("📢 Live Signal (latest candle)")

box = st.container()
if action == "BUY":
    box.success(
        f"✅ **BUY NOW** — {ts} | Price: {latest['Close']:.5f}  | "
        f"Tech={tech_component:.2f}  ML={ml_component:.2f}  News={news_component:.2f}  → Score={composite:.2f}"
    )
elif action == "SELL":
    box.error(
        f"❌ **SELL NOW** — {ts} | Price: {latest['Close']:.5f}  | "
        f"Tech={tech_component:.2f}  ML={ml_component:.2f}  News={news_component:.2f}  → Score={composite:.2f}"
    )
else:
    box.info(
        f"⏳ **WAIT** — {ts} | Price: {latest['Close']:.5f}  | "
        f"Tech={tech_component:.2f}  ML={ml_component:.2f}  News={news_component:.2f}  → Score={composite:.2f}"
    )

# ------------- Stable Plotly chart -------------
st.subheader("📈 Price Chart (stable, zoom preserved)")

view = tech.tail(shown).copy()
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
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["RSI14"], name="RSI14", mode="lines"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Bar(x=plot_df["t_plot"], y=plot_df["MACD_hist"], name="MACD hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["MACD"], name="MACD", mode="lines"), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["MACD_signal"], name="Signal", mode="lines"), row=3, col=1)

# Signal markers
sig_pts = plot_df[(plot_df["EMA20"].notna()) & (plot_df.index.isin(tech.index[tech["TechSignal"].isin(["BUY","SELL"])]))]
if not sig_pts.empty:
    marks = tech.loc[sig_pts.index, "TechSignal"]
    fig.add_trace(go.Scatter(
        x=sig_pts["t_plot"],
        y=sig_pts["Close"],
        mode="markers+text",
        text=marks,
        textposition="top center",
        marker=dict(symbol=["triangle-up" if s=="BUY" else "triangle-down" for s in marks],
                    size=12),
        name="Tech Signals"
    ), row=1, col=1)

fig.update_layout(
    height=780,
    xaxis_rangeslider_visible=False,
    uirevision=f"{pair}_{interval}_stable_v2",   # keeps your zoom on refresh
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=10, r=10, t=30, b=10)
)
st.plotly_chart(fig, use_container_width=True)

# ------------- Recent signals -------------
st.subheader("🧾 Recent BUY/SELL (exact time)")
recent = tech[tech["TechSignal"].isin(["BUY","SELL"])][["TechSignal","Close"]].tail(40).copy()
if not recent.empty:
    recent["Time"] = recent.index.strftime("%Y-%m-%d %H:%M:%S %Z")
    st.dataframe(recent.rename(columns={"TechSignal":"Signal","Close":"Price"})[["Time","Signal","Price"]])
else:
    st.info("No recent technical signals in the shown window.")

# ------------- News panel -------------
with st.expander("📰 Latest headlines & sentiment"):
    if headlines:
        news_df = pd.DataFrame(headlines, columns=["Headline","Sentiment (-1..1)"])
        st.dataframe(news_df)
    else:
        st.write("No headlines available right now.")

st.caption("Signals are for educational use only — not financial advice.")
