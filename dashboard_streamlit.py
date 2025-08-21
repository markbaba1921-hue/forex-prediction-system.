# app.py
"""
Forex Pro — Stable Live Signals (Multiple pairs, Technical / ML / Composite, Bollinger, Supertrend)
Paste this entire file into your repo and run with Streamlit.
"""

import warnings
warnings.filterwarnings("ignore")

import time
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# App settings
# -------------------------
st.set_page_config(page_title="Forex Pro — Signals", layout="wide")
st.title("⚡ Forex Pro — Stable Live Signals (Technical | ML | Composite)")

APP_TZ = "Africa/Addis_Ababa"   # change to your timezone if needed

# -------------------------
# Utilities & helpers
# -------------------------
analyzer = SentimentIntensityAnalyzer()

def tz_localize_and_convert_index(df: pd.DataFrame, tz: str):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)
    return df

@st.cache_data(ttl=25)
def fetch_ohlc(pair: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch single-pair OHLC from yfinance, flatten columns, localize index to APP_TZ.
    """
    df = yf.download(tickers=pair, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    df = tz_localize_and_convert_index(df, APP_TZ)
    return df

def compute_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """Classic Supertrend implementation returning level and direction (+1 up, -1 down)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    hl2 = (high + low) / 2
    # True range
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(df)):
        if (basic_ub.iloc[i] < final_ub.iloc[i-1]) or (close.iloc[i-1] > final_ub.iloc[i-1]):
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]

        if (basic_lb.iloc[i] > final_lb.iloc[i-1]) or (close.iloc[i-1] < final_lb.iloc[i-1]):
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]

    # determine trend
    supertrend = final_ub.copy()
    trend = np.ones(len(df))  # 1 = up, -1 = down
    trend[0] = 1
    for i in range(1, len(df)):
        if close.iloc[i] > final_ub.iloc[i-1]:
            trend[i] = 1
        elif close.iloc[i] < final_lb.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        # supertrend line uses the appropriate band
        supertrend.iloc[i] = final_lb.iloc[i] if trend[i] == 1 else final_ub.iloc[i]

    out = pd.Series(supertrend, index=df.index)
    return out, pd.Series(trend, index=df.index)

def add_indicators(df: pd.DataFrame):
    """Add all indicators used: EMA20/50, RSI14, MACD, ATR, Bollinger, Supertrend"""
    x = df.copy()
    # EMA
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    # RSI
    x["RSI14"] = ta.momentum.rsi(x["Close"], window=14)
    # MACD - using ta
    macd = ta.trend.MACD(x["Close"])
    x["MACD"] = macd.macd()
    x["MACD_signal"] = macd.macd_signal()
    x["MACD_hist"] = x["MACD"] - x["MACD_signal"]
    # ATR
    x["ATR14"] = ta.volatility.average_true_range(x["High"], x["Low"], x["Close"], window=14)
    # Bollinger
    ma20 = x["Close"].rolling(window=20).mean()
    std20 = x["Close"].rolling(window=20).std()
    x["BB_mid"] = ma20
    x["BB_up"] = ma20 + 2 * std20
    x["BB_low"] = ma20 - 2 * std20
    x["BB_width"] = (x["BB_up"] - x["BB_low"]) / x["BB_mid"]
    # Supertrend
    try:
        st_line, st_dir = compute_supertrend(x, period=10, multiplier=3.0)
        x["Supertrend"] = st_line
        x["Supertrend_dir"] = st_dir
    except Exception:
        x["Supertrend"] = np.nan
        x["Supertrend_dir"] = 0
    return x.replace([np.inf, -np.inf], np.nan).dropna()

def tech_signal_logic(df: pd.DataFrame, threshold=0.6):
    """
    Determine technical signal + tech score by voting / confirmations:
    returns df with TechScore (0..1) and TechSignal {BUY,SELL,WAIT} and NewTech flag.
    """
    s = df.copy()
    cond_ema = s["EMA20"] > s["EMA50"]
    cond_rsi_buy = s["RSI14"] < 70
    cond_rsi_sell = s["RSI14"] > 30
    cond_macd = s["MACD"] > s["MACD_signal"]
    cond_super_up = s["Supertrend_dir"] == 1
    cond_super_dn = s["Supertrend_dir"] == -1
    cond_vol = (s["ATR14"] / s["Close"]) > 0.0004  # filter very quiet markets

    # Compute tech votes for buy/sell
    buy_votes = (cond_ema & cond_rsi_buy & cond_macd).astype(int) + cond_super_up.astype(int) + cond_vol.astype(int)
    sell_votes = (~cond_ema & cond_rsi_sell & (~cond_macd)).astype(int) + cond_super_dn.astype(int) + cond_vol.astype(int)

    max_votes = 5.0
    s["tech_score"] = (buy_votes - sell_votes) / max_votes  # -1 .. +1 sentiment (positive = buy)
    # normalized for threshold test: tech_conf = (buy_votes)/(max_votes)
    s["tech_conf_buy"] = (buy_votes / max_votes).clip(0,1)
    s["tech_conf_sell"] = (sell_votes / max_votes).clip(0,1)

    s["TechSignal"] = "WAIT"
    s.loc[s["tech_conf_buy"] >= threshold, "TechSignal"] = "BUY"
    s.loc[s["tech_conf_sell"] >= threshold, "TechSignal"] = "SELL"

    s["PrevTech"] = s["TechSignal"].shift(1).fillna("WAIT")
    s["NewTech"] = (s["TechSignal"] != "WAIT") & (s["TechSignal"] != s["PrevTech"])
    return s

# -------------------------
# ML Model (lightweight)
# -------------------------
def prepare_ml_features(df: pd.DataFrame):
    df2 = df.copy()
    df2["ret1"] = df2["Close"].pct_change(1)
    df2["ret3"] = df2["Close"].pct_change(3)
    df2["ema_diff"] = df2["EMA20"] - df2["EMA50"]
    df2["macd_hist"] = df2["MACD_hist"]
    df2["bbw"] = df2["BB_width"]
    df2["atr_ratio"] = df2["ATR14"] / df2["Close"]
    df2["rsi"] = df2["RSI14"]
    df2 = df2.dropna()
    # label: next bar up/down
    df2["future_ret"] = df2["Close"].shift(-1) / df2["Close"] - 1
    df2["y"] = (df2["future_ret"] > 0).astype(int)
    feats = ["ret1","ret3","ema_diff","macd_hist","bbw","atr_ratio","rsi"]
    return df2, feats

@st.cache_data(ttl=120)
def train_light_model(df_prepared, feats):
    # If not enough rows, return nothing
    if len(df_prepared) < 200:
        return None, 0.5
    train = df_prepared.iloc[:-50]
    X = train[feats]
    y = train["y"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42, n_jobs=1))
    ])
    model.fit(X, y)
    # quick validation
    valid = df_prepared.iloc[-50:-1]
    if len(valid) > 5:
        ytrue = valid["y"]
        ypred = model.predict(valid[feats])
        acc = float((ypred == ytrue).mean())
    else:
        acc = 0.5
    return model, acc

def predict_ml_prob(model, last_row, feats):
    if model is None:
        return 0.5
    X = last_row[feats].values.reshape(1, -1)
    return float(model.predict_proba(X)[0,1])

# -------------------------
# News sentiment (simple)
# -------------------------
def fetch_news_sentiment(pair: str):
    """
    Use Yahoo RSS for the currencies in the pair to get rough sentiment.
    Returns normalized score 0..1 (0 negative, 1 positive).
    """
    try:
        # parse pair: EURUSD=X -> EUR, USD
        base = pair.split("=")[0][:6]
        if len(base) >= 6:
            a = base[:3]
            b = base[3:6]
        else:
            a = base
            b = base
        s = []
        for cur in [a, b]:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={cur}%3DX&region=US&lang=en-US"
            d = feedparser.parse(url)
            for e in (d.entries or [])[:8]:
                title = e.get("title","")
                if title:
                    s.append(analyzer.polarity_scores(title)["compound"])
        if not s:
            return 0.5  # neutral
        avg = float(np.mean(s))
        # normalize -1..1 to 0..1
        return (avg + 1)/2.0
    except Exception:
        return 0.5

# -------------------------
# UI controls
# -------------------------
st.sidebar.header("Settings")
pairs = st.sidebar.multiselect("Pairs (max 3)", ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X"], default=["EURUSD=X"])
if len(pairs) == 0:
    st.sidebar.warning("Choose at least one pair.")
algo_mode = st.sidebar.radio("Strategy mode", ["Technical","ML","Composite"], index=2)
tech_threshold = st.sidebar.slider("Tech confidence threshold (0..1)", 0.1, 0.9, 0.6, 0.05)
composite_weights = st.sidebar.slider("Composite weight for Technical (0..1)", 0.0, 1.0, 0.5, 0.05)
refresh_sec = st.sidebar.slider("Auto-refresh (sec)", 5, 120, 20, step=5)
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","60m","1d"], index=2)
period = st.sidebar.selectbox("History", ["1d","5d","1mo","3mo","6mo","1y"], index=1)
max_pairs = 3
if len(pairs) > max_pairs:
    st.sidebar.error(f"Select at most {max_pairs} pairs.")
st.sidebar.markdown("Tip: For fast signals use 1m + refresh 5-15s; Yahoo may limit history for tiny intervals.")

# autorefresh (soft)
st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_main")

# -------------------------
# Main loop: display per pair as tabs
# -------------------------
tabs = st.tabs(pairs if pairs else ["No pair selected"])
for i, pair in enumerate(pairs):
    tab = tabs[i]
    with tab:
        st.subheader(f"Pair: {pair}  —  Interval: {interval}")
        # Fetch
        df = fetch_ohlc(pair, period, interval)
        if df.empty or len(df) < 60:
            st.error("No or insufficient data for this pair/interval. Try different options.")
            continue

        # Indicators
        df_ind = add_indicators(df)
        if df_ind.empty:
            st.error("Could not compute indicators.")
            continue

        # Tech signals
        tech_df = tech_signal_logic(df_ind, threshold=tech_threshold)

        # ML
        df_feat, feats = prepare_ml_features(tech_df)
        model, model_acc = train_light_model(df_feat, feats)

        # ML prediction probability for latest bar
        ml_prob = 0.5
        if model is not None and len(df_feat) > 0:
            last_idx = df_feat.index[-1]
            ml_prob = predict_ml_prob(model, df_feat.loc[[last_idx]], feats)

        # News
        news_score = fetch_news_sentiment(pair)

        # Composite decision per latest candle
        latest = tech_df.iloc[-1]
        tech_signal = latest["TechSignal"]
        tech_conf = latest["tech_conf_buy"] if tech_signal == "BUY" else (latest["tech_conf_sell"] if tech_signal == "SELL" else 0.0)
        tech_score_norm = (latest["tech_score"] + 1) / 2.0  # map -1..1 to 0..1
        # composite weights: technical = composite_weights, ML = (1 - composite_weights)*0.7, news small remainder
        w_tech = composite_weights
        w_ml = (1 - composite_weights) * 0.8
        w_news = (1 - composite_weights) * 0.2
        composite_score = w_tech * tech_score_norm + w_ml * ml_prob + w_news * news_score

        # Decide according to mode
        if algo_mode == "Technical":
            action = tech_signal
        elif algo_mode == "ML":
            action = "BUY" if ml_prob >= 0.55 else ("SELL" if ml_prob <= 0.45 else "WAIT")
        else:  # composite
            action = "BUY" if composite_score >= 0.58 else ("SELL" if composite_score <= 0.42 else "WAIT")

        # output header with NOW info
        ts = latest.name.strftime("%Y-%m-%d %H:%M:%S %Z")
        if action == "BUY":
            st.success(f"✅ {algo_mode} → BUY NOW  — {ts}  | Price: {latest['Close']:.5f}")
        elif action == "SELL":
            st.error(f"❌ {algo_mode} → SELL NOW  — {ts}  | Price: {latest['Close']:.5f}")
        else:
            st.info(f"⏳ {algo_mode} → WAIT  — {ts}  | Price: {latest['Close']:.5f}")

        # show scores
        st.write(f"Technical score: {tech_score_norm:.2f}  |  ML up-prob: {ml_prob:.2f} (acc~{model_acc:.2f})  |  News: {news_score:.2f}  |  Composite: {composite_score:.2f}")

        # Chart view selection
        last_n = st.slider(f"Show last N candles for {pair}", 150, 3000, 600, step=50, key=f"slider_{pair}")

        # Prepare plot_df (Plotly requires naive timestamps for stable display)
        view = tech_df.tail(last_n).copy()
        plot_df = view.copy()
        plot_df["t_plot"] = plot_df.index.tz_convert(APP_TZ).tz_localize(None)

        # Build Plotly with subplots: Candles, RSI, MACD
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.62, 0.18, 0.20], vertical_spacing=0.03)

        # Candles
        fig.add_trace(go.Candlestick(
            x=plot_df["t_plot"], open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name="Candles"
        ), row=1, col=1)

        # EMAs & Bollinger mid
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["EMA20"], name="EMA20", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["EMA50"], name="EMA50", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["BB_mid"], name="BB_mid", line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["BB_up"], name="BB_up", line=dict(width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["BB_low"], name="BB_low", line=dict(width=1, dash="dash")), row=1, col=1)

        # Markers: Tech signals
        sigs = plot_df[plot_df["TechSignal"].isin(["BUY","SELL"])]
        if not sigs.empty:
            fig.add_trace(go.Scatter(
                x=sigs["t_plot"], y=sigs["Close"], mode="markers+text",
                text=sigs["TechSignal"], textposition="top center",
                marker=dict(symbol=["triangle-up" if s=="BUY" else "triangle-down" for s in sigs["TechSignal"]],
                            size=10, color=["green" if s=="BUY" else "red" for s in sigs["TechSignal"]]),
                name="Tech Signals"
            ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["RSI14"], name="RSI14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(go.Bar(x=plot_df["t_plot"], y=plot_df["MACD_hist"], name="MACD_hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["t_plot"], y=plot_df["MACD_signal"], name="MACD_signal"), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                          uirevision=f"{pair}_{interval}_stable_v3", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Recent signals table
        st.subheader("Recent Tech Signals (timestamped)")
        recent = tech_df[tech_df["TechSignal"].isin(["BUY","SELL"])][["TechSignal","Close"]].tail(50).copy()
        if not recent.empty:
            recent["Time"] = recent.index.strftime("%Y-%m-%d %H:%M:%S %Z")
            recent = recent.rename(columns={"TechSignal":"Signal","Close":"Price"})[["Time","Signal","Price"]]
            st.dataframe(recent)
        else:
            st.info("No technical signals found in this period.")

        # Optional: show headlines (small)
        with st.expander("Latest headlines (small sample)"):
            try:
                # quick fetch via feedparser for pair currencies
                s = fetch_news_sentiment(pair)
                if isinstance(s, float):
                    st.write(f"News sentiment score: {s:.2f} (0..1)")
                else:
                    st.write("News unavailable")
            except Exception:
                st.write("News fetch failed (ok).")

        st.markdown("---")

# end for each pair

st.markdown("### Notes")
st.markdown("""
- This app combines Technical indicators, a lightweight ML predictor, and headline sentiment into a composite.  
- For **true per-second ticks** you need a broker feed (OANDA, IG, etc). This app uses Yahoo via `yfinance` (best near-1m updates).  
- **No trading guarantee** — signals are educational only.
""")
