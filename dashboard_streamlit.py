import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

# Page configuration
st.set_page_config(
    page_title="Forex Pro Signals", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy {
        color: green;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .signal-sell {
        color: red;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .signal-neutral {
        color: orange;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Forex pairs for quick selection
FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", 
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X"
]

# ----------------------
# Fetch data safely
# ----------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(symbol, period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Flatten MultiIndex if exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# ----------------------
# Compute indicators
# ----------------------
def add_indicators(df):
    # Trend indicators
    df["EMA20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["EMA100"] = EMAIndicator(close=df["Close"], window=100).ema_indicator()
    
    # MACD
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()
    
    # RSI
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["Stoch_%K"] = stoch.stoch()
    df["Stoch_%D"] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    
    # Average True Range (ATR)
    df["ATR"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"]).average_true_range()
    
    # ADX
    df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"]).adx()
    
    # VWAP
    df["VWAP"] = VolumeWeightedAveragePrice(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    ).volume_weighted_average_price()
    
    return df

# ----------------------
# Generate signals with confidence score
# ----------------------
def generate_signals(df):
    signals = []
    confidence_scores = []
    
    for i in range(1, len(df)):
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # EMA Crossover
        if df["EMA20"].iloc[i] > df["EMA50"].iloc[i]:
            buy_signals += 1
        else:
            sell_signals += 1
        total_signals += 1
        
        # RSI
        if df["RSI"].iloc[i] < 30:
            buy_signals += 1
        elif df["RSI"].iloc[i] > 70:
            sell_signals += 1
        total_signals += 1
        
        # MACD
        if df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]:
            buy_signals += 1
        else:
            sell_signals += 1
        total_signals += 1
        
        # Stochastic
        if df["Stoch_%K"].iloc[i] < 20 and df["Stoch_%D"].iloc[i] < 20:
            buy_signals += 1
        elif df["Stoch_%K"].iloc[i] > 80 and df["Stoch_%D"].iloc[i] > 80:
            sell_signals += 1
        total_signals += 1
        
        # Price relative to Bollinger Bands
        if df["Close"].iloc[i] < df["BB_Low"].iloc[i]:
            buy_signals += 1
        elif df["Close"].iloc[i] > df["BB_High"].iloc[i]:
            sell_signals += 1
        total_signals += 1
        
        # Determine final signal
        if buy_signals > sell_signals:
            signals.append("BUY")
            confidence_scores.append(round(buy_signals / total_signals * 100, 1))
        elif sell_signals > buy_signals:
            signals.append("SELL")
            confidence_scores.append(round(sell_signals / total_signals * 100, 1))
        else:
            signals.append("NEUTRAL")
            confidence_scores.append(0)
    
    # Add empty values for the first row
    signals.insert(0, "")
    confidence_scores.insert(0, 0)
    
    df["Signal"] = signals
    df["Confidence"] = confidence_scores
    return df

# ----------------------
# Plot chart with signals
# ----------------------
def plot_chart(df, symbol):
    # Create subplots
    fig = go.Figure()
    
    # Main price chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ))
    
    # EMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color="blue", width=1), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="orange", width=1), name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA100"], line=dict(color="purple", width=1), name="EMA100"))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], line=dict(color="gray", width=1, dash='dash'), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], line=dict(color="gray", width=1), name="BB Middle", opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], line=dict(color="gray", width=1, dash='dash'), name="BB Lower", fill='tonexty', opacity=0.1))
    
    # Buy/Sell markers
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Low"] * 0.998, 
        mode="markers", 
        marker=dict(color="green", size=12, symbol="triangle-up"),
        name="BUY"
    ))
    
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["High"] * 1.002, 
        mode="markers", 
        marker=dict(color="red", size=12, symbol="triangle-down"),
        name="SELL"
    ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart with Signals",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig

# ----------------------
# Plot indicator subplots
# ----------------------
def plot_indicators(df):
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")))
    
    # Add reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="RSI Indicator",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=300
    )
    
    return fig

# ----------------------
# Display performance metrics
# ----------------------
def display_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Current Price", f"{current_price:.5f}", 
                 f"{price_change:.5f} ({price_change_pct:.2f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        rsi_value = df["RSI"].iloc[-1]
        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        st.metric("RSI", f"{rsi_value:.2f}", rsi_status)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("ATR", f"{df['ATR'].iloc[-1]:.5f}", "Volatility")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        adx_value = df["ADX"].iloc[-1]
        trend_strength = "Strong" if adx_value > 25 else "Weak" if adx_value < 20 else "Moderate"
        st.metric("ADX", f"{adx_value:.2f}", trend_strength)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Streamlit UI
# ----------------------
st.markdown("<h1 class='main-header'>📈 Forex Pro Signals — Advanced Analysis</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Symbol selection
    selected_symbol = st.selectbox("Select Forex Pair", FOREX_PAIRS, index=0)
    custom_symbol = st.text_input("Or enter custom symbol", "EURUSD=X")
    
    symbol = custom_symbol if custom_symbol else selected_symbol
    
    # Time settings
    period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=2)
    
    # Strategy settings
    st.subheader("Strategy Settings")
    use_ema = st.checkbox("Use EMA Crossovers", value=True)
    use_rsi = st.checkbox("Use RSI", value=True)
    use_macd = st.checkbox("Use MACD", value=True)
    use_stoch = st.checkbox("Use Stochastic", value=True)
    use_bb = st.checkbox("Use Bollinger Bands", value=True)
    
    # Info
    st.info("""
    This dashboard provides technical analysis for Forex pairs using multiple indicators.
    Signals are generated based on indicator convergence.
    """)

# Main content
if st.button("Analyze Market", type="primary"):
    with st.spinner("Fetching data and analyzing..."):
        df = fetch_data(symbol, period, interval)
        
        if df is None or df.empty:
            st.error("No data returned. Try a different timeframe or symbol.")
        else:
            df = add_indicators(df)
            df = generate_signals(df)
            
            # Latest Signal
            latest = df.iloc[-1]
            st.subheader("📢 Latest Signal")
            
            if latest["Signal"] == "BUY":
                st.markdown(f"<p class='signal-buy'>BUY signal with {latest['Confidence']}% confidence</p>", unsafe_allow_html=True)
            elif latest["Signal"] == "SELL":
                st.markdown(f"<p class='signal-sell'>SELL signal with {latest['Confidence']}% confidence</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='signal-neutral'>NEUTRAL - No strong signal detected</p>", unsafe_allow_html=True)
            
            # Display metrics
            display_metrics(df)
            
            # Plot Chart
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicators plot
            fig_indicators = plot_indicators(df)
            st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Show data
            st.subheader("📊 Historical Data with Signals")
            display_df = df.tail(10).copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_df.style.format({
                'Open': '{:.5f}', 'High': '{:.5f}', 'Low': '{:.5f}', 'Close': '{:.5f}',
                'EMA20': '{:.5f}', 'EMA50': '{:.5f}', 'EMA100': '{:.5f}',
                'RSI': '{:.2f}', 'ATR': '{:.5f}', 'ADX': '{:.2f}'
            }))
            
            # Export option
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"{symbol}_forex_data.csv",
                mime="text/csv",
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Forex Pro Signals Dashboard | This is for educational purposes only. Trading involves risk.</p>
</div>
""", unsafe_allow_html=True)
