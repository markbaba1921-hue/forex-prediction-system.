import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Forex Trading System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
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
    }
    .feature-importance-plot {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
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
# Data Ingestion
# ----------------------
@st.cache_data(ttl=300)
def fetch_data(symbol, period="5d", interval="5m"):
    """Fetch OHLC data from Yahoo Finance"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

def fetch_fundamental_data():
    """Fetch fundamental economic data (mock implementation)"""
    # In a real implementation, you would fetch from APIs like FRED, IMF, etc.
    fundamentals = {
        'US_GDP': 1.05,  # % change
        'US_Unemployment': 3.7,  # %
        'US_InterestRate': 5.5,  # %
        'EU_GDP': 0.8,
        'EU_Unemployment': 6.5,
        'EU_InterestRate': 4.5,
    }
    return fundamentals

def fetch_news_sentiment(symbol="EURUSD"):
    """Fetch and analyze news sentiment (mock implementation)"""
    # In a real implementation, you would use NewsAPI, Twitter API, etc.
    news_items = [
        "ECB maintains hawkish stance amid inflation concerns",
        "US job growth exceeds expectations, dollar strengthens",
        "Brexit negotiations continue to impact GBP volatility"
    ]
    
    sentiments = []
    for news in news_items:
        analysis = TextBlob(news)
        sentiments.append(analysis.sentiment.polarity)
    
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment, news_items

# ----------------------
# Feature Engineering
# ----------------------
def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    # Trend indicators
    df["EMA20"] = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["EMA100"] = ta.trend.EMAIndicator(close=df["Close"], window=100).ema_indicator()
    
    # MACD
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()
    
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["Stoch_%K"] = stoch.stoch()
    df["Stoch_%D"] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    
    # Average True Range (ATR)
    df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"]).average_true_range()
    
    # ADX
    df["ADX"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"]).adx()
    
    # Volume indicators
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    
    return df

def create_composite_features(df, fundamental_data, sentiment_score):
    """Create composite features from technical, fundamental and sentiment data"""
    df = df.copy()
    
    # Price momentum composite
    df['Price_Momentum'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    
    # Volatility composite
    df['Volatility_Ratio'] = df['ATR'] / df['Close'] * 100
    
    # Add fundamental data (repeated for each row)
    df['US_GDP'] = fundamental_data.get('US_GDP', 0)
    df['US_InterestRate'] = fundamental_data.get('US_InterestRate', 0)
    
    # Add sentiment data
    df['News_Sentiment'] = sentiment_score
    
    # Interest rate differential (example for EURUSD)
    df['Rate_Diff'] = fundamental_data.get('EU_InterestRate', 0) - fundamental_data.get('US_InterestRate', 0)
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    return df

# ----------------------
# ML Model (Simulated)
# ----------------------
def generate_ml_signals(df):
    """Generate trading signals using a simulated ML model"""
    # In a real implementation, you would use a trained model
    # This is a simplified simulation based on multiple factors
    
    signals = []
    probabilities = []
    features = []
    
    for i in range(len(df)):
        # Feature vector (simulated model inputs)
        feature_vector = [
            df['RSI'].iloc[i] if i < len(df) else 50,
            df['MACD'].iloc[i] if i < len(df) else 0,
            df['Stoch_%K'].iloc[i] if i < len(df) else 50,
            df['Price_Momentum'].iloc[i] if i < len(df) else 0,
            df['Volatility_Ratio'].iloc[i] if i < len(df) else 0,
            df['News_Sentiment'].iloc[i] if i < len(df) else 0,
            df['Rate_Diff'].iloc[i] if i < len(df) else 0,
        ]
        features.append(feature_vector)
        
        # Simulated model output (in reality, this would come from a trained model)
        # Weighted combination of indicators
        score = (
            0.3 * (70 - df['RSI'].iloc[i])/50 +  # RSI contribution
            0.2 * np.tanh(df['MACD'].iloc[i] * 10) +  # MACD contribution
            0.2 * (df['Stoch_%K'].iloc[i] - 50)/50 +  # Stochastic contribution
            0.1 * np.tanh(df['Price_Momentum'].iloc[i]/5) +  # Momentum contribution
            0.1 * df['News_Sentiment'].iloc[i] * 2 +  # Sentiment contribution
            0.1 * np.tanh(df['Rate_Diff'].iloc[i]/2)  # Rate diff contribution
        )
        
        # Convert to probability
        probability = 1 / (1 + np.exp(-score * 5))
        
        # Determine signal
        if probability > 0.6:
            signals.append("BUY")
            probabilities.append(probability)
        elif probability < 0.4:
            signals.append("SELL")
            probabilities.append(1 - probability)
        else:
            signals.append("NEUTRAL")
            probabilities.append(0.5)
    
    df["ML_Signal"] = signals
    df["ML_Probability"] = probabilities
    
    # Feature importance (simulated)
    feature_importance = {
        'RSI': 0.3,
        'MACD': 0.2,
        'Stochastic': 0.2,
        'Price Momentum': 0.1,
        'News Sentiment': 0.1,
        'Rate Differential': 0.1
    }
    
    return df, feature_importance

# ----------------------
# Risk Management
# ----------------------
def calculate_position_size(balance, risk_per_trade, entry_price, stop_loss):
    """Calculate position size based on risk management rules"""
    risk_amount = balance * (risk_per_trade / 100)
    price_diff = abs(entry_price - stop_loss)
    position_size = risk_amount / price_diff
    return position_size

def apply_risk_management(df, balance=10000, risk_per_trade=1):
    """Apply risk management to generate trading decisions"""
    trades = []
    current_balance = balance
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    for i, row in df.iterrows():
        # Check if we need to exit a position
        if position != 0:
            # Check stop loss
            if (position > 0 and row['Low'] <= stop_loss) or (position < 0 and row['High'] >= stop_loss):
                pnl = position * (stop_loss - entry_price) if position > 0 else position * (entry_price - stop_loss)
                current_balance += pnl
                trades.append({
                    'Entry_Time': i,
                    'Exit_Time': i,
                    'Type': 'STOP_LOSS',
                    'Price': stop_loss,
                    'PnL': pnl,
                    'Balance': current_balance
                })
                position = 0
            
            # Check take profit
            elif (position > 0 and row['High'] >= take_profit) or (position < 0 and row['Low'] <= take_profit):
                pnl = position * (take_profit - entry_price) if position > 0 else position * (entry_price - take_profit)
                current_balance += pnl
                trades.append({
                    'Entry_Time': i,
                    'Exit_Time': i,
                    'Type': 'TAKE_PROFIT',
                    'Price': take_profit,
                    'PnL': pnl,
                    'Balance': current_balance
                })
                position = 0
        
        # Check if we should enter a new position
        if position == 0 and row['ML_Signal'] in ['BUY', 'SELL']:
            entry_price = row['Close']
            atr = row['ATR']
            
            # Set stop loss and take profit based on ATR
            if row['ML_Signal'] == 'BUY':
                stop_loss = entry_price - 2 * atr
                take_profit = entry_price + 3 * atr
                position = calculate_position_size(current_balance, risk_per_trade, entry_price, stop_loss)
            else:  # SELL
                stop_loss = entry_price + 2 * atr
                take_profit = entry_price - 3 * atr
                position = -calculate_position_size(current_balance, risk_per_trade, entry_price, stop_loss)
            
            trades.append({
                'Entry_Time': i,
                'Exit_Time': None,
                'Type': 'ENTER',
                'Price': entry_price,
                'PnL': 0,
                'Balance': current_balance
            })
    
    # Close any open position at the end
    if position != 0:
        exit_price = df.iloc[-1]['Close']
        pnl = position * (exit_price - entry_price) if position > 0 else position * (entry_price - exit_price)
        current_balance += pnl
        trades.append({
            'Entry_Time': trades[-1]['Entry_Time'],
            'Exit_Time': df.index[-1],
            'Type': 'CLOSE_END',
            'Price': exit_price,
            'PnL': pnl,
            'Balance': current_balance
        })
    
    return pd.DataFrame(trades), current_balance

# ----------------------
# Visualization
# ----------------------
def plot_trading_signals(df, trades_df):
    """Plot price chart with trading signals and annotations"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price with Signals', 'Technical Indicators'),
        row_width=[0.7, 0.3]
    )
    
    # Price data
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # EMA lines
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='blue', width=1), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='orange', width=1), name='EMA50'), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1, dash='dash'), name='BB Lower', fill='tonexty'), row=1, col=1)
    
    # Add trade entries and exits
    if trades_df is not None and not trades_df.empty:
        entries = trades_df[trades_df['Type'] == 'ENTER']
        exits = trades_df[trades_df['Type'].isin(['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_END'])]
        
        # Entry points
        fig.add_trace(go.Scatter(
            x=entries['Entry_Time'], y=entries['Price'],
            mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Entry'
        ), row=1, col=1)
        
        # Exit points
        fig.add_trace(go.Scatter(
            x=exits['Exit_Time'], y=exits['Price'],
            mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Exit'
        ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title='Trading Signals with Risk Management',
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=800,
        showlegend=True
    )
    
    return fig

def plot_equity_curve(trades_df):
    """Plot equity curve from trades"""
    if trades_df is None or trades_df.empty:
        return None
        
    equity_df = trades_df[trades_df['Type'].isin(['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_END'])].copy()
    if equity_df.empty:
        return None
        
    equity_df['Cumulative_PnL'] = equity_df['PnL'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df['Exit_Time'], y=equity_df['Cumulative_PnL'], 
                            mode='lines', name='Equity Curve'))
    
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Profit/Loss ($)',
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_feature_importance(feature_importance):
    """Plot feature importance chart"""
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h'
    ))
    
    fig.update_layout(
        title='Simulated Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        template="plotly_white",
        height=400
    )
    
    return fig

# ----------------------
# Streamlit UI
# ----------------------
st.markdown("<h1 class='main-header'>Advanced Forex Trading System</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <p>ML-powered trading signals with comprehensive risk management</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Symbol selection
    selected_symbol = st.selectbox("Select Forex Pair", FOREX_PAIRS, index=0)
    custom_symbol = st.text_input("Or enter custom symbol", "EURUSD=X")
    symbol = custom_symbol if custom_symbol else selected_symbol
    
    # Time settings
    period = st.selectbox("Time Period", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["5m", "15m", "30m", "1h", "1d"], index=1)
    
    # Risk management settings
    st.subheader("Risk Management")
    initial_balance = st.number_input("Initial Balance ($)", min_value=1000, max_value=100000, value=10000, step=1000)
    risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    # Strategy settings
    st.subheader("Strategy Settings")
    use_ml = st.checkbox("Use ML Signals", value=True)
    use_risk_management = st.checkbox("Apply Risk Management", value=True)
    
    # Info
    st.info("""
    This system uses simulated ML models for demonstration.
    Always test strategies thoroughly before live trading.
    """)

# Main content
if st.button("Run Analysis", type="primary"):
    with st.spinner("Collecting data and generating signals..."):
        # Data Ingestion
        df = fetch_data(symbol, period, interval)
        
        if df is None or df.empty:
            st.error("No data returned. Try a different timeframe or symbol.")
        else:
            # Fetch additional data
            fundamental_data = fetch_fundamental_data()
            sentiment_score, news_items = fetch_news_sentiment(symbol)
            
            # Feature Engineering
            df = add_technical_indicators(df)
            df = create_composite_features(df, fundamental_data, sentiment_score)
            
            # Generate ML Signals
            if use_ml:
                df, feature_importance = generate_ml_signals(df)
                
                # Display latest signal
                latest = df.iloc[-1]
                st.subheader("📊 Latest ML Signal")
                
                if latest["ML_Signal"] == "BUY":
                    st.markdown(f"<p class='signal-buy'>BUY signal with {latest['ML_Probability']*100:.1f}% confidence</p>", unsafe_allow_html=True)
                elif latest["ML_Signal"] == "SELL":
                    st.markdown(f"<p class='signal-sell'>SELL signal with {(1-latest['ML_Probability'])*100:.1f}% confidence</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='signal-neutral'>NEUTRAL - No strong signal detected</p>", unsafe_allow_html=True)
            
            # Apply Risk Management
            trades_df = None
            final_balance = initial_balance
            
            if use_risk_management and use_ml:
                trades_df, final_balance = apply_risk_management(df, initial_balance, risk_per_trade)
                
                # Display performance metrics
                st.subheader("💰 Performance Metrics")
                
                if trades_df is not None and not trades_df.empty:
                    closed_trades = trades_df[trades_df['Type'].isin(['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE_END'])]
                    
                    if not closed_trades.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_trades = len(closed_trades)
                            st.metric("Total Trades", total_trades)
                        
                        with col2:
                            winning_trades = len(closed_trades[closed_trades['PnL'] > 0])
                            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                            st.metric("Win Rate", f"{win_rate:.1f}%")
                        
                        with col3:
                            total_pnl = closed_trades['PnL'].sum()
                            st.metric("Total PnL", f"${total_pnl:.2f}")
                        
                        with col4:
                            balance_change = ((final_balance - initial_balance) / initial_balance) * 100
                            st.metric("Balance Change", f"{balance_change:.1f}%")
            
            # Display charts
            st.subheader("📈 Price Chart with Signals")
            fig = plot_trading_signals(df, trades_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Equity curve
            if trades_df is not None:
                equity_fig = plot_equity_curve(trades_df)
                if equity_fig:
                    st.plotly_chart(equity_fig, use_container_width=True)
            
            # Feature importance
            if use_ml:
                st.subheader("📋 Feature Importance")
                importance_fig = plot_feature_importance(feature_importance)
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # News sentiment
            st.subheader("📰 News Sentiment")
            st.metric("Average Sentiment Score", f"{sentiment_score:.3f}")
            
            for news in news_items:
                st.write(f"- {news}")
            
            # Show data
            st.subheader("📊 Processed Data")
            with st.expander("View processed data with features"):
                display_df = df.tail(10).copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
                st.dataframe(display_df.style.format({
                    'Open': '{:.5f}', 'High': '{:.5f}', 'Low': '{:.5f}', 'Close': '{:.5f}',
                    'EMA20': '{:.5f}', 'EMA50': '{:.5f}', 'EMA100': '{:.5f}',
                    'RSI': '{:.2f}', 'ATR': '{:.5f}', 'ADX': '{:.2f}',
                    'ML_Probability': '{:.3f}'
                }))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Disclaimer:</strong> This is a simulated trading system for educational purposes only. 
    Trading financial instruments involves risk and is not suitable for all investors.</p>
    <p>Always test strategies thoroughly with backtesting and paper trading before risking real capital.</p>
</div>
""", unsafe_allow_html=True)
