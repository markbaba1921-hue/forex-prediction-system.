import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Forex Prediction System", layout="wide")
st.title("💹 Forex Prediction System")

# -------------------------------
# Sidebar controls
# -------------------------------
pair = st.sidebar.text_input("Forex Pair (Yahoo format)", "EURUSD=X")
interval = st.sidebar.selectbox("Interval", ["1h","30m","15m","1d"], index=0)
start = st.sidebar.date_input("Start date", pd.to_datetime("2022-01-01"))
end   = st.sidebar.date_input("End date",   pd.to_datetime("today"))
window = st.sidebar.slider("Sliding window (candles)", 20, 150, 64, 1)
epochs = st.sidebar.slider("Epochs", 1, 10, 2, 1)
batch  = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)

# -------------------------------
# Load & clean data
# -------------------------------
@st.cache_data
def load_data(pair, start, end, interval):
    df = yf.download(pair, start=start, end=end, interval=interval, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    df = df[["Close"]].dropna()
    df["Close"] = df["Close"].astype(float)
    df = df[~df["Close"].isna()]
    return df

data = load_data(pair, start, end, interval)

if data.empty:
    st.error("No data returned. Try a larger date range or a different interval/pair (e.g., EURUSD=X).")
    st.stop()

st.subheader("📊 Close price")
st.line_chart(data["Close"])

# Guard: enough rows for the chosen window?
if len(data) <= window + 1:
    st.warning(f"Not enough rows ({len(data)}) for window={window}. Increase date range or lower the window.")
    st.stop()

# -------------------------------
# Prepare supervised dataset
# -------------------------------
data = data.dropna()
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[["Close"]])

X, y = [], []
for i in range(window, len(scaled_close)):
    X.append(scaled_close[i-window:i, 0])
    y.append(scaled_close[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

st.write(f"Prepared dataset: X={X.shape}, y={y.shape}")

# -------------------------------
# Build model
# -------------------------------
def build_model(input_steps: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(input_steps, 1)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

if st.button("🔮 Train & Predict"):
    with st.spinner("Training LSTM…"):
        model = build_model(window)
        model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)

    # Predict on training window (demo)
    preds = model.predict(X, verbose=0)
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_inv = data["Close"].values[window:]

    # Plot
    fig, ax = plt.subplots()
    idx = data.index[window:]
    ax.plot(idx, actual_inv, label="Actual")
    ax.plot(idx, preds_inv, label="Predicted")
    ax.legend()
    ax.set_title("Predicted vs Actual (training window)")
    st.pyplot(fig)

    # Direction accuracy (simple)
    def direction_accuracy(a, b):
        a = np.sign(np.diff(a))
        b = np.sign(np.diff(b))
        n = min(len(a), len(b))
        return float((a[:n] == b[:n]).mean()) if n > 0 else float("nan")

    da = direction_accuracy(actual_inv, preds_inv)
    st.success(f"Direction accuracy: {da:.3f}")
