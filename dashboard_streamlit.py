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
# Sidebar Input
# -------------------------------
symbol = st.sidebar.text_input("Forex Pair (Yahoo Finance format)", "EURUSD=X")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, interval="1h")
    data.dropna(inplace=True)
    return data

data = load_data(symbol, start_date, end_date)

st.subheader("📊 Historical Data")
st.line_chart(data["Close"])

# -------------------------------
# Preprocessing
# -------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data[["Close"]])

X, y = [], []
window = 50
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# -------------------------------
# Model
# -------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")

if st.button("🔮 Train Model"):
    with st.spinner("Training... this may take a minute"):
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    st.success("Model trained!")

    # Predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Plot results
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data.index[window:], data["Close"].values[window:], label="Actual")
    ax.plot(data.index[window:], predictions, label="Predicted")
    ax.legend()
    st.pyplot(fig)
