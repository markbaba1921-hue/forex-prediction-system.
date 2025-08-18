# 💹 Forex Prediction System

A complete Forex trading prediction system built with Python, TensorFlow/Keras, and Streamlit.

## 🚀 Features
- **Data Acquisition**  
  - Download historical Forex data (EUR/USD, etc.) via Yahoo Finance.  
  - Upload Forex chart screenshots → extract data with OCR (pytesseract + OpenCV).  
  - Merge OCR + API data.  

- **Preprocessing**  
  - Handle missing values, normalize data.  
  - Create technical indicators (RSI, MACD, Bollinger Bands, Moving Averages).  
  - Convert to supervised learning format.  

- **Model Training**  
  - LSTM model for short & medium-term predictions.  
  - Save trained model for reuse.  

- **Evaluation**  
  - Metrics: RMSE, MAE, direction accuracy.  
  - Visualize actual vs predicted prices.  

- **Real-Time Dashboard**  
  - Streamlit web app for predictions.  
  - Live Forex price fetching.  
  - Rolling forecasts & signals.  

- **Alerts**  
  - Email / Telegram alerts for strong Buy/Sell signals.  

## ⚙️ Installation
```bash
pip install -r requirements.txt
