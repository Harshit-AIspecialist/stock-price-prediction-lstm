import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction System")
st.markdown("LSTM-based Deep Learning Model")

# Load model (ONLY ONCE)
model = load_model("stock_lstm_model.keras")

@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start="2018-01-01", end="2024-01-01")
    return df

symbol = st.text_input("Enter Stock Symbol (Yahoo format)", "RELIANCE.NS")
df = load_data(symbol)

if df.empty:
    st.error("Invalid stock symbol")
    st.stop()

# Feature engineering
df['Return'] = df['Close'].pct_change()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()
df['EMA_20'] = df['Close'].ewm(span=20).mean()
df['Volatility'] = df['Return'].rolling(20).std()
df.dropna(inplace=True)

features = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'Volatility']
data = df[features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length=60):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
    return np.array(X)

X = create_sequences(scaled_data)

predictions = model.predict(X)

dummy = np.zeros((len(predictions), scaled_data.shape[1]))
dummy[:, 0] = predictions.flatten()
predicted_prices = scaler.inverse_transform(dummy)[:, 0]
actual_prices = df['Close'].values[-len(predicted_prices):]

st.subheader("📊 Actual vs Predicted Price")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(actual_prices, label="Actual Price")
ax.plot(predicted_prices, label="Predicted Price")
ax.legend()
st.pyplot(fig)

st.warning("""
⚠️ Disclaimer:
This app is for educational purposes only.
Not financial advice.
""")
