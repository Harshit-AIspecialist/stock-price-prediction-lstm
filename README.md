# 📈 Stock Price Prediction using LSTM (Deep Learning)

This project implements a **Stock Price Prediction System** using **Long Short-Term Memory (LSTM)** neural networks.  
It uses real historical stock data and predicts future stock prices based on past trends and technical indicators.

⚠️ **Disclaimer**: This project is for educational purposes only and should not be considered financial advice.

---

## 🚀 Project Overview

- **Model Type**: LSTM (Deep Learning – Time Series)
- **Task**: Stock price prediction (regression)
- **Frameworks**: TensorFlow, Keras, Scikit-learn
- **Deployment**: Streamlit Web App
- **Data Source**: Yahoo Finance (`yfinance`)

---

## 📊 Features Used

The model is trained using the following features:
- Closing Price
- 20-day Simple Moving Average (SMA)
- 50-day Simple Moving Average (SMA)
- 20-day Exponential Moving Average (EMA)
- Volatility (Rolling Standard Deviation)

---

## 🧠 Model Architecture

- LSTM (64 units, return sequences)
- Dropout (0.2)
- LSTM (32 units)
- Dropout (0.2)
- Dense (1 output)

Loss Function: **Mean Squared Error (MSE)**  
Optimizer: **Adam**

---

## 📁 Project Structure

```
stock-price-prediction-lstm/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Project dependencies
├── stock_lstm_model.keras     # Trained LSTM model
├── README.md                  # Project documentation
```

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Harshit-AIspecialist/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🌐 Web App Features

- Enter any stock symbol (Yahoo Finance format, e.g. `RELIANCE.NS`, `AAPL`)
- View **Actual vs Predicted price**
- Interactive visualization
- Clean UI using Streamlit

---

## 📉 Model Performance

- RMSE and MAE are used for evaluation
- Model captures short-term price trends reasonably well
- Performance may vary across different stocks

---

## ⚠️ Disclaimer

Stock markets are volatile and influenced by many external factors.  
This model **does not guarantee accuracy** and should not be used for real trading decisions.

---

## 👨‍💻 Author

**Harshit Kumawat**  
AI & Machine Learning Enthusiast  
GitHub: https://github.com/Harshit-AIspecialist
