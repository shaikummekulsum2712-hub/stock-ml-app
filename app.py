import streamlit as st
from data import fetch_data
from model import add_features, train_model

st.title("Stock Trend Predictor")

stock = st.text_input("Enter Stock Symbol", "^NSEI")

df = fetch_data(stock)
df = add_features(df)

model, acc = train_model(df)

latest = df.iloc[-1][["MA10", "MA50", "RSI", "Returns"]].values.reshape(1, -1)
prediction = model.predict(latest)

st.write("Prediction:", "UP 📈" if prediction[0] == 1 else "DOWN 📉")
st.write(f"Model Accuracy: {acc:.2f}")

st.subheader("Price Chart")
st.line_chart(df["Close"])

st.subheader("RSI Indicator")
st.line_chart(df["RSI"])

# Simple Trading Insight
rsi_value = df["RSI"].iloc[-1]

if rsi_value < 30:
    st.success("Oversold → Possible BUY zone 🟢")

elif rsi_value > 70:
    st.error("Overbought → Possible SELL zone 🔴")

