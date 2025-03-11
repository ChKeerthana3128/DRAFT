import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate financial health score
def calculate_financial_health(income, expenses, investments, liabilities):
    savings = income - expenses
    score = ((savings + investments) / (income + 1)) * 100 - (liabilities / (income + 1)) * 50
    return max(0, min(100, score))

# Load stock dataset (Ensure the file is available)
def load_stock_data():
    try:
        df = pd.read_csv("stock_data.csv")  # Ensure this CSV exists in the directory
        return df
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return None

# Function to train a stock price prediction model
def train_stock_model(df):
    df = df.dropna()
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    st.write("### Model Performance")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    
    return model

# Streamlit App Layout
st.title("💰 WealthWise Dashboard")
st.sidebar.header("User Input Parameters")

# User input fields
income = st.sidebar.number_input("Monthly Income (₹)", min_value=0, step=1000)
expenses = st.sidebar.number_input("Monthly Expenses (₹)", min_value=0, step=500)
investments = st.sidebar.number_input("Total Investments (₹)", min_value=0, step=1000)
liabilities = st.sidebar.number_input("Total Liabilities (₹)", min_value=0, step=1000)

# Calculate Financial Health Score
if st.sidebar.button("Calculate Financial Health"):
    score = calculate_financial_health(income, expenses, investments, liabilities)
    st.sidebar.write(f"Your Financial Health Score: **{score:.2f}**")

# Tabs for different functionalities
option = st.selectbox("Choose Module", ["Personal Finance", "Stock Investments"])

if option == "Personal Finance":
    st.subheader("📊 Financial Overview")
    st.write(f"💰 **Disposable Income:** ₹{income - expenses:.2f}")
    st.write(f"📈 **Estimated Wealth Growth (10% return):** ₹{investments * 1.1:.2f}")
    st.write("🔍 Smart Savings Plan: Save at least 20% of your income for better financial health!")

elif option == "Stock Investments":
    st.subheader("📈 Stock Investment Insights")
    stock_data = load_stock_data()
    
    if stock_data is not None:
        st.write("### Sample Stock Data")
        st.dataframe(stock_data.head())
        
        model = train_stock_model(stock_data)
        
        # Prediction Input
        st.write("### Predict Stock Closing Price")
        open_price = st.number_input("Open Price", min_value=0.0)
        high_price = st.number_input("High Price", min_value=0.0)
        low_price = st.number_input("Low Price", min_value=0.0)
        volume = st.number_input("Volume", min_value=0)
        
        if st.button("Predict Price"):
            prediction = model.predict([[open_price, high_price, low_price, volume]])
            st.success(f"Predicted Closing Price: ₹{prediction[0]:.2f}")

st.write("🛠 Built with Streamlit | 🚀 AI-Powered Insights")
