import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# 2. Page Configuration
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# 3. Data Loading (Only for Stock Investments)
@st.cache_data
def load_stock_data(csv_path="archive (3) 2/NIFTY CONSUMPTION_daily_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Stock CSV not found! Ensure 'NIFTY CONSUMPTION_daily_data.csv' exists.")
        return None
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"date": "Date", "close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
        df['Symbol'] = "NIFTY CONSUMPTION"
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().all():
            st.error("🚨 Invalid date format in stock data!")
            return None
        df = df.sort_values(by='Date').dropna()
        return df
    except Exception as e:
        st.error(f"🚨 Error loading stock data: {str(e)}")
        return None

# 4. Model Training (Only for Stock Investments)
@st.cache_resource
def train_stock_model(data):
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    X = data[['Day', 'Month', 'Year']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

# 5. Predictive Functions (For Personal Finance, using form inputs directly)
def calculate_financial_health_score(income, total_expenses, debt, discretionary):
    """🌡️ Gauge your financial strength!"""
    if income <= 0:
        return 0
    savings = max(0, income - total_expenses)
    savings_ratio = savings / income
    debt_ratio = debt / income
    discretionary_ratio = discretionary / income
    savings_score = min(25, (savings_ratio / 0.2) * 25)
    debt_score = max(0, 50 - (debt_ratio * 100))
    discretionary_score = max(0, 25 - (discretionary_ratio * 125))
    return max(0, min(100, savings_score + debt_score + discretionary_score))

def predict_disposable_income(income, total_expenses):
    """Simplified calculation: Disposable Income = Income - Total Expenses"""
    return max(0, income - total_expenses)

def forecast_wealth_growth(income, total_expenses, savings_rate, years, income_growth=0.0, expense_growth=0.0):
    """📈 Project your wealth ascent!"""
    wealth = 0.0
    current_income, current_expenses = income, total_expenses
    for _ in range(years + 1):
        disposable = max(0, current_income - current_expenses)
        annual_savings = disposable * (savings_rate / 100)
        wealth += annual_savings
        current_income *= (1 + income_growth / 100)
        current_expenses *= (1 + expense_growth / 100)
    return wealth

def wealth_trajectory(income, total_expenses, savings_rate, years, income_growth, expense_growth):
    trajectory = []
    wealth = 0.0
    current_income, current_expenses = income, total_expenses
    for _ in range(years + 1):
        disposable = max(0, current_income - current_expenses)
        annual_savings = disposable * (savings_rate / 100)
        wealth += annual_savings
        trajectory.append(wealth)
        current_income *= (1 + income_growth / 100)
        current_expenses *= (1 + expense_growth / 100)
    return trajectory

def smart_savings_plan(income, total_expenses, years_to_retirement):
    """🧠 Craft your retirement blueprint!"""
    dream_fund = max(100000.0, total_expenses * 12 * 20) if total_expenses > 0 else 100000.0
    annual_target = dream_fund / years_to_retirement if years_to_retirement > 0 else dream_fund
    savings_rate = min(max((annual_target / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    expense_growth = 2.5
    return dream_fund, savings_rate, income_growth, expense_growth

# 6. Insight Generators (For Stock Investments)
def portfolio_advice(risk_tolerance):
    """💼 Your investment playbook!"""
    if risk_tolerance == "Low":
        return {
            "Overview": "🌳 Calm and steady wins the race!",
            "Picks": [
                {"Type": "Blue-Chip", "Name": "HDFC Bank 🏦", "Why": "Rock-solid dividends."},
                {"Type": "Blue-Chip", "Name": "TCS 💻", "Why": "Stable IT powerhouse."},
                {"Type": "Bonds", "Name": "RBI Bonds 📜", "Why": "Safe haven returns."}
            ]
        }
    elif risk_tolerance == "Medium":
        return {
            "Overview": "⚖️ Balanced growth with flair!",
            "Picks": [
                {"Type": "Large Cap", "Name": "Reliance Industries 🏭", "Why": "Diversified titan."},
                {"Type": "Mid Cap", "Name": "Bajaj Finance 📈", "Why": "Growth with grit."},
                {"Type": "Real Estate", "Name": "DLF 🏡", "Why": "Property steady climber."},
                {"Type": "Mutual Fund", "Name": "SBI Bluechip Fund 🌟", "Why": "Smart diversification."}
            ]
        }
    else:
        return {
            "Overview": "🚀 Bold moves, big wins!",
            "Picks": [
                {"Type": "Small Cap", "Name": "Paytm 💳", "Why": "Fintech frontier."},
                {"Type": "Small Cap", "Name": "Zomato 🍽️", "Why": "Food tech rocket."},
                {"Type": "Crypto", "Name": "Bitcoin ₿", "Why": "High-octane reward chase!"}
            ]
        }

# 7. Main Application
def main():
    # Page title
    st.title("WealthWise Dashboard")

    # Load stock data (for Stock Investments tab)
    stock_data = load_stock_data()
    if stock_data is None:
        st.warning("Stock Investments tab will not function without stock data. Proceeding with Personal Finance tab.")

    # Train stock model if data is available
    stock_model, stock_r2 = None, 0.0
    if stock_data is not None:
        stock_model, stock_r2 = train_stock_model(stock_data)

    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Personal Finance"
    if 'finance_submit' not in st.session_state:
        st.session_state.finance_submit = False
    if 'stock_submit' not in st.session_state:
        st.session_state.stock_submit = False
    if 'input_data' not in st.session_state:
        st.session_state.input_data = None
    if 'total_expenses' not in st.session_state:
        st.session_state.total_expenses = None
    if 'years_to_retirement' not in st.session_state:
        st.session_state.years_to_retirement = None
    if 'debt' not in st.session_state:
        st.session_state.debt = None
    if 'discretionary' not in st.session_state:
        st.session_state.discretionary = None
    if 'horizon' not in st.session_state:
        st.session_state.horizon = 45
    if 'risk_tolerance' not in st.session_state:
        st.session_state.risk_tolerance = "High"
    if 'predicted_price' not in st.session_state:
        st.session_state.predicted_price = None

    # Define tabs
    tab1, tab2 = st.tabs(["💵 Personal Finance", "📈 Stock Investments"])

    # --- Personal Finance Dashboard (Your Code) ---
    with tab1:
        st.header("💵 Your Money Mastery Hub")
        
        with st.sidebar:
            st.subheader("Personal Finance")
            st.subheader("🌡️ Personal Health")
            st.metric("Score", "N/A")
            st.subheader("🏦 Future Wealth")
            st.metric("At Retirement (₹)", "N/A")

        with st.form(key="finance_form"):
            income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, value=0.0, step=1000.0)
            expenses = st.number_input("📊 Monthly Expenses (₹)", min_value=0.0, value=0.0, step=500.0)
            savings_rate = st.slider("🎯 Savings Rate (%)", 0.0, 100.0, 10.0, step=1.0)
            submit = st.form_submit_button("🚀 Analyze Finances")

        if submit:
            disposable_income = max(0, income - expenses)
            st.write(f"💸 Disposable Income: ₹{disposable_income:,.2f}")

    # --- Stock Investments Dashboard (Your Code) ---
    with tab2:
        st.header("📈 Stock Market Quest")
        
        with st.sidebar:
            st.subheader("Stock Investments")
            st.write(f"📊 Stock Model R²: {stock_r2:.2f}")
            st.subheader("📌 Predicted Price")
            st.metric("Predicted Price (₹)", "N/A")
            st.subheader("💡 Investment Insights")
            st.write("🎯 Risk Level: High")

        with st.form(key="stock_form"):
            horizon = st.number_input("⏳ Investment Horizon (Months)", min_value=1, max_value=60, value=12)
            risk_tolerance = st.radio("🎲 Risk Appetite", ["Low", "Medium", "High"])
            stock_submit = st.form_submit_button("🚀 Analyze Stock Investments")

        if stock_submit:
            st.write(f"📌 Investment Horizon: {horizon} months")
            st.write(f"🎯 Risk Tolerance: {risk_tolerance}")

    st.markdown("---")
    st.write("✨ Powered by WealthWise | Built with ❤️ by xAI")

if __name__ == "__main__":
    main()
