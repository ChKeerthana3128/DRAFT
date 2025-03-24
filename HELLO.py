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

st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_stock_data(csv_path="archive (3) 2/NIFTY CONSUMPTION_daily_data.csv"):
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"date": "Date", "close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
        df['Symbol'] = "NIFTY CONSUMPTION"
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.sort_values(by='Date').dropna()
    except Exception as e:
        st.error(f"🚨 Error loading stock data: {str(e)}")
        return None

@st.cache_resource
def train_stock_model(data):
    data['Day'], data['Month'], data['Year'] = data['Date'].dt.day, data['Date'].dt.month, data['Date'].dt.year
    X, y = data[['Day', 'Month', 'Year']], data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

def predict_disposable_income(income, total_expenses):
    return max(0, income - total_expenses)

def forecast_wealth_growth(income, total_expenses, savings_rate, years, income_growth=0.0, expense_growth=0.0):
    wealth, current_income, current_expenses = 0.0, income, total_expenses
    for _ in range(years + 1):
        disposable, annual_savings = max(0, current_income - current_expenses), max(0, current_income - current_expenses) * (savings_rate / 100)
        wealth += annual_savings
        current_income *= (1 + income_growth / 100)
        current_expenses *= (1 + expense_growth / 100)
    return wealth

def main():
    st.title("WealthWise Dashboard")

    stock_data = load_stock_data()
    stock_model, stock_r2 = None, 0.0
    if stock_data is not None:
        stock_model, stock_r2 = train_stock_model(stock_data)

    if 'active_tab' not in st.session_state:
        st.session_state.update({
            'active_tab': "Personal Finance", 'finance_submit': False, 'stock_submit': False, 
            'input_data': None, 'total_expenses': None, 'years_to_retirement': None, 
            'debt': None, 'discretionary': None, 'horizon': 45, 'risk_tolerance': "High", 
            'predicted_price': None
        })

    tab1, tab2 = st.tabs(["💵 Personal Finance", "📈 Stock Investments"])

    with tab1:
        st.session_state.active_tab = "Personal Finance"
        st.header("💵 Your Money Mastery Hub")
        st.markdown("Shape your financial destiny with style! 🌈")

        with st.sidebar:
            st.subheader("Personal Finance")
            st.write("📊 Finance Model R²: N/A (Using form inputs directly)")
            st.metric("Score", "N/A")
            st.metric("Monthly (₹)", "N/A")
            st.metric("At Retirement (₹)", "N/A")

        with st.form(key="finance_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Name", "")
                age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, value=0.0, step=1000.0)
                dependents = st.number_input("👨‍👩‍👧 Dependents", min_value=0, value=0)
                rent = st.number_input("🏠 Rent (₹)", min_value=0.0, value=0.0, step=500.0)
                loan_repayment = st.number_input("💳 Loan Repayment (₹)", min_value=0.0, value=0.0, step=500.0)
            with col2:
                insurance = st.number_input("🛡️ Insurance (₹)", min_value=0.0, value=0.0, step=100.0)
                groceries = st.number_input("🛒 Groceries (₹)", min_value=0.0, value=0.0, step=100.0)
                transport = st.number_input("🚗 Transport (₹)", min_value=0.0, value=0.0, step=100.0)
                healthcare = st.number_input("🏥 Healthcare (₹)", min_value=0.0, value=0.0, step=100.0)
                education = st.number_input("📚 Education (₹)", min_value=0.0, value=0.0, step=100.0)
                eating_out = st.number_input("🍽️ Eating Out (₹)", min_value=0.0, value=0.0, step=100.0)
                entertainment = st.number_input("🎬 Entertainment (₹)", min_value=0.0, value=0.0, step=100.0)
                utilities = st.number_input("💡 Utilities (₹)", min_value=0.0, value=0.0, step=100.0)
                savings_rate = st.number_input("🎯 Savings Rate (%)", min_value=0.0, max_value=100.0, value=10.0)

            retirement_age = st.slider("👴 Retirement Age", int(age), 100, value=min(62, max(int(age), 62)))
            submit = st.form_submit_button("🚀 Analyze My Finances")

        if submit:
            st.session_state.finance_submit = True
            st.session_state.input_data = {
                "Income": income, "Age": age, "Dependents": dependents, "Rent": rent, "Loan_Repayment": loan_repayment,
                "Insurance": insurance, "Groceries": groceries, "Transport": transport, "Healthcare": healthcare,
                "Education": education, "Eating_Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
                "Desired_Savings_Percentage": savings_rate
            }
            st.session_state.total_expenses = sum([rent, loan_repayment, insurance, groceries, transport, healthcare, education, eating_out, entertainment, utilities])
            st.session_state.years_to_retirement = max(0, retirement_age - age)
            st.session_state.debt = rent + loan_repayment
            st.session_state.discretionary = eating_out + entertainment + utilities

        if st.session_state.finance_submit:
            st.subheader("🌍 Wealth Roadmap")
            dream_fund, suggested_rate, income_growth, expense_growth = smart_savings_plan(income, st.session_state.total_expenses, st.session_state.years_to_retirement)
            desired_fund = st.number_input("💎 Desired Retirement Fund (₹)", min_value=100000.0, value=max(100000.0, dream_fund), step=100000.0)
            savings_rate = st.slider("🎯 Savings Rate (%)", 0.0, 100.0, suggested_rate, step=1.0)
            income_growth = st.slider("📈 Income Growth (%)", 0.0, 10.0, income_growth, step=0.5)
            expense_growth = st.slider("📉 Expense Growth (%)", 0.0, 10.0, expense_growth, step=0.5)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Spending Snapshot")
                spending = pd.Series({"Rent": rent, "Loan": loan_repayment, "Insurance": insurance, "Groceries": groceries,
                                      "Transport": transport, "Healthcare": healthcare, "Education": education,
                                      "Eating Out": eating_out, "Entertainment": entertainment, "Utilities": utilities})
                fig, ax = plt.subplots(figsize=(8, 5))
                spending.plot(kind="bar", ax=ax)
                ax.set_title("Monthly Spending (₹)")
                ax.set_ylabel("Amount (₹)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                st.subheader("🌱 Wealth Growth")
                trajectory = wealth_trajectory(income, st.session_state.total_expenses, savings_rate, st.session_state.years_to_retirement, income_growth, expense_growth)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(st.session_state.years_to_retirement + 1), trajectory, marker="o", label="Wealth")
                ax.axhline(y=desired_fund, color="red", linestyle="--", label="Goal")
                ax.set_title("Wealth Growth (₹)")
                ax.set_xlabel("Years")
                ax.set_ylabel("Savings (₹)")
                ax.legend()
                st.pyplot(fig)

    with tab2:
        st.session_state.active_tab = "Stock Investments"
        st.header("📈 Stock Market Quest")
        st.markdown("Conquer the NIFTY CONSUMPTION index! 🌠")

        with st.sidebar:
            st.subheader("Stock Investments")
            st.write(f"📊 Stock Model R²: {stock_r2:.2f}")

            if st.session_state.stock_submit and stock_model is not None:
                future = pd.DataFrame({"Day": [1], "Month": [st.session_state.horizon % 12 or 12], "Year": [2025 + st.session_state.horizon // 12]})
                predicted_price = stock_model.predict(future)[0]
                st.session_state.predicted_price = predicted_price
            else:
                predicted_price = 0.0
                st.session_state.predicted_price = 0.0

            st.subheader("📌 Predicted Price")
            st.metric(f"In {st.session_state.horizon} Months (₹)", f"₹{predicted_price:,.2f}")

            st.subheader("💡 Investment Insights")
            st.write(f"🎯 Risk Level: {st.session_state.risk_tolerance}")
            st.write(f"📈 Horizon: {st.session_state.horizon} months")
            if stock_data is not None and st.session_state.stock_submit:
                predicted_growth = predicted_price - stock_data['Close'].iloc[-1]
            else:
                predicted_growth = 0.0
            st.write(f"💰 Predicted Growth: ₹{predicted_growth:,.2f}")

        with st.form(key="stock_form"):
            horizon = st.number_input("⏳ Investment Horizon (Months)", min_value=1, max_value=60, value=45)
            risk_tolerance = st.radio("🎲 Risk Appetite", ["Low", "Medium", "High"])
            stock_submit = st.form_submit_button("🚀 Analyze Stock Investments")

        if stock_submit:
            st.session_state.stock_submit = True
            st.session_state.horizon = horizon
            st.session_state.risk_tolerance = risk_tolerance

        if st.session_state.stock_submit:
            st.subheader("🔮 Price Prediction")
            future = pd.DataFrame({"Day": [1], "Month": [st.session_state.horizon % 12 or 12], "Year": [2025 + st.session_state.horizon // 12]})
            if stock_model is not None:
                predicted_price = stock_model.predict(future)[0]
            else:
                predicted_price = 0.0
            st.write(f"📌 Predicted Price in {st.session_state.horizon} months: **₹{predicted_price:,.2f}**")

            st.subheader("💼 Investment Playbook")
            portfolio = portfolio_advice(st.session_state.risk_tolerance)
            st.write(f"**Strategy**: {portfolio['Overview']}")
            for pick in portfolio["Picks"]:
                st.write(f"- {pick['Type']}: **{pick['Name']}** - {pick['Why']}")

            st.subheader("📉 NIFTY CONSUMPTION Trend")
            if stock_data is not None:
                fig = px.line(stock_data, x='Date', y='Close', title="Price Trend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Stock data unavailable. Please ensure 'NIFTY CONSUMPTION_daily_data.csv' is present.")

            if stock_data is not None:
                stock_subset = stock_data.copy()
                stock_subset['SMA_30'] = stock_subset['Close'].rolling(window=30).mean()
                stock_subset['Volatility'] = stock_subset['Close'].pct_change().rolling(window=30).std()

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📏 Moving Average")
                    fig_ma = px.line(stock_subset, x='Date', y=['Close', 'SMA_30'], title="30-Day SMA")
                    st.plotly_chart(fig_ma, use_container_width=True)
                with col2:
                    st.subheader("🌩️ Volatility")
                    fig_vol = px.line(stock_subset, x='Date', y='Volatility', title="30-Day Volatility")
                    st.plotly_chart(fig_vol, use_container_width=True)

            if stock_model is not None:
                if not os.path.exists("models"):
                    os.makedirs("models")
                joblib.dump(stock_model, "models/stock_model.pkl")

    st.markdown("---")
    st.write("✨ Powered by WealthWise | Built with ❤️ by xAI")

if __name__ == "__main__":
    main()
