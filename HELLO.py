import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# Set page configuration with a vibrant theme
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for colorful styling
st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}
    .sidebar .sidebar-content {background-color: #e6f3ff;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stMetric {background-color: #fff3e6; border: 1px solid #ffa500; border-radius: 8px; padding: 10px;}
    .stExpander {background-color: #f9f9f9; border-radius: 8px;}
    h1, h2 {color: #2E8B57;}
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_finance_data(csv_path="financial_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Finance CSV not found! Please upload 'financial_data.csv'.")
        return None
    try:
        data = pd.read_csv(csv_path)
        combined_col = "Miscellaneous (Eating_Out,Entertainmentand Utilities)\n"
        if combined_col not in data.columns:
            st.error(f"🚨 Missing '{combined_col}' column. Available: {data.columns.tolist()}")
            return None
        data[['Eating_Out', 'Entertainment', 'Utilities']] = data[combined_col].apply(
            lambda x: [x/3]*3 if pd.notna(x) and x > 0 else [0, 0, 0]
        ).apply(pd.Series)
        data = data.drop(columns=[combined_col])
        if "Education\n" in data.columns:
            data = data.rename(columns={"Education\n": "Education"})
        elif "Education" not in data.columns:
            data["Education"] = 0
        required_cols = ["Income", "Age", "Dependents", "Occupation", "City_Tier", "Rent", "Loan_Repayment", 
                         "Insurance", "Groceries", "Transport", "Healthcare", "Education", "Eating_Out", 
                         "Entertainment", "Utilities", "Desired_Savings_Percentage", "Disposable_Income"]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            st.error(f"🚨 Missing columns: {missing}")
            return None
        return data
    except Exception as e:
        st.error(f"🚨 Error loading finance data: {str(e)}")
        return None

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

# --- Model Training ---
@st.cache_resource
def train_finance_model(data):
    features = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", "Groceries", 
                "Transport", "Healthcare", "Education", "Eating_Out", "Entertainment", "Utilities", 
                "Desired_Savings_Percentage"]
    X = pd.get_dummies(data[features + ["Occupation", "City_Tier"]], columns=["Occupation", "City_Tier"])
    y = data["Disposable_Income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

# --- Predictive Functions ---
def prepare_finance_input(input_data, model):
    features = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", "Groceries", 
                "Transport", "Healthcare", "Education", "Eating_Out", "Entertainment", "Utilities", 
                "Desired_Savings_Percentage"]
    input_dict = {col: input_data.get(col, 0.0 if col != "Desired_Savings_Percentage" else 10.0) for col in features}
    input_df = pd.DataFrame([input_dict], columns=features)
    input_df["Occupation"] = "Unknown"
    input_df["City_Tier"] = "Unknown"
    input_df = pd.get_dummies(input_df, columns=["Occupation", "City_Tier"])
    for feature in model.feature_names_in_:
        if feature not in input_df.columns:
            input_df[feature] = 0
    return input_df[model.feature_names_in_]

def calculate_financial_health_score(income, total_expenses, debt, discretionary):
    """🌟 Calculate your financial wellness score (0-100) based on real-world benchmarks."""
    if income <= 0:
        return 0
    savings = max(0, income - total_expenses)
    savings_ratio = savings / income
    debt_ratio = debt / income
    discretionary_ratio = discretionary / income
    # Adjusted to align with CSV patterns
    savings_score = min(25, (savings_ratio / 0.2) * 25)  # 25 points for 20% savings
    debt_score = max(0, 50 - (debt_ratio * 100))  # 50 points, heavy debt penalty
    discretionary_score = max(0, 25 - (discretionary_ratio * 125))  # 25 points, moderate penalty
    return max(0, min(100, savings_score + debt_score + discretionary_score))

def predict_disposable_income(model, input_data):
    return max(0, model.predict(prepare_finance_input(input_data, model))[0])

def forecast_wealth_growth(income, total_expenses, savings_rate, years, income_growth=0.0, expense_growth=0.0):
    """📈 Project your financial future with optimism!"""
    wealth = 0.0
    current_income, current_expenses = income, total_expenses
    for _ in range(years + 1):
        annual_savings = max(0, (current_income * (savings_rate / 100)) - current_expenses)
        wealth += annual_savings
        current_income *= (1 + income_growth / 100)
        current_expenses *= (1 + expense_growth / 100)
    return wealth

def wealth_trajectory(income, total_expenses, savings_rate, years, income_growth, expense_growth):
    trajectory = []
    wealth = 0.0
    current_income, current_expenses = income, total_expenses
    for _ in range(years + 1):
        annual_savings = max(0, (current_income * (savings_rate / 100)) - current_expenses)
        wealth += annual_savings
        trajectory.append(wealth)
        current_income *= (1 + income_growth / 100)
        current_expenses *= (1 + expense_growth / 100)
    return trajectory

def smart_savings_plan(income, total_expenses, years_to_retirement):
    """🧠 Craft a clever savings strategy!"""
    dream_fund = total_expenses * 12 * 20
    annual_target = dream_fund / years_to_retirement if years_to_retirement > 0 else dream_fund
    savings_rate = min(max((annual_target / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    expense_growth = 2.5
    return dream_fund, savings_rate, income_growth, expense_growth

# --- Insight Generators ---
def financial_wisdom(health_score, debt, discretionary, income, total_expenses):
    insights = [f"🌟 Your Financial Health Score: {health_score:.1f}/100"]
    status = "🎉 Excellent" if health_score >= 70 else "👍 Moderate" if health_score >= 40 else "⚠️ Needs Work"
    insights.append(f"Status: {status}")
    debt_ratio = debt / income if income > 0 else 0
    if debt_ratio > 0.36:
        insights.append(f"💸 Debt (₹{debt:,.2f}) is {debt_ratio:.1%} of income—above 36%. Time to strategize!")
    discretionary_ratio = discretionary / income if income > 0 else 0
    if discretionary_ratio > 0.15:
        insights.append(f"🎭 Discretionary (₹{discretionary:,.2f}) at {discretionary_ratio:.1%}—over 15%. Trim the fun?")
    savings_ratio = max(0, income - total_expenses) / income if income > 0 else 0
    if savings_ratio < 0.2:
        insights.append(f"💰 Savings at {savings_ratio:.1%}—below 20%. Boost it up!")
    return insights

def portfolio_advice(risk_tolerance):
    """💼 Your personalized investment playbook!"""
    if risk_tolerance == "Low":
        return {
            "Overview": "🌳 Steady growth with minimal risk!",
            "Picks": [
                {"Type": "Blue-Chip", "Name": "HDFC Bank 🏦", "Why": "Reliable dividends, strong banking leader."},
                {"Type": "Blue-Chip", "Name": "TCS 💻", "Why": "Stable IT giant with global reach."},
                {"Type": "Bonds", "Name": "RBI Bonds 📜", "Why": "Safe, guaranteed returns."}
            ]
        }
    elif risk_tolerance == "Medium":
        return {
            "Overview": "⚖️ Balanced growth with calculated risks!",
            "Picks": [
                {"Type": "Large Cap", "Name": "Reliance Industries 🏭", "Why": "Diversified powerhouse."},
                {"Type": "Mid Cap", "Name": "Bajaj Finance 📈", "Why": "High-growth financial player."},
                {"Type": "Real Estate", "Name": "DLF 🏡", "Why": "Steady property appreciation."},
                {"Type": "Mutual Fund", "Name": "SBI Bluechip Fund 🌟", "Why": "Diversified with moderate risk."}
            ]
        }
    else:
        return {
            "Overview": "🚀 High stakes, high rewards!",
            "Picks": [
                {"Type": "Small Cap", "Name": "Paytm 💳", "Why": "Fintech innovator with big potential."},
                {"Type": "Small Cap", "Name": "Zomato 🍽️", "Why": "Rapidly expanding food tech."},
                {"Type": "Crypto", "Name": "Bitcoin ₿", "Why": "High-risk, high-reward asset."}
            ]
        }

# --- Main Application ---
def main():
    finance_data = load_finance_data()
    stock_data = load_stock_data()
    if finance_data is None or stock_data is None:
        st.stop()

    finance_model, finance_r2 = train_finance_model(finance_data)

    # Sidebar
    st.sidebar.title("🌟 WealthWise Insights")
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"📊 Finance Model R²: {finance_r2:.2f}")

    tabs = st.tabs(["💵 Personal Finance", "📈 Stock Investments"])

    # --- Personal Finance Dashboard ---
    with tabs[0]:
        st.header("💵 Your Personal Finance Journey")
        st.markdown("Plan your financial future with flair! 🌈")

        with st.form(key="finance_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Name", "Amit Sharma")
                age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
                dependents = st.number_input("👨‍👩‍👧 Dependents", min_value=0, value=0)
                rent = st.number_input("🏠 Rent (₹)", min_value=0.0, value=15000.0, step=500.0)
                loan_repayment = st.number_input("💳 Loan Repayment (₹)", min_value=0.0, value=0.0, step=500.0)
            with col2:
                insurance = st.number_input("🛡️ Insurance (₹)", min_value=0.0, value=2000.0, step=100.0)
                groceries = st.number_input("🛒 Groceries (₹)", min_value=0.0, value=8000.0, step=100.0)
                transport = st.number_input("🚗 Transport (₹)", min_value=0.0, value=3000.0, step=100.0)
                healthcare = st.number_input("🏥 Healthcare (₹)", min_value=0.0, value=1500.0, step=100.0)
                education = st.number_input("📚 Education (₹)", min_value=0.0, value=0.0, step=100.0)
                eating_out = st.number_input("🍽️ Eating Out (₹)", min_value=0.0, value=2000.0, step=100.0)
                entertainment = st.number_input("🎬 Entertainment (₹)", min_value=0.0, value=1500.0, step=100.0)
                utilities = st.number_input("💡 Utilities (₹)", min_value=0.0, value=4000.0, step=100.0)
                savings_rate = st.number_input("🎯 Savings Rate (%)", min_value=0.0, max_value=100.0, value=10.0)

            retirement_age = st.slider("👴 Retirement Age", int(age), 62, min(62, age + 30))
            submit = st.form_submit_button("🚀 Analyze My Finances")

        if submit:
            input_data = {
                "Income": income, "Age": age, "Dependents": dependents, "Rent": rent, "Loan_Repayment": loan_repayment,
                "Insurance": insurance, "Groceries": groceries, "Transport": transport, "Healthcare": healthcare,
                "Education": education, "Eating_Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
                "Desired_Savings_Percentage": savings_rate
            }
            total_expenses = sum([rent, loan_repayment, insurance, groceries, transport, healthcare, education, eating_out, entertainment, utilities])
            years_to_retirement = max(0, retirement_age - age)
            debt = rent + loan_repayment
            discretionary = eating_out + entertainment + utilities

            st.sidebar.subheader("🌡️ Financial Health")
            health_score = calculate_financial_health_score(income, total_expenses, debt, discretionary)
            st.sidebar.metric("Score", f"{health_score:.1f}/100", delta=f"{health_score-50:.1f}")

            st.sidebar.subheader("💸 Disposable Income")
            disposable = predict_disposable_income(finance_model, input_data)
            st.sidebar.metric("Monthly (₹)", f"₹{disposable:,.2f}")

            st.subheader("🌍 Wealth Roadmap")
            dream_fund, suggested_rate, income_growth, expense_growth = smart_savings_plan(income, total_expenses, years_to_retirement)
            desired_fund = st.number_input("💎 Desired Retirement Fund (₹)", min_value=100000.0, value=dream_fund, step=100000.0)
            savings_rate = st.slider("🎯 Savings Rate (%)", 0.0, 100.0, suggested_rate, step=1.0)
            income_growth = st.slider("📈 Income Growth (%)", 0.0, 10.0, income_growth, step=0.5)
            expense_growth = st.slider("📉 Expense Growth (%)", 0.0, 10.0, expense_growth, step=0.5)

            wealth = forecast_wealth_growth(income, total_expenses, savings_rate, years_to_retirement, income_growth, expense_growth)
            st.sidebar.subheader("🏦 Future Wealth")
            st.sidebar.metric("At Retirement (₹)", f"₹{wealth:,.2f}")

            with st.sidebar.expander("💡 Financial Wisdom"):
                for tip in financial_wisdom(health_score, debt, discretionary, income, total_expenses):
                    st.write(tip)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Spending Snapshot")
                spending = pd.Series({"Rent": rent, "Loan": loan_repayment, "Insurance": insurance, "Groceries": groceries,
                                      "Transport": transport, "Healthcare": healthcare, "Education": education,
                                      "Eating Out": eating_out, "Entertainment": entertainment, "Utilities": utilities})
                fig, ax = plt.subplots(figsize=(8, 5))
                spending.plot(kind="bar", ax=ax, color="#87CEEB")
                ax.set_title("Monthly Spending (₹)", color="#2E8B57")
                ax.set_ylabel("Amount (₹)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                st.subheader("🌱 Wealth Growth")
                trajectory = wealth_trajectory(income, total_expenses, savings_rate, years_to_retirement, income_growth, expense_growth)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(years_to_retirement + 1), trajectory, marker="o", color="#32CD32", label="Wealth")
                ax.axhline(y=desired_fund, color="red", linestyle="--", label="Goal")
                ax.set_title("Wealth Growth (₹)", color="#2E8B57")
                ax.set_xlabel("Years")
                ax.set_ylabel("Savings (₹)")
                ax.legend()
                st.pyplot(fig)

    # --- Stock Investment Dashboard ---
    with tabs[1]:
        st.header("📈 Stock Market Adventure")
        st.markdown("Explore NIFTY CONSUMPTION and plan your investments! 🌠")

        horizon = st.sidebar.slider("⏳ Investment Horizon (Months)", 1, 60, 12)
        risk_tolerance = st.sidebar.radio("🎲 Risk Appetite", ["Low", "Medium", "High"])

        st.subheader("📉 NIFTY CONSUMPTION Trend")
        fig = px.line(stock_data, x='Date', y='Close', title="Price Trend", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        stock_subset = stock_data.copy()
        stock_subset['SMA_30'] = stock_subset['Close'].rolling(window=30).mean()
        stock_subset['Volatility'] = stock_subset['Close'].pct_change().rolling(window=30).std()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📏 Moving Average")
            fig_ma = px.line(stock_subset, x='Date', y=['Close', 'SMA_30'], title="30-Day SMA", template="plotly_dark")
            st.plotly_chart(fig_ma, use_container_width=True)
        with col2:
            st.subheader("🌩️ Volatility")
            fig_vol = px.line(stock_subset, x='Date', y='Volatility', title="30-Day Volatility", template="plotly_dark")
            st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("🔮 Price Prediction")
        stock_subset['Day'] = stock_subset['Date'].dt.day
        stock_subset['Month'] = stock_subset['Date'].dt.month
        stock_subset['Year'] = stock_subset['Date'].dt.year
        X = stock_subset[['Day', 'Month', 'Year']]
        y = stock_subset['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        stock_model = RandomForestRegressor(n_estimators=100, random_state=42)
        stock_model.fit(X_train, y_train)
        future = pd.DataFrame({"Day": [1], "Month": [horizon % 12 or 12], "Year": [2025 + horizon // 12]})
        predicted_price = stock_model.predict(future)[0]
        st.write(f"📌 Predicted Price in {horizon} months: **₹{predicted_price:,.2f}**")

        st.subheader("💼 Investment Playbook")
        portfolio = portfolio_advice(risk_tolerance)
        st.write(f"**Strategy**: {portfolio['Overview']}")
        for pick in portfolio["Picks"]:
            st.write(f"- {pick['Type']}: **{pick['Name']}** - {pick['Why']}")

        if not os.path.exists("models"):
            os.makedirs("models")
        joblib.dump(stock_model, "models/stock_model.pkl")

    st.markdown("---")
    st.write("✨ Powered by WealthWise | Built with ❤️ by xAI")

if __name__ == "__main__":
    main()
