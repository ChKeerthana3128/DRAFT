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

# Page Configuration
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# Data Loading Functions
@st.cache_data
def load_stock_data(csv_path="archive (3) 2/NIFTY CONSUMPTION_daily_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Stock CSV not found!")
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

@st.cache_data
def load_survey_data(csv_path="survey_data.csv"):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        # Convert categorical ranges to numeric midpoints for training
        def parse_range(value):
            if pd.isna(value) or value == "I don’t save":
                return 0
            if "Above" in value:
                return float(value.split("₹")[1].replace(",", "")) + 500
            if "₹" in value:
                bounds = value.split("₹")[1].split("-")
                return (float(bounds[0]) + float(bounds[1])) / 2
            return float(value)
        
        df["Income"] = df["How much pocket money or income do you receive per month (in ₹)?"].apply(parse_range)
        df["Essentials"] = df["How much do you spend monthly on essentials (e.g., food, transport, books)?"].apply(parse_range)
        df["Non_Essentials"] = df["How much do you spend monthly on non-essentials (e.g., entertainment, eating out)?"].apply(parse_range)
        df["Debt_Payment"] = df["If yes to debt, how much do you pay monthly (in ₹)?"].apply(parse_range)
        df["Savings"] = df["How much of your pocket money/income do you save each month (in ₹)?"].apply(parse_range)
        return df
    except Exception as e:
        st.error(f"Error loading survey data: {str(e)}")
        return None

# Model Training for Survey Data
@st.cache_resource
def train_survey_model(survey_data):
    features = ["Income", "Essentials", "Non_Essentials", "Debt_Payment"]
    target = "Savings"
    X = survey_data[features].fillna(0)
    y = survey_data[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

# Existing Predictive Functions
def calculate_financial_health_score(income, total_expenses, debt, discretionary):
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
    return max(0, income - total_expenses)

def forecast_wealth_growth(income, total_expenses, savings_rate, years, income_growth=0.0, expense_growth=0.0):
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
    dream_fund = max(100000.0, total_expenses * 12 * 20) if total_expenses > 0 else 100000.0
    annual_target = dream_fund / years_to_retirement if years_to_retirement > 0 else dream_fund
    savings_rate = min(max((annual_target / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    expense_growth = 2.5
    return dream_fund, savings_rate, income_growth, expense_growth

# Investment Suggestions
def get_investment_suggestions(risk_tolerance, horizon_months, disposable_income, survey_data=None):
    base_advice = portfolio_advice(risk_tolerance)
    suggestions = {"Overview": base_advice["Overview"], "Picks": []}
    investable_amount = disposable_income * 0.5
    
    if survey_data is not None:
        similar_users = survey_data[survey_data["What is your risk tolerance for investing?"] == risk_tolerance]
        common_goal = similar_users["What is your main goal for saving or investing?"].mode()[0] if not similar_users.empty else "No specific goal"
    else:
        common_goal = "No specific goal"

    for pick in base_advice["Picks"]:
        if risk_tolerance == "Low":
            amount = min(investable_amount, 5000)
        elif risk_tolerance == "Medium":
            amount = min(investable_amount, 10000)
        else:
            amount = investable_amount
        suggestions["Picks"].append({
            "Type": pick["Type"],
            "Name": pick["Name"],
            "Why": pick["Why"],
            "Suggested Amount (₹)": f"₹{amount:,.2f}"
        })
    suggestions["Goal Alignment"] = f"Matches peers aiming for: {common_goal}"
    suggestions["Horizon"] = f"Best for {horizon_months // 12}-year horizon" if horizon_months >= 12 else f"Best for <1-year horizon"
    return suggestions

def portfolio_advice(risk_tolerance):
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

# Main Application
def main():
    st.title("WealthWise Dashboard")

    # Load data
    stock_data = load_stock_data()
    survey_data = load_survey_data()
    if stock_data is None:
        st.warning("Stock Investments tab will not function without stock data.")
    if survey_data is None:
        st.warning("Survey data unavailable. Practical Representation will be limited.")

    # Train models
    stock_model, stock_r2 = None, 0.0
    if stock_data is not None:
        stock_model, stock_r2 = train_stock_model(stock_data)

    survey_model, survey_r2 = None, 0.0
    if survey_data is not None:
        survey_model, survey_r2 = train_survey_model(survey_data)

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
        st.session_state.horizon = 24
    if 'risk_tolerance' not in st.session_state:
        st.session_state.risk_tolerance = "Medium"
    if 'predicted_price' not in st.session_state:
        st.session_state.predicted_price = None

    # Sidebar: Needed Investments
    with st.sidebar:
        st.subheader("💡 Needed Investments")
        st.write("Personalized suggestions based on your profile")
        if st.session_state.active_tab == "Personal Finance" and st.session_state.finance_submit and st.session_state.input_data:
            disposable = predict_disposable_income(st.session_state.input_data["Income"], st.session_state.total_expenses)
            horizon_months = st.session_state.years_to_retirement * 12
            risk_tolerance = st.session_state.risk_tolerance
            suggestions = get_investment_suggestions(risk_tolerance, horizon_months, disposable, survey_data)
            st.write(f"**Strategy**: {suggestions['Overview']}")
            for pick in suggestions["Picks"]:
                st.write(f"- {pick['Type']}: **{pick['Name']}** - {pick['Why']} (Invest: {pick['Suggested Amount (₹)']})")
            st.write(f"**Goal**: {suggestions['Goal Alignment']}")
            st.write(f"**Horizon**: {suggestions['Horizon']}")
        elif st.session_state.active_tab == "Stock Investments" and st.session_state.stock_submit:
            disposable = 750  # Survey-based default (mode of savings)
            suggestions = get_investment_suggestions(st.session_state.risk_tolerance, st.session_state.horizon, disposable, survey_data)
            st.write(f"**Strategy**: {suggestions['Overview']}")
            for pick in suggestions["Picks"]:
                st.write(f"- {pick['Type']}: **{pick['Name']}** - {pick['Why']} (Invest: {pick['Suggested Amount (₹)']})")
            st.write(f"**Goal**: {suggestions['Goal Alignment']}")
            st.write(f"**Horizon**: {suggestions['Horizon']}")
        else:
            st.write("Submit your details to see investment suggestions!")

    # Tabs Definition
    tab1, tab2 = st.tabs(["💵 Personal Finance", "📈 Stock Investments"])

    # Personal Finance Tab
    with tab1:
        st.session_state.active_tab = "Personal Finance"
        st.header("💵 Your Money Mastery Hub")
        st.markdown("Shape your financial destiny with style! 🌈")

        with st.form(key="finance_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Name", "")
                age = st.number_input("🎂 Age", min_value=18, max_value=100, value=20)
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, value=3500.0, step=1000.0)
                dependents = st.number_input("👨‍👩‍👧 Dependents", min_value=0, value=0)
                rent = st.number_input("🏠 Rent (₹)", min_value=0.0, value=0.0, step=500.0)
                loan_repayment = st.number_input("💳 Loan Repayment (₹)", min_value=0.0, value=0.0, step=500.0)
            with col2:
                insurance = st.number_input("🛡️ Insurance (₹)", min_value=0.0, value=0.0, step=100.0)
                groceries = st.number_input("🛒 Groceries (₹)", min_value=0.0, value=2000.0, step=100.0)
                transport = st.number_input("🚗 Transport (₹)", min_value=0.0, value=0.0, step=100.0)
                healthcare = st.number_input("🏥 Healthcare (₹)", min_value=0.0, value=0.0, step=100.0)
                education = st.number_input("📚 Education (₹)", min_value=0.0, value=0.0, step=100.0)
                eating_out = st.number_input("🍽️ Eating Out (₹)", min_value=0.0, value=750.0, step=100.0)
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

            # New Section: Practical Representation
            st.subheader("📈 Practical Representation")
            st.markdown("Insights from survey-trained model")
            if survey_model is not None and st.session_state.input_data:
                input_df = pd.DataFrame({
                    "Income": [income],
                    "Essentials": [groceries + transport + healthcare + education],
                    "Non_Essentials": [eating_out + entertainment + utilities],
                    "Debt_Payment": [loan_repayment]
                })
                predicted_savings = survey_model.predict(input_df)[0]
                st.write(f"**Predicted Monthly Savings (Survey-Based)**: ₹{predicted_savings:,.2f}")
                st.write(f"**Model Accuracy (R²)**: {survey_r2:.2f}")
                if survey_data is not None:
                    avg_savings = survey_data["Savings"].mean()
                    st.write(f"**Average Savings Among Peers**: ₹{avg_savings:,.2f}")
                    if predicted_savings > avg_savings:
                        st.success("You're saving more than the average survey respondent!")
                    else:
                        st.warning("Consider increasing savings to match peers.")
            else:
                st.write("Survey model unavailable. Please ensure survey_data.csv is present.")

    # Stock Investments Tab
    with tab2:
        st.session_state.active_tab = "Stock Investments"
        st.header("📈 Stock Market Quest")
        st.markdown("Conquer the NIFTY CONSUMPTION index! 🌠")

        with st.form(key="stock_form"):
            horizon = st.number_input("⏳ Investment Horizon (Months)", min_value=1, max_value=60, value=24)
            risk_tolerance = st.radio("🎲 Risk Appetite", ["Low", "Medium", "High"], index=1)
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
                st.write("Stock data unavailable.")

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

            # New Section: Practical Representation
            st.subheader("📈 Practical Representation")
            st.markdown("Insights from survey-trained model")
            if survey_model is not None and survey_data is not None:
                avg_income = survey_data["Income"].mean()
                avg_essentials = survey_data["Essentials"].mean()
                avg_non_essentials = survey_data["Non_Essentials"].mean()
                avg_debt = survey_data["Debt_Payment"].mean()
                input_df = pd.DataFrame({
                    "Income": [avg_income],
                    "Essentials": [avg_essentials],
                    "Non_Essentials": [avg_non_essentials],
                    "Debt_Payment": [avg_debt]
                })
                predicted_savings = survey_model.predict(input_df)[0]
                st.write(f"**Predicted Savings for Average Survey User**: ₹{predicted_savings:,.2f}")
                st.write(f"**Model Accuracy (R²)**: {survey_r2:.2f}")
                risk_dist = survey_data["What is your risk tolerance for investing?"].value_counts(normalize=True) * 100
                st.write(f"**Peer Risk Tolerance**:")
                for risk, pct in risk_dist.items():
                    st.write(f"- {risk}: {pct:.1f}%")
            else:
                st.write("Survey model unavailable. Please ensure survey_data.csv is present.")

    st.markdown("---")
    st.write("✨ Powered by WealthWise | Built with ❤️ by xAI")

if __name__ == "__main__":
    main()
