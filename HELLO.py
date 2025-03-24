# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import PageTemplate, BaseDocTemplate, Frame
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import os
import warnings

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

def load_survey_data(csv_path="survey_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Survey CSV not found!")
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]

        def parse_range(value):
            if pd.isna(value) or value in ["I don’t save", ""]:
                return 0
            if "Above" in value:
                return float(value.split("₹")[1].replace(",", "")) + 500
            if "₹" in value:
                bounds = value.split("₹")[1].split("-")
                if len(bounds) == 2:
                    return (float(bounds[0].replace(",", "")) + float(bounds[1].replace(",", ""))) / 2
                return float(bounds[0].replace(",", ""))
            return float(value)

        df["Income"] = df["How much pocket money or income do you receive per month (in ₹)?"].apply(parse_range)
        df["Essentials"] = df["How much do you spend monthly on essentials (e.g., food, transport, books)?"].apply(parse_range)
        df["Non_Essentials"] = df["How much do you spend monthly on non-essentials (e.g., entertainment, eating out)?"].apply(parse_range)
        df["Debt_Payment"] = df["If yes to debt, how much do you pay monthly (in ₹)?"].apply(parse_range)
        df["Savings"] = df["How much of your pocket money/income do you save each month (in ₹)?"].apply(parse_range)
        return df
    except Exception as e:
        st.error(f"🚨 Error loading survey data: {str(e)}")
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

# Predictive and Utility Functions
# ... (previous code remains unchanged up to smart_savings_plan)

def smart_savings_plan(income, total_expenses, years_to_retirement):
    """🧠 Craft your retirement blueprint!"""
    dream_fund = max(100000.0, total_expenses * 12 * 20) if total_expenses > 0 else 100000.0
    annual_target = dream_fund / years_to_retirement if years_to_retirement > 0 else dream_fund
    savings_rate = min(max((annual_target / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    expense_growth = 2.5
    return dream_fund, savings_rate, income_growth, expense_growth

# Predictive and Utility Functions
def predict_savings(model, income, essentials, non_essentials, debt_payment):
    input_df = pd.DataFrame({"Income": [income], "Essentials": [essentials], "Non_Essentials": [non_essentials], "Debt_Payment": [debt_payment]})
    return model.predict(input_df)[0]

def calculate_savings_goal(goal_amount, horizon_years):
    return goal_amount / (horizon_years * 12) if horizon_years > 0 else goal_amount

def get_investment_recommendations(income, savings, goal, risk_tolerance, horizon_years):
    investable = savings * 0.5
    recommendations = []
    if risk_tolerance == "Low":
        options = [
            {"Company": "HDFC Bank", "Type": "Blue-Chip", "Min_Invest": 500, "Goal": "Emergency fund"},
            {"Company": "TCS", "Type": "Blue-Chip", "Min_Invest": 500, "Goal": "Wealth growth"},
            {"Company": "RBI Bonds", "Type": "Bonds", "Min_Invest": 1000, "Goal": "Future expenses"}
        ]
    elif risk_tolerance == "Medium":
        options = [
            {"Company": "Reliance Industries", "Type": "Large Cap", "Min_Invest": 1000, "Goal": "Wealth growth"},
            {"Company": "Bajaj Finance", "Type": "Mid Cap", "Min_Invest": 1500, "Goal": "Future expenses"},
            {"Company": "SBI Bluechip Fund", "Type": "Mutual Fund", "Min_Invest": 500, "Goal": "Emergency fund"}
        ]
    else:
        options = [
            {"Company": "Paytm", "Type": "Small Cap", "Min_Invest": 2000, "Goal": "Wealth growth"},
            {"Company": "Zomato", "Type": "Small Cap", "Min_Invest": 2000, "Goal": "Future expenses"},
            {"Company": "Bitcoin", "Type": "Crypto", "Min_Invest": 5000, "Goal": "No specific goal"}
        ]
    
    for opt in options:
        if investable >= opt["Min_Invest"] and (goal == opt["Goal"] or goal == "No specific goal"):
            amount = min(investable, opt["Min_Invest"] * (horizon_years if horizon_years > 1 else 1))
            recommendations.append({"Company": opt["Company"], "Type": opt["Type"], "Amount": amount})
    return recommendations

def generate_pdf(name, income, savings, goal, risk_tolerance, horizon_years, recommendations, peer_savings, tips):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Investment Plan for {name}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Income: ₹{income:,.2f}", styles['Normal']))
    story.append(Paragraph(f"Predicted Savings: ₹{savings:,.2f}", styles['Normal']))
    story.append(Paragraph(f"Goal: {goal}", styles['Normal']))
    story.append(Paragraph(f"Risk Tolerance: {risk_tolerance}", styles['Normal']))
    story.append(Paragraph(f"Investment Horizon: {horizon_years} years", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Investment Recommendations", styles['Heading2']))
    data = [["Company", "Type", "Amount (₹)"]] + [[r["Company"], r["Type"], f"₹{r['Amount']:,.2f}"] for r in recommendations]
    story.append(Table(data))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Peer Average Savings: ₹{peer_savings:,.2f}", styles['Normal']))
    story.append(Paragraph("Budget Tips", styles['Heading2']))
    for tip in tips:
        story.append(Paragraph(f"- {tip}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

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

    # Load survey data (for Personalized Investment tab)
    survey_data = load_survey_data()
    if survey_data is not None:
        survey_model, survey_r2 = train_survey_model(survey_data)
    else:
        survey_model, survey_r2 = None, 0.0

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

    # Define tabs (corrected indentation)
    tab1, tab2, tab3 = st.tabs(["💵 Personal Finance", "📈 Stock Investments", "🎯 Personalized Investment"])

    # --- Personal Finance Dashboard ---
    with tab1:
        st.session_state.active_tab = "Personal Finance"
        st.header("💵 Your Money Mastery Hub")
        st.markdown("Shape your financial destiny with style! 🌈")

        # Sidebar for Personal Finance
        with st.sidebar:
            st.subheader("Personal Finance")
            st.write("📊 Finance Model R²: N/A (Using form inputs directly)")

            st.subheader("🌡️ Financial Health")
            if st.session_state.finance_submit and st.session_state.input_data:
                health_score = calculate_financial_health_score(
                    st.session_state.input_data["Income"],
                    st.session_state.total_expenses,
                    st.session_state.debt,
                    st.session_state.discretionary
                )
                st.metric("Score", f"{health_score:.1f}/100", delta=f"{health_score-50:.1f}")
            else:
                st.metric("Score", "N/A")

            st.subheader("💸 Disposable Income")
            if st.session_state.finance_submit and st.session_state.input_data:
                disposable = predict_disposable_income(
                    st.session_state.input_data["Income"],
                    st.session_state.total_expenses
                )
                st.metric("Monthly (₹)", f"₹{disposable:,.2f}")
            else:
                st.metric("Monthly (₹)", "N/A")

            st.subheader("🏦 Future Wealth")
            if st.session_state.finance_submit and st.session_state.input_data:
                dream_fund, suggested_rate, income_growth, expense_growth = smart_savings_plan(
                    st.session_state.input_data["Income"],
                    st.session_state.total_expenses,
                    st.session_state.years_to_retirement
                )
                wealth = forecast_wealth_growth(
                    st.session_state.input_data["Income"],
                    st.session_state.total_expenses,
                    suggested_rate,
                    st.session_state.years_to_retirement,
                    income_growth,
                    expense_growth
                )
                st.metric("At Retirement (₹)", f"₹{wealth:,.2f}")
            else:
                st.metric("At Retirement (₹)", "N/A")

        # Main content
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

        # Update session state on form submission
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

        # Main content after submission
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

    # --- Stock Investments Dashboard ---
    with tab2:
        st.session_state.active_tab = "Stock Investments"
        st.header("📈 Stock Market Quest")
        st.markdown("Conquer the NIFTY CONSUMPTION index! 🌠")

        # Sidebar for Stock Investments
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

        # Main content with form
        with st.form(key="stock_form"):
            horizon = st.number_input("⏳ Investment Horizon (Months)", min_value=1, max_value=60, value=45)
            risk_tolerance = st.radio("🎲 Risk Appetite", ["Low", "Medium", "High"])
            stock_submit = st.form_submit_button("🚀 Analyze Stock Investments")

        # Update session state on form submission
        if stock_submit:
            st.session_state.stock_submit = True
            st.session_state.horizon = horizon
            st.session_state.risk_tolerance = risk_tolerance

        # Display results only after submission
        if st.session_state.stock_submit:
            # Price Prediction
            st.subheader("🔮 Price Prediction")
            future = pd.DataFrame({"Day": [1], "Month": [st.session_state.horizon % 12 or 12], "Year": [2025 + st.session_state.horizon // 12]})
            if stock_model is not None:
                predicted_price = stock_model.predict(future)[0]
            else:
                predicted_price = 0.0
            st.write(f"📌 Predicted Price in {st.session_state.horizon} months: **₹{predicted_price:,.2f}**")

            # Investment Playbook
            st.subheader("💼 Investment Playbook")
            portfolio = portfolio_advice(st.session_state.risk_tolerance)
            st.write(f"**Strategy**: {portfolio['Overview']}")
            for pick in portfolio["Picks"]:
                st.write(f"- {pick['Type']}: **{pick['Name']}** - {pick['Why']}")

            # NIFTY CONSUMPTION Trend
            st.subheader("📉 NIFTY CONSUMPTION Trend")
            if stock_data is not None:
                fig = px.line(stock_data, x='Date', y='Close', title="Price Trend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Stock data unavailable. Please ensure 'NIFTY CONSUMPTION_daily_data.csv' is present.")

            # Moving Average and Volatility
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

    # --- Personalized Investment Tab ---
    with tab3:
        st.session_state.active_tab = "Personalized Investment"
        st.header("🎯 Personalized Investment Planner")
        st.markdown("Tailor your investment journey with insights from peers!")

        with st.form(key="investment_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Name", "")
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                essentials = st.number_input("🍲 Essentials (₹)", min_value=0.0, step=100.0)
                non_essentials = st.number_input("🎉 Non-Essentials (₹)", min_value=0.0, step=100.0)
                debt_payment = st.number_input("💳 Debt Payment (₹)", min_value=0.0, step=100.0)
            with col2:
                goal = st.selectbox("🎯 Investment Goal", ["No specific goal", "Emergency fund", "Future expenses (e.g., education, travel)", "Wealth growth"])
                goal_amount = st.number_input("💎 Goal Amount (₹)", min_value=0.0, step=1000.0, value=50000.0)
                risk_tolerance = st.selectbox("🎲 Risk Tolerance", ["Low", "Medium", "High"])
                horizon_years = st.slider("⏳ Investment Horizon (Years)", 1, 10, 3)
            submit = st.form_submit_button("🚀 Get Your Plan")

        if submit and survey_data is not None and survey_model is not None:
            # Map form risk tolerance to survey data format
            risk_mapping = {
                "Low": "Low (prefer safe options)",
                "Medium": "Medium (okay with some risk)",
                "High": "High (comfortable with big risks)"
            }
            survey_risk_tolerance = risk_mapping[risk_tolerance]
            # 1. Personalized Investment Recommendations
            predicted_savings = predict_savings(survey_model, income, essentials, non_essentials, debt_payment)
            recommendations = get_investment_recommendations(income, predicted_savings, goal, risk_tolerance, horizon_years)
            st.subheader("💼 Investment Recommendations")
            for rec in recommendations:
                st.write(f"- Invest ₹{rec['Amount']:,.2f} in **{rec['Company']}** ({rec['Type']})")

            # 2. Savings Goal Calculator
            monthly_savings_needed = calculate_savings_goal(goal_amount, horizon_years)
            st.subheader("🎯 Savings Goal")
            st.write(f"To reach ₹{goal_amount:,.2f} in {horizon_years} years, save ₹{monthly_savings_needed:,.2f}/month.")

            # 3. Peer Comparison Dashboard
            peer_avg_savings = survey_data["Savings"].mean()
            st.subheader("📊 Peer Comparison")
            fig, ax = plt.subplots()
            ax.bar(["Your Savings", "Peer Average"], [predicted_savings, peer_avg_savings], color=["#1f77b4", "#ff7f0e"])
            ax.set_ylabel("Amount (₹)")
            st.pyplot(fig)

            # 4. Investment Education Module
            st.subheader("📚 Investment Basics")
            familiarity = survey_data["How familiar are you with stock investments?"].mode()[0]
            if familiarity == "Not at all":
                st.write("**New to Investing?** Start with safe options like Blue-Chip stocks (e.g., HDFC Bank) which offer stability.")
            elif familiarity in ["Slightly familiar", "Moderately familiar"]:
                st.write("**Building Knowledge?** Mutual Funds (e.g., SBI Bluechip) balance risk and reward.")
            else:
                st.write("**Experienced?** Explore high-growth options like Small Caps (e.g., Paytm).")
            # 5. Risk Tolerance Assessment (Fixed)
            st.subheader("🎲 Your Risk Profile")
            risk_count = survey_data["What is your risk tolerance for investing?"].value_counts()
            st.write(f"Your risk tolerance ({risk_tolerance}) matches {risk_count[survey_risk_tolerance]}/{len(survey_data)} peers ({risk_count[survey_risk_tolerance]/len(survey_data)*100:.1f}%).")

            # 6. from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import PageTemplate, BaseDocTemplate, Frame
from reportlab.lib.units import inch
import io

def generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_savings, tips):
    buffer = io.BytesIO()
    
    class MyDocTemplate(BaseDocTemplate):
        def __init__(self, filename, **kwargs):
            BaseDocTemplate.__init__(self, filename, **kwargs)
            frame = Frame(0.5*inch, 0.5*inch, 7.5*inch, 10*inch, id='normal')
            self.addPageTemplates([PageTemplate(id='AllPages', frames=frame, onPage=self.header_footer)])
        
        def header_footer(self, canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica-Bold", 12)
            canvas.setFillColor(colors.darkblue)
            canvas.drawString(0.5*inch, 10.5*inch, "💰 WealthWise Investment Plan")
            canvas.setFont("Helvetica", 10)
            canvas.setFillColor(colors.black)
            canvas.drawRightString(8*inch, 10.5*inch, f"For: {name}")
            canvas.line(0.5*inch, 10.4*inch, 8*inch, 10.4*inch)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.gray)
            canvas.drawString(0.5*inch, 0.3*inch, "✨ Powered by WealthWise | Built with ❤️ by xAI")
            canvas.drawRightString(8*inch, 0.3*inch, f"Page {doc.page}")
            canvas.line(0.5*inch, 0.4*inch, 8*inch, 0.4*inch)
            canvas.restoreState()

    doc = MyDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = styles['Title']
    title_style.fontSize = 16
    title_style.textColor = colors.darkblue
    heading_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=12, textColor=colors.darkgreen)
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    tip_style = ParagraphStyle('Tip', parent=styles['Normal'], fontSize=10, textColor=colors.darkred, leftIndent=20)

    story = []

    story.append(Paragraph(f"Your Personalized Investment Plan, {name}", title_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Financial Summary", heading_style))
    summary_data = [
        ["Income", f"₹{income:,.2f}"],
        ["Predicted Savings", f"₹{predicted_savings:,.2f}"],
        ["Goal", goal],
        ["Risk Tolerance", risk_tolerance],
        ["Investment Horizon", f"{horizon_years} years"]
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 5*inch])
    summary_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ])
    story.append(summary_table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Investment Recommendations", heading_style))
    rec_data = [["Company", "Type", "Amount (₹)"]] + [[r["Company"], r["Type"], f"₹{r['Amount']:,.2f}"] for r in recommendations]
    rec_table = Table(rec_data, colWidths=[2*inch, 2*inch, 2*inch])
    rec_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ])
    story.append(rec_table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Budget Tips", heading_style))
    if not tips:
        story.append(Paragraph("You're already on track! Keep up the good work.", normal_style))
    for tip in tips:
        story.append(Paragraph(f"• {tip}", tip_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Personalized Insights", heading_style))
    insights = []
    savings_ratio = predicted_savings / income if income > 0 else 0
    if savings_ratio < 0.1:
        insights.append("Your savings rate is below 10%. Consider automating savings to build wealth faster.")
    elif savings_ratio > 0.3:
        insights.append("Excellent savings rate (>30%)! You’re well-positioned for aggressive investments.")
    if horizon_years > 5 and risk_tolerance == "Low":
        insights.append("With a long horizon, you might explore medium-risk options for higher returns.")
    if income < peer_savings:
        insights.append("Your income is below peer average savings. Small side hustles could boost your funds!")
    for insight in insights:
        story.append(Paragraph(f"• {insight}", normal_style))
    if not insights:
        story.append(Paragraph("Your plan is solid—stay the course!", normal_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Visual Insights", heading_style))
    savings_vs_goal = f"Your savings (₹{predicted_savings:,.2f}) vs. Goal (₹{goal_amount:,.2f}): A bar chart shows you’re {(predicted_savings / goal_amount * 100 if goal_amount > 0 else 0):.1f}% toward your goal."
    peer_comparison = f"Peer Comparison: Your savings (₹{predicted_savings:,.2f}) vs. Peer Average (₹{peer_savings:,.2f})—you’re {'above' if predicted_savings > peer_savings else 'below'} average."
    story.append(Paragraph(savings_vs_goal, normal_style))
    story.append(Paragraph(peer_comparison, normal_style))
    story.append(Spacer(1, 0.2*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

            # 7. Budget Optimization Tips
    st.subheader("💡 Budget Optimization Tips")
    median_non_essentials = survey_data["Non_Essentials"].median()
    if non_essentials > median_non_essentials:
    st.write(f"- Cut ₹{non_essentials - median_non_essentials:,.2f} from non-essentials (peer median: ₹{median_non_essentials:,.2f}).")
    else:
    st.write("- Your spending is optimized compared to peers!")
            # 8. Goal-Based Investment Horizon Planner
    st.subheader("⏳ Horizon-Based Plan")
    if horizon_years <= 1:
    st.write("Short-term: Stick to low-risk options like bonds.")
    elif horizon_years <= 3:
        st.write("Medium-term: Balance with mid-cap stocks or mutual funds.")
    else:
        st.write("Long-term: Diversify into stocks for higher returns.")
    else:
        st.write("Please submit the form and ensure survey data is available.")

    st.markdown("---")
    st.write("✨ Powered by WealthWise | Built with ❤️ by xAI")

if __name__ == "__main__":
    main()
