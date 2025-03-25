import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from fpdf import FPDF
import io
import os

# Page Configuration
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# Simulated Investment Dataset
investment_data = pd.DataFrame({
    "Company": [
        "Reliance Industries", "HDFC Bank", "Bajaj Finance", "SBI Bluechip Fund",
        "Paytm", "Zomato", "Bitcoin", "Ethereum"
    ],
    "Category": [
        "Large Cap", "Large Cap", "Medium Cap", "Medium Cap",
        "Low Cap", "Low Cap", "Crypto", "Crypto"
    ],
    "Min_Invest": [1000, 500, 1500, 500, 2000, 2000, 5000, 3000],
    "Risk": ["Medium", "Low", "Medium", "Medium", "High", "High", "High", "High"],
    "Goal": [
        "Wealth growth", "Emergency fund", "Future expenses", "Emergency fund",
        "Wealth growth", "Future expenses", "No specific goal", "Wealth growth"
    ],
    "Expected_Return": [8.5, 6.0, 10.0, 7.5, 15.0, 14.0, 20.0, 18.0],
    "Volatility": [15.0, 10.0, 20.0, 12.0, 30.0, 28.0, 50.0, 45.0]
})
investment_data["Risk_Encoded"] = investment_data["Risk"].map({"Low": 0, "Medium": 1, "High": 2})
investment_data["Goal_Encoded"] = investment_data["Goal"].map({
    "Wealth growth": 0, "Emergency fund": 1, "Future expenses": 2, "No specific goal": 3
})

# Data Loading Functions
@st.cache_data
def load_stock_data(csv_path="NIFTY CONSUMPTION_daily_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Stock CSV not found! Please upload 'NIFTY CONSUMPTION_daily_data.csv'")
        return None
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['Date'].isnull().all():
            st.error("🚨 Invalid date format in stock data!")
            return None
        df = df[['Date', 'open', 'high', 'low', 'close', 'volume']].sort_values(by='Date').dropna()
        return df
    except Exception as e:
        st.error(f"🚨 Error loading stock data: {str(e)}")
        return None

@st.cache_data
def load_survey_data(csv_path="survey_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Survey CSV not found! Please upload 'survey_data.csv'.")
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
                    return (float(bounds[0].replace(",", "")) + float(bounds[1].replace(",",""))) / 2
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

@st.cache_data
def load_financial_data(csv_path="financial_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Financial CSV not found! Please upload 'financial_data.csv'.")
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip().replace('"', '') for col in df.columns]
        col_map = {col.lower(): col for col in df.columns}
        required_cols = ["income", "projected_savings"]
        missing_cols = [col for col in required_cols if col not in col_map]
        if missing_cols:
            st.error(f"🚨 'financial_data.csv' is missing required columns: {', '.join(missing_cols)}")
            return None
        df = df.rename(columns={col_map["income"]: "income", col_map["projected_savings"]: "Projected_Savings"})
        expense_cols = ["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Healthcare", 
                       "Education", "Miscellaneous (Eating_Out,Entertainmentand Utilities)"]
        available_expense_cols = [col for col in expense_cols if col in df.columns]
        if available_expense_cols:
            df["Total_Expenses"] = df[available_expense_cols].sum(axis=1)
        else:
            df["Total_Expenses"] = 0
        return df
    except Exception as e:
        st.error(f"🚨 Error loading financial data: {str(e)}")
        return None

# Model Training Functions
@st.cache_resource
def train_stock_model(data):
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    X = data[['Day', 'Month', 'Year']]
    y = data['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training stock prediction model..."):
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
    with st.spinner("Training savings prediction model..."):
        model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

@st.cache_resource
def train_retirement_model(financial_data):
    features = ["income", "Total_Expenses"]
    target = "Projected_Savings"
    X = financial_data[features].fillna(0)
    y = financial_data[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training retirement savings model..."):
        model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))@st.cache_resource
def train_investment_model(data):
    X = data[["Min_Invest", "Risk_Encoded", "Goal_Encoded", "Expected_Return", "Volatility"]]
    y = (data["Expected_Return"] / data["Volatility"]) * (1 - data["Risk_Encoded"] * 0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training investment recommendation model..."):
        model.fit(X, y)
    return model

# Predictive and Utility Functions
def predict_savings(model, income, essentials, non_essentials, debt_payment):
    input_df = pd.DataFrame({
        "Income": [income],
        "Essentials": [essentials],
        "Non_Essentials": [non_essentials],
        "Debt_Payment": [debt_payment]
    })
    return model.predict(input_df)[0]

def predict_retirement_savings(model, income, expenses):
    input_df = pd.DataFrame({"income": [income], "Total_Expenses": [expenses]})
    return model.predict(input_df)[0]

def calculate_savings_goal(goal_amount, horizon_years):
    return goal_amount / (horizon_years * 12) if horizon_years > 0 else goal_amount

def forecast_retirement_savings(income, savings, years, growth_rate=5.0):
    wealth = savings
    monthly_savings = savings
    for _ in range(years * 12):
        wealth = wealth * (1 + growth_rate / 1200) + monthly_savings
    return wealth

def predict_investment_strategy(model, invest_amount, risk_tolerance, horizon_years, goal):
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    goal_map = {"Wealth growth": 0, "Emergency fund": 1, "Future expenses": 2, "No specific goal": 3}
    risk_encoded = risk_map[risk_tolerance]
    goal_encoded = goal_map[goal]
    
    input_data = investment_data[["Min_Invest", "Risk_Encoded", "Goal_Encoded", "Expected_Return", "Volatility"]].copy()
    input_data["Expected_Return"] = input_data["Expected_Return"] * (1 + horizon_years * 0.05)
    input_data["Volatility"] = input_data["Volatility"] * (1 - horizon_years * 0.02)
    
    scores = model.predict(input_data)
    investment_data["Suitability_Score"] = scores
    
    filtered = investment_data[
        (investment_data["Min_Invest"] <= invest_amount) &
        (investment_data["Risk_Encoded"] <= risk_encoded) &
        ((investment_data["Goal_Encoded"] == goal_encoded) | (investment_data["Goal"] == "No specific goal"))
    ]
    
    recommendations = {}
    for category in filtered["Category"].unique():
        category_recs = filtered[filtered["Category"] == category].sort_values("Suitability_Score", ascending=False).head(1)
        recommendations[category] = [
            {"Company": row["Company"], "Amount": invest_amount / len(filtered["Category"].unique())}
            for _, row in category_recs.iterrows()
        ]return recommendations

# PDF Generation with FPDF
def generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_savings, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"WealthWise Investment Plan for {name}", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "Powered by WealthWise | Built with love by xAI", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Financial Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Income: INR {income:,.2f}", ln=True)
    pdf.cell(0, 10, f"Predicted Savings: INR {predicted_savings:,.2f}", ln=True)
    pdf.cell(0, 10, f"Goal: {goal}", ln=True)
    pdf.cell(0, 10, f"Risk Tolerance: {risk_tolerance}", ln=True)
    pdf.cell(0, 10, f"Investment Horizon: {horizon_years} years", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Investment Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)
    for category, recs in recommendations.items():
        if recs:
            pdf.cell(0, 10, f"{category}:", ln=True)
            for rec in recs:
                pdf.cell(0, 10, f"  - {rec['Company']}: INR {rec['Amount']:,.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Budget Tips", ln=True)
    pdf.set_font("Arial", "", 10)
    for tip in tips:
        pdf.cell(0, 10, f"- {tip}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Peer Comparison", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Your Savings: INR {predicted_savings:,.2f} | Peer Average: INR {peer_savings:,.2f}", ln=True)
    buffer = io.BytesIO()
    buffer.write(pdf.output(dest='S'))
    buffer.seek(0)
    return buffer

# Main Application
def main():
    st.title("💰 WealthWise Dashboard")
    st.markdown("Your ultimate wealth management companion! 🚀")

    # Load data
    stock_data = load_stock_data()
    survey_data = load_survey_data()
    financial_data = load_financial_data()

    # Train models
    stock_model, stock_r2 = None, 0.0
    if stock_data is not None:
        stock_model, stock_r2 = train_stock_model(stock_data)
    survey_model, survey_r2 = None, 0.0
    if survey_data is not None:
        survey_model, survey_r2 = train_survey_model(survey_data)
    retirement_model, retirement_r2 = None, 0.0
    if financial_data is not None:
        retirement_model, retirement_r2 = train_retirement_model(financial_data)
    investment_model = train_investment_model(investment_data)

    # Sidebar
    with st.sidebar:
        st.header("Dashboard Insights")
        st.info("Explore your financial future with these tools!")
        if stock_data is not None:
            st.metric("Stock Model Accuracy (R²)", f"{stock_r2:.2f}")
        if survey_data is not None:
            st.metric("Savings Model Accuracy (R²)", f"{survey_r2:.2f}")
        if financial_data is not None:
            st.metric("Retirement Model Accuracy (R²)", f"{retirement_r2:.2f}")# Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Stock Investments", "🎯 Personalized Investment", "🏡 Retirement Planning"])

    with tab1:
        st.header("📈 Stock Market Adventure")
        st.markdown("Navigate the NIFTY CONSUMPTION index with precision! 🌟")
        with st.form(key="stock_form"):
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("⏳ Investment Horizon (Months)", 1, 60, 12, help="How long will you invest?")
                invest_amount = st.number_input("💰 Amount to Invest (₹)", min_value=1000.0, value=6000.0, step=500.0, help="How much are you investing?")
            with col2:
                risk_tolerance = st.selectbox("🎲 Risk Appetite", ["Low", "Medium", "High"], help="Your comfort with risk")
                goal = st.selectbox("🎯 Goal", ["Wealth growth", "Emergency fund", "Future expenses"], help="What’s your aim?")
            submit = st.form_submit_button("🚀 Explore Market")
        if submit and stock_data is not None and stock_model is not None:
            with st.spinner("Analyzing your investment strategy..."):
                future = pd.DataFrame({"Day": [1], "Month": [horizon % 12 or 12], "Year": [2025 + horizon // 12]})
                predicted_price = stock_model.predict(future)[0]
                current_price = stock_data['close'].iloc[-1]
                growth = predicted_price - current_price
                horizon_years = horizon // 12 or 1
                recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goal)
            st.subheader("🔮 Market Forecast")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Price (₹)", f"₹{predicted_price:,.2f}", f"{growth:,.2f}")
            col2.metric("Growth Potential", f"{(growth/current_price)*100:.1f}%", "🚀" if growth > 0 else "📉")
            with st.expander("📊 Price Trend", expanded=True):
                fig = px.line(stock_data, x='Date', y='close', title="NIFTY CONSUMPTION Trend", 
                             hover_data=['open', 'high', 'low', 'volume'])
                fig.update_traces(line_color='#00ff00')
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("💡 Your Investment Strategy")
            progress = min(1.0, invest_amount / 100000)
            st.progress(progress)
            any_recommendations = False
            for category in ["Large Cap", "Medium Cap", "Low Cap", "Crypto"]:
                recs = recommendations.get(category, [])
                if recs:
                    any_recommendations = True
                    with st.expander(f"{category} Options"):
                        for rec in recs:
                            st.write(f"- **{rec['Company']}**: Invest ₹{rec['Amount']:,.2f}")
            if not any_recommendations:
                st.info("No investment options match your criteria. Try increasing your investment amount or adjusting your risk tolerance/goal.")with tab2:
        st.header("🎯 Your Investment Journey")
        st.markdown("Craft a personalized plan for wealth growth! 🌈")
        with st.form(key="investment_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Your Name", help="Who’s planning their wealth?")
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                essentials = st.number_input("🍲 Essentials (₹)", min_value=0.0, step=100.0, help="Food, transport, etc.")
                non_essentials = st.number_input("🎉 Non-Essentials (₹)", min_value=0.0, step=100.0, help="Fun stuff!")
                debt_payment = st.number_input("💳 Debt Payment (₹)", min_value=0.0, step=100.0)
            with col2:
                goal = st.selectbox("🎯 Goal", ["No specific goal", "Emergency fund", "Future expenses", "Wealth growth"])
                goal_amount = st.number_input("💎 Goal Amount (₹)", min_value=0.0, step=1000.0, value=50000.0)
                risk_tolerance = st.selectbox("🎲 Risk Tolerance", ["Low", "Medium", "High"])
                horizon_years = st.slider("⏳ Horizon (Years)", 1, 10, 3)
            submit = st.form_submit_button("🚀 Get Your Plan")
        if submit and survey_data is not None and survey_model is not None:
            with st.spinner("Crafting your personalized plan..."):
                predicted_savings = predict_savings(survey_model, income, essentials, non_essentials, debt_payment)
                recommendations = predict_investment_strategy(investment_model, predicted_savings, risk_tolerance, horizon_years, goal)
                monthly_savings_needed = calculate_savings_goal(goal_amount, horizon_years)
                peer_avg_savings = survey_data["Savings"].mean()
            st.subheader("💼 Your Investment Options")
            for category in ["Large Cap", "Medium Cap", "Low Cap", "Crypto"]:
                recs = recommendations.get(category, [])
                if recs:
                    with st.expander(f"{category} Investments"):
                        for rec in recs:
                            st.write(f"- *{rec['Company']}*: ₹{rec['Amount']:,.2f}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Savings Progress")
                progress = min(1.0, predicted_savings / goal_amount) if goal_amount > 0 else 0
                st.progress(progress)
                st.write(f"₹{predicted_savings:,.2f} / ₹{goal_amount:,.2f}")
            with col2:
                st.subheader("📊 Peer Benchmark")
                st.bar_chart({"You": predicted_savings, "Peers": peer_avg_savings})
            with st.expander("💡 Budget Tips", expanded=True):
                tips = []
                median_non_essentials = survey_data["Non_Essentials"].median()
                if non_essentials > median_non_essentials:
                    tips.append(f"Reduce non-essentials by INR {non_essentials - median_non_essentials:,.2f} (peer median: INR {median_non_essentials:,.2f}).")
                if debt_payment > income * 0.3:
                    tips.append("Debt payment exceeds 30% of income - consider refinancing or cutting expenses.")
                if predicted_savings < monthly_savings_needed:
                    shortfall = monthly_savings_needed - predicted_savings
                    tips.append(f"Boost savings by INR {shortfall:,.2f}/month to meet your goal.")
                else:
                    tips.append("Great job! Your savings exceed your goal - consider investing more.")
                for tip in tips:
                    st.write(f"- {tip}")
            st.subheader("🎲 Risk Tolerance Assessment")
            risk_map = {"Low": "Safe", "Medium": "Balanced", "High": "Aggressive"}
            st.write(f"Your Profile: *{risk_map[risk_tolerance]}*")
            if risk_tolerance == "Low" and horizon_years > 5:
                st.info("Long horizon with low risk? You could explore medium-risk options for better returns.")
            elif risk_tolerance == "High" and horizon_years < 3:
                st.warning("Short horizon with high risk? Consider safer options to protect your funds.")
            pdf_buffer = generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_avg_savings, tips)
            st.download_button("📥 Download Your Plan", pdf_buffer, f"{name}_investment_plan.pdf", "application/pdf")

    with tab3:
        st.header("🏡 Retirement Planning")
        st.markdown("Secure your golden years with smart savings! 🌞")
        with st.form(key="retirement_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 Current Age", min_value=18, max_value=100, value=30)
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                current_savings = st.number_input("🏦 Current Savings (₹)", min_value=0.0, step=1000.0)
            with col2:
                retirement_age = st.slider("👴 Retirement Age", age, 100, 65)
                monthly_expenses = st.number_input("💸 Expected Monthly Expenses (₹)", min_value=0.0, step=500.0)
            submit = st.form_submit_button("🚀 Plan My Retirement")
        if submit and financial_data is not None and retirement_model is not None:
            with st.spinner("Projecting your retirement using financial_data.csv..."):
                years_to_retirement = retirement_age - age
                predicted_savings = predict_retirement_savings(retirement_model, income, monthly_expenses)
                retirement_wealth = forecast_retirement_savings(income, predicted_savings + current_savings, years_to_retirement)
                retirement_goal = monthly_expenses * 12 * 20
            st.subheader("🌟 Retirement Outlook")
            col1, col2 = st.columns(2)
            col1.metric("Projected Wealth", f"₹{retirement_wealth:,.2f}")
            col2.metric("Retirement Goal", f"₹{retirement_goal:,.2f}", f"{'Surplus' if retirement_wealth > retirement_goal else 'Shortfall'}: ₹{abs(retirement_wealth - retirement_goal):,.2f}")
            st.subheader("📈 Savings Trajectory")
            trajectory = [forecast_retirement_savings(income, predicted_savings + current_savings, y) for y in range(years_to_retirement + 1)]
            fig = px.line(x=range(years_to_retirement + 1), y=trajectory, labels={"x": "Years", "y": "Wealth (₹)"}, title="Retirement Growth")
            fig.add_hline(y=retirement_goal, line_dash="dash", line_color="red", annotation_text="Goal")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("💡 Retirement Tips")
            if retirement_wealth < retirement_goal:
                shortfall = (retirement_goal - retirement_wealth) / (years_to_retirement * 12)
                st.write(f"- Increase monthly savings by ₹{shortfall:,.2f} to meet your goal.")
            st.write("- Assumes a 5% annual growth rate—adjust investments for higher returns if needed.")

    st.markdown("---")
    st.write("Powered by WealthWise | Built with love by xAI")

if __name__ == "__main__":
    main()
