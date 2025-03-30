import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import requests
from fpdf import FPDF
import io
import os
import tempfile
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# Simulated Investment Dataset
investment_data = pd.DataFrame({
    "Company": ["Reliance Industries", "HDFC Bank", "Bajaj Finance", "SBI Bluechip Fund",
                "Paytm", "Zomato", "Bitcoin", "Ethereum"],
    "Category": ["Large Cap", "Large Cap", "Medium Cap", "Medium Cap",
                 "Low Cap", "Low Cap", "Crypto", "Crypto"],
    "Min_Invest": [1000, 500, 1500, 500, 2000, 2000, 5000, 3000],
    "Risk": ["Medium", "Low", "Medium", "Medium", "High", "High", "High", "High"],
    "Goal": ["Wealth growth", "Emergency fund", "Future expenses", "Emergency fund",
             "Wealth growth", "Future expenses", "No specific goal", "Wealth growth"],
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
    return model, r2_score(y_test, model.predict(X_test))

@st.cache_resource
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

def predict_investment_strategy(model, invest_amount, risk_tolerance, horizon_years, goals):
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    goal_map = {"Wealth growth": 0, "Emergency fund": 1, "Future expenses": 2, "No specific goal": 3}
    risk_encoded = risk_map[risk_tolerance]
    goal_encoded_list = [goal_map[goal] for goal in goals]
    
    input_data = investment_data[["Min_Invest", "Risk_Encoded", "Goal_Encoded", "Expected_Return", "Volatility"]].copy()
    input_data["Expected_Return"] = input_data["Expected_Return"] * (1 + horizon_years * 0.05)
    input_data["Volatility"] = input_data["Volatility"] * (1 - horizon_years * 0.02)
    
    scores = model.predict(input_data)
    investment_data["Suitability_Score"] = scores
    
    filtered = investment_data[
        (investment_data["Min_Invest"] <= invest_amount) &
        (investment_data["Risk_Encoded"] <= risk_encoded) &
        (
            investment_data["Goal_Encoded"].isin(goal_encoded_list) | 
            (investment_data["Goal_Encoded"] == goal_map["No specific goal"])
        )
    ]
    
    recommendations = {}
    for category in filtered["Category"].unique():
        category_recs = filtered[filtered["Category"] == category].sort_values("Suitability_Score", ascending=False).head(1)
        recommendations[category] = [
            {"Company": row["Company"], "Amount": invest_amount / len(filtered["Category"].unique()) if len(filtered["Category"].unique()) > 0 else invest_amount}
            for _, row in category_recs.iterrows()
        ]
    return recommendations

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
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Fetch Real-Time Stock Data
def get_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if "Time Series (5min)" not in data:
            return None, "Error: Invalid symbol, API key, or rate limit reached."
        time_series = data["Time Series (5min)"]
        df = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# Fetch Market News
def get_market_news(api_key, tickers="AAPL"):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if "feed" not in data or not data["feed"]:
            return None, "No news available. Free key has limited access—try a premium key!"
        return data["feed"], None
    except Exception as e:
        return None, f"Error fetching news: {str(e)}"

# Main Application
def main():
    st.title("💰 WealthWise Dashboard")
    st.markdown("Hello, WealthWise!")
    
    # Initialize session state at the start
    if 'doubloons' not in st.session_state:
        st.session_state.doubloons = 0
    if 'badges' not in st.session_state:
        st.session_state.badges = []
    if 'island' not in st.session_state:
        st.session_state.island = 0
        
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
            st.metric("Retirement Model Accuracy (R²)", f"{retirement_r2:.2f}")
        st.markdown("### 🔑 Your Market Data Pass")
        api_key = st.text_input("Paste Your Key Here", value="", type="password")
        st.markdown("""
        **How to Get It:**  
        1. Visit [Alpha Vantage](https://www.alphavantage.co/).  
        2. Click 'Get Free API Key' and sign up.  
        3. Copy the code and paste it here!
        """)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Stock Investments", "🎯 Personalized Investment", "🏡 Retirement Planning", "🌐 Live Market Insights", "🏴‍☠️ Pirate’s Treasure Quest"])
    
    with tab1:
        st.header("📈 Stock Market Adventure")
        with st.form(key="stock_form"):
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("⏳ Investment Horizon (Months)", 1, 60, 12)
                invest_amount = st.number_input("💰 Amount to Invest (₹)", min_value=1000.0, value=6000.0, step=500.0)
            with col2:
                risk_tolerance = st.selectbox("🎲 Risk Appetite", ["Low", "Medium", "High"])
                goals = st.multiselect("🎯 Goals", ["Wealth growth", "Emergency fund", "Future expenses", "No specific goal"], default=["Wealth growth"])
            submit = st.form_submit_button("🚀 Explore Market")
        if submit and stock_data is not None and stock_model is not None:
            with st.spinner("Analyzing your investment strategy..."):
                future = pd.DataFrame({"Day": [1], "Month": [horizon % 12 or 12], "Year": [2025 + horizon // 12]})
                predicted_price = stock_model.predict(future)[0]
                current_price = stock_data['close'].iloc[-1]
                growth = predicted_price - current_price
                horizon_years = horizon // 12 or 1
                recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goals)
            st.subheader("🔮 Market Forecast")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Price (₹)", f"₹{predicted_price:,.2f}", f"{growth:,.2f}")
            col2.metric("Growth Potential", f"{(growth/current_price)*100:.1f}%", "🚀" if growth > 0 else "📉")
            with st.expander("📊 Price Trend", expanded=True):
                fig = px.line(stock_data, x='Date', y='close', title="NIFTY CONSUMPTION Trend")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("💡 Your Investment Strategy")
            for category, recs in recommendations.items():
                if recs:
                    with st.expander(f"{category} Options"):
                        for rec in recs:
                            st.write(f"- **{rec['Company']}**: Invest ₹{rec['Amount']:,.2f}")
    
    with tab2:
        st.header("🎯 Your Investment Journey")
        with st.form(key="investment_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Your Name")
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                essentials = st.number_input("🍲 Essentials (₹)", min_value=0.0, step=100.0)
                non_essentials = st.number_input("🎉 Non-Essentials (₹)", min_value=0.0, step=100.0)
                debt_payment = st.number_input("💳 Debt Payment (₹)", min_value=0.0, step=100.0)
            with col2:
                goals = st.multiselect("🎯 Goals", ["Wealth growth", "Emergency fund", "Future expenses", "No specific goal"], default=["Wealth growth"])
                goal_amount = st.number_input("💎 Total Goal Amount (₹)", min_value=0.0, step=1000.0, value=50000.0)
                risk_tolerance = st.selectbox("🎲 Risk Tolerance", ["Low", "Medium", "High"])
                horizon_years = st.slider("⏳ Horizon (Years)", 1, 10, 3)
                invest_percent = st.slider("💸 % of Savings to Invest", 0, 100, 50)
            submit = st.form_submit_button("🚀 Get Your Plan")
        if submit and survey_data is not None and survey_model is not None:
            predicted_savings = predict_savings(survey_model, income, essentials, non_essentials, debt_payment)
            invest_amount = predicted_savings * (invest_percent / 100)
            recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goals)
            st.subheader("💰 Your Monthly Breakdown")
            breakdown_data = {"Essentials": essentials, "Non-Essentials": non_essentials, "Debt Payment": debt_payment, "Savings": predicted_savings}
            fig = px.pie(values=list(breakdown_data.values()), names=list(breakdown_data.keys()), title="Spending vs. Savings")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("💼 Your Investment Options")
            for category, recs in recommendations.items():
                if recs:
                    with st.expander(f"{category} Investments"):
                        for rec in recs:
                            st.write(f"- *{rec['Company']}*: ₹{rec['Amount']:,.2f}")
            pdf_buffer = generate_pdf(name, income, predicted_savings, ", ".join(goals), risk_tolerance, horizon_years, recommendations, survey_data["Savings"].mean(), [])
            st.download_button("📥 Download Your Plan", pdf_buffer, f"{name}_investment_plan.pdf", "application/pdf")
    
    with tab3:
        st.header("🏡 Retirement Planning")
        with st.form(key="retirement_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 Current Age", min_value=18, max_value=100, value=30)
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                current_savings = st.number_input("🏦 Current Savings (₹)", min_value=0.0, step=1000.0)
            with col2:
                retirement_age = st.slider("👴 Retirement Age", age + 1, 100, 65)
                monthly_expenses = st.number_input("💸 Expected Monthly Expenses (₹)", min_value=0.0, step=500.0)
                inflation_rate = st.slider("📈 Expected Inflation Rate (%)", 0.0, 10.0, 3.0)
            submit = st.form_submit_button("🚀 Plan My Retirement")
        if submit and financial_data is not None and retirement_model is not None:
            years_to_retirement = retirement_age - age
            future_expenses = monthly_expenses * (1 + inflation_rate / 100) ** years_to_retirement
            retirement_goal = future_expenses * 12 * 20
            predicted_savings = predict_retirement_savings(retirement_model, income, monthly_expenses)
            retirement_wealth = forecast_retirement_savings(income, predicted_savings + current_savings, years_to_retirement)
            st.subheader("🌟 Retirement Outlook")
            col1, col2 = st.columns(2)
            col1.metric("Projected Wealth", f"₹{retirement_wealth:,.2f}")
            col2.metric("Inflation-Adjusted Goal", f"₹{retirement_goal:,.2f}")
    
    with tab4:
        st.header("🌐 Live Market Insights")
        if not api_key:
            st.error("Please add your Alpha Vantage key in the sidebar!")
        else:
            portfolio_input = st.text_area("Enter stock symbols (one per line):", "AAPL\nMSFT")
            portfolio = [symbol.strip().upper() for symbol in portfolio_input.split("\n") if symbol.strip()]
            if st.button("Track Portfolio & News"):
                for symbol in portfolio:
                    df, error = get_stock_data(symbol, api_key)
                    if df is not None:
                        latest_price = df["Close"].iloc[0]
                        st.metric(f"{symbol} Current Price", f"${latest_price:.2f}")
    
  with tab5:
        st.header("🏴‍☠️ Pirate’s Treasure Quest")
        st.markdown("""
        **Ahoy, Captain!** The Treasure of WealthWise be out there—gold doubloons, crew badges, and the secrets o’ riches! 
        Sail me four islands, plunder their quests, and claim the ultimate loot! Argh!
        """)
        
        # Crew's Loot (Rewards)
        st.subheader("💰 Yer Crew’s Loot")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Doubloons Plundered: {st.session_state.doubloons} 💰")
        with col2:
            st.write("Crew Badges: " + ", ".join([f"⚔️ {badge}" for badge in st.session_state.badges]) or "None yet, ye scallywag!")
        
        # Treasure Trail (Progress)
        st.subheader("🗺️ The Treasure Trail")
        total_islands = 4
        st.progress(st.session_state.island / total_islands)
        st.write(f"Islands Conquered: {st.session_state.island} of {total_islands}")
        
        # Island Raids (Quests)
        st.subheader("🏝️ Island Raids")
        
        # Island 1: Stock Reef
        if st.session_state.island >= 0:
            st.write("**1. Stock Reef: Plunder the NIFTY’s Gold!**")
            st.info("Guess the NIFTY CONSUMPTION price within 500 doubloons! Spy the latest haul in Stock Reef [Tab 1].")
            guess = st.slider("Yer Guess (₹)", 0, 10000, 5000, step=100, key="stock_guess")
            if stock_data is not None:
                if st.button("Fire the Cannon!", key="cannon_with_data"):
                    latest_price = stock_data['close'].iloc[-1]
                    if abs(guess - latest_price) <= 500:
                        st.success(f"BOOM! Nailed it, Captain! Latest haul: ₹{latest_price:.2f}. +20 doubloons!")
                        st.session_state.doubloons += 20
                        st.session_state.badges.append("Reef Raider")
                        st.session_state.island = 1
                        st.balloons()
                    else:
                        st.error(f"Argh! Missed the mark! Latest price: ₹{latest_price:.2f}. Lower yer sails and try again!")
            else:
                if st.button("Fire the Cannon!", key="cannon_no_data"):
                    st.warning("No map to Stock Reef yet! For now, guess ₹5000—me demo treasure!")
                    if abs(guess - 5000) <= 500:
                        st.success("BOOM! Ye hit me demo stash! +20 doubloons!")
                        st.session_state.doubloons += 20
                        st.session_state.badges.append("Reef Raider")
                        st.session_state.island = 1
                        st.balloons()

        # Island 2: Savings Cove
        if st.session_state.island >= 1:
            st.write("**2. Savings Cove: Bury Yer Treasure!**")
            st.info("Sail to Savings Cove [Tab 2], set a ₹50,000 goal, and bury it deep!")
            if st.button("I Buried It!", key="savings_cove_button"):
                st.success("Shiver me timbers! Ye’ve stashed ₹50,000! +20 doubloons!")
                st.session_state.doubloons += 20
                st.session_state.badges.append("Cove Captain")
                st.session_state.island = 2
                st.snow()
                st.write("Psst… me parrot says ye’re a savvy saver now!")
        else:
            st.write("**2. Savings Cove** (Fog o’ the Sea)")

        # Island 3: Retirement Bay
        if st.session_state.island >= 2:
            st.write("**3. Retirement Bay: Chart Yer Sunset!**")
            st.info("Drop anchor at Retirement Bay [Tab 3] and set yer sails for age 65!")
            if st.button("Course Charted!", key="retirement_bay_button"):
                st.success("Yo ho ho! Retirement at 65 be secured! +20 doubloons!")
                st.session_state.doubloons += 20
                st.session_state.badges.append("Bay Buccaneer")
                st.session_state.island = 3
                st.balloons()
        else:
            st.write("**3. Retirement Bay** (Fog o’ the Sea)")

        # Island 4: Market Lagoon
        if st.session_state.island >= 3:
            st.write("**4. Market Lagoon: Spy the Horizon!**")
            st.info("Sneak to Market Lagoon [Tab 4], use yer spyglass (API key), and track any stock!")
            if st.button("Stock Spotted!", key="market_lagoon_button"):
                st.success("Argh! Ye’ve spied the market’s secrets! +20 doubloons!")
                st.session_state.doubloons += 20
                st.session_state.badges.append("Lagoon Lookout")
                st.session_state.island = 4
                st.snow()
        else:
            st.write("**4. Market Lagoon** (Fog o’ the Sea)")

        # Grand Finale
        if st.session_state.island >= total_islands:
            st.subheader("🏴‍☠️ Ye’ve Found the Treasure!")
            st.markdown(f"""
            **Blimey, Captain!** Ye’ve plundered all islands and claimed the Treasure of WealthWise! 
            Yer haul: **{st.session_state.doubloons} doubloons** and a legendary crew!
            """)
            st.download_button(
                "🗝️ Claim Yer Treasure Map",
                data="Secret Code: WEALTHY2025\nHint: More riches await ye in the future!",
                file_name="pirate_treasure_map.txt",
                key="treasure_map_button"
            )
            st.write("Show this code to me crew later for a surprise, arr!")

if __name__ == "__main__":
    main()
