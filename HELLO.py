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
from reportlab.lib.styles import getSampleStyleSheet
import io
import os
import warnings

warnings.filterwarnings("ignore")

# 2. Page Configuration (Moved to the top as the first Streamlit command)
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# 3. Data Loading Functions
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
        df = df.sort_values(by='Date').dropna()
        return df
    except Exception as e:
        st.error(f"🚨 Error loading stock data: {str(e)}")
        return None

@st.cache_data
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

# 4. Model Training
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

# 5. Predictive and Utility Functions
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

# 6. Main Application
def main():
    st.title("WealthWise Dashboard")

    # Load data
    stock_data = load_stock_data()
    survey_data = load_survey_data()
    if survey_data is not None:
        survey_model, survey_r2 = train_survey_model(survey_data)
    else:
        survey_model, survey_r2 = None, 0.0

    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Personal Finance"

    # Define tabs
    tab1, tab2, tab3 = st.tabs(["💵 Personal Finance", "📈 Stock Investments", "🎯 Personalized Investment"])

    # --- Personal Finance Tab (Simplified) ---
    with tab1:
        st.session_state.active_tab = "Personal Finance"
        st.header("💵 Your Money Mastery Hub")
        with st.form(key="finance_form"):
            income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
            essentials = st.number_input("Essentials (₹)", min_value=0.0, step=100.0)
            non_essentials = st.number_input("Non-Essentials (₹)", min_value=0.0, step=100.0)
            debt_payment = st.number_input("Debt Payment (₹)", min_value=0.0, step=100.0)
            submit = st.form_submit_button("🚀 Analyze")
        if submit:
            st.write("Basic analysis placeholder (expand as needed).")

    # --- Stock Investments Tab (Simplified) ---
    with tab2:
        st.session_state.active_tab = "Stock Investments"
        st.header("📈 Stock Market Quest")
        if stock_data is not None:
            st.write("Stock data placeholder (expand as needed).")
        else:
            st.write("Stock data unavailable.")

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

            # 5. Risk Tolerance Assessment
            st.subheader("🎲 Your Risk Profile")
            risk_count = survey_data["What is your risk tolerance for investing?"].value_counts()
            st.write(f"Your risk tolerance ({risk_tolerance}) matches {risk_count[risk_tolerance]}/{len(survey_data)} peers ({risk_count[risk_tolerance]/len(survey_data)*100:.1f}%).")

            # 6. Downloadable PDF Report
            st.subheader("📄 Download Your Plan")
            tips = []
            if predicted_savings < monthly_savings_needed:
                tips.append("Reduce non-essential spending to meet your goal.")
            if debt_payment > peer_avg_savings:
                tips.append("Pay off debt faster to boost savings.")
            if predicted_savings > income * 0.2:
                tips.append("Great savings! Invest surplus in your chosen options.")
            pdf_buffer = generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_avg_savings, tips)
            st.download_button("Download PDF", pdf_buffer, f"Investment_Plan_{name}.pdf", "application/pdf")

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
