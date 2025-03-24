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

# Page Configuration
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# Data Loading
@st.cache_data
def load_stock_data(csv_path="archive (3) 2/NIFTY CONSUMPTION_daily_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Stock CSV not found! Please upload 'NIFTY CONSUMPTION_daily_data.csv'.")
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

# Model Training
@st.cache_resource
def train_stock_model(data):
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    X = data[['Day', 'Month', 'Year']]
    y = data['Close']
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
    for tip in tips:
        story.append(Paragraph(f"• {tip}", tip_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Peer Comparison", heading_style))
    story.append(Paragraph(f"Your Savings: ₹{predicted_savings:,.2f} | Peer Average: ₹{peer_savings:,.2f}", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Main Application
def main():
    st.title("💰 WealthWise Dashboard")
    st.markdown("Your interactive financial companion! 🚀")

    # Load data
    stock_data = load_stock_data()
    survey_data = load_survey_data()

    # Train models
    stock_model, stock_r2 = None, 0.0
    if stock_data is not None:
        stock_model, stock_r2 = train_stock_model(stock_data)
    
    survey_model, survey_r2 = None, 0.0
    if survey_data is not None:
        survey_model, survey_r2 = train_survey_model(survey_data)

    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        st.info("Use the tabs below to explore your financial future!")
        if stock_data is not None:
            st.metric("Stock Model Accuracy (R²)", f"{stock_r2:.2f}")
        if survey_data is not None:
            st.metric("Savings Model Accuracy (R²)", f"{survey_r2:.2f}")

    # Tabs
    tab1, tab2 = st.tabs(["📈 Stock Investments", "🎯 Personalized Investment"])

    # Stock Investments Tab
    with tab1:
        st.header("📈 Stock Market Adventure")
        st.markdown("Predict and conquer the NIFTY CONSUMPTION index! 🌟")

        with st.form(key="stock_form"):
            horizon = st.slider("⏳ Investment Horizon (Months)", 1, 60, 12, help="How long do you plan to invest?")
            risk_tolerance = st.selectbox("🎲 Risk Appetite", ["Low", "Medium", "High"], help="How much risk can you handle?")
            submit = st.form_submit_button("🚀 Explore Market")

        if submit and stock_data is not None and stock_model is not None:
            with st.spinner("Analyzing market trends..."):
                future = pd.DataFrame({"Day": [1], "Month": [horizon % 12 or 12], "Year": [2025 + horizon // 12]})
                predicted_price = stock_model.predict(future)[0]
                current_price = stock_data['Close'].iloc[-1]
                growth = predicted_price - current_price

            st.subheader("🔮 Market Forecast")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Price (₹)", f"₹{predicted_price:,.2f}", f"{growth:,.2f}")
            col2.metric("Growth Potential", f"{(growth/current_price)*100:.1f}%", "🚀" if growth > 0 else "📉")

            with st.expander("📊 Price Trend", expanded=True):
                fig = px.line(stock_data, x='Date', y='Close', title="NIFTY CONSUMPTION Trend", 
                             hover_data=['Open', 'High', 'Low', 'Volume'])
                fig.update_traces(line_color='#00ff00')
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("💡 Investment Tips")
            tips = {
                "Low": "Stick to stable blue-chip stocks like HDFC Bank for steady gains!",
                "Medium": "Mix it up with large caps (e.g., Reliance) and mutual funds!",
                "High": "Go bold with small caps (e.g., Paytm) or crypto for big wins!"
            }
            st.success(tips[risk_tolerance])

    # Personalized Investment Tab
    with tab2:
        st.header("🎯 Your Investment Journey")
        st.markdown("Plan your financial future with personalized insights! 🌈")

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
            with st.spinner("Crafting your plan..."):
                predicted_savings = predict_savings(survey_model, income, essentials, non_essentials, debt_payment)
                recommendations = get_investment_recommendations(income, predicted_savings, goal, risk_tolerance, horizon_years)
                monthly_savings_needed = calculate_savings_goal(goal_amount, horizon_years)
                peer_avg_savings = survey_data["Savings"].mean()

            st.subheader("💼 Your Investment Options")
            for rec in recommendations:
                st.write(f"- Invest ₹{rec['Amount']:,.2f} in **{rec['Company']}** ({rec['Type']})")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Savings Goal")
                st.metric("Monthly Savings Needed", f"₹{monthly_savings_needed:,.2f}")
            with col2:
                st.subheader("📊 Peer Benchmark")
                st.bar_chart({"You": predicted_savings, "Peers": peer_avg_savings})

            with st.expander("💡 Budget Tips"):
                tips = []
                median_non_essentials = survey_data["Non_Essentials"].median()
                if non_essentials > median_non_essentials:
                    tips.append(f"Cut ₹{non_essentials - median_non_essentials:,.2f} from non-essentials (peer median: ₹{median_non_essentials:,.2f}).")
                else:
                    tips.append("Your spending is optimized—great job!")
                for tip in tips:
                    st.write(f"- {tip}")

            pdf_buffer = generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_avg_savings, tips)
            st.download_button("📥 Download Your Plan", pdf_buffer, f"{name}_investment_plan.pdf", "application/pdf")

    st.markdown("---")
    st.write("✨ Powered by WealthWise | Built with ❤️ by xAI")

if __name__ == "__main__":
    main()
