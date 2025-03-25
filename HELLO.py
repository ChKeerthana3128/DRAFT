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
import requests  # New import for API calls
import time  # For refresh timing

# ... (Keep your existing imports and initial setup code)

# Alpha Vantage API Key (Replace with your own key)
API_KEY = "O597WKEOZNCKCLSO"  # Get this from alphavantage.co

# Real-Time Data Fetching Functions
@st.cache_data(ttl=60)  # Cache for 60 seconds to avoid hitting API limits
def fetch_real_time_price(symbol):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"])
        else:
            st.warning(f"Could not fetch real-time price for {symbol}. Using fallback.")
            return None
    except Exception as e:
        st.error(f"Error fetching price: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache news for 5 minutes
def fetch_market_news():
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=finance&apikey={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "feed" in data:
            return data["feed"][:5]  # Return top 5 news items
        return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Portfolio Tracking Function
def calculate_portfolio_value(portfolio, real_time_prices):
    total_value = 0
    performance = []
    for holding in portfolio:
        symbol = holding["symbol"]
        shares = holding["shares"]
        purchase_price = holding["purchase_price"]
        current_price = real_time_prices.get(symbol, purchase_price)  # Fallback to purchase price if API fails
        value = shares * current_price
        total_value += value
        gain_loss = (current_price - purchase_price) * shares
        performance.append({
            "Symbol": symbol,
            "Shares": shares,
            "Current Price": current_price,
            "Value": value,
            "Gain/Loss": gain_loss
        })
    return total_value, performance

# ... (Keep your existing data loading and model training functions)

# Main Application (Modified tab1 only)
def main():
    st.title("💰 WealthWise Dashboard")
    st.markdown("Your ultimate wealth management companion! 🚀")

    # Load data (ensure this runs before tabs)
    stock_data = load_stock_data()
    survey_data = load_survey_data()
    financial_data = load_financial_data()

    # Train models (ensure this runs before tabs)
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

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Stock Investments", "🎯 Personalized Investment", "🏡 Retirement Planning"])

    with tab1:
        st.header("📈 Stock Market Adventure")
        st.markdown("Navigate the NIFTY CONSUMPTION index with precision and real-time insights! 🌟")

        # Portfolio Input Section
        st.subheader("📊 Live Portfolio Tracking")
        portfolio = []
        with st.expander("Add Your Holdings", expanded=False):
            num_holdings = st.number_input("Number of Stocks in Portfolio", min_value=0, max_value=10, value=0)
            for i in range(num_holdings):
                col1, col2, col3 = st.columns(3)
                with col1:
                    symbol = st.text_input(f"Stock Symbol {i+1}", value="RELIANCE.NS", key=f"symbol_{i}")
                with col2:
                    shares = st.number_input(f"Shares {i+1}", min_value=0.0, step=1.0, key=f"shares_{i}")
                with col3:
                    purchase_price = st.number_input(f"Purchase Price {i+1} (₹)", min_value=0.0, step=100.0, key=f"price_{i}")
                portfolio.append({"symbol": symbol, "shares": shares, "purchase_price": purchase_price})

        # Fetch Real-Time Prices for Portfolio
        if portfolio:
            with st.spinner("Fetching real-time portfolio data..."):
                real_time_prices = {holding["symbol"]: fetch_real_time_price(holding["symbol"]) or holding["purchase_price"] 
                                   for holding in portfolio}
                total_value, performance = calculate_portfolio_value(portfolio, real_time_prices)
            st.metric("Total Portfolio Value", f"₹{total_value:,.2f}")
            st.write("### Portfolio Performance")
            perf_df = pd.DataFrame(performance)
            st.dataframe(perf_df.style.format({
                "Current Price": "₹{:.2f}",
                "Value": "₹{:.2f}",
                "Gain/Loss": "₹{:.2f}"
            }).highlight_max(subset=["Gain/Loss"], color="green").highlight_min(subset=["Gain/Loss"], color="red"))
            st.write("*Portfolio updates every 60 seconds.*")
            time.sleep(1)  # Simulate delay

        # Existing Investment Form with Real-Time Data
        st.subheader("🔮 Plan Your Investment")
        with st.form(key="stock_form"):
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("⏳ Investment Horizon (Months)", 1, 60, 12, help="How long will you invest?")
                invest_amount = st.number_input("💰 Amount to Invest (₹)", min_value=1000.0, value=6000.0, step=500.0)
            with col2:
                risk_tolerance = st.selectbox("🎲 Risk Appetite", ["Low", "Medium", "High"])
                goal = st.selectbox("🎯 Goal", ["Wealth growth", "Emergency fund", "Future expenses"])
            submit = st.form_submit_button("🚀 Explore Market")

        if submit and stock_data is not None and stock_model is not None:
            with st.spinner("Analyzing your investment strategy..."):
                future = pd.DataFrame({"Day": [1], "Month": [horizon % 12 or 12], "Year": [2025 + horizon // 12]})
                predicted_price = stock_model.predict(future)[0]
                current_price = fetch_real_time_price("NIFTYBEES.NS") or stock_data['close'].iloc[-1]  # Real-time NIFTY
                growth = predicted_price - current_price
                horizon_years = horizon // 12 or 1
                recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goal)

            st.subheader("🔮 Market Forecast")
            col1, col2 = st.columns(2)
            col1.metric("Real-Time Price (₹)", f"₹{current_price:,.2f}")
            col2.metric("Predicted Price (₹)", f"₹{predicted_price:,.2f}", f"{growth:,.2f}")
            col3, col4 = st.columns(2)
            col3.metric("Growth Potential", f"{(growth/current_price)*100:.1f}%", "🚀" if growth > 0 else "📉")

            with st.expander("📊 Price Trend", expanded=True):
                fig = px.line(stock_data, x='Date', y='close', title="NIFTY CONSUMPTION Trend (Historical)", 
                             hover_data=['open', 'high', 'low', 'volume'])
                fig.add_scatter(x=[stock_data['Date'].iloc[-1]], y=[current_price], mode="markers", 
                               name="Real-Time", marker=dict(color="red", size=10))
                fig.update_traces(line_color='#00ff00')
                st.plotly_chart(fig, use_container_width=True)

            # Real-Time Market News
            st.subheader("📰 Market News")
            news_items = fetch_market_news()
            if news_items:
                for item in news_items:
                    st.write(f"**{item['title']}** ({item['time_published'][:10]})")
                    st.write(f"{item['summary'][:200]}... [Read More]({item['url']})")
            else:
                st.info("No recent market news available.")

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
                            real_time_price = fetch_real_time_price(rec["Company"] + ".NS") or rec["Amount"] / 10
                            st.write(f"- **{rec['Company']}**: Invest ₹{rec['Amount']:,.2f} (Current Price: ₹{real_time_price:,.2f})")
            if not any_recommendations:
                st.info("No investment options match your criteria. Try adjusting your inputs.")

    # ... (Keep tab2 and tab3 as they are)

    st.markdown("---")
    st.write("Powered by WealthWise | Built with love by xAI")

if __name__ == "__main__":
    main()
