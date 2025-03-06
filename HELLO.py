# 1. Imports
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

# 2. Page Configuration and Styling
st.set_page_config(page_title="💰 WealthWise Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for white text and black background
st.markdown("""
    <style>
    .main {background-color: #000000;}
    .sidebar .sidebar-content {background-color: #333333;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stMetric {background-color: #333333; border: 1px solid #4682b4; border-radius: 8px; padding: 10px; color: #FFFFFF !important;}
    .stExpander {background-color: #333333; border-radius: 8px; color: #FFFFFF !important;}
    h1, h2, h3, h4, h5, h6 {color: #FFFFFF !important;}
    .stMarkdown, .stText, p, div, span, label {color: #FFFFFF !important;}
    .sidebar .stMarkdown, .sidebar .stText, .sidebar p, .sidebar div, .sidebar span, .sidebar label {color: #FFFFFF !important;}
    .css-1d391kg, .css-1v0mbdj {color: #FFFFFF !important;}
    </style>
""", unsafe_allow_html=True)

# 3. Data Loading
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

# 4. Model Training
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

# 5. Predictive Functions
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

def predict_disposable_income(model, input_data):
    return max(0, model.predict(prepare_finance_input(input_data, model))[0])

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
    dream_fund = total_expenses * 12 * 20
    annual_target = dream_fund / years_to_retirement if years_to_retirement > 0 else dream_fund
    savings_rate = min(max((annual_target / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    expense_growth = 2.5
    return dream_fund, savings_rate, income_growth, expense_growth

# 6. Insight Generators (Not displayed in sidebar as per request)
def financial_wisdom(health_score, debt, discretionary, income, total_expenses):
    insights = [f"🌟 Your Financial Health Score: {health_score:.1f}/100"]
    status = "🎉 Excellent" if health_score >= 70 else "👍 Moderate" if health_score >= 40 else "⚠️ Needs Work"
    insights.append(f"Status: {status}")
    debt_ratio = debt / income if income > 0 else 0
    if debt_ratio > 0.36:
        insights.append(f"💸 Debt (₹{debt:,.2f}) is {debt_ratio:.1%}—above 36%. Time to strategize!")
    discretionary_ratio = discretionary / income if income > 0 else 0
    if discretionary_ratio > 0.15:
        insights.append(f"🎭 Discretionary (₹{discretionary:,.2f}) at {discretionary_ratio:.1%}—over 15%. Trim it?")
    savings_ratio = max(0, income - total_expenses) / income if income > 0 else 0
    if savings_ratio < 0.2:
        insights.append(f"💰 Savings at {savings_ratio:.1%}—below 20%. Let’s grow it!")
    return insights

def wealth_management_insights(income, total_expenses, savings_rate, years_to_retirement, wealth, desired_fund):
    insights = [f"🏦 With {savings_rate:.1f}% savings over {years_to_retirement} years, you’ll reach ₹{wealth:,.2f}."]
    shortfall = desired_fund - wealth if wealth < desired_fund else 0
    if shortfall > 0:
        extra_rate = (shortfall / (income * years_to_retirement)) * 100 if income > 0 and years_to_retirement > 0 else 0
        insights.append(f"📉 Short ₹{shortfall:,.2f}. Increase savings by {extra_rate:.1f}% or cut costs.")
    else:
        insights.append(f"🎯 Surplus of ₹{-shortfall:,.2f}! Invest the extra wisely.")
    if years_to_retirement > 20:
        insights.append("⏳ Long haul—3% income growth can amplify your wealth!")
    elif years_to_retirement < 10:
        insights.append("⏰ Tight timeline—ramp up savings now!")
    return insights

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
    # Load data
    finance_data = load_finance_data()
    stock_data = load_stock_data()
    if finance_data is None or stock_data is None:
        st.stop()

    # Train models
    finance_model, finance_r2 = train_finance_model(finance_data)
    stock_model, stock_r2 = train_stock_model(stock_data)

    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Personal Finance"
    if 'submit' not in st.session_state:
        st.session_state.submit = False
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

    # --- Personal Finance Dashboard ---
    with tab1:
        st.session_state.active_tab = "Personal Finance"
        st.header("💵 Your Money Mastery Hub")
        st.markdown("Shape your financial destiny with style! 🌈")

        # Sidebar for Personal Finance
        with st.sidebar:
            st.subheader("Personal Finance")
            st.write(f"📊 Finance Model R²: {finance_r2:.2f}")

            st.subheader("🌡️ Financial Health")
            if st.session_state.submit and st.session_state.input_data:
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
            if st.session_state.submit and st.session_state.input_data:
                disposable = predict_disposable_income(finance_model, st.session_state.input_data)
                st.metric("Monthly (₹)", f"₹{disposable:,.2f}")
            else:
                st.metric("Monthly (₹)", "N/A")

            st.subheader("🏦 Future Wealth")
            if st.session_state.submit and st.session_state.input_data:
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

        # Update session state on form submission
        if submit:
            st.session_state.submit = True
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

        # Main content
        if st.session_state.submit:
            st.subheader("🌍 Wealth Roadmap")
            dream_fund, suggested_rate, income_growth, expense_growth = smart_savings_plan(income, st.session_state.total_expenses, st.session_state.years_to_retirement)
            desired_fund = st.number_input("💎 Desired Retirement Fund (₹)", min_value=100000.0, value=dream_fund, step=100000.0)
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
                spending.plot(kind="bar", ax=ax, color="#87CEEB")
                ax.set_title("Monthly Spending (₹)")
                ax.set_ylabel("Amount (₹)")
                plt.xticks(rotation=45)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color("#FFFFFF")
                ax.title.set_color("#FFFFFF")
                ax.yaxis.label.set_color("#FFFFFF")
                st.pyplot(fig)
            with col2:
                st.subheader("🌱 Wealth Growth")
                trajectory = wealth_trajectory(income, st.session_state.total_expenses, savings_rate, st.session_state.years_to_retirement, income_growth, expense_growth)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(st.session_state.years_to_retirement + 1), trajectory, marker="o", color="#32CD32", label="Wealth")
                ax.axhline(y=desired_fund, color="red", linestyle="--", label="Goal")
                ax.set_title("Wealth Growth (₹)")
                ax.set_xlabel("Years")
                ax.set_ylabel("Savings (₹)")
                ax.legend()
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color("#FFFFFF")
                ax.title.set_color("#FFFFFF")
                ax.xaxis.label.set_color("#FFFFFF")
                ax.yaxis.label.set_color("#FFFFFF")
                ax.legend().set_title("")
                for text in ax.legend().get_texts():
                    text.set_color("#FFFFFF")
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

            st.subheader("📌 Predicted Price")
            future = pd.DataFrame({"Day": [1], "Month": [st.session_state.horizon % 12 or 12], "Year": [2025 + st.session_state.horizon // 12]})
            predicted_price = stock_model.predict(future)[0]
            st.session_state.predicted_price = predicted_price
            st.metric(f"In {st.session_state.horizon} Months (₹)", f"₹{predicted_price:,.2f}")

            st.subheader("💡 Investment Insights")
            st.write(f"🎯 Risk Level: {st.session_state.risk_tolerance}")
            st.write(f"📈 Horizon: {st.session_state.horizon} months")
            predicted_growth = predicted_price - stock_data['Close'].iloc[-1]
            st.write(f"💰 Predicted Growth: ₹{predicted_growth:,.2f}")

        # Main content
        horizon = st.number_input("⏳ Investment Horizon (Months)", min_value=1, max_value=60, value=45)
        risk_tolerance = st.radio("🎲 Risk Appetite", ["Low", "Medium", "High"])

        # Update session state for stock investments
        st.session_state.horizon = horizon
        st.session_state.risk_tolerance = risk_tolerance

        st.subheader("📉 NIFTY CONSUMPTION Trend")
        fig = px.line(stock_data, x='Date', y='Close', title="Price Trend", template="plotly_dark")
        fig.update_layout(
            title_font_color="#FFFFFF",
            xaxis_title_font_color="#FFFFFF",
            yaxis_title_font_color="#FFFFFF",
            xaxis_tickfont_color="#FFFFFF",
            yaxis_tickfont_color="#FFFFFF"
        )
        st.plotly_chart(fig, use_container_width=True)

        stock_subset = stock_data.copy()
        stock_subset['SMA_30'] = stock_subset['Close'].rolling(window=30).mean()
        stock_subset['Volatility'] = stock_subset['Close'].pct_change().rolling(window=30).std()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📏 Moving Average")
            fig_ma = px.line(stock_subset, x='Date', y=['Close', 'SMA_30'], title="30-Day SMA", template="plotly_dark")
            fig_ma.update_layout(
                title_font_color="#FFFFFF",
                xaxis_title_font_color="#FFFFFF",
                yaxis_title_font_color="#FFFFFF",
                xaxis_tickfont_color="#FFFFFF",
                yaxis_tickfont_color="#FFFFFF",
                legend_font_color="#FFFFFF"
            )
            st.plotly_chart(fig_ma, use_container_width=True)
        with col2:
            st.subheader("🌩️ Volatility")
            fig_vol = px.line(stock_subset, x='Date', y='Volatility', title="30-Day Volatility", template="plotly_dark")
            fig_vol.update_layout(
                title_font_color="#FFFFFF",
                xaxis_title_font_color="#FFFFFF",
                yaxis_title_font_color="#FFFFFF",
                xaxis_tickfont_color="#FFFFFF",
                yaxis_tickfont_color="#FFFFFF",
                legend_font_color="#FFFFFF"
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("🔮 Price Prediction")
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
