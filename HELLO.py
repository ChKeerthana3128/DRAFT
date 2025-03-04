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

# Set page configuration at the top
st.set_page_config(page_title="AI Financial & Investment Dashboard (INR)", layout="wide")

# --- Data Loading for Personal Finance ---
@st.cache_data
def load_finance_data(csv_path="financial_data.csv"):
    """Load and preprocess the personal finance dataset."""
    if not os.path.exists(csv_path):
        st.error(f"Finance CSV file not found at {csv_path}. Ensure 'financial_data.csv' exists.")
        return None
    
    try:
        data = pd.read_csv(csv_path)
        combined_column_name = "Miscellaneous (Eating_Out,Entertainmentand Utilities)\n"
        
        if combined_column_name not in data.columns:
            st.error(f"Column '{combined_column_name}' not found. Available columns: {data.columns.tolist()}")
            return None

        def distribute_combined_value(value):
            return [value / 3] * 3 if pd.notna(value) and value > 0 else [0, 0, 0]

        distributed_values = data[combined_column_name].apply(distribute_combined_value)
        data[['Eating_Out', 'Entertainment', 'Utilities']] = pd.DataFrame(distributed_values.tolist(), index=data.index)
        data = data.drop(columns=[combined_column_name])

        if "Education\n" in data.columns:
            data = data.rename(columns={"Education\n": "Education"})
        elif "Education" not in data.columns:
            data["Education"] = 0

        required_cols = ["Income", "Age", "Dependents", "Occupation", "City_Tier", "Rent", "Loan_Repayment", 
                         "Insurance", "Groceries", "Transport", "Healthcare", "Education", "Eating_Out", 
                         "Entertainment", "Utilities", "Desired_Savings_Percentage", "Disposable_Income"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in finance data: {missing_cols}")
            return None

        return data
    except Exception as e:
        st.error(f"Error loading finance CSV: {str(e)}")
        return None

# --- Data Loading for Stock Data ---
@st.cache_data
def load_stock_data(csv_path="archive (3) 2"):
    """Load and preprocess the stock dataset."""
    if not os.path.exists(csv_path):
        st.error(f"Stock CSV file not found at {csv_path}. Ensure "archive (3) 2")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading stock CSV: {str(e)}")
        return None

# --- Model Training for Personal Finance ---
def train_finance_model(data):
    """Train a LinearRegression model to predict Disposable_Income."""
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", "Groceries", 
                    "Transport", "Healthcare", "Education", "Eating_Out", "Entertainment", "Utilities", 
                    "Desired_Savings_Percentage"]
    categorical_cols = ["Occupation", "City_Tier"]
    
    X = pd.get_dummies(data[feature_cols + categorical_cols], columns=categorical_cols)
    y = data["Disposable_Income"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2

@st.cache_resource
def get_trained_finance_model(data):
    return train_finance_model(data)

# --- Predictive Functions for Personal Finance ---
def prepare_finance_input(input_data, model):
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", "Groceries", 
                    "Transport", "Healthcare", "Education", "Eating_Out", "Entertainment", "Utilities", 
                    "Desired_Savings_Percentage"]
    input_dict = {col: input_data.get(col, 0.0 if col != "Desired_Savings_Percentage" else 10.0) for col in feature_cols}
    
    input_df = pd.DataFrame([input_dict], columns=feature_cols)
    input_df["Occupation"] = "Unknown"
    input_df["City_Tier"] = "Unknown"
    
    input_df = pd.get_dummies(input_df, columns=["Occupation", "City_Tier"])
    trained_features = model.feature_names_in_
    for feature in trained_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[trained_features]
    
    return input_df

def calculate_financial_health_score(income, savings, debt, discretionary):
    savings_ratio = savings / income if income > 0 else 0
    debt_ratio = debt / income if income > 0 else 0
    discretionary_ratio = discretionary / income if income > 0 else 0
    score = (savings_ratio * 50) - (debt_ratio * 30) - (discretionary_ratio * 20)
    return max(0, min(100, score))

def predict_disposable_income(model, input_data):
    input_df = prepare_finance_input(input_data, model)
    return max(0, model.predict(input_df)[0])

def predict_future_savings(income, total_expenses, savings_rate, years, income_growth_rate=0.0, expense_growth_rate=0.0):
    total_savings = 0.0
    current_income = income
    current_expenses = total_expenses
    
    for _ in range(years + 1):
        annual_savings = max(0, (current_income * (savings_rate / 100)) - current_expenses)
        total_savings += annual_savings
        current_income *= (1 + income_growth_rate / 100)
        current_expenses *= (1 + expense_growth_rate / 100)
    
    return total_savings

def get_savings_trajectory(income, total_expenses, savings_rate, years, income_growth_rate, expense_growth_rate):
    trajectory = []
    current_income = income
    current_expenses = total_expenses
    total_savings = 0.0
    
    for _ in range(years + 1):
        annual_savings = max(0, (current_income * (savings_rate / 100)) - current_expenses)
        total_savings += annual_savings
        trajectory.append(total_savings)
        current_income *= (1 + income_growth_rate / 100)
        current_expenses *= (1 + expense_growth_rate / 100)
    
    return trajectory

def suggest_wealth_management_params(income, total_expenses, years_to_retirement):
    suggested_fund = total_expenses * 12 * 20
    annual_savings_needed = suggested_fund / years_to_retirement if years_to_retirement > 0 else suggested_fund
    suggested_savings_rate = min(max((annual_savings_needed / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    suggested_income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    suggested_expense_growth = 2.5
    return suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth

# --- Insight Generators ---
def generate_wealth_management_insights(income, total_expenses, savings_rate, years_to_retirement, projected_savings, desired_retirement_fund):
    shortfall = desired_retirement_fund - projected_savings if projected_savings < desired_retirement_fund else 0
    insights = []
    insights.append(f"With a {savings_rate:.1f}% savings rate over {years_to_retirement} years, you'll save ₹{projected_savings:,.2f}.")
    if shortfall > 0:
        additional_rate = (shortfall / (income * years_to_retirement)) * 100 if income > 0 and years_to_retirement > 0 else 0
        insights.append(f"To reach ₹{desired_retirement_fund:,.2f}, increase savings rate by {additional_rate:.1f}% or reduce expenses.")
    else:
        insights.append("You’re on track to exceed your retirement goal—consider investing surplus!")
    if years_to_retirement > 20:
        insights.append("Long horizon: A 3% income growth rate could boost savings.")
    elif years_to_retirement < 10:
        insights.append("Short horizon: Focus on higher savings now.")
    return insights

def generate_financial_health_insights(health_score, debt, discretionary, income):
    insights = []
    insights.append(f"Your Financial Health Score is {health_score:.1f}/100.")
    if health_score < 40:
        insights.append("Low score: Prioritize debt reduction and savings.")
    elif health_score < 70:
        insights.append("Moderate score: Room to improve—balance spending and savings.")
    else:
        insights.append("Excellent score: Maintain your financial discipline!")
    debt_ratio = debt / income if income > 0 else 0
    if debt_ratio > 0.3:
        insights.append(f"Debt (₹{debt:,.2f}) exceeds 30% of income—consider refinancing.")
    discretionary_ratio = discretionary / income if income > 0 else 0
    if discretionary_ratio > 0.2:
        insights.append(f"Discretionary spending (₹{discretionary:,.2f}) is high—review entertainment and utilities.")
    return insights

# --- App Logic ---
def main():
    # Load data
    finance_data = load_finance_data()
    stock_data = load_stock_data()
    
    if finance_data is None or stock_data is None:
        st.stop()

    # Train finance model
    finance_model, finance_r2 = get_trained_finance_model(finance_data)

    # Tabs for navigation
    tab1, tab2 = st.tabs(["Personal Finance Dashboard", "Stock Investment Dashboard"])

    # --- Personal Finance Dashboard ---
    with tab1:
        st.header("Personal Finance Dashboard (INR)")
        st.markdown("Analyze your personal finances and plan for retirement.")

        # Sidebar for Personal Finance
        st.sidebar.title("Personal Finance Insights")
        st.sidebar.subheader("Model Accuracy")
        st.sidebar.write(f"R² Score: {finance_r2:.2f}")

        with st.form(key="finance_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", "Amit Sharma")
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                income = st.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
                dependents = st.number_input("Dependents", min_value=0, value=0)
                rent = st.number_input("Rent (₹)", min_value=0.0, value=15000.0, step=500.0)
                loan_repayment = st.number_input("Loan Repayment (₹)", min_value=0.0, value=0.0, step=500.0)
            with col2:
                insurance = st.number_input("Insurance (₹)", min_value=0.0, value=2000.0, step=100.0)
                groceries = st.number_input("Groceries (₹)", min_value=0.0, value=8000.0, step=100.0)
                transport = st.number_input("Transport (₹)", min_value=0.0, value=3000.0, step=100.0)
                healthcare = st.number_input("Healthcare (₹)", min_value=0.0, value=1500.0, step=100.0)
                education = st.number_input("Education (₹)", min_value=0.0, value=0.0, step=100.0)
                eating_out = st.number_input("Eating Out (₹)", min_value=0.0, value=2000.0, step=100.0)
                entertainment = st.number_input("Entertainment (₹)", min_value=0.0, value=1500.0, step=100.0)
                utilities = st.number_input("Utilities (₹)", min_value=0.0, value=4000.0, step=100.0)
                desired_savings_percentage = st.number_input("Desired Savings %", min_value=0.0, max_value=100.0, value=10.0)
            
            retirement_age = st.slider("Retirement Age", int(age), 62, min(62, age + 30))
            submit_finance = st.form_submit_button("Analyze My Finances")

        if submit_finance:
            input_data = {
                "Income": income, "Age": age, "Dependents": dependents, "Rent": rent, "Loan_Repayment": loan_repayment,
                "Insurance": insurance, "Groceries": groceries, "Transport": transport, "Healthcare": healthcare,
                "Education": education, "Eating_Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
                "Desired_Savings_Percentage": desired_savings_percentage
            }
            
            total_expenses = rent + loan_repayment + insurance + groceries + transport + healthcare + education + eating_out + entertainment + utilities
            years_to_retirement = max(0, retirement_age - age)
            debt = rent + loan_repayment
            discretionary = eating_out + entertainment + utilities
            savings = income * (desired_savings_percentage / 100)

            # Sidebar Metrics
            st.sidebar.subheader("Financial Health Score")
            health_score = calculate_financial_health_score(income, savings, debt, discretionary)
            st.sidebar.metric("Score", f"{health_score:.1f}/100")
            st.sidebar.write(f"Status: {'Excellent' if health_score >= 70 else 'Moderate' if health_score >= 40 else 'Low'}")

            st.sidebar.subheader("Predicted Disposable Income")
            predicted_disposable = predict_disposable_income(finance_model, input_data)
            st.sidebar.metric("Monthly (₹)", f"₹{predicted_disposable:,.2f}")

            st.subheader("Wealth Management")
            suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth = suggest_wealth_management_params(income, total_expenses, years_to_retirement)
            desired_retirement_fund = st.number_input("Desired Retirement Fund (₹)", min_value=100000.0, value=suggested_fund, step=100000.0)
            savings_rate_filter = st.slider("Savings Rate (%)", 0.0, 100.0, suggested_savings_rate, step=1.0)
            income_growth_rate = st.slider("Income Growth Rate (%)", 0.0, 10.0, suggested_income_growth, step=0.5)
            expense_growth_rate = st.slider("Expense Growth Rate (%)", 0.0, 10.0, suggested_expense_growth, step=0.5)

            projected_savings = predict_future_savings(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
            st.sidebar.subheader("Projected Savings at Retirement")
            st.sidebar.metric("Total (₹)", f"₹{projected_savings:,.2f}")

            required_savings_rate = (desired_retirement_fund / (income * years_to_retirement)) * 100 if income > 0 and years_to_retirement > 0 else 0
            st.sidebar.subheader("Required Savings Rate")
            st.sidebar.metric("To Meet Goal (%)", f"{required_savings_rate:.2f}%")

            with st.sidebar.expander("Wealth Management Insights"):
                for insight in generate_wealth_management_insights(income, total_expenses, savings_rate_filter, years_to_retirement, projected_savings, desired_retirement_fund):
                    st.write(f"- {insight}")

            with st.sidebar.expander("Financial Health Insights"):
                for insight in generate_financial_health_insights(health_score, debt, discretionary, income):
                    st.write(f"- {insight}")

            # Main Visuals
            st.subheader("Spending Breakdown")
            spending_data = pd.Series({"Rent": rent, "Loan": loan_repayment, "Insurance": insurance, "Groceries": groceries,
                                       "Transport": transport, "Healthcare": healthcare, "Education": education,
                                       "Eating Out": eating_out, "Entertainment": entertainment, "Utilities": utilities})
            fig, ax = plt.subplots(figsize=(10, 5))
            spending_data.plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title("Monthly Spending (INR)")
            ax.set_ylabel("Amount (₹)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("Savings Growth Projection")
            trajectory = get_savings_trajectory(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
            years = np.arange(years_to_retirement + 1)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(years, trajectory, marker="o", color="green", label="Projected Savings")
            ax.axhline(y=desired_retirement_fund, color="red", linestyle="--", label="Retirement Goal")
            ax.set_title("Savings Growth (INR)")
            ax.set_xlabel("Years to Retirement")
            ax.set_ylabel("Savings (₹)")
            ax.legend()
            st.pyplot(fig)

    # --- Stock Investment Dashboard ---
    with tab2:
        st.header("Stock Investment Dashboard")
        st.markdown("Analyze stock performance and get investment predictions.")

        # Sidebar for Stock Options
        st.sidebar.title("Stock Investment Options")
        selected_stock = st.sidebar.selectbox("Choose a Stock", stock_data['Symbol'].unique())
        horizon = st.sidebar.slider("Investment Horizon (Months)", 1, 60, 12)
        risk_tolerance = st.sidebar.radio("Risk Level", ["Low", "Medium", "High"])

        stock_subset = stock_data[stock_data['Symbol'] == selected_stock]

        # Stock Price Trend
        st.subheader(f"{selected_stock} Stock Performance")
        fig = px.line(stock_subset, x='Date', y='Close', title=f"{selected_stock} Stock Price Trend", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Moving Average & Volatility
        st.subheader("Moving Averages & Volatility")
        stock_subset['SMA_30'] = stock_subset['Close'].rolling(window=30).mean()
        stock_subset['Volatility'] = stock_subset['Close'].pct_change().rolling(window=30).std()

        fig_ma = px.line(stock_subset, x='Date', y=['Close', 'SMA_30'], title="30-Day Moving Average", template="plotly_dark")
        fig_vol = px.line(stock_subset, x='Date', y='Volatility', title="Stock Volatility", template="plotly_dark")
        st.plotly_chart(fig_ma, use_container_width=True)
        st.plotly_chart(fig_vol, use_container_width=True)

        # AI-Based Stock Price Prediction
        st.subheader("AI-Based Stock Price Prediction")
        stock_subset['Day'] = stock_subset['Date'].dt.day
        stock_subset['Month'] = stock_subset['Date'].dt.month
        stock_subset['Year'] = stock_subset['Date'].dt.year

        X = stock_subset[['Day', 'Month', 'Year']]
        y = stock_subset['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        stock_model = RandomForestRegressor(n_estimators=100, random_state=42)
        stock_model.fit(X_train, y_train)

        future_days = pd.DataFrame({"Day": [1], "Month": [horizon], "Year": [2025]})
        predicted_price = stock_model.predict(future_days)[0]
        st.write(f"📌 **Predicted Price for {selected_stock} in {horizon} months:** ₹{predicted_price:.2f}")

        # Portfolio Recommendation
        st.subheader("Investment Portfolio Recommendation")
        if risk_tolerance == "Low":
            recommendation = "✅ Invest in Blue-Chip Stocks & Bonds (HDFC, TCS, Infosys, Govt Bonds)"
        elif risk_tolerance == "Medium":
            recommendation = "✅ Invest in Diversified Portfolio (Large Cap, Mid Cap, Real Estate, Mutual Funds)"
        else:
            recommendation = "✅ Invest in High-Growth Stocks & Startups (Small Cap, Crypto, Emerging Markets)"
        st.write(recommendation)

        # Financial Ratios & ESG (assuming available in data)
        st.subheader("Financial Ratios & ESG Ranking")
        available_ratios = [col for col in ['PE_ratio', 'EPS_ratio', 'PS_ratio', 'PB_ratio', 'NetProfitMargin_ratio', 'roe_ratio', 'current_ratio', 'ESG_ranking'] if col in stock_subset.columns]
        if available_ratios:
            st.dataframe(stock_subset[available_ratios])
        else:
            st.write("No financial ratios or ESG data available in the dataset.")

        # Save Stock Model
        if not os.path.exists("models"):
            os.makedirs("models")
        joblib.dump(stock_model, "models/stock_price_model.pkl")

    st.markdown("---")
    st.write("Powered by Streamlit")

if __name__ == "__main__":
    main()
