import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set page configuration
st.set_page_config(page_title="AI Financial Dashboard", layout="wide")

# Load the dataset (replace with your actual file path or upload mechanism)
data = pd.read_csv("financial_data.csv")  # Replace with your dataset file path

# Helper functions
def calculate_financial_health_score(row):
    """Calculate a financial health score based on key financial metrics."""
    savings_ratio = row["Desired_Savings"] / row["Income"] if row["Income"] > 0 else 0
    debt_ratio = (row["Rent"] + row["Loan_Repayment"]) / row["Income"] if row["Income"] > 0 else 0
    discretionary_spending = (row["Eating_Out"] + row["Entertainment"]) / row["Income"] if row["Income"] > 0 else 0
    
    # Weighted score (savings good, debt bad, discretionary moderate)
    score = (savings_ratio * 50) - (debt_ratio * 30) - (discretionary_spending * 20)
    return max(0, min(100, score))  # Normalize to 0-100

def predict_future_savings(income, expenses, savings_rate, years):
    """Predict future savings based on current financials."""
    annual_savings = income * savings_rate - expenses
    return annual_savings * years

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Financial Health", "Wealth Management", "Insights"])

# Main App
st.title("AI-Driven Financial Dashboard")
st.markdown("Welcome to your personalized financial assistant! Explore your financial health, savings potential, and wealth management strategies.")

if page == "Dashboard":
    st.header("Financial Overview")
    
    # User selection
    st.subheader("Select Profile")
    user_idx = st.selectbox("Choose a user profile by index", range(len(data)))
    user_data = data.iloc[user_idx]
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Income", f"${user_data['Income']:.2f}")
    with col2:
        st.metric("Age", int(user_data["Age"]))
    with col3:
        st.metric("Dependents", int(user_data["Dependents"]))
    
    # Spending Breakdown
    st.subheader("Spending Breakdown")
    spending_cols = ["Rent", "Insurance", "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"]
    spending_data = user_data[spending_cols]
    
    fig, ax = plt.subplots()
    spending_data.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Monthly Spending Breakdown")
    ax.set_ylabel("Amount ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Financial Health":
    st.header("Financial Health Analysis")
    
    # User selection
    user_idx = st.selectbox("Select user profile", range(len(data)), key="health_idx")
    user_data = data.iloc[user_idx]
    
    # Financial Health Score
    health_score = calculate_financial_health_score(user_data)
    st.metric("Financial Health Score", f"{health_score:.1f}/100")
    
    # Insights based on score
    if health_score < 40:
        st.error("⚠️ Your financial health needs attention. Consider reducing debt or discretionary spending.")
    elif health_score < 70:
        st.warning("⚠️ Your financial health is moderate. Boost savings or cut unnecessary expenses.")
    else:
        st.success("✅ Your financial health is strong! Keep up the good work.")
    
    # Expense vs Income Visualization
    st.subheader("Income vs Expenses")
    total_expenses = user_data[["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Eating_Out", 
                                "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"]].sum()
    fig, ax = plt.subplots()
    plt.pie([user_data["Income"] - total_expenses, total_expenses], labels=["Disposable Income", "Expenses"], autopct="%1.1f%%", colors=["#4CAF50", "#FF5722"])
    ax.set_title("Income Allocation")
    st.pyplot(fig)

elif page == "Wealth Management":
    st.header("Wealth Management & Retirement Planning")
    
    # User selection
    user_idx = st.selectbox("Select user profile", range(len(data)), key="wealth_idx")
    user_data = data.iloc[user_idx]
    
    # Inputs for planning
    years_to_retirement = st.slider("Years until retirement", 1, 40, 30)
    desired_retirement_fund = st.number_input("Desired retirement fund ($)", min_value=10000, value=1000000, step=10000)
    
    # Current financials
    income = user_data["Income"]
    total_expenses = user_data[["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Eating_Out", 
                                "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"]].sum()
    current_savings_rate = user_data["Desired_Savings_Percentage"] / 100
    
    # Prediction
    future_savings = predict_future_savings(income, total_expenses, current_savings_rate, years_to_retirement)
    st.write(f"With your current savings rate ({user_data['Desired_Savings_Percentage']:.2f}%), you could save **${future_savings:.2f}** by retirement.")
    
    # Required savings rate
    required_savings = desired_retirement_fund / (income * years_to_retirement) if income > 0 else 0
    st.write(f"To reach ${desired_retirement_fund:,} in {years_to_retirement} years, you need a savings rate of **{required_savings*100:.2f}%**.")
    
    # Visualization of savings growth
    st.subheader("Savings Growth Over Time")
    years = np.arange(1, years_to_retirement + 1)
    savings_trajectory = [predict_future_savings(income, total_expenses, current_savings_rate, y) for y in years]
    
    fig, ax = plt.subplots()
    ax.plot(years, savings_trajectory, marker="o", color="green")
    ax.axhline(y=desired_retirement_fund, color="red", linestyle="--", label="Goal")
    ax.set_title("Projected Savings Growth")
    ax.set_xlabel("Years")
    ax.set_ylabel("Savings ($)")
    ax.legend()
    st.pyplot(fig)

elif page == "Insights":
    st.header("Personalized Financial Insights")
    
    # User selection
    user_idx = st.selectbox("Select user profile", range(len(data)), key="insights_idx")
    user_data = data.iloc[user_idx]
    
    # Potential Savings Analysis
    st.subheader("Where You Can Save")
    potential_savings_cols = ["Potential_Savings_Groceries", "Potential_Savings_Transport", "Potential_Savings_Eating_Out", 
                              "Potential_Savings_Entertainment", "Potential_Savings_Utilities", "Potential_Savings_Healthcare", 
                              "Potential_Savings_Education", "Potential_Savings_Miscellaneous"]
    savings_data = user_data[potential_savings_cols]
    
    fig, ax = plt.subplots()
    savings_data.plot(kind="bar", ax=ax, color="orange")
    ax.set_title("Potential Monthly Savings")
    ax.set_ylabel("Amount ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Personalized Recommendations
    st.subheader("Recommendations")
    if user_data["Eating_Out"] > user_data["Income"] * 0.1:
        st.write("🍽️ *Cut back on eating out* - You’re spending more than 10% of your income here!")
    if user_data["Loan_Repayment"] > 0:
        st.write("💳 *Prioritize debt repayment* - Clearing loans will free up more disposable income.")
    if user_data["Desired_Savings_Percentage"] < 10:
        st.write("💰 *Increase your savings rate* - Aim for at least 10% to build a stronger financial future.")

# Footer
st.markdown("---")
st.write("Built with ❤️ by [Your Name] using Streamlit & AI/ML")
