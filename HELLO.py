import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="AI Financial Dashboard", layout="wide")

# Load dataset (paste your dataset as a string or upload it)
@st.cache_data
def load_data():
    # Replace this with your actual dataset loading logic
    # For now, assuming it's provided as a CSV string or file
    data = pd.read_csv("financial_data.csv")  # Replace with your dataset file path
    return data

data = load_data()

# Preprocessing and Training Model
def train_model(data):
    # Features for prediction (excluding target and non-numeric columns)
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    X = data[feature_cols]
    y = data["Disposable_Income"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse

# Train the model once and cache it
@st.cache_resource
def get_trained_model():
    model, r2, rmse = train_model(data)
    return model, r2, rmse

model, r2_score_val, rmse_val = get_trained_model()

# Helper functions
def calculate_financial_health_score(row):
    """Calculate a financial health score based on key financial metrics."""
    savings_ratio = row["Desired_Savings"] / row["Income"] if row["Income"] > 0 else 0
    debt_ratio = (row["Rent"] + row["Loan_Repayment"]) / row["Income"] if row["Income"] > 0 else 0
    discretionary_spending = (row["Eating_Out"] + row["Entertainment"]) / row["Income"] if row["Income"] > 0 else 0
    
    score = (savings_ratio * 50) - (debt_ratio * 30) - (discretionary_spending * 20)
    return max(0, min(100, score))

def predict_disposable_income(model, user_data):
    """Predict disposable income using the trained model."""
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    input_data = user_data[feature_cols].values.reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return prediction

def predict_future_savings(income, expenses, savings_rate, years):
    """Predict future savings based on current financials."""
    annual_savings = income * savings_rate - expenses
    return annual_savings * years

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Financial Health", "Wealth Management", "Insights"])

# Main App
st.title("AI-Driven Financial Dashboard")
st.markdown("Your personalized financial assistant powered by machine learning. Explore predictions, insights, and wealth strategies.")

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
    
    # Predicted Disposable Income
    predicted_income = predict_disposable_income(model, user_data)
    st.metric("Predicted Disposable Income", f"${predicted_income:.2f}", delta=f"{predicted_income - user_data['Disposable_Income']:.2f}")
    
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
        st.error("⚠️ Financial Health Alert: High debt or low savings detected. Reduce discretionary spending or debt.")
    elif health_score < 70:
        st.warning("⚠️ Moderate Health: Increase savings or optimize expenses for better stability.")
    else:
        st.success("✅ Strong Financial Health: Maintain your current habits!")
    
    # Expense vs Income Visualization
    st.subheader("Income vs Expenses")
    total_expenses = user_data[["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Eating_Out", 
                                "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"]].sum()
    fig, ax = plt.subplots()
    plt.pie([user_data["Disposable_Income"], total_expenses], labels=["Disposable Income", "Expenses"], autopct="%1.1f%%", colors=["#4CAF50", "#FF5722"])
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
    
    # Prediction using trained model
    predicted_disposable = predict_disposable_income(model, user_data)
    future_savings = predict_future_savings(income, total_expenses, current_savings_rate, years_to_retirement)
    st.write(f"With your current savings rate ({user_data['Desired_Savings_Percentage']:.2f}%), you could save **${future_savings:.2f}** by retirement.")
    
    # Required savings rate
    required_savings = desired_retirement_fund / (income * years_to_retirement) if income > 0 else 0
    st.write(f"To reach ${desired_retirement_fund:,} in {years_to_retirement} years, you need a savings rate of **{required_savings*100:.2f}%**.")
    
    # Visualization
    st.subheader("Savings Growth Over Time")
    years = np.arange(1, years_to_retirement + 1)
    savings_trajectory = [predict_future_savings(income, total_expenses, current_savings_rate, y) for y in years]
    
    fig, ax = plt.subplots()
    ax.plot(years, savings_trajectory, marker="o", color="green", label="Current Trajectory")
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
    st.subheader("Potential Savings Opportunities")
    potential_savings_cols = ["Potential_Savings_Groceries", "Potential_Savings_Transport", "Potential_Savings_Eating_Out", 
                              "Potential_Savings_Entertainment", "Potential_Savings_Utilities", "Potential_Savings_Healthcare", 
                              "Potential_Savings_Education", "Potential_Savings_Miscellaneous"]
    savings_data = user_data[potential_savings_cols]
    
    fig, ax = plt.subplots()
    savings_data.plot(kind="bar", ax=ax, color="orange")
    ax.set_title("Potential Monthly Savings by Category")
    ax.set_ylabel("Amount ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Structured Insights
    st.subheader("Actionable Recommendations")
    predicted_disposable = predict_disposable_income(model, user_data)
    total_potential_savings = savings_data.sum()
    
    st.write(f"**Predicted Disposable Income:** ${predicted_disposable:.2f}")
    st.write(f"**Total Potential Monthly Savings:** ${total_potential_savings:.2f}")
    
    if user_data["Eating_Out"] > user_data["Income"] * 0.1:
        st.write("- 🍽️ *Reduce Eating Out*: Spending exceeds 10% of income. Cut back to save more!")
    if user_data["Loan_Repayment"] > 0:
        st.write("- 💳 *Focus on Debt*: Pay off loans to increase disposable income.")
    if user_data["Desired_Savings_Percentage"] < 10:
        st.write("- 💰 *Boost Savings*: Increase your savings rate to at least 10% for long-term security.")
    if total_potential_savings > user_data["Disposable_Income"] * 0.2:
        st.write("- 📉 *Optimize Spending*: Significant savings potential exists—review your budget!")

# Model Performance
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"R² Score: {r2_score_val:.2f}")
st.sidebar.write(f"RMSE: ${rmse_val:.2f}")

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit & AI/ML")
