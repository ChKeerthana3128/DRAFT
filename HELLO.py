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

# Load and Cache Dataset
@st.cache_data
def load_data():
    # Replace with your dataset file path or paste the data directly
    data = pd.read_csv("financial_data.csv")  # Replace with actual file path
    return data

data = load_data()

# Model Training
def train_model(data):
    """Train a Linear Regression model to predict Disposable Income."""
    # Features for prediction
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    X = data[feature_cols]
    y = data["Disposable_Income"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse

# Cache trained model
@st.cache_resource
def get_trained_model():
    model, r2, rmse = train_model(data)
    return model, r2, rmse

model, r2_score_val, rmse_val = get_trained_model()

# Helper Functions
def calculate_financial_health_score(input_data):
    """Calculate financial health score based on user inputs."""
    income = input_data["Income"]
    savings = input_data["Desired_Savings"]
    debt = input_data["Rent"] + input_data["Loan_Repayment"]
    discretionary = input_data["Eating_Out"] + input_data["Entertainment"]
    
    savings_ratio = savings / income if income > 0 else 0
    debt_ratio = debt / income if income > 0 else 0
    discretionary_ratio = discretionary / income if income > 0 else 0
    
    score = (savings_ratio * 50) - (debt_ratio * 30) - (discretionary_ratio * 20)
    return max(0, min(100, score))

def predict_disposable_income(model, input_data):
    """Predict disposable income using trained model."""
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    input_df = pd.DataFrame([input_data], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    return prediction

def predict_future_savings(income, total_expenses, savings_rate, years):
    """Predict future savings based on current financials."""
    annual_savings = income * (savings_rate / 100) - total_expenses
    return annual_savings * years

# Sidebar
st.sidebar.title("AI Financial Dashboard")
st.sidebar.markdown("Enter your details to get personalized financial insights.")
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"R² Score: {r2_score_val:.2f}")
st.sidebar.write(f"RMSE: ${rmse_val:.2f}")

# Main App
st.title("Your AI-Powered Financial Assistant")
st.markdown("Enter your financial details below to receive predictions, health scores, and wealth management insights.")

# User Input Form
with st.form(key="financial_form"):
    st.subheader("Your Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "John Doe")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
        dependents = st.number_input("Number of Dependents", min_value=0, value=0)
        rent = st.number_input("Rent ($)", min_value=0.0, value=1000.0, step=50.0)
        loan_repayment = st.number_input("Loan Repayment ($)", min_value=0.0, value=0.0, step=50.0)
    with col2:
        insurance = st.number_input("Insurance ($)", min_value=0.0, value=100.0, step=10.0)
        groceries = st.number_input("Groceries ($)", min_value=0.0, value=400.0, step=10.0)
        transport = st.number_input("Transport ($)", min_value=0.0, value=200.0, step=10.0)
        eating_out = st.number_input("Eating Out ($)", min_value=0.0, value=150.0, step=10.0)
        entertainment = st.number_input("Entertainment ($)", min_value=0.0, value=100.0, step=10.0)
        utilities = st.number_input("Utilities ($)", min_value=0.0, value=150.0, step=10.0)
        healthcare = st.number_input("Healthcare ($)", min_value=0.0, value=100.0, step=10.0)
        education = st.number_input("Education ($)", min_value=0.0, value=0.0, step=10.0)
        miscellaneous = st.number_input("Miscellaneous ($)", min_value=0.0, value=50.0, step=10.0)
        desired_savings_percentage = st.number_input("Desired Savings Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    
    submit_button = st.form_submit_button(label="Get Insights")

# Process Inputs and Provide Insights
if submit_button:
    # Create input dictionary
    input_data = {
        "Income": income,
        "Age": age,
        "Dependents": dependents,
        "Rent": rent,
        "Loan_Repayment": loan_repayment,
        "Insurance": insurance,
        "Groceries": groceries,
        "Transport": transport,
        "Eating_Out": eating_out,
        "Entertainment": entertainment,
        "Utilities": utilities,
        "Healthcare": healthcare,
        "Education": education,
        "Miscellaneous": miscellaneous,
        "Desired_Savings_Percentage": desired_savings_percentage
    }
    
    # Calculate total expenses and desired savings
    total_expenses = sum([rent, loan_repayment, insurance, groceries, transport, eating_out, 
                          entertainment, utilities, healthcare, education, miscellaneous])
    desired_savings = income * (desired_savings_percentage / 100)
    input_data["Desired_Savings"] = desired_savings
    
    # Predictions and Metrics
    st.header(f"Financial Insights for {name}")
    
    # Financial Health Score
    health_score = calculate_financial_health_score(input_data)
    st.subheader("Financial Health Score")
    st.metric("Score", f"{health_score:.1f}/100")
    if health_score < 40:
        st.error("⚠️ Low Financial Health: High debt or low savings. Action required!")
    elif health_score < 70:
        st.warning("⚠️ Moderate Health: Room for improvement in savings or spending.")
    else:
        st.success("✅ Excellent Health: Keep up the good work!")
    
    # Disposable Income Prediction
    predicted_disposable = predict_disposable_income(model, input_data)
    st.subheader("Predicted Disposable Income")
    st.metric("Monthly Disposable Income", f"${predicted_disposable:.2f}")
    
    # Spending Breakdown Visualization
    st.subheader("Spending Breakdown")
    spending_data = pd.Series({
        "Rent": rent, "Insurance": insurance, "Groceries": groceries, "Transport": transport,
        "Eating_Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
        "Healthcare": healthcare, "Education": education, "Miscellaneous": miscellaneous
    })
    fig, ax = plt.subplots()
    spending_data.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Your Monthly Spending Breakdown")
    ax.set_ylabel("Amount ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Wealth Management
    st.subheader("Wealth Management")
    years_to_retirement = st.slider("Years until retirement", 1, 40, 30, key="retirement_slider")
    desired_retirement_fund = st.number_input("Desired retirement fund ($)", min_value=10000, value=1000000, step=10000, key="retirement_fund")
    
    future_savings = predict_future_savings(income, total_expenses, desired_savings_percentage, years_to_retirement)
    st.write(f"With your current savings rate ({desired_savings_percentage:.1f}%), you could save **${future_savings:.2f}** by retirement.")
    
    required_savings_rate = (desired_retirement_fund / (income * years_to_retirement)) * 100 if income > 0 else 0
    st.write(f"To reach ${desired_retirement_fund:,} in {years_to_retirement} years, aim for a savings rate of **{required_savings_rate:.2f}%**.")
    
    # Savings Growth Plot
    years = np.arange(1, years_to_retirement + 1)
    savings_trajectory = [predict_future_savings(income, total_expenses, desired_savings_percentage, y) for y in years]
    fig, ax = plt.subplots()
    ax.plot(years, savings_trajectory, marker="o", color="green", label="Current Trajectory")
    ax.axhline(y=desired_retirement_fund, color="red", linestyle="--", label="Goal")
    ax.set_title("Projected Savings Growth")
    ax.set_xlabel("Years")
    ax.set_ylabel("Savings ($)")
    ax.legend()
    st.pyplot(fig)
    
    # Actionable Insights
    st.subheader("Personalized Recommendations")
    if eating_out > income * 0.1:
        st.write("- 🍽️ *Reduce Eating Out*: You're spending over 10% of your income here.")
    if loan_repayment > 0:
        st.write("- 💳 *Prioritize Debt*: Paying off loans will boost your disposable income.")
    if desired_savings_percentage < 10:
        st.write("- 💰 *Increase Savings*: Aim for at least 10% to secure your future.")
    if total_expenses > income * 0.8:
        st.write("- 📉 *Cut Expenses*: Your spending exceeds 80% of income—review your budget.")

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit & AI/ML | Powered by your data insights")
