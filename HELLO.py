import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="AI Financial Dashboard (INR)", layout="wide")

# Load and Normalize Dataset
@st.cache_data
def load_and_normalize_data():
    # Replace with your dataset file path
    data = pd.read_csv("financial_data.csv")  # Replace with actual file path
    
    # Select numeric columns for normalization
    numeric_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage", 
                    "Disposable_Income"]
    
    # Normalize numeric columns using MinMaxScaler
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data, scaler

data, scaler = load_and_normalize_data()

# Model Training
def train_model(data):
    """Train a Linear Regression model on normalized data."""
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    X = data[feature_cols]
    y = data["Disposable_Income"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse

@st.cache_resource
def get_trained_model():
    model, r2, rmse = train_model(data)
    return model, r2, rmse

model, r2_score_val, rmse_val = get_trained_model()

# Helper Functions
def normalize_input(input_data, scaler):
    """Normalize user input data using the same scaler as training data."""
    numeric_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    input_df = pd.DataFrame([input_data], columns=numeric_cols)
    input_normalized = scaler.transform(input_df[numeric_cols])
    return pd.DataFrame(input_normalized, columns=numeric_cols)

def denormalize_value(value, scaler, column_idx):
    """Denormalize a single value back to original scale."""
    min_val = scaler.data_min_[column_idx]
    max_val = scaler.data_max_[column_idx]
    return value * (max_val - min_val) + min_val

def calculate_financial_health_score(input_data):
    """Calculate financial health score."""
    income = input_data["Income"]
    savings = input_data["Desired_Savings"]
    debt = input_data["Rent"] + input_data["Loan_Repayment"]
    discretionary = input_data["Eating_Out"] + input_data["Entertainment"]
    
    savings_ratio = savings / income if income > 0 else 0
    debt_ratio = debt / income if income > 0 else 0
    discretionary_ratio = discretionary / income if income > 0 else 0
    
    score = (savings_ratio * 50) - (debt_ratio * 30) - (discretionary_ratio * 20)
    return max(0, min(100, score))

def predict_disposable_income(model, input_data, scaler):
    """Predict disposable income and denormalize result."""
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    normalized_input = normalize_input(input_data, scaler)
    prediction_normalized = model.predict(normalized_input[feature_cols])[0]
    
    # Denormalize prediction (Disposable_Income is the last column in numeric_cols)
    disposable_idx = 15  # Index of Disposable_Income in numeric_cols
    prediction_denormalized = denormalize_value(prediction_normalized, scaler, disposable_idx)
    return prediction_denormalized

def predict_future_savings(income, total_expenses, savings_rate, years):
    """Predict future savings in INR."""
    annual_savings = income * (savings_rate / 100) - total_expenses
    return annual_savings * years

# Sidebar Layout
st.sidebar.title("Financial Insights")
st.sidebar.markdown("Your key financial metrics and predictions.")

# Main App
st.title("AI Financial Dashboard (INR)")
st.markdown("Enter your financial details to get personalized insights in Indian Rupees.")

# User Input Form (Main Area)
with st.form(key="financial_form"):
    st.subheader("Enter Your Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Amit Sharma")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
        dependents = st.number_input("Number of Dependents", min_value=0, value=0)
        rent = st.number_input("Rent (₹)", min_value=0.0, value=15000.0, step=500.0)
        loan_repayment = st.number_input("Loan Repayment (₹)", min_value=0.0, value=0.0, step=500.0)
    with col2:
        insurance = st.number_input("Insurance (₹)", min_value=0.0, value=2000.0, step=100.0)
        groceries = st.number_input("Groceries (₹)", min_value=0.0, value=8000.0, step=100.0)
        transport = st.number_input("Transport (₹)", min_value=0.0, value=3000.0, step=100.0)
        eating_out = st.number_input("Eating Out (₹)", min_value=0.0, value=4000.0, step=100.0)
        entertainment = st.number_input("Entertainment (₹)", min_value=0.0, value=2000.0, step=100.0)
        utilities = st.number_input("Utilities (₹)", min_value=0.0, value=2500.0, step=100.0)
        healthcare = st.number_input("Healthcare (₹)", min_value=0.0, value=1500.0, step=100.0)
        education = st.number_input("Education (₹)", min_value=0.0, value=0.0, step=100.0)
        miscellaneous = st.number_input("Miscellaneous (₹)", min_value=0.0, value=1000.0, step=100.0)
        desired_savings_percentage = st.number_input("Desired Savings Percentage (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    
    submit_button = st.form_submit_button(label="Analyze My Finances")

# Process Inputs and Display Results
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
    
    total_expenses = sum([rent, loan_repayment, insurance, groceries, transport, eating_out, 
                          entertainment, utilities, healthcare, education, miscellaneous])
    desired_savings = income * (desired_savings_percentage / 100)
    input_data["Desired_Savings"] = desired_savings
    
    # Sidebar: Financial Health Score
    st.sidebar.subheader("Financial Health")
    health_score = calculate_financial_health_score(input_data)
    st.sidebar.metric("Health Score", f"{health_score:.1f}/100")
    if health_score < 40:
        st.sidebar.error("⚠️ Low Health: Act now!")
    elif health_score < 70:
        st.sidebar.warning("⚠️ Moderate: Optimize!")
    else:
        st.sidebar.success("✅ Excellent!")
    
    # Sidebar: Disposable Income Prediction
    predicted_disposable = predict_disposable_income(model, input_data, scaler)
    st.sidebar.subheader("Predicted Disposable Income")
    st.sidebar.metric("Monthly (₹)", f"₹{predicted_disposable:,.2f}")
    
    # Sidebar: Wealth Management
    st.sidebar.subheader("Wealth Management")
    years_to_retirement = st.sidebar.slider("Years to Retirement", 1, 40, 30)
    desired_retirement_fund = st.sidebar.number_input("Desired Retirement Fund (₹)", min_value=100000.0, value=5000000.0, step=100000.0)
    
    future_savings = predict_future_savings(income, total_expenses, desired_savings_percentage, years_to_retirement)
    st.sidebar.write(f"Projected Savings: **₹{future_savings:,.2f}**")
    required_savings_rate = (desired_retirement_fund / (income * years_to_retirement)) * 100 if income > 0 else 0
    st.sidebar.write(f"Required Savings Rate: **{required_savings_rate:.2f}%**")
    
    # Main Area: Detailed Insights
    st.header(f"Financial Insights for {name}")
    
    # Spending Breakdown
    st.subheader("Spending Breakdown")
    spending_data = pd.Series({
        "Rent": rent, "Insurance": insurance, "Groceries": groceries, "Transport": transport,
        "Eating_Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
        "Healthcare": healthcare, "Education": education, "Miscellaneous": miscellaneous
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    spending_data.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Your Monthly Spending (INR)")
    ax.set_ylabel("Amount (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Savings Growth Plot
    st.subheader("Savings Growth Projection")
    years = np.arange(1, years_to_retirement + 1)
    savings_trajectory = [predict_future_savings(income, total_expenses, desired_savings_percentage, y) for y in years]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, savings_trajectory, marker="o", color="green", label="Current Trajectory")
    ax.axhline(y=desired_retirement_fund, color="red", linestyle="--", label="Goal")
    ax.set_title("Projected Savings Growth (INR)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Savings (₹)")
    ax.legend()
    st.pyplot(fig)
    
    # Actionable Recommendations
    st.subheader("Personalized Recommendations")
    if eating_out > income * 0.1:
        st.write("- 🍽️ *Reduce Eating Out*: Spending exceeds 10% of income (₹{eating_out:,.2f}).")
    if loan_repayment > 0:
        st.write("- 💳 *Clear Debt*: Loan repayment (₹{loan_repayment:,.2f}) reduces your disposable income.")
    if desired_savings_percentage < 10:
        st.write("- 💰 *Boost Savings*: Increase your savings rate from {desired_savings_percentage:.1f}% to at least 10%.")
    if total_expenses > income * 0.8:
        st.write("- 📉 *Cut Expenses*: Spending (₹{total_expenses:,.2f}) is over 80% of income—review your budget!")

# Model Performance in Sidebar
st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"R² Score: {r2_score_val:.2f}")
st.sidebar.write(f"RMSE (Normalized): {rmse_val:.2f}")

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit & AI/ML | All amounts in Indian Rupees (₹)")
