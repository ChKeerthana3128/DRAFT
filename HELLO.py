import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="AI Financial Dashboard (INR)", layout="wide")

# Load Dataset from CSV File
@st.cache_data
def load_data(csv_path="financial_data.csv"):
    """Load the dataset from a CSV file and clean column names."""
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}. Please ensure 'financial_data.csv' exists.")
        return None
    
    try:
        data = pd.read_csv(csv_path)
        
        # Define the exact column name with newline as it appears in your data
        combined_column_name = "Miscellaneous (Eating_Out,Entertainmentand Utilities)\n"
        
        if combined_column_name not in data.columns:
            st.error(f"Column '{combined_column_name}' not found. Available columns: {data.columns.tolist()}")
            return None

        # Split Miscellaneous into Eating_Out, Entertainment, Utilities
        def distribute_combined_value(value):
            return [value / 3] * 3 if pd.notna(value) and value > 0 else [0, 0, 0]

        distributed_values = data[combined_column_name].apply(distribute_combined_value)
        data[['Eating_Out', 'Entertainment', 'Utilities']] = pd.DataFrame(distributed_values.tolist(), index=data.index)
        data = data.drop(columns=[combined_column_name])

        # Clean Education column (it has a newline in your data)
        if "Education\n" in data.columns:
            data = data.rename(columns={"Education\n": "Education"})
        elif "Education" not in data.columns:
            data["Education"] = 0

        required_cols = ["Income", "Age", "Dependents", "Occupation", "City_Tier", "Rent", "Loan_Repayment", 
                         "Insurance", "Groceries", "Transport", "Healthcare", "Education", "Eating_Out", 
                         "Entertainment", "Utilities", "Desired_Savings_Percentage", "Disposable_Income"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            return None

        return data
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

# Load data
data = load_data()
if data is None:
    st.stop()

# Model Training
def train_model(data):
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
def get_trained_model():
    model, r2 = train_model(data)
    return model, r2

model, r2_score_val = get_trained_model()

# Helper Functions
def prepare_input(input_data):
    """Prepare user input for prediction."""
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

def calculate_financial_health_score(income, savings, debt, miscellaneous):
    """Calculate financial health score (0-100)."""
    savings_ratio = savings / income if income > 0 else 0
    debt_ratio = debt / income if income > 0 else 0
    discretionary_ratio = miscellaneous / income if income > 0 else 0
    score = (savings_ratio * 50) - (debt_ratio * 30) - (discretionary_ratio * 20)
    return max(0, min(100, score))

def predict_disposable_income(model, input_data):
    """Predict disposable income."""
    input_df = prepare_input(input_data)
    return max(0, model.predict(input_df)[0])

def predict_future_savings(income, total_expenses, savings_rate, years, income_growth_rate=0.0, expense_growth_rate=0.0):
    """Calculate projected savings over time."""
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
    """Generate savings trajectory for plotting."""
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
    """Suggest retirement planning parameters."""
    suggested_fund = total_expenses * 12 * 20  # 20 years of expenses
    annual_savings_needed = suggested_fund / years_to_retirement if years_to_retirement > 0 else suggested_fund
    suggested_savings_rate = min(max((annual_savings_needed / income) * 100 if income > 0 else 10.0, 5.0), 50.0)
    suggested_income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    suggested_expense_growth = 2.5
    return suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth

# Sidebar Layout
st.sidebar.title("Financial Insights (INR)")
st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"R² Score: {r2_score_val:.2f}")

# Main App
st.title("AI Financial Dashboard (INR)")
st.markdown("Enter your financial details for personalized insights in Indian Rupees.")

# User Input Form
with st.form(key="financial_form"):
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
    submit_button = st.form_submit_button("Analyze My Finances")

# Process Inputs and Display Results
if submit_button:
    input_data = {
        "Income": income, "Age": age, "Dependents": dependents, "Rent": rent, "Loan_Repayment": loan_repayment,
        "Insurance": insurance, "Groceries": groceries, "Transport": transport, "Healthcare": healthcare,
        "Education": education, "Eating_Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
        "Desired_Savings_Percentage": desired_savings_percentage
    }
    
    total_expenses = rent + loan_repayment + insurance + groceries + transport + healthcare + education + eating_out + entertainment + utilities
    years_to_retirement = max(0, retirement_age - age)
    debt = rent + loan_repayment
    
    # Financial Health Score
    savings = income * (desired_savings_percentage / 100)
    miscellaneous = eating_out + entertainment + utilities
    health_score = calculate_financial_health_score(income, savings, debt, miscellaneous)
    st.sidebar.subheader("Financial Health Score")
    st.sidebar.metric("Score", f"{health_score:.1f}/100")
    st.sidebar.write("Status: " + ("Low" if health_score < 40 else "Moderate" if health_score < 70 else "Excellent"))

    # Predicted Disposable Income
    predicted_disposable = predict_disposable_income(model, input_data)
    st.sidebar.subheader("Predicted Disposable Income")
    st.sidebar.metric("Monthly (₹)", f"₹{predicted_disposable:,.2f}")

    # Wealth Management
    suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth = suggest_wealth_management_params(income, total_expenses, years_to_retirement)
    st.subheader("Wealth Management")
    desired_retirement_fund = st.number_input("Desired Retirement Fund (₹)", min_value=100000.0, value=float(suggested_fund), step=100000.0)
    savings_rate_filter = st.slider("Savings Rate (%)", 0.0, 100.0, suggested_savings_rate, step=1.0)
    income_growth_rate = st.slider("Income Growth Rate (%)", 0.0, 10.0, suggested_income_growth, step=0.5)
    expense_growth_rate = st.slider("Expense Growth Rate (%)", 0.0, 10.0, suggested_expense_growth, step=0.5)

    # Projected Savings
    projected_savings = predict_future_savings(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
    st.sidebar.subheader("Projected Savings")
    st.sidebar.metric("At Retirement (₹)", f"₹{projected_savings:,.2f}")

    # Required Savings Rate
    required_savings_rate = (desired_retirement_fund / (income * years_to_retirement)) * 100 if income > 0 and years_to_retirement > 0 else 0
    st.sidebar.subheader("Required Savings Rate")
    st.sidebar.metric("To Meet Goal (%)", f"{required_savings_rate:.2f}%")

    # Detailed Insights
    st.header(f"Financial Insights for {name}")
    
    # Spending Breakdown
    st.subheader("Spending Breakdown")
    spending_data = pd.Series({"Rent": rent, "Loan": loan_repayment, "Insurance": insurance, "Groceries": groceries,
                               "Transport": transport, "Healthcare": healthcare, "Education": education,
                               "Eating Out": eating_out, "Entertainment": entertainment, "Utilities": utilities})
    fig, ax = plt.subplots()
    spending_data.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Monthly Spending (INR)")
    ax.set_ylabel("Amount (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Savings Growth Projection
    st.subheader("Savings Growth Projection")
    trajectory = get_savings_trajectory(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
    years = np.arange(years_to_retirement + 1)
    fig, ax = plt.subplots()
    ax.plot(years, trajectory, marker="o", color="green", label="Projected Savings")
    ax.axhline(y=desired_retirement_fund, color="red", linestyle="--", label="Goal")
    ax.set_title("Savings Growth (INR)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Savings (₹)")
    ax.legend()
    st.pyplot(fig)

    # Recommendations
    st.subheader("Recommendations")
    if miscellaneous > income * 0.2:
        st.write(f"- Reduce discretionary spending (₹{miscellaneous:,.2f} > 20% of income).")
    if debt > 0:
        st.write(f"- Address debt (₹{debt:,.2f}) to improve disposable income.")
    if savings_rate_filter < 10:
        st.write(f"- Increase savings rate from {savings_rate_filter:.1f}% to at least 10%.")

st.markdown("---")
st.write("Powered by Streamlit")
