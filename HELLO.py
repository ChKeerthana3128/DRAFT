import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="AI Financial Dashboard (INR)", layout="wide")

# Load Dataset from CSV
@st.cache_data
def load_data():
    """Load the dataset from a CSV file without normalization."""
    data = pd.read_csv("financial_data.csv")  # Replace with your dataset path
    return data

# Load data
data = load_data()

# Model Training
def train_model(data):
    """Train a Linear Regression model on raw data."""
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
    
    return model, r2

@st.cache_resource
def get_trained_model():
    model, r2 = train_model(data)
    return model, r2

# Train the model
model, r2_score_val = get_trained_model()

# Helper Functions
def prepare_input(input_data):
    """Prepare user input data for prediction without normalization."""
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    input_df = pd.DataFrame({col: [input_data[col]] for col in feature_cols})
    return input_df

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

def predict_disposable_income(model, input_data):
    """Predict disposable income using raw data."""
    input_df = prepare_input(input_data)
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    prediction = model.predict(input_df[feature_cols])[0]
    return prediction

def predict_future_savings(income, total_expenses, savings_rate, years, income_growth_rate=0.0, expense_growth_rate=0.0):
    """Predict future savings in INR with growth rates."""
    savings_trajectory = []
    current_income = income
    current_expenses = total_expenses
    
    for year in range(years + 1):
        annual_savings = current_income * (savings_rate / 100) - current_expenses
        savings_trajectory.append(annual_savings * year if year > 0 else 0)
        current_income *= (1 + income_growth_rate / 100)
        current_expenses *= (1 + expense_growth_rate / 100)
    
    return savings_trajectory[-1]

def get_savings_trajectory(income, total_expenses, savings_rate, years, income_growth_rate, expense_growth_rate):
    """Get savings trajectory for plotting."""
    savings_trajectory = []
    current_income = income
    current_expenses = total_expenses
    
    for year in range(years + 1):
        annual_savings = current_income * (savings_rate / 100) - current_expenses
        savings_trajectory.append(annual_savings * year if year > 0 else 0)
        current_income *= (1 + income_growth_rate / 100)
        current_expenses *= (1 + expense_growth_rate / 100)
    
    return savings_trajectory

def suggest_wealth_management_params(income, total_expenses, years_to_retirement):
    """Suggest Wealth Management parameters based on years to retirement."""
    # Desired Retirement Fund: 20x annual expenses
    suggested_fund = total_expenses * 12 * 20
    
    # Savings Rate: Adjusted to reach fund in given years
    annual_savings_needed = suggested_fund / years_to_retirement if years_to_retirement > 0 else suggested_fund
    suggested_savings_rate = (annual_savings_needed / income) * 100 if income > 0 else 10.0
    suggested_savings_rate = min(max(suggested_savings_rate, 5.0), 50.0)  # Cap between 5% and 50%
    
    # Income Growth Rate: Based on time horizon
    suggested_income_growth = 3.0 if years_to_retirement > 20 else 2.0 if years_to_retirement > 10 else 1.0
    
    # Expense Growth Rate: Assume inflation
    suggested_expense_growth = 2.5
    
    return suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth

# Sidebar Layout
st.sidebar.title("Financial Insights")
st.sidebar.markdown("Your key financial metrics in INR.")
st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"R² Score: {r2_score_val:.2f}")

# Main App
st.title("AI Financial Dashboard (INR)")
st.markdown("Enter your financial details to get personalized predictions and insights in Indian Rupees.")

# User Input Form
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
    
    # Retirement Age Input
    st.subheader("Retirement Planning")
    default_retirement_age = min(62, age + 30)  # Default to 30 years from now or 62 if sooner
    retirement_age = st.slider("Retirement Age (up to 62)", int(age), 62, default_retirement_age)
    
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
        "Desired_Savings_Percentage": desired_savings_percentage,
        "Desired_Savings": income * (desired_savings_percentage / 100)
    }
    
    total_expenses = sum([rent, loan_repayment, insurance, groceries, transport, eating_out, 
                          entertainment, utilities, healthcare, education, miscellaneous])
    years_to_retirement = max(0, retirement_age - int(age))  # Calculate years from retirement age
    
    # Suggest Wealth Management parameters based on retirement age
    suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth = suggest_wealth_management_params(income, total_expenses, years_to_retirement)
    
    # Sidebar: Financial Health Score
    st.sidebar.subheader("Financial Health")
    health_score = calculate_financial_health_score(input_data)
    st.sidebar.metric("Score", f"{health_score:.1f}/100")
    if health_score < 40:
        st.sidebar.error("⚠️ Low: Take action!")
    elif health_score < 70:
        st.sidebar.warning("⚠️ Moderate: Room to improve!")
    else:
        st.sidebar.success("✅ Excellent!")
    
    # Sidebar: Disposable Income Prediction
    predicted_disposable = predict_disposable_income(model, input_data)
    st.sidebar.subheader("Predicted Disposable Income")
    st.sidebar.metric("Monthly (₹)", f"₹{predicted_disposable:,.2f}")
    
    # Wealth Management Section with Dynamic Adjustments
    st.subheader("Wealth Management")
    st.write(f"Plan for retirement at age {retirement_age} ({years_to_retirement} years from now):")
    
    # Initialize session state for Wealth Management parameters
    if 'desired_retirement_fund' not in st.session_state:
        st.session_state.desired_retirement_fund = suggested_fund
    if 'savings_rate_filter' not in st.session_state:
        st.session_state.savings_rate_filter = suggested_savings_rate
    if 'income_growth_rate' not in st.session_state:
        st.session_state.income_growth_rate = suggested_income_growth
    if 'expense_growth_rate' not in st.session_state:
        st.session_state.expense_growth_rate = suggested_expense_growth
    
    # Update Wealth Management parameters based on retirement age
    def update_wealth_params():
        years = max(0, st.session_state.retirement_age - int(age))
        suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth = suggest_wealth_management_params(income, total_expenses, years)
        st.session_state.desired_retirement_fund = suggested_fund
        st.session_state.savings_rate_filter = suggested_savings_rate
        st.session_state.income_growth_rate = suggested_income_growth
        st.session_state.expense_growth_rate = suggested_expense_growth
    
    # Wealth Management Filters with Callbacks
    st.session_state.retirement_age = st.slider("Retirement Age", int(age), 62, retirement_age, on_change=update_wealth_params)
    years_to_retirement = max(0, st.session_state.retirement_age - int(age))
    desired_retirement_fund = st.number_input("Desired Retirement Fund (₹)", min_value=100000.0, value=float(st.session_state.desired_retirement_fund), step=100000.0, key="fund")
    savings_rate_filter = st.slider("Adjust Savings Rate (%)", 0.0, 100.0, st.session_state.savings_rate_filter, step=1.0, key="savings")
    income_growth_rate = st.slider("Annual Income Growth Rate (%)", 0.0, 10.0, st.session_state.income_growth_rate, step=0.5, key="income_growth")
    expense_growth_rate = st.slider("Annual Expense Growth Rate (%)", 0.0, 10.0, st.session_state.expense_growth_rate, step=0.5, key="expense_growth")
    
    # Update session state when sliders are manually adjusted
    st.session_state.desired_retirement_fund = desired_retirement_fund
    st.session_state.savings_rate_filter = savings_rate_filter
    st.session_state.income_growth_rate = income_growth_rate
    st.session_state.expense_growth_rate = expense_growth_rate
    
    # Calculate savings with adjusted parameters
    future_savings = predict_future_savings(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
    st.sidebar.subheader("Wealth Management Results")
    st.sidebar.write(f"Projected Savings: **₹{future_savings:,.2f}**")
    required_savings_rate = (desired_retirement_fund / (income * years_to_retirement)) * 100 if income > 0 and years_to_retirement > 0 else 0
    st.sidebar.write(f"Required Savings Rate (at current income): **{required_savings_rate:.2f}%**")
    
    # Main Area: Detailed Insights
    st.header(f"Financial Insights for {name}")
    
    # Spending Breakdown
    st.subheader("Spending Breakdown")
    spending_data = pd.Series({
        "Rent": rent, "Insurance": insurance, "Groceries": groceries, "Transport": transport,
        "Eating Out": eating_out, "Entertainment": entertainment, "Utilities": utilities,
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
    years = np.arange(1, years_to_retirement + 1) if years_to_retirement > 0 else np.arange(1, 2)
    savings_trajectory = get_savings_trajectory(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, savings_trajectory[1:], marker="o", color="green", label="Projected Savings")
    ax.axhline(y=desired_retirement_fund, color="red", linestyle="--", label="Goal")
    ax.set_title("Projected Savings Growth (INR)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Savings (₹)")
    ax.legend()
    st.pyplot(fig)
    
    # Actionable Recommendations
    st.subheader("Personalized Recommendations")
    if eating_out > income * 0.1:
        st.write(f"- 🍽️ *Reduce Eating Out*: Spending exceeds 10% of income (₹{eating_out:,.2f}).")
    if loan_repayment > 0:
        st.write(f"- 💳 *Clear Debt*: Loan repayment (₹{loan_repayment:,.2f}) reduces your disposable income.")
    if desired_savings_percentage < 10:
        st.write(f"- 💰 *Boost Savings*: Increase your savings rate from {desired_savings_percentage:.1f}% to at least 10%.")
    if total_expenses > income * 0.8:
        st.write(f"- 📉 *Cut Expenses*: Spending (₹{total_expenses:,.2f}) is over 80% of income—review your budget!")

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit & AI/ML | All amounts in Indian Rupees (₹)")
