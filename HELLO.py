import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Financial & Investment Dashboard (INR)", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_finance_data(csv_path="financial_data.csv"):
    if not os.path.exists(csv_path):
        st.error(f"Finance CSV file not found at {csv_path}.")
        return None
    
    try:
        data = pd.read_csv(csv_path)
        combined_column_name = "Miscellaneous (Eating_Out,Entertainmentand Utilities)\n"
        
        if combined_column_name not in data.columns:
            st.error(f"Column '{combined_column_name}' not found. Available: {data.columns.tolist()}")
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
            st.error(f"Missing columns: {missing_cols}")
            return None

        return data
    except Exception as e:
        st.error(f"Error loading finance CSV: {str(e)}")
        return None

# --- Model Training ---
def train_finance_model(data):
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

# --- Predictive Functions ---
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

def calculate_financial_health_score(income, total_expenses, debt, discretionary):
    """Adjusted to better align with CSV Financial_Health_Score."""
    if income <= 0:
        return 0

    actual_savings = max(0, income - total_expenses)
    savings_ratio = actual_savings / income
    debt_ratio = debt / income
    discretionary_ratio = discretionary / income
    expense_ratio = total_expenses / income

    # Adjusted to approximate CSV (trial based on sample data)
    savings_score = min(30, (savings_ratio / 0.2) * 30)  # Max 30, softer curve
    debt_score = max(0, 40 - (debt_ratio * 80))  # Max 40, stronger debt penalty
    discretionary_score = max(0, 30 - (discretionary_ratio * 150))  # Max 30, stricter discretionary

    score = savings_score + debt_score + discretionary_score
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
def generate_financial_health_insights(health_score, debt, discretionary, income, total_expenses):
    insights = []
    insights.append(f"Your Financial Health Score is {health_score:.1f}/100.")
    if health_score < 40:
        insights.append("Low score: High expenses or debt may be limiting savings.")
    elif health_score < 70:
        insights.append("Moderate score: Room to improve—balance spending and savings.")
    else:
        insights.append("Excellent score: Maintain your financial discipline!")

    debt_ratio = debt / income if income > 0 else 0
    if debt_ratio > 0.36:
        insights.append(f"Debt (₹{debt:,.2f}) is {debt_ratio:.1%} of income—above 36%.")
    
    discretionary_ratio = discretionary / income if income > 0 else 0
    if discretionary_ratio > 0.15:
        insights.append(f"Discretionary (₹{discretionary:,.2f}) is {discretionary_ratio:.1%}—above 15%.")

    expense_ratio = total_expenses / income if income > 0 else 0
    if expense_ratio > 0.8:
        insights.append(f"Expenses (₹{total_expenses:,.2f}) are {expense_ratio:.1%}—over 80%.")
    
    savings_ratio = max(0, income - total_expenses) / income if income > 0 else 0
    if savings_ratio < 0.2:
        insights.append(f"Savings are {savings_ratio:.1%}—below 20% ideal.")
    
    return insights

# --- Main Application ---
def main():
    finance_data = load_finance_data()
    if finance_data is None:
        st.stop()

    finance_model, finance_r2 = get_trained_finance_model(finance_data)

    st.sidebar.title("Dashboard Insights")
    st.sidebar.subheader("Finance Model Accuracy")
    st.sidebar.write(f"R² Score: {finance_r2:.2f}")

    st.header("Personal Finance Dashboard (INR)")
    st.markdown("Analyze your personal finances and plan for retirement.")

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

        st.sidebar.subheader("Financial Health Score")
        health_score = calculate_financial_health_score(income, total_expenses, debt, discretionary)
        st.sidebar.metric("Score", f"{health_score:.1f}/100")
        st.sidebar.write(f"Status: {'Excellent' if health_score >= 70 else 'Moderate' if health_score >= 40 else 'Low'}")

        st.sidebar.subheader("Predicted Disposable Income")
        predicted_disposable = predict_disposable_income(finance_model, input_data)
        st.sidebar.metric("Monthly (₹)", f"₹{predicted_disposable:,.2f}")

        st.subheader("Wealth Management")
        suggested_fund, suggested_savings_rate, suggested_income_growth, suggested_expense_growth = suggest_wealth_management_params(income, total_expenses, years_to_retirement)
        desired_retirement_fund = st.number_input("Desired Retirement Fund (₹)", min_value=100000.0, value=suggested_fund, step=100000.0, key="retirement_fund")
        savings_rate_filter = st.slider("Savings Rate (%)", 0.0, 100.0, suggested_savings_rate, step=1.0, key="savings_rate")
        income_growth_rate = st.slider("Income Growth Rate (%)", 0.0, 10.0, suggested_income_growth, step=0.5, key="income_growth")
        expense_growth_rate = st.slider("Expense Growth Rate (%)", 0.0, 10.0, suggested_expense_growth, step=0.5, key="expense_growth")

        projected_savings = predict_future_savings(income, total_expenses, savings_rate_filter, years_to_retirement, income_growth_rate, expense_growth_rate)
        st.sidebar.subheader("Projected Savings at Retirement")
        st.sidebar.metric("Total (₹)", f"₹{projected_savings:,.2f}")

        required_savings_rate = (desired_retirement_fund / (income * years_to_retirement)) * 100 if income > 0 and years_to_retirement > 0 else 0
        st.sidebar.subheader("Required Savings Rate")
        st.sidebar.metric("To Meet Goal (%)", f"{required_savings_rate:.2f}%")

        with st.sidebar.expander("Financial Health Insights"):
            for insight in generate_financial_health_insights(health_score, debt, discretionary, income, total_expenses):
                st.write(f"- {insight}")

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

    st.markdown("---")
    st.write("Powered by Streamlit")

if __name__ == "__main__":
    main()
