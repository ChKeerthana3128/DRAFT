import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 🏡 **UI Configuration**
st.set_page_config(page_title="Artha", layout="wide")
st.title("Artha - AI-Based Financial Dashboard")

# 📌 **Load Dataset with Error Handling**
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("financial_data.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Error: The dataset file 'financial_data.csv' is missing.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    
    # 🔹 **Data Preprocessing**
    def preprocess_data(df):
        df.fillna(0, inplace=True)
        numeric_cols = ['Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous',
                        'Desired_Savings', 'Disposable_Income']
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        df["Debt_to_Income_Ratio"] = df["Loan_Repayment"] / df["Income"]
        df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
        df["Disposable_Income_Percentage"] = df["Disposable_Income"] / df["Income"]
        return df

    df = preprocess_data(df)

    # 🔹 **Train Multiple Linear Regression Model**
    X = df[['Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
            'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']]
    y = df['Desired_Savings']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    df['Predicted_Savings'] = model.predict(X)

    # 🏠 **Sidebar - User Input**
    st.sidebar.header("📌 Enter Your Details")
    name = st.sidebar.text_input("👤 Name", "John Doe")
    age = st.sidebar.number_input("🎂 Age", min_value=18, max_value=100, value=30)
    income = st.sidebar.number_input("💵 Annual Salary", min_value=10000, max_value=1000000, value=50000, step=1000)
    
    st.sidebar.markdown("---")

    # 🔹 **Sidebar Insights & Recommendations**
    with st.sidebar.expander("📊 Wealth Management Insights"):
        st.write("- Plan your financial goals effectively.")
        st.write("- Allocate savings wisely based on your income.")

    with st.sidebar.expander("💡 Financial Health Insights"):
        st.write("- Monitor your debt-to-income ratio.")
        st.write("- Optimize discretionary spending for better savings.")

    # 🎯 **Financial Goals**
    st.sidebar.subheader("🎯 Financial Goals")
    retirement_age = st.sidebar.number_input("At what age do you plan to retire?", min_value=age, max_value=100, step=1)
    goal = st.sidebar.selectbox("What are you planning for?", ["Retirement", "Buying a Car", "Buying a House"])

    # 🔹 **Apply Filters Button**
    apply_filters = st.sidebar.button("✅ Apply Filters")

    if apply_filters:
        # 🔹 **Financial Health Score Calculation**
        def calculate_financial_health_score(row):
            debt_to_income = row['Loan_Repayment'] / row['Income'] if row['Income'] > 0 else 1
            savings_rate = row['Desired_Savings'] / row['Income'] if row['Income'] > 0 else 0
            discretionary_spending = (row['Eating_Out'] + row['Entertainment'] + row['Miscellaneous']) / row['Income'] if row['Income'] > 0 else 0
            score = 100 - (debt_to_income * 40 + discretionary_spending * 30 - savings_rate * 30)
            return max(0, min(100, score))

        df['Financial_Health_Score'] = df.apply(calculate_financial_health_score, axis=1)

        # 🔹 **Predictive Insights**
        years_until_retirement = max(0, retirement_age - age)
        suggested_savings = (income * 0.15) * years_until_retirement if goal == "Retirement" else income * 0.25

        st.subheader("📌 Financial Planning Insights")
        st.write(f"For your goal: **{goal}**, you should aim to save approximately **₹{suggested_savings:,.2f}** over the next {years_until_retirement} years.")

        st.subheader("📚 Financial Knowledge")
        st.markdown(
            "A strong financial plan includes budgeting, investing, and saving for long-term goals. "
            "Start by allocating a portion of your salary to different financial buckets: "
            "essential expenses, discretionary spending, and savings. "
            "The key to financial success is consistency in saving and making informed investment choices."
        )

        # 📊 **Data Visualization with Table**
        st.subheader("📊 Financial Data Analysis")

        # 🔹 **Financial Health Score Distribution**
        st.subheader("📈 Financial Health Score Distribution")
        fig = px.histogram(df, x='Financial_Health_Score', nbins=20, title='📈 Financial Health Score Distribution')
        st.plotly_chart(fig)
        st.dataframe(df[['Income', 'Financial_Health_Score', 'Savings_Rate', 'Debt_to_Income_Ratio']]
                     .sort_values(by='Financial_Health_Score', ascending=False))

        # 🔹 **Income vs Predicted Savings**
        st.subheader("💡 Income vs Predicted Savings")
        fig2 = px.scatter(df, x='Income', y='Predicted_Savings', color='Financial_Health_Score', title='💡 Income vs Predicted Savings')
        st.plotly_chart(fig2)
        st.dataframe(df[['Income', 'Predicted_Savings', 'Disposable_Income_Percentage']]
                     .sort_values(by='Predicted_Savings', ascending=False))

        st.markdown("---")
        st.caption("🚀 AI-Powered Financial Insights - Created by AKVSS")
# Load dataset
file_path = "financial_data.csv"
df = pd.read_csv(file_path)

# Function to predict financial health
def predict_savings(trend_data):
    X = np.arange(len(trend_data)).reshape(-1, 1)
    y = trend_data.values
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array([[len(trend_data) + i] for i in range(1, 7)])
    return model.predict(future_X)

# Streamlit UI

st.title("💰 Interactive Financial Health & Wealth Management Dashboard")

# User Inputs
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=18, max_value=100, value=25)
salary = st.number_input("Enter your monthly salary (in $):", min_value=500, max_value=100000, value=3000)

# Select columns to visualize
selected_col = st.selectbox("Select a financial metric to analyze:", df.columns[1:])
st.write(f"Showing trends for {selected_col}")

# Trend Analysis
fig = px.line(df, x=df.index, y=selected_col, title=f"Trend Analysis of {selected_col}")
st.plotly_chart(fig, use_container_width=True)

# Predictive Alerts
if selected_col in df.columns:
    predicted_values = predict_savings(df[selected_col])
    st.subheader("📊 Predictive Alerts")
    st.write(f"If your current trend continues, your {selected_col} will change as follows:")
    st.write(pd.DataFrame(predicted_values, columns=[f"{selected_col} Projection"], index=["+1M", "+2M", "+3M", "+4M", "+5M", "+6M"]))
    if predicted_values[-1] < df[selected_col].iloc[-1] * 0.8:
        st.warning("⚠ Warning: Your financial health is declining! Consider reducing expenses.")
    else:
        st.success("✅ Your financial health looks stable!")

# Budgeting Suggestions
st.subheader("📌 Personalized Financial Tips")
st.write("Based on your inputs, here are some suggestions:")
if salary < 2000:
    st.write("- Consider increasing your income sources or reducing non-essential expenses.")
elif salary < 5000:
    st.write("- Aim to save at least 20% of your salary monthly for long-term stability.")
else:
    st.write("- Diversify investments and plan for retirement early.")

# Summary
st.sidebar.title("User Details")
st.sidebar.write(f"**Name:** {name}")
st.sidebar.write(f"**Age:** {age}")
st.sidebar.write(f"**Salary:** ${salary}")

st.sidebar.info("🔔 Get insights to manage your wealth effectively!")


# Load dataset
file_path = "financial_data.csv"
df = pd.read_csv(file_path)

# Ensure relevant columns exist and have valid values
df['Desired_Savings_Percentage'] = pd.to_numeric(df.get('Desired_Savings_Percentage', pd.Series()), errors='coerce')
df['Desired_Savings'] = pd.to_numeric(df.get('Desired_Savings', pd.Series()), errors='coerce')
df.dropna(subset=['Desired_Savings_Percentage', 'Desired_Savings'], inplace=True)

# Function to calculate financial health score
def calculate_financial_health(debt, income, savings, desired_savings):
    debt_to_income = debt / income if income > 0 else 0
    savings_gap = (desired_savings - savings) / desired_savings if desired_savings > 0 else 0
    score = (1 - debt_to_income) * 50 + (1 - savings_gap) * 50
    return max(0, min(100, score))

# Function to predict financial health
def predict_financial_health(trend_data):
    if len(trend_data) < 2:
        return np.linspace(trend_data.iloc[-1] if len(trend_data) > 0 else 1000, 
                           trend_data.iloc[-1] * 1.05 if len(trend_data) > 0 else 1050, 6)  # Safe fallback
    X = np.arange(len(trend_data)).reshape(-1, 1)
    y = trend_data.values
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array([[len(trend_data) + i] for i in range(1, 7)])
    return model.predict(future_X)

# Streamlit UI
st.title("💰 Interactive Financial Health & Wealth Management Dashboard")

# User Inputs with unique keys
name = st.text_input("Enter your name:", key="name_input")
age = st.number_input("Enter your age:", min_value=18, max_value=100, value=25, key="age_input")
income = st.number_input("Enter your monthly income (in $):", min_value=500, max_value=100000, value=3000, key="income_input")
debt = st.number_input("Enter your total monthly debt (in $):", min_value=0, max_value=100000, value=500, key="debt_input")
savings = st.number_input("Enter your total monthly savings (in $):", min_value=0, max_value=100000, value=500, key="savings_input")
desired_savings = st.number_input("Enter your desired monthly savings (in $):", min_value=0, max_value=100000, value=1000, key="desired_savings_input")

# Calculate Financial Health Score
score = calculate_financial_health(debt, income, savings, desired_savings)
st.subheader("📊 Financial Health Score")
st.metric(label="Your Financial Health Score", value=f"{score:.2f}/100")

# Predictive Alerts
if not df.empty and 'Desired_Savings' in df.columns and df['Desired_Savings'].notnull().sum() > 0:
    predicted_values = predict_financial_health(df['Desired_Savings'])
else:
    predicted_values = np.linspace(savings, savings * 1.05, 6)  # Safe default prediction

st.subheader("📈 Predictive Alerts")
st.write("If your current trend continues, your savings will change as follows:")
st.write(pd.DataFrame(predicted_values, columns=["Savings Projection"], index=["+1M", "+2M", "+3M", "+4M", "+5M", "+6M"]))
if predicted_values[-1] < savings * 0.8:
    st.warning("⚠ Warning: Your savings are projected to decline! Consider adjusting your spending.")
else:
    st.success("✅ Your savings trend looks stable!")

# AI-Generated Recommendations
st.subheader("💡 AI-Generated Recommendations")
if score < 50:
    st.write("- Reduce unnecessary expenses and prioritize paying off debts.")
    st.write("- Increase your savings rate by at least 10%.")
elif score < 80:
    st.write("- Maintain a balanced approach to savings and investments.")
    st.write("- Consider diversifying income sources.")
else:
    st.write("- You have a strong financial foundation! Look into long-term investments.")

# Summary
st.sidebar.title("User Details")
st.sidebar.write(f"**Name:** {name}")
st.sidebar.write(f"**Age:** {age}")
st.sidebar.write(f"**Income:** ${income}")
st.sidebar.write(f"**Debt:** ${debt}")
st.sidebar.write(f"**Savings:** ${savings}")
st.sidebar.write(f"**Desired Savings:** ${desired_savings}")
st.sidebar.info("🔔 Get AI-driven insights to optimize your financial health!")
