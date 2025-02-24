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

# Embedded Dataset (Your provided data)
@st.cache_data
def load_and_normalize_data():
    """Load and normalize the embedded dataset."""
    data = pd.DataFrame({
        "Income": [44637.25, 26858.60, 50367.61, 101455.60, 24875.28, 24875.61, 106119.69, 55431.34, 20606.73, 46304.79, 20706.82, 20668.56, 36407.22, 6492.02, 7547.85, 19132.10, 13342.21, 38574.65, 14509.13, 9692.52, 96903.57, 25042.55, 31665.25, 9596.50, 19408.11, 32783.83, 11946.07],
        "Age": [49, 34, 35, 21, 52, 57, 53, 40, 32, 30, 31, 31, 54, 22, 39, 45, 62, 21, 55, 46, 54, 52, 37, 52, 52, 37, 19],
        "Dependents": [0, 2, 1, 0, 4, 4, 0, 1, 4, 1, 0, 4, 3, 3, 4, 2, 4, 4, 4, 0, 4, 4, 0, 4, 3, 3, 4],
        "Rent": [13391.17, 5371.72, 7555.14, 15218.34, 4975.06, 4975.12, 21223.94, 11086.27, 6182.02, 9260.96, 3106.02, 4133.71, 10922.17, 973.80, 1509.57, 3826.42, 2668.44, 7714.93, 4352.74, 1938.50, 19380.71, 5008.51, 9499.58, 1919.30, 5822.43, 6556.77, 2389.21],
        "Loan_Repayment": [0.0, 0.0, 4612.10, 6809.44, 3112.61, 0.0, 0.0, 8307.70, 3562.49, 0.0, 0.0, 0.0, 4583.25, 0.0, 0.0, 2525.41, 2540.48, 0.0, 829.44, 0.0, 9481.17, 2898.04, 0.0, 0.0, 2832.16, 0.0, 0.0],
        "Insurance": [2206.49, 869.52, 2201.80, 4889.42, 635.91, 1038.23, 4360.20, 1755.44, 1018.59, 2130.25, 1031.78, 754.45, 970.79, 184.24, 330.19, 590.19, 352.89, 1838.34, 438.60, 217.59, 3585.07, 914.22, 1034.49, 320.45, 516.12, 946.62, 413.30],
        "Groceries": [6658.77, 2818.44, 6313.22, 14690.15, 3034.33, 3250.07, 12790.39, 8194.48, 3066.87, 5065.63, 3037.27, 2597.40, 4685.67, 955.50, 1032.73, 2515.72, 1920.10, 5126.82, 1506.25, 1306.54, 14257.03, 3170.69, 4019.47, 1092.98, 2486.69, 4815.61, 1363.79],
        "Transport": [2636.97, 1543.02, 3221.40, 7106.13, 1276.16, 1760.16, 6345.74, 3353.00, 1170.79, 2500.05, 1459.28, 1123.04, 2503.09, 494.82, 568.24, 1164.07, 961.38, 2614.48, 811.70, 562.10, 6271.72, 1947.50, 2180.72, 627.38, 1346.56, 1788.11, 641.89],
        "Eating_Out": [1651.80, 649.38, 1513.81, 5040.25, 692.83, 1049.07, 4390.91, 1741.91, 688.06, 2040.31, 451.41, 878.88, 1328.96, 309.24, 266.44, 661.75, 622.27, 1267.55, 515.58, 396.06, 2829.05, 1098.41, 1373.34, 429.72, 956.59, 1519.47, 475.93],
        "Entertainment": [1536.18, 1050.24, 1723.31, 2858.19, 660.19, 751.02, 2626.77, 1219.68, 418.23, 1016.61, 785.29, 643.15, 822.88, 229.83, 277.47, 660.75, 331.70, 1802.74, 414.23, 427.91, 3879.58, 839.59, 876.08, 379.45, 930.29, 1452.26, 497.31],
        "Utilities": [2911.79, 1626.14, 3368.46, 6128.55, 1092.69, 1024.31, 6202.55, 2361.98, 921.23, 2945.43, 1548.08, 1408.90, 2335.94, 471.87, 510.60, 1159.58, 985.63, 2081.39, 766.34, 723.64, 4066.69, 1321.02, 1686.60, 515.08, 1041.33, 1557.94, 699.28],
        "Healthcare": [1546.91, 1137.35, 2178.52, 4571.12, 1169.10, 1022.30, 3807.10, 2574.45, 830.24, 1649.76, 819.10, 1019.25, 1153.75, 222.65, 235.71, 919.43, 531.56, 1196.04, 656.65, 305.65, 3956.93, 1066.04, 1579.08, 417.07, 592.63, 1175.05, 404.44],
        "Education": [0.0, 1551.72, 3160.03, 0.0, 1445.22, 2003.85, 0.0, 5520.44, 1939.23, 3085.99, 0.0, 1506.36, 1940.13, 590.05, 521.98, 1442.91, 1043.20, 3475.98, 1449.96, 0.0, 6227.47, 1470.45, 0.0, 540.69, 1604.70, 2225.12, 782.14],
        "Miscellaneous": [831.53, 564.24, 628.37, 2526.06, 515.51, 402.22, 2776.77, 850.04, 446.07, 919.53, 410.03, 483.81, 570.12, 179.09, 133.49, 461.51, 286.26, 843.80, 285.02, 155.46, 2131.73, 407.47, 623.56, 258.72, 452.33, 973.88, 130.97],
        "Desired_Savings_Percentage": [13.89, 7.16, 14.00, 16.46, 7.53, 5.94, 17.34, 11.97, 6.30, 12.13, 8.35, 8.25, 7.53, 7.15, 5.46, 8.02, 8.14, 6.91, 7.24, 5.22, 13.55, 5.75, 6.84, 5.45, 6.12, 9.37, 5.61],
        "Disposable_Income": [11265.63, 9676.82, 13891.45, 31617.95, 6265.70, 7599.27, 41595.33, 8465.94, 362.92, 15690.26, 8058.57, 6119.61, 4590.47, 1880.93, 2161.46, 3204.35, 1098.29, 10612.58, 2482.61, 3659.07, 20836.43, 4900.62, 8792.34, 3095.66, 826.27, 9773.00, 4147.83]
    })
    
    # Numeric columns for normalization
    numeric_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage", 
                    "Disposable_Income"]
    
    # Normalize numeric columns
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
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
    # Create DataFrame with only the feature columns expected by the scaler and model
    input_df = pd.DataFrame({col: [input_data[col]] for col in feature_cols})
    input_normalized = scaler.transform(input_df)
    return pd.DataFrame(input_normalized, columns=feature_cols)

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
    normalized_input = normalize_input(input_data, scaler)
    feature_cols = ["Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance", 
                    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", 
                    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage"]
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
st.sidebar.markdown("Your key financial metrics in INR.")

# Main App
st.title("AI Financial Dashboard (INR)")
st.markdown("Enter your financial details to get personalized predictions and insights in Indian Rupees.")

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
        "Desired_Savings_Percentage": desired_savings_percentage,
        "Desired_Savings": income * (desired_savings_percentage / 100)  # Calculated separately
    }
    
    total_expenses = sum([rent, loan_repayment, insurance, groceries, transport, eating_out, 
                          entertainment, utilities, healthcare, education, miscellaneous])
    
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
        st.write(f"- 🍽️ *Reduce Eating Out*: Spending exceeds 10% of income (₹{eating_out:,.2f}).")
    if loan_repayment > 0:
        st.write(f"- 💳 *Clear Debt*: Loan repayment (₹{loan_repayment:,.2f}) reduces your disposable income.")
    if desired_savings_percentage < 10:
        st.write(f"- 💰 *Boost Savings*: Increase your savings rate from {desired_savings_percentage:.1f}% to at least 10%.")
    if total_expenses > income * 0.8:
        st.write(f"- 📉 *Cut Expenses*: Spending (₹{total_expenses:,.2f}) is over 80% of income—review your budget!")

# Model Performance in Sidebar
st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"R² Score: {r2_score_val:.2f}")
st.sidebar.write(f"RMSE (Normalized): {rmse_val:.2f}")

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit & AI/ML | All amounts in Indian Rupees (₹)")
