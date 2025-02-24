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
    # Assumes the dataset is saved as 'financial_data.csv' in the working directory
    data = pd.read_csv("financial_data.csv")
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
    """Prepare user input data for prediction without
