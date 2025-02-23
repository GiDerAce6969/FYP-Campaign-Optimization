import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog

# Streamlit App Title
st.title("Campaign Reach Optimization & SHAP Analysis")

# User Inputs for Campaign Parameters
st.sidebar.header("Campaign Settings")
NUM_CAMPAIGNS = st.sidebar.slider("Number of Campaigns", min_value=3, max_value=10, value=5)
TOTAL_CUSTOMERS = st.sidebar.number_input("Total Customers", min_value=50000, max_value=500000, value=250000, step=5000)
BUDGET_CONSTRAINTS = st.sidebar.number_input("Total Budget ($)", min_value=50000, max_value=500000, value=100000, step=5000)

# User Adjustable Campaign Data
st.sidebar.header("Campaign Parameters")
MAX_CUSTOMERS_PER_CAMPAIGN = []
EXPECTED_REACH_RATE = []
COST_PER_CUSTOMER = []

for i in range(NUM_CAMPAIGNS):
    with st.sidebar.expander(f"Campaign {i+1} Settings"):
        MAX_CUSTOMERS_PER_CAMPAIGN.append(st.number_input(
            f"Max Customers for Campaign {i+1}", 
            min_value=25000, max_value=50000, 
            value=np.random.randint(25000, 50000), 
            key=f"max_customers_{i}"
        ))
        
        EXPECTED_REACH_RATE.append(st.slider(
            f"Expected Reach Rate for Campaign {i+1}", 
            min_value=0.1, max_value=1.0, 
            value=np.round(np.random.uniform(0.6, 0.9), 2), 
            key=f"reach_rate_{i}"
        ))
        
        COST_PER_CUSTOMER.append(st.number_input(
            f"Cost Per Customer ($) for Campaign {i+1}", 
            min_value=1.0, max_value=5.0, 
            value=np.round(np.random.uniform(1.5, 3.0), 2), 
            key=f"cost_per_customer_{i}"
        ))


# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    np.random.seed(None)  # Ensure randomness each run
    historical_reach = np.random.randint(25000, 50000, NUM_CAMPAIGNS)
    ad_spend = np.random.randint(20000, 50000, NUM_CAMPAIGNS)
    data = {
        "Campaign": [f"Campaign {i+1}" for i in range(NUM_CAMPAIGNS)],
        "Historical Reach": historical_reach,
        "Ad Spend": ad_spend,
        "Engagement Rate": np.round(np.random.uniform(0.2, 0.8, NUM_CAMPAIGNS), 2),
        "Competitor Ad Spend": np.random.randint(15000, 45000, NUM_CAMPAIGNS),
        "Seasonality Factor": np.random.choice([0.9, 1.0, 1.1], NUM_CAMPAIGNS),
        "Repeat Customer Rate": np.round(np.random.uniform(0.1, 0.6, NUM_CAMPAIGNS), 2),
    }
    df = pd.DataFrame(data)

# Feature Engineering
X = df.drop(columns=["Campaign"])
y = np.random.rand(len(df)) * 100

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
model = RandomForestRegressor(n_estimators=100, random_state=None)
model.fit(X_train, y_train)
df["Predicted Reach Rate"] = model.predict(X) / 100

# SHAP Analysis
explainer = shap.Explainer(model, X)  # No check_additivity here
shap_values = explainer(X, check_additivity=False)  # Apply when calling the explainer


# Display SHAP Summary Plot
st.subheader("Feature Importance (SHAP Analysis)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)

# Ensure each campaign gets at least 25,000 customers first
base_allocation = np.minimum(MAX_CUSTOMERS_PER_CAMPAIGN, 25000)
remaining_customers = TOTAL_CUSTOMERS - base_allocation.sum()

# Calculate available slots per campaign
available_capacity = np.array(MAX_CUSTOMERS_PER_CAMPAIGN) - base_allocation

# Distribute the remaining customers proportionally based on available capacity
if remaining_customers > 0:
    proportions = available_capacity / available_capacity.sum()
    additional_allocation = np.floor(proportions * remaining_customers).astype(int)
    additional_allocation[np.argmax(available_capacity)] += remaining_customers - additional_allocation.sum()
else:
    additional_allocation = np.zeros(NUM_CAMPAIGNS, dtype=int)

# Final allocation
allocated_customers = base_allocation + additional_allocation

# Simulation Results
allocation_df = pd.DataFrame({
    'Campaign': df['Campaign'],
    'Allocated Customers': allocated_customers,
    'Predicted Reach Rate': df["Predicted Reach Rate"],
    'Estimated Reached Customers': (allocated_customers * df["Predicted Reach Rate"].values).astype(int),
    'Total Cost': (allocated_customers * np.array(COST_PER_CUSTOMER[:len(df)])).astype(int)
})

st.subheader("Campaign Allocation Results")
st.write(allocation_df)

# Performance Analysis
total_reached = allocation_df['Estimated Reached Customers'].sum()
efficiency = total_reached / TOTAL_CUSTOMERS
total_cost = allocation_df['Total Cost'].sum()

st.subheader("Performance Analysis")
st.write(f"Total Estimated Reached Customers: {total_reached}")
st.write(f"Overall Efficiency of Campaigns: {efficiency:.2%}")
st.write(f"Total Campaign Cost: ${total_cost}")

# Visualization
st.subheader("Expected vs Realized Reach Rate")
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(df))
plt.bar(index, EXPECTED_REACH_RATE[:len(df)], bar_width, label='Expected Reach Rate', alpha=0.6, color='blue')
plt.bar(index + bar_width, df['Predicted Reach Rate'], bar_width, label='Realized Reach Rate', alpha=0.6, color='red')
plt.xlabel('Campaigns')
plt.ylabel('Reach Rate')
plt.xticks(index + bar_width / 2, df['Campaign'])
plt.legend()
st.pyplot(fig)
