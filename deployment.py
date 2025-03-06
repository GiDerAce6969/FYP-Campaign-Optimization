import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
from google import genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# Page configuration
st.set_page_config(
    page_title="Campaign Optimization AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 50px; 
        margin-bottom: 0.5rem;
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        margin-top: 0.5rem !important;
        color: #2c3e50;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API
@st.cache_resource
def initialize_api():
    try:
        api_key = st.secrets["gemini_api"]
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize API: {e}")
        return None

client = initialize_api()

# Database functions
def init_db():
    """Initialize SQLite database for storing campaign data"""
    conn = sqlite3.connect('campaign_data.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY,
        name TEXT,
        date TEXT,
        historical_reach INTEGER,
        ad_spend REAL,
        engagement_rate REAL,
        competitor_ad_spend REAL,
        seasonality_factor REAL,
        repeat_customer_rate REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS optimizations (
        id INTEGER PRIMARY KEY,
        date TEXT,
        campaign_id INTEGER,
        allocated_customers INTEGER,
        predicted_reach_rate REAL,
        estimated_reached_customers INTEGER,
        total_cost REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
    )
    ''')
    
    conn.commit()
    return conn

# Enhanced sample data generation
@st.cache_data(ttl=3600)
def load_sample_data(num_campaigns=5):
    """Load sample data for demonstration with added complexity"""
    np.random.seed(42)
    historical_reach = np.random.randint(25000, 50000, num_campaigns)
    ad_spend = np.random.randint(20000, 50000, num_campaigns)
    data = {
        "Campaign": [f"Campaign {i+1}" for i in range(num_campaigns)],
        "Historical Reach": historical_reach,
        "Ad Spend": ad_spend,
        "Engagement Rate": np.round(np.random.uniform(0.2, 0.8, num_campaigns), 2),
        "Competitor Ad Spend": np.random.randint(15000, 45000, num_campaigns),
        "Seasonality Factor": np.random.choice([0.9, 1.0, 1.1], num_campaigns),
        "Repeat Customer Rate": np.round(np.random.uniform(0.1, 0.6, num_campaigns), 2),
    }
    df = pd.DataFrame(data)
    
    # Add new risk calculation feature
    df['Campaign Risk'] = (
        df['Engagement Rate'].std() / df['Engagement Rate'].mean() * 100
    )
    
    # Calculate additional metrics
    df['Efficiency Score'] = (df['Historical Reach'] / df['Ad Spend']) * df['Engagement Rate']
    df['Potential Growth'] = df['Repeat Customer Rate'] * df['Seasonality Factor']
    
    return df

# Advanced Visualization Functions
def create_advanced_dashboard(df):
    """Create an advanced, interactive dashboard with multiple visualizations"""
    st.header("Comprehensive Campaign Performance Dashboard")
    
    # Tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Dimensional Analysis", 
        "Correlation Insights", 
        "Performance Radar", 
        "Detailed Campaign Metrics"
    ])
    
    with tab1:
        # Multi-dimensional scatter plot with interactive features
        st.subheader("Campaign Performance Landscape")
        
        scatter_fig = px.scatter(
            df, 
            x="Historical Reach", 
            y="Engagement Rate",
            size="Ad Spend",
            color="Campaign Risk",
            hover_name="Campaign",
            title="Campaign Performance Multidimensional View",
            labels={
                "Historical Reach": "Historical Reach",
                "Engagement Rate": "Engagement Rate",
                "Ad Spend": "Ad Spend Size",
                "Campaign Risk": "Campaign Risk"
            },
            size_max=60
        )
        scatter_fig.update_layout(height=600)
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with tab2:
        # Correlation Heatmap
        st.subheader("Campaign Metrics Correlation")
        
        corr_columns = [
            'Historical Reach', 
            'Ad Spend', 
            'Engagement Rate', 
            'Competitor Ad Spend', 
            'Seasonality Factor', 
            'Repeat Customer Rate',
            'Campaign Risk'
        ]
        
        corr_matrix = df[corr_columns].corr()
        
        plt.figure(figsize=(10, 8))
        correlation_fig = sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            linewidths=0.5, 
            fmt=".2f",
            square=True,
            center=0
        )
        plt.title("Correlation Between Campaign Metrics")
        st.pyplot(plt.gcf())
    
    with tab3:
        # Radar Chart for Campaign Comparison
        st.subheader("Campaign Performance Radar")
        
        scaler = StandardScaler()
        radar_columns = [
            'Historical Reach', 
            'Engagement Rate', 
            'Ad Spend', 
            'Repeat Customer Rate'
        ]
        
        radar_data = scaler.fit_transform(df[radar_columns])
        
        radar_fig = go.Figure()
        
        for i, campaign in enumerate(df['Campaign']):
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_data[i],
                theta=radar_columns,
                fill='toself',
                name=campaign
            ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )
            ),
            showlegend=True,
            title="Normalized Campaign Performance Comparison"
        )
        
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab4:
        # Detailed Campaign Metrics with Interactive Table
        st.subheader("Comprehensive Campaign Metrics")
        
        styled_df = df.style.highlight_max(
            subset=['Efficiency Score', 'Potential Growth'], 
            color='lightgreen'
        ).format({
            'Efficiency Score': "{:.4f}",
            'Potential Growth': "{:.4f}",
            'Campaign Risk': "{:.2f}%"
        })
        
        st.dataframe(styled_df)

# Enhanced Optimization Function
def optimize_campaign_with_agentic_ai(df, budget, total_customers):
    """Enhanced campaign optimization with Agentic AI insights"""
    st.header("Advanced Campaign Optimization")
    
    st.subheader("AI Optimization Strategy")
    
    # Calculate optimization metrics
    df['Optimization Potential'] = (
        df['Historical Reach'] / df['Ad Spend'] * 
        df['Engagement Rate'] * 
        df['Seasonality Factor']
    )
    
    # Sort campaigns by optimization potential
    optimized_campaigns = df.sort_values('Optimization Potential', ascending=False)
    
    # Simulate budget allocation
    total_optimization_potential = optimized_campaigns['Optimization Potential'].sum()
    optimized_campaigns['Allocated Budget'] = (
        optimized_campaigns['Optimization Potential'] / total_optimization_potential * budget
    )
    
    # Estimated reach calculation
    optimized_campaigns['Estimated Reach'] = (
        optimized_campaigns['Allocated Budget'] / 
        optimized_campaigns['Ad Spend'] * 
        optimized_campaigns['Historical Reach']
    )
    
    # Visualization of optimization results
    fig = px.bar(
        optimized_campaigns, 
        x='Campaign', 
        y='Allocated Budget', 
        color='Estimated Reach',
        title='AI-Optimized Budget Allocation',
        labels={'Allocated Budget': 'Budget Allocation', 'Estimated Reach': 'Potential Reach'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed optimization results
    st.subheader("Optimization Breakdown")
    optimization_results = optimized_campaigns[[
        'Campaign', 
        'Allocated Budget', 
        'Estimated Reach', 
        'Optimization Potential'
    ]].style.format({
        'Allocated Budget': "${:,.2f}",
        'Estimated Reach': "{:,.0f}",
        'Optimization Potential': "{:.4f}"
    })
    
    st.dataframe(optimization_results)
    
    # AI Insights
    st.subheader("AI Strategic Recommendations")
    recommendations = [
        f"🎯 Prioritize {optimized_campaigns.iloc[0]['Campaign']} with highest optimization potential",
        f"💡 Potential budget reallocation could increase overall reach by {optimized_campaigns['Estimated Reach'].sum() / df['Historical Reach'].sum():.2%}",
        "🔍 Consider adjusting strategies for lower-performing campaigns",
        f"📊 Top 2 campaigns ({', '.join(optimized_campaigns.head(2)['Campaign'])}) show most promise"
    ]
    
    for rec in recommendations:
        st.markdown(rec)

# Export and additional utility functions
def export_data(df, filename, export_type='csv'):
    """Export campaign data to CSV or Excel"""
    if export_type == 'csv':
        df.to_csv(f"{filename}.csv", index=False)
        st.success(f"Data exported to {filename}.csv")
    elif export_type == 'excel':
        df.to_excel(f"{filename}.xlsx", index=False)
        st.success(f"Data exported to {filename}.xlsx")

# Existing optimization function
def optimize_allocation(df, MAX_CUSTOMERS_PER_CAMPAIGN, EXPECTED_REACH_RATE, COST_PER_CUSTOMER, BUDGET_CONSTRAINTS, TOTAL_CUSTOMERS):
    # [Previous implementation remains unchanged]
    pass

# Main application logic
def main():
    st.title("Campaign Optimization AI 🚀")
    
    # Initialize database
    conn = init_db()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Analysis Mode",
        [
            "Campaign Dashboard 📊", 
            "Optimization Engine 🎯", 
            "AI Insights 🤖", 
            "Scenario Comparison 📈"
        ]
    )
    
    # Load sample data
    df = load_sample_data()
    
    if page == "Campaign Dashboard 📊":
        # Advanced dashboard visualization
        create_advanced_dashboard(df)
        
        # Export functionality
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to CSV"):
                export_data(df, "campaign_data", 'csv')
        with col2:
            if st.button("Export to Excel"):
                export_data(df, "campaign_data", 'excel')
    
    elif page == "Optimization Engine 🎯":
        # Budget and customers input
        budget = st.sidebar.slider(
            "Total Marketing Budget", 
            min_value=50000, 
            max_value=500000, 
            value=200000
        )
        total_customers = st.sidebar.slider(
            "Total Target Customers", 
            min_value=10000, 
            max_value=200000, 
            value=100000
        )
        
        if st.sidebar.button("Run AI-Powered Optimization"):
            optimize_campaign_with_agentic_ai(df, budget, total_customers)
    
    elif page == "AI Insights 🤖":
        st.header("Campaign Strategies Recommendation Powered By Gemini")
        
        user_query = st.text_area(
            "Ask about your campaign strategy with AI:",
            "Example: What are the key factors affecting campaign reach and how can I improve my marketing efficiency?"
        )
        
        if st.button("Generate Strategic Insights"):
            if client:
                try:
                    with st.spinner("Analyzing campaign data..."):
                        prompt = f"""
                        Campaign Data Analysis:
                        {df.to_string()}
                        
                        User Strategic Query: {user_query}
                        
                        Provide data-driven marketing strategy insights.
                        Focus on actionable recommendations.
                        """
                        
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt
                        )
                        
                        st.markdown("### 🧠 AI Strategic Insights")
                        st.write(response.text)
                
                except Exception as e:
                    st.error(f"AI Insight Generation Error: {e}")
                    st.write("Fallback Insights:")
                    st.write("- Review campaigns with high engagement rates")
                    st.write("- Consider reallocating budget from low-performing campaigns")
            else:
                st.error("AI integration currently unavailable")
    
    elif page == "Scenario Comparison 📈":
        st.header("Campaign Scenario Simulator")
        st.info("""
        🔬 Compare different marketing allocation scenarios.
        Experiment with budget and targeting strategies.
        """)
        
        # Placeholder for scenario comparison features
        scenario_type = st.selectbox(
            "Select Scenario Type",
            ["Budget Variation", "Target Audience Change", "Seasonal Impact"]
        )
        
        st.write(f"Scenario Comparison for: {scenario_type}")
        # Future implementation of detailed scenario analysis

if __name__ == "__main__":
    main()
