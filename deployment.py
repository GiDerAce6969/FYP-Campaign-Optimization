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

def upload_dataset():
    """Allow users to upload their own campaign dataset and store in session state"""
    st.subheader("Upload Your Campaign Data")
    
    # Check if data is already in session state
    if 'user_data' in st.session_state and st.session_state['user_data'] is not None:
        st.success(f"Using your previously uploaded dataset with {len(st.session_state['user_data'])} campaigns.")
        st.button("Upload New Dataset", on_click=lambda: st.session_state.pop('user_data', None))
        return st.session_state['user_data']
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", 
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        try:
            # Display a loading message
            with st.spinner("Processing your dataset..."):
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Basic validation
                required_columns = [
                    "Campaign", "Historical Reach", "Ad Spend", 
                    "Engagement Rate", "Competitor Ad Spend", 
                    "Seasonality Factor", "Repeat Customer Rate"
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure your dataset has the following columns: " + 
                           ", ".join(required_columns))
                    
                    # Show sample data format
                    st.subheader("Sample Data Format")
                    st.dataframe(load_sample_data(3))
                    
                    return None
                
                # Calculate additional metrics if they don't exist
                if 'Campaign Risk' not in df.columns:
                    df['Campaign Risk'] = (
                        df['Engagement Rate'].std() / df['Engagement Rate'].mean() * 100
                    )
                
                if 'Efficiency Score' not in df.columns:
                    df['Efficiency Score'] = (df['Historical Reach'] / df['Ad Spend']) * df['Engagement Rate']
                
                if 'Potential Growth' not in df.columns:
                    df['Potential Growth'] = df['Repeat Customer Rate'] * df['Seasonality Factor']
                
                # Store in session state
                st.session_state['user_data'] = df
                
                # Display success message
                st.success(f"Dataset with {len(df)} campaigns successfully loaded!")
                
                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                return df
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    
    return None

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

def simulate_scenario(df, scenario_type, parameters):
    """
    Simulate different marketing scenarios based on the selected type and parameters
    
    Args:
        df: Original campaign dataframe
        scenario_type: Type of scenario to simulate
        parameters: Dictionary of scenario-specific parameters
    
    Returns:
        Modified dataframe based on scenario
    """
    # Create a copy of the dataframe to avoid modifying the original
    scenario_df = df.copy()
    
    if scenario_type == "Budget Variation":
        # Apply budget adjustment factor to ad spend
        budget_factor = parameters.get('budget_factor', 1.0)
        scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * budget_factor
        
        # Estimate new reach based on adjusted spend
        # Using a logarithmic relationship between spend and reach
        scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * (
            1 + np.log1p(budget_factor - 1) * parameters.get('elasticity', 0.7)
        )
        
        # Adjust engagement based on diminishing returns
        if budget_factor > 1:
            # Slight decrease in engagement with higher spend (saturation)
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * (
                1 - (budget_factor - 1) * 0.1
            )
        else:
            # Slight increase in engagement with lower spend (targeting)
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * (
                1 + (1 - budget_factor) * 0.05
            )
            
    elif scenario_type == "Target Audience Change":
        # Apply audience targeting adjustments
        audience_focus = parameters.get('audience_focus', 'Broad')
        
        if audience_focus == 'Narrow':
            # Higher engagement but lower reach
            scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * 0.7
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * 1.4
            scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * 0.9
        elif audience_focus == 'Balanced':
            # Moderate adjustments
            scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * 0.9
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * 1.2
            scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * 1.0
        else:  # Broad
            # Higher reach but lower engagement
            scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * 1.3
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * 0.8
            scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * 1.1
    
    elif scenario_type == "Seasonal Impact":
        # Apply seasonal factors
        season = parameters.get('season', 'Normal')
        
        if season == 'Peak':
            # High season - better performance but more expensive
            scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * 1.25
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * 1.15
            scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * 1.2
        elif season == 'Low':
            # Off season - cheaper but lower performance
            scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * 0.8
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * 0.9
            scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * 0.85
        else:  # Normal
            # Standard season
            scenario_df['Estimated Reach'] = scenario_df['Historical Reach'] * 1.0
            scenario_df['Adjusted Engagement'] = scenario_df['Engagement Rate'] * 1.0
            scenario_df['Adjusted Ad Spend'] = scenario_df['Ad Spend'] * 1.0
    
    # Calculate adjusted metrics for all scenarios
    scenario_df['ROI'] = (scenario_df['Estimated Reach'] * scenario_df['Adjusted Engagement']) / scenario_df['Adjusted Ad Spend']
    scenario_df['Efficiency Score'] = (scenario_df['Estimated Reach'] / scenario_df['Adjusted Ad Spend']) * scenario_df['Adjusted Engagement']
    
    return scenario_df

def display_scenario_comparison(original_df, scenario_df, scenario_type, parameters):
    """Display comparison between original data and scenario simulation"""
    st.subheader(f"Scenario Analysis: {scenario_type}")
    
    # Display scenario parameters
    st.write("Scenario Parameters:")
    for param, value in parameters.items():
        st.write(f"- **{param.replace('_', ' ').title()}**: {value}")
    
    # Key metrics comparison
    original_total_reach = original_df['Historical Reach'].sum()
    scenario_total_reach = scenario_df['Estimated Reach'].sum()
    
    original_total_spend = original_df['Ad Spend'].sum()
    scenario_total_spend = scenario_df['Adjusted Ad Spend'].sum()
    
    # Create metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Reach", 
            f"{scenario_total_reach:,.0f}",
            f"{(scenario_total_reach - original_total_reach) / original_total_reach:.1%}"
        )
    
    with col2:
        st.metric(
            "Total Ad Spend", 
            f"${scenario_total_spend:,.2f}",
            f"{(scenario_total_spend - original_total_spend) / original_total_spend:.1%}"
        )
    
    with col3:
        original_efficiency = original_total_reach / original_total_spend
        scenario_efficiency = scenario_total_reach / scenario_total_spend
        
        st.metric(
            "Reach Efficiency", 
            f"{scenario_efficiency:.2f} per $",
            f"{(scenario_efficiency - original_efficiency) / original_efficiency:.1%}"
        )
    
    # Visual comparison
    st.subheader("Campaign Performance Comparison")
    
    # Prepare comparison data
    comparison_data = pd.DataFrame({
        'Campaign': original_df['Campaign'],
        'Original Reach': original_df['Historical Reach'],
        'Scenario Reach': scenario_df['Estimated Reach'],
        'Original Spend': original_df['Ad Spend'],
        'Scenario Spend': scenario_df['Adjusted Ad Spend'],
    })
    
    # Create reach comparison chart
    reach_fig = px.bar(
        comparison_data,
        x='Campaign',
        y=['Original Reach', 'Scenario Reach'],
        barmode='group',
        title='Reach Comparison by Campaign',
        labels={'value': 'Reach', 'variable': 'Scenario'}
    )
    st.plotly_chart(reach_fig, use_container_width=True)
    
    # Create spend comparison chart
    spend_fig = px.bar(
        comparison_data,
        x='Campaign',
        y=['Original Spend', 'Scenario Spend'],
        barmode='group',
        title='Spend Comparison by Campaign',
        labels={'value': 'Ad Spend ($)', 'variable': 'Scenario'}
    )
    st.plotly_chart(spend_fig, use_container_width=True)
    
    # Display detailed comparison table
    st.subheader("Detailed Metrics Comparison")
    
    detailed_comparison = pd.DataFrame({
        'Campaign': original_df['Campaign'],
        'Original Reach': original_df['Historical Reach'],
        'Scenario Reach': scenario_df['Estimated Reach'],
        'Reach Δ%': (scenario_df['Estimated Reach'] - original_df['Historical Reach']) / original_df['Historical Reach'] * 100,
        'Original Spend': original_df['Ad Spend'],
        'Scenario Spend': scenario_df['Adjusted Ad Spend'],
        'Spend Δ%': (scenario_df['Adjusted Ad Spend'] - original_df['Ad Spend']) / original_df['Ad Spend'] * 100,
        'Original Engagement': original_df['Engagement Rate'],
        'Scenario Engagement': scenario_df['Adjusted Engagement'],
        'Engagement Δ%': (scenario_df['Adjusted Engagement'] - original_df['Engagement Rate']) / original_df['Engagement Rate'] * 100
    })
    
    # Format the table
    styled_comparison = detailed_comparison.style.format({
        'Original Reach': '{:,.0f}',
        'Scenario Reach': '{:,.0f}',
        'Reach Δ%': '{:.1f}%',
        'Original Spend': '${:,.2f}',
        'Scenario Spend': '${:,.2f}',
        'Spend Δ%': '{:.1f}%',
        'Original Engagement': '{:.2f}',
        'Scenario Engagement': '{:.2f}',
        'Engagement Δ%': '{:.1f}%'
    }).background_gradient(
        subset=['Reach Δ%', 'Engagement Δ%'], 
        cmap='RdYlGn'
    )
    
    st.dataframe(styled_comparison)
    
    # AI-generated insights
    st.subheader("Scenario Insights")
    
    insights = []
    
    # Generate insights based on scenario results
    if scenario_total_reach > original_total_reach and scenario_total_spend <= original_total_spend:
        insights.append("✅ This scenario achieves higher reach with the same or lower budget")
    elif scenario_total_reach > original_total_reach and scenario_total_spend > original_total_spend:
        if (scenario_total_reach / original_total_reach) > (scenario_total_spend / original_total_spend):
            insights.append("✅ This scenario is more efficient despite higher costs")
        else:
            insights.append("⚠️ This scenario increases reach but at a disproportionately higher cost")
    elif scenario_total_reach < original_total_reach and scenario_total_spend < original_total_spend:
        if (scenario_total_reach / original_total_reach) > (scenario_total_spend / original_total_spend):
            insights.append("✅ This scenario saves budget while maintaining relative efficiency")
        else:
            insights.append("⚠️ This scenario reduces costs but significantly impacts campaign performance")
    
    # Campaign-specific insights
    best_campaign = comparison_data.loc[
        comparison_data['Reach Δ%'].idxmax() if 'Reach Δ%' in comparison_data.columns 
        else comparison_data.index[0]
    ]
    
    insights.append(f"🔍 {best_campaign['Campaign']} shows the most improvement in this scenario")
    
    for insight in insights:
        st.markdown(insight)
    
    # Recommendation based on scenario
    st.subheader("Recommendation")
    
    if scenario_efficiency > original_efficiency * 1.1:
        st.success("This scenario shows significant improvements and is recommended for implementation")
    elif scenario_efficiency > original_efficiency:
        st.info("This scenario shows moderate improvements and could be considered with careful monitoring")
    else:
        st.warning("This scenario does not improve overall efficiency compared to the current approach")

def validate_campaign_query(query):
    """
    Validate if the query is related to campaign optimization
    
    Uses a pattern-based approach to determine campaign relevance
    """
    import re
    
    # Convert query to lowercase
    lower_query = query.lower()
    
    # Check if the query is too short or generic
    if len(lower_query.strip()) < 10:
        return False, "Please provide a more detailed question about campaign optimization."
    
    # Regex patterns for campaign-related queries
    campaign_patterns = [
        r'\b(campaign|marketing|ad|advertis(ing|e))\b',
        r'\b(optimize|improve|strategy|reach|engagement)\b',
        r'\b(budget|spend|performance|target(ing)?)\b',
        r'\b(customer|audience|conversion)\b'
    ]
    
    # Check if any campaign-related pattern matches
    if any(re.search(pattern, lower_query) for pattern in campaign_patterns):
        return True, ""
    
    # Specific blocking for non-campaign topics
    non_campaign_patterns = [
        r'\b(news|current\s*events|politics|celebrity|personal)\b',
        r'\b(trump|biden|president|world\s*leader)\b',
        r'\b(sports|entertainment|gossip)\b'
    ]
    
    # Block queries matching non-campaign patterns
    if any(re.search(pattern, lower_query) for pattern in non_campaign_patterns):
        return False, "I am designed to provide insights specifically for campaign optimization. Please ask a question related to marketing campaigns, strategy, or performance."
    
    # Generic fallback for queries that don't match campaign patterns
    return False, "Please ask a specific question about campaign optimization. For example: 'How can I improve my campaign engagement?' or 'What strategies can increase marketing reach?'"

def ai_insights_section(df):
    """
    Enhanced AI Insights section with dataset upload and context-aware insights
    """
    st.header("Campaign Strategies Recommendation Powered By Gemini")
    
    # Dataset context display
    st.subheader("Current Dataset Overview")
    
    # Display basic dataset statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Campaigns", len(df))
    
    with col2:
        st.metric("Avg Engagement Rate", f"{df['Engagement Rate'].mean():.2%}")
    
    with col3:
        st.metric("Total Ad Spend", f"${df['Ad Spend'].sum():,.0f}")
    
    # Dataset details with expandable section
    with st.expander("Dataset Details"):
        st.write("### Campaign Metrics Summary")
        
        # Modified summary statistics calculation
        summary_stats_dict = {
            'Historical Reach': ['mean', 'min', 'max'],
            'Ad Spend': ['mean', 'min', 'max'],
            'Engagement Rate': ['mean', 'median'],
            'Competitor Ad Spend': ['mean'],
            'Repeat Customer Rate': ['mean']
        }
        
        # Dynamically create summary based on available columns
        summary_data = {}
        for col, agg_funcs in summary_stats_dict.items():
            if col in df.columns:
                summary_data[col] = df[col].agg(agg_funcs)
        
        # Convert to DataFrame
        summary_stats = pd.DataFrame(summary_data).T
        
        # Rename columns to be consistent
        if len(summary_stats.columns) == 3:
            summary_stats.columns = ['Mean', 'Min', 'Max']
        elif len(summary_stats.columns) == 2:
            summary_stats.columns = ['Mean', 'Median']
        
        # Format numeric columns
        summary_stats = summary_stats.apply(lambda x: x.apply(lambda val: f'{val:.4f}'))
        
        st.dataframe(summary_stats)
    
    # AI Query Section
    st.subheader("Ask AI About Your Campaigns")
    
    # Preset context options
    context_options = [
        "General Strategy Recommendations",
        "Budget Optimization",
        "Audience Targeting",
        "Engagement Improvement",
        "Competitive Analysis"
    ]
    
    # Context selection columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            "Specific Campaign Strategy Question:",
            placeholder="What strategies can improve my campaign performance?"
        )
    
    with col2:
        context_focus = st.selectbox(
            "Query Context", 
            context_options,
            index=0
        )
    
    # Gemini AI Integration
    if st.button("Generate Strategic Insights"):
        # Validate the query first
        is_valid, error_message = validate_campaign_query(user_query)
        
        if not is_valid:
            st.warning(error_message)
        elif client:
            try:
                with st.spinner("Analyzing campaign data..."):
                    # Prepare comprehensive data context
                    data_context = f"""
                    Campaign Dataset Overview:
                    - Total Campaigns: {len(df)}
                    - Metrics: {', '.join(df.columns)}
                    
                    Key Statistics:
                    - Average Engagement Rate: {df['Engagement Rate'].mean():.2f}
                    - Total Ad Spend: ${df['Ad Spend'].sum():,.2f}
                    - Average Historical Reach: {df['Historical Reach'].mean():,.0f}
                    - Repeat Customer Rate: {df['Repeat Customer Rate'].mean():.2f}
                    
                    Top Performing Campaigns:
                    {df.nlargest(3, 'Engagement Rate')[['Campaign', 'Engagement Rate', 'Ad Spend']].to_string()}
                    
                    Campaign Performance Distribution:
                    - Engagement Rate Range: {df['Engagement Rate'].min():.2f} - {df['Engagement Rate'].max():.2f}
                    - Ad Spend Range: ${df['Ad Spend'].min():,.0f} - ${df['Ad Spend'].max():,.0f}
                    
                    Specific Context: {context_focus}
                    
                    Specific User Query: {user_query}
                    
                    Request:
                    1. Provide data-driven marketing strategy insights
                    2. Focus on actionable recommendations
                    3. Explain reasoning using the provided campaign metrics
                    4. Tailor advice to the specified context: {context_focus}
                    """
                    
                    # Generate AI response
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=data_context
                    )
                    
                    # Display AI insights
                    st.markdown("### 🧠 AI Strategic Insights")
                    st.write(response.text)
                    
                    # Optional: Highlight key takeaways
                    st.subheader("Key Recommendations")
                    key_points = response.text.split('\n')[:5]  # First 5 lines as key points
                    for point in key_points:
                        if point.strip():  # Skip empty lines
                            st.markdown(f"- {point}")
            
            except Exception as e:
                st.error(f"AI Insight Generation Error: {e}")
                st.write("Fallback Insights:")
                st.write("- Review campaigns with high engagement rates")
                st.write("- Consider reallocating budget from low-performing campaigns")
        else:
            st.error("AI integration currently unavailable")
            
def main():
    st.title("Campaign Optimization AI 🚀")
    
    # Initialize session state for data persistence if not already done
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "Use Sample Data"
    
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
    
    # Data source selection with session state
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Use Sample Data", "Upload Your Own Data"],
        index=0 if st.session_state['data_source'] == "Use Sample Data" else 1,
        key="data_source_radio"
    )
    
    # Update session state
    st.session_state['data_source'] = data_source
    
    # Load appropriate data based on session state
    if data_source == "Use Sample Data":
        # Clear any existing user data when switching to sample data
        if 'user_data' in st.session_state:
            st.session_state.pop('user_data', None)
        df = load_sample_data()
        st.sidebar.info("Using sample data for demonstration purposes.")
    else:
        # Try to load from session state first, then from upload
        if 'user_data' in st.session_state and st.session_state['user_data'] is not None:
            df = st.session_state['user_data']
            st.sidebar.success(f"Using your uploaded dataset with {len(df)} campaigns")
            
            # Add option to use a different dataset
            if st.sidebar.button("Upload Different Dataset", key="upload_different_dataset"):
                st.session_state.pop('user_data', None)
                st.experimental_rerun()
        else:
            # Call the upload function
            uploaded_df = upload_dataset()
            if uploaded_df is not None:
                df = uploaded_df
            else:
                # Fall back to sample data if upload fails or isn't completed
                df = load_sample_data()
                st.sidebar.warning("Using sample data. Please upload your dataset to see your own campaign metrics.")
    
    # Page-specific content
    if page == "Campaign Dashboard 📊":
        # Advanced dashboard visualization
        create_advanced_dashboard(df)
        
        # Export functionality
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to CSV", key="export_csv"):
                export_data(df, "campaign_data", 'csv')
        with col2:
            if st.button("Export to Excel", key="export_excel"):
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
        
        if st.sidebar.button("Run AI-Powered Optimization", key="run_optimization"):
            optimize_campaign_with_agentic_ai(df, budget, total_customers)
    
    elif page == "AI Insights 🤖":
        # Existing AI Insights section with unique key for button
        ai_insights_section(df)
        st.header("Campaign Strategies Recommendation Powered By Gemini")
        
        user_query = st.text_area(
            "Ask about your campaign strategy with AI:",
            "Example: What are the key factors affecting campaign reach and how can I improve my marketing efficiency?"
        )
        
        if st.button("Generate Strategic Insights", key="generate_ai_insights"):
            # Validate the query first
            is_valid, error_message = validate_campaign_query(user_query)
            
            if not is_valid:
                st.warning(error_message)
            elif client:
                try:
                    with st.spinner("Analyzing campaign data..."):
                        prompt = f"""
                        Campaign Data Overview:
                        Total Campaigns: {len(df)}
                        Metrics: {', '.join(df.columns)}
                        
                        Key Statistics:
                        - Average Engagement Rate: {df['Engagement Rate'].mean():.2f}
                        - Total Ad Spend: ${df['Ad Spend'].sum():,.2f}
                        - Average Historical Reach: {df['Historical Reach'].mean():,.0f}
                        
                        User Strategic Query: {user_query}
                        
                        Provide concise, data-driven marketing strategy insights.
                        Focus on actionable recommendations based on the campaign data.
                        Explain your reasoning using the provided metrics.
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
        Experiment with budget and targeting strategies to visualize potential outcomes.
        """)
        
        # Scenario selection
        scenario_type = st.selectbox(
            "Select Scenario Type",
            ["Budget Variation", "Target Audience Change", "Seasonal Impact"]
        )
        
        # Dynamic scenario parameters based on selection
        parameters = {}
        
        if scenario_type == "Budget Variation":
            col1, col2 = st.columns(2)
            
            with col1:
                budget_factor = st.slider(
                    "Budget Adjustment Factor", 
                    min_value=0.5, 
                    max_value=2.0, 
                    value=1.0, 
                    step=0.1,
                    help="Multiply current budget by this factor (e.g., 1.5 = 50% increase)"
                )
                parameters['budget_factor'] = budget_factor
                
            with col2:
                elasticity = st.slider(
                    "Spend-Reach Elasticity", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.7,
                    step=0.1,
                    help="How responsive reach is to spend changes (higher = more responsive)"
                )
                parameters['elasticity'] = elasticity
                
        elif scenario_type == "Target Audience Change":
            audience_focus = st.radio(
                "Audience Targeting Strategy",
                ["Broad", "Balanced", "Narrow"],
                horizontal=True,
                help="Broad = larger audience, lower engagement; Narrow = smaller audience, higher engagement"
            )
            parameters['audience_focus'] = audience_focus
            
        elif scenario_type == "Seasonal Impact":
            season = st.radio(
                "Seasonal Period",
                ["Peak", "Normal", "Low"],
                horizontal=True,
                help="How seasonal factors affect campaign performance"
            )
            parameters['season'] = season
        
        # Run comparison button
        if st.button("Run Scenario Comparison", key="run_scenario_comparison"):
            with st.spinner("Simulating scenario..."):
                # Add slight delay for visual effect
                time.sleep(0.5)
                
                # Run the scenario simulation
                scenario_df = simulate_scenario(df, scenario_type, parameters)
                
                # Display comparison
                display_scenario_comparison(df, scenario_df, scenario_type, parameters)
                
                # Export option
                st.download_button(
                    label="Export Scenario Results",
                    data=scenario_df.to_csv().encode('utf-8'),
                    file_name=f"campaign_scenario_{scenario_type.lower().replace(' ', '_')}.csv",
                    mime='text/csv',
                    key="export_scenario_results"
                )

if __name__ == "__main__":
    main()
