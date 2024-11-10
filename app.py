import streamlit as st
import pandas as pd
import joblib

# Load models
success_model = joblib.load('C:/Users/namik/OneDrive/Desktop/bindu/best_model_success.pkl')
turnover_model = joblib.load('C:/Users/namik/OneDrive/Desktop/bindu/best_model_turnover.pkl')

# Define a threshold for success probability adjustment
THRESHOLD = 0 # Adjust this value to change the level of conservativeness

# Function to get user input
def get_user_input():
    st.title("Startup Prediction Dashboard")
    
    # Categorical features
    industry = st.selectbox('Industry', ['Tech', 'Healthcare', 'Finance', 'Retail', 'Other'])
    location = st.selectbox('Location', ['USA', 'India', 'Europe', 'Asia', 'Other'])
    funding_type = st.selectbox('Funding Type', ['Seed', 'Series A', 'Series B', 'Series C', 'Other'])
    education_level = st.selectbox('Education Level', ['High School', 'Undergraduate', 'Postgraduate', 'PhD'])
    product_diff = st.selectbox('Product Differentiation', ['High', 'Medium', 'Low'])

    # Numerical features
    founding_year = st.number_input('Founding Year', min_value=1900, max_value=2024, value=2020)
    initial_funding = st.number_input('Initial Funding', min_value=0, value=100000)
    funding_rounds = st.number_input('Funding Rounds', min_value=0, value=1)
    revenue_growth_rate = st.number_input('Revenue Growth Rate (%)', min_value=0.0, value=10.0)
    profit_margin = st.number_input('Profit Margin (%)', min_value=0.0, value=5.0)
    annual_revenue = st.number_input('Annual Revenue', min_value=0, value=500000)
    burn_rate = st.number_input('Burn Rate', min_value=0, value=100000)
    valuation = st.number_input('Valuation', min_value=0, value=1000000)
    debt_equity_ratio = st.number_input('Debt Equity Ratio', min_value=0.0, value=0.5)
    market_size_growth = st.number_input('Market Size Growth Rate (%)', min_value=0.0, value=5.0)
    competitive_density = st.number_input('Competitive Density', min_value=0.0, value=0.5)
    market_adoption_rate = st.number_input('Market Adoption Rate (%)', min_value=0.0, value=20.0)
    partnerships = st.number_input('Partnerships', min_value=0, value=3)
    founder_experience = st.number_input('Founder Experience (years)', min_value=0, value=5)
    team_experience = st.number_input('Team Experience (years)', min_value=0, value=5)
    tech_stack = st.number_input('Technology Stack Score', min_value=0, value=7)

    # Create a dictionary for the input data
    input_data = {
        'Industry': industry,
        'Location': location,
        'Funding_Type': funding_type,
        'Education_Level': education_level,
        'Product_Differentiation': product_diff,
        'Founding_Year': founding_year,
        'Initial_Funding': initial_funding,
        'Funding_Rounds': funding_rounds,
        'Revenue_Growth_Rate': revenue_growth_rate,
        'Profit_Margin': profit_margin,
        'Annual_Revenue': annual_revenue,
        'Burn_Rate': burn_rate,
        'Valuation': valuation,
        'Debt_Equity_Ratio': debt_equity_ratio,
        'Market_Size_Growth': market_size_growth,
        'Competitive_Density': competitive_density,
        'Market_Adoption_Rate': market_adoption_rate,
        'Partnerships': partnerships,
        'Founder_Experience': founder_experience,
        'Team_Experience': team_experience,
        'Technology_Stack': tech_stack
    }

    return pd.DataFrame([input_data])

# User input
user_input = get_user_input()

# Predictions
if st.button('Predict Success Probability'):
    # Check if the model supports predict_proba
    if hasattr(success_model, "predict_proba"):
        success_prediction = success_model.predict_proba(user_input)[:, 1][0] * 100  # Probability prediction
    else:
        success_prediction = success_model.predict(user_input)[0]  # Use predict method if proba not available
    
    # Apply threshold adjustment
    success_prediction = min(100, max(0, success_prediction - THRESHOLD))  # Ensure it's within 0-100%

    st.subheader(f"Predicted Success Probability: {success_prediction:.2f}%")

    # Predict turnover for the next 5 years
    turnover_prediction = turnover_model.predict(user_input)

    st.subheader("Predicted Turnover for Next 5 Years")
    for i, year in enumerate([f'Year {i+1}' for i in range(5)]):
        st.write(f"{year}: {turnover_prediction[0, i]:.2f}")

