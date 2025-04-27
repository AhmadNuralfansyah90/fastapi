import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
import logging

from app.schemas.startup import StartupInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the exact industries and regions from the training data
# These must match EXACTLY what was used during model training
# Based on the model's feature_names_in_ log:
INDUSTRIES = [
    'AI', 'Cybersecurity', 'E-Commerce', 'EdTech', 
    'FinTech', 'Gaming', 'HealthTech', 'IoT'
    # Note: 'SaaS' was not in the original training data
]

REGIONS = [
    'Asia', 'Australia', 'Europe', 'North America', 'South America'
]

STAGES = ['Early Stage', 'Growth Stage', 'Mature Stage']

# Mapping to standardize input values to match training data
INDUSTRY_MAPPING = {
    'e-commerce': 'E-Commerce',
    'e-Commerce': 'E-Commerce',
    'ecommerce': 'E-Commerce',
    'Ecommerce': 'E-Commerce',
    'E-commerce': 'E-Commerce',
    'fintech': 'FinTech',
    'Fintech': 'FinTech',
    'healthtech': 'HealthTech',
    'Healthtech': 'HealthTech',
    'edtech': 'EdTech',
    'Edtech': 'EdTech',
    'saas': 'AI',  # Map SaaS to AI as a fallback
    'Saas': 'AI',  # Map SaaS to AI as a fallback
    'SaaS': 'AI',  # Map SaaS to AI as a fallback
    'ai': 'AI',
    'Ai': 'AI',
    'iot': 'IoT',
    'Iot': 'IoT',
    'IOT': 'IoT',
    'gaming': 'Gaming',
    'cybersecurity': 'Cybersecurity',
    'Cybersecurity': 'Cybersecurity',
    'cyber security': 'Cybersecurity',
    'Cyber Security': 'Cybersecurity',
    'CyberSecurity': 'Cybersecurity',
}

def preprocess_startup_data(startup: StartupInput, scaler: StandardScaler) -> pd.DataFrame:
    """
    Preprocess a single startup input for prediction.
    
    This function:
    1. Creates a DataFrame with the same structure as training data
    2. Applies one-hot encoding to categorical features
    3. Scales numeric features
    4. Returns the preprocessed features as a DataFrame
    
    Args:
        startup: The startup input data
        scaler: The fitted scaler for numeric features
        
    Returns:
        A pandas DataFrame of preprocessed features ready for prediction
    """
    try:
        # Standardize industry name to match training data
        standardized_industry = INDUSTRY_MAPPING.get(startup.industry, startup.industry)
        
        # If the standardized industry is not in our list, map to a default
        if standardized_industry not in INDUSTRIES:
            logger.warning(f"Industry '{standardized_industry}' not found in training data. Mapping to 'AI' as default.")
            standardized_industry = 'AI'
            
        logger.info(f"Processing startup: {startup.industry} (standardized to {standardized_industry}) in {startup.region}")
        
        # Create a DataFrame with numeric features
        data = {
            'Funding Rounds': startup.funding_rounds,
            'Funding Amount (M USD)': startup.funding_amount,
            'Valuation (M USD)': startup.valuation,
            'Revenue (M USD)': startup.revenue,
            'Employees': startup.employees,
            'Market Share (%)': startup.market_share,
            'Profitable': startup.profitable,
            'Age': startup.age,
            'Funding Efficiency': startup.valuation / max(startup.funding_amount, 0.001),
            'Revenue per Employee': startup.revenue / max(startup.employees, 1)
        }
        
        # Map exit status to numeric
        exit_status_mapping = {
            'Private': 0,
            'IPO': 1,
            'Acquired': 2
        }
        data['Exit Status Numeric'] = exit_status_mapping.get(startup.exit_status, 0)
        
        # Create DataFrame with numeric features
        df_numeric = pd.DataFrame([data])
        logger.info(f"Numeric features created: {df_numeric.columns.tolist()}")
        
        # Scale numeric features
        numeric_columns = df_numeric.columns.tolist()
        df_numeric_scaled = pd.DataFrame(
            scaler.transform(df_numeric),
            columns=numeric_columns
        )
        
        # Create one-hot encoded features for categorical variables
        # Industry - EXACTLY matching the training data columns
        for industry in INDUSTRIES:
            df_numeric_scaled[f'Industry_{industry}'] = 1 if standardized_industry == industry else 0
        
        # Region - EXACTLY matching the training data columns
        # Standardize region name first
        standardized_region = startup.region
        if standardized_region not in REGIONS:
            logger.warning(f"Region '{standardized_region}' not found in training data. Mapping to 'North America' as default.")
            standardized_region = 'North America'
            
        for region in REGIONS:
            df_numeric_scaled[f'Region_{region}'] = 1 if standardized_region == region else 0
        
        # Stage (derived from age)
        stage = ""
        if startup.age <= 3:
            stage = "Early Stage"
        elif startup.age <= 7:
            stage = "Growth Stage"
        else:
            stage = "Mature Stage"
        
        for s in STAGES:
            df_numeric_scaled[f'Stage_{s}'] = 1 if stage == s else 0
        
        # Ensure columns are in the exact same order as the model expects
        expected_columns = [
            'Funding Rounds', 'Funding Amount (M USD)', 'Valuation (M USD)',
            'Revenue (M USD)', 'Employees', 'Market Share (%)', 'Profitable',
            'Age', 'Funding Efficiency', 'Revenue per Employee', 'Exit Status Numeric',
            'Industry_AI', 'Industry_Cybersecurity', 'Industry_E-Commerce', 
            'Industry_EdTech', 'Industry_FinTech', 'Industry_Gaming', 
            'Industry_HealthTech', 'Industry_IoT', 'Region_Asia', 
            'Region_Australia', 'Region_Europe', 'Region_North America', 
            'Region_South America', 'Stage_Early Stage', 'Stage_Growth Stage', 
            'Stage_Mature Stage'
        ]
        
        # Create a new DataFrame with exactly the expected columns
        final_df = pd.DataFrame(columns=expected_columns)
        
        # Copy values from our processed DataFrame to the final one
        for col in expected_columns:
            if col in df_numeric_scaled.columns:
                final_df[col] = df_numeric_scaled[col]
            else:
                # This should not happen if our lists are correct
                logger.warning(f"Column {col} not found in processed data. Setting to 0.")
                final_df[col] = 0
        
        logger.info(f"Final DataFrame shape: {final_df.shape}")
        logger.info(f"Final columns: {final_df.columns.tolist()}")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise
