import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

def inspect_model(model_path: str) -> Dict[str, Any]:
    """
    Inspect a scikit-learn model and return its properties.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary with model properties
    """
    try:
        model = joblib.load(model_path)
        
        # Basic model info
        model_info = {
            "model_type": type(model).__name__,
        }
        
        # KMeans specific properties
        if hasattr(model, 'cluster_centers_'):
            model_info["n_clusters"] = model.n_clusters
            model_info["cluster_centers_shape"] = model.cluster_centers_.shape
            model_info["inertia"] = model.inertia_
            model_info["n_iter"] = model.n_iter_
        
        return model_info
    except Exception as e:
        logger.error(f"Error inspecting model: {str(e)}")
        return {"error": str(e)}

def inspect_scaler(scaler_path: str) -> Dict[str, Any]:
    """
    Inspect a scikit-learn scaler and return its properties.
    
    Args:
        scaler_path: Path to the saved scaler file
        
    Returns:
        Dictionary with scaler properties
    """
    try:
        scaler = joblib.load(scaler_path)
        
        # Basic scaler info
        scaler_info = {
            "scaler_type": type(scaler).__name__,
        }
        
        # StandardScaler specific properties
        if hasattr(scaler, 'mean_'):
            scaler_info["mean_shape"] = scaler.mean_.shape
            scaler_info["scale_shape"] = scaler.scale_.shape
            
            # If feature names were stored during fit
            if hasattr(scaler, 'feature_names_in_'):
                scaler_info["feature_names"] = scaler.feature_names_in_.tolist()
        
        return scaler_info
    except Exception as e:
        logger.error(f"Error inspecting scaler: {str(e)}")
        return {"error": str(e)}

def create_example_input(model_path: str, scaler_path: str) -> Dict[str, Any]:
    """
    Create an example input based on the model and scaler.
    
    Args:
        model_path: Path to the saved model file
        scaler_path: Path to the saved scaler file
        
    Returns:
        Dictionary with example input values
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        example = {
            "funding_rounds": 3,
            "funding_amount": 50.0,
            "valuation": 200.0,
            "revenue": 30.0,
            "employees": 150,
            "market_share": 2.5,
            "profitable": 1,
            "age": 5,
            "industry": "FinTech",
            "region": "North America",
            "exit_status": "Private"
        }
        
        return example
    except Exception as e:
        logger.error(f"Error creating example input: {str(e)}")
        return {"error": str(e)}
