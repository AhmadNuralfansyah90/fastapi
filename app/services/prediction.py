import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile, Depends
from typing import List, Tuple, Dict, Any
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

from app.schemas.startup import StartupInput, StartupPrediction
from app.core.config import settings
from app.utils.preprocessing import preprocess_startup_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the prediction service with the trained model and scaler.
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            if hasattr(self.model, 'cluster_centers_'):
                logger.info(f"Model has {len(self.model.cluster_centers_)} clusters")
                logger.info(f"Cluster centers shape: {self.model.cluster_centers_.shape}")
                
                if hasattr(self.model, 'feature_names_in_'):
                    logger.info(f"Model feature names: {self.model.feature_names_in_.tolist()}")
                    
                    self.feature_names = self.model.feature_names_in_.tolist()
                else:
                    self.feature_names = None
            
            self.cluster_descriptions = self._generate_cluster_descriptions()
        except Exception as e:
            logger.error(f"Failed to load model or scaler: {str(e)}")
            raise RuntimeError(f"Failed to load model or scaler: {str(e)}")
    
    def _generate_cluster_descriptions(self) -> Dict[int, str]:
        """
        Generate descriptions for each cluster based on the model.
        In a real application, these would be derived from cluster analysis.
        """
        cluster_count = len(self.model.cluster_centers_) if hasattr(self.model, 'cluster_centers_') else 4
        
        descriptions = {
            0: "High-growth startups with significant funding and high valuations",
            1: "Early-stage startups with moderate funding and growth potential",
            2: "Mature profitable startups with stable revenue and market share",
        }
        
        for i in range(3, cluster_count):
            descriptions[i] = f"Cluster {i} - Additional startup segment"
            
        return descriptions
    
    def predict_single(self, startup: StartupInput) -> Tuple[int, float]:
        """
        Predict the cluster for a single startup.
        Returns the cluster ID and a confidence score.
        """
        try:
            df_features = preprocess_startup_data(startup, self.scaler)
            logger.info(f"Preprocessed features shape: {df_features.shape}")
            
            if self.feature_names:
                missing_features = [f for f in self.feature_names if f not in df_features.columns]
                extra_features = [f for f in df_features.columns if f not in self.feature_names]
                
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                if extra_features:
                    logger.warning(f"Extra features: {extra_features}")
                    
                df_features = df_features[self.feature_names]
            
            cluster = self.model.predict(df_features)[0]
            logger.info(f"Predicted cluster: {cluster}")
            
            distances = self.model.transform(df_features)[0]
            min_distance = distances[cluster]
            max_distance = np.max(distances)
            confidence = 1 - (min_distance / max_distance) if max_distance > 0 else 1.0
            logger.info(f"Confidence score: {confidence}")
            
            return cluster, confidence
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise
    
    def predict_batch(self, startups: List[StartupInput]) -> List[StartupPrediction]:
        """
        Predict clusters for multiple startups.
        """
        results = []
        for i, startup in enumerate(startups):
            try:
                cluster, confidence = self.predict_single(startup)
                results.append(
                    StartupPrediction(
                        startup_index=i,
                        cluster=int(cluster),
                        confidence=float(confidence),
                        cluster_description=self.get_cluster_description(cluster)
                    )
                )
            except Exception as e:
                logger.error(f"Error predicting for startup {i}: {str(e)}")
                # Continue with next startup instead of failing the whole batch                continue
        
        return results
    
    def predict_from_csv(self, file: UploadFile) -> List[StartupPrediction]:
        """
        Predict clusters for startups from a CSV file.
        """
        contents = file.file.read()
        buffer = io.BytesIO(contents)
        df = pd.read_csv(buffer)
        logger.info(f"CSV file loaded with {len(df)} rows")
        
        startups = []
        for _, row in df.iterrows():
            try:
                startup = StartupInput(
                    funding_rounds=row.get('Funding Rounds', 0),
                    funding_amount=row.get('Funding Amount (M USD)', 0.0),
                    valuation=row.get('Valuation (M USD)', 0.0),
                    revenue=row.get('Revenue (M USD)', 0.0),
                    employees=row.get('Employees', 0),
                    market_share=row.get('Market Share (%)', 0.0),
                    profitable=row.get('Profitable', 0),
                    age=row.get('Age', 0),
                    industry=row.get('Industry', ''),
                    region=row.get('Region', ''),
                    exit_status=row.get('Exit Status', 'Private')
                )
                startups.append(startup)
            except Exception as e:
                logger.error(f"Error parsing row: {str(e)}")
                continue
        
        logger.info(f"Converted {len(startups)} valid rows from CSV")
        
        return self.predict_batch(startups)
    
    def get_cluster_description(self, cluster_id: int) -> str:
        """
        Get the description for a specific cluster.
        """
        return self.cluster_descriptions.get(
            cluster_id, 
            "Unknown cluster type"
        )
    
    def get_all_cluster_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all clusters.
        """

        
        cluster_info = []
        for cluster_id, description in self.cluster_descriptions.items():
            info = {
                "cluster_id": cluster_id,
                "size": 100 + cluster_id * 25,  
                "description": description,
                "avg_funding": 50.0 + cluster_id * 30.0,  
                "avg_valuation": 200.0 + cluster_id * 150.0,  
                "avg_revenue": 30.0 + cluster_id * 20.0,  
                "profitability_rate": 0.3 + cluster_id * 0.15,  
                "common_industries": [
                    "FinTech", 
                    "HealthTech", 
                    "E-Commerce"
                ][:2 + cluster_id]  
            }
            cluster_info.append(info)
        
        return cluster_info

def get_prediction_service() -> PredictionService:
    """
    Dependency to get the prediction service.
    """
    return PredictionService(
        model_path=settings.MODEL_PATH,
        scaler_path=settings.SCALER_PATH
    )
