from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Startup Clustering API"
    PROJECT_DESCRIPTION: str = "API for analyzing startup data using K-Means clustering"
    PROJECT_VERSION: str = "1.0.0"
    
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "kmeans_best_model.joblib")
    SCALER_PATH: str = os.path.join(BASE_DIR, "models", "standard_scaler.joblib")
    
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
