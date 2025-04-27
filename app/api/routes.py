from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Optional

from app.schemas.startup import (
    StartupInput, 
    StartupPredictionResponse, 
    StartupBatchInput,
    StartupBatchPredictionResponse,
    ClusterInfoResponse
)
from app.services.prediction import PredictionService, get_prediction_service

router = APIRouter()

@router.post("/predict", response_model=StartupPredictionResponse, tags=["Predictions"])
async def predict_cluster(
    startup: StartupInput,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict the cluster for a single startup based on its features.
    """
    try:
        cluster, confidence = prediction_service.predict_single(startup)
        return {
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_description": prediction_service.get_cluster_description(cluster)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict-batch", response_model=StartupBatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    startups: StartupBatchInput,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict clusters for multiple startups.
    """
    try:
        results = prediction_service.predict_batch(startups.startups)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.post("/predict-from-csv", response_model=StartupBatchPredictionResponse, tags=["Predictions"])
async def predict_from_csv(
    file: UploadFile = File(...),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict clusters for startups from a CSV file.
    """
    try:
        results = prediction_service.predict_from_csv(file)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction error: {str(e)}")

@router.get("/clusters", response_model=List[ClusterInfoResponse], tags=["Clusters"])
async def get_cluster_info(
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Get information about all clusters.
    """
    try:
        return prediction_service.get_all_cluster_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cluster info: {str(e)}")
