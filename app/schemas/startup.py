from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union

class StartupInput(BaseModel):
    funding_rounds: int = Field(..., description="Number of funding rounds")
    funding_amount: float = Field(..., description="Total funding amount in million USD")
    valuation: float = Field(..., description="Valuation in million USD")
    revenue: float = Field(..., description="Revenue in million USD")
    employees: int = Field(..., description="Number of employees")
    market_share: float = Field(..., description="Market share percentage")
    profitable: int = Field(..., description="Profitability status (0=Not profitable, 1=Profitable)")
    age: int = Field(..., description="Age of the startup in years")
    industry: str = Field(..., description="Industry of the startup (AI, Cybersecurity, E-Commerce, EdTech, FinTech, Gaming, HealthTech, IoT)")
    region: str = Field(..., description="Region where the startup operates (Asia, Australia, Europe, North America, South America)")
    exit_status: str = Field(..., description="Exit status (Private, IPO, Acquired)")
    
    @validator('profitable')
    def validate_profitable(cls, v):
        if v not in [0, 1]:
            raise ValueError('profitable must be 0 or 1')
        return v
    
    @validator('exit_status')
    def validate_exit_status(cls, v):
        valid_statuses = ['Private', 'IPO', 'Acquired']
        if v not in valid_statuses:
            raise ValueError(f'exit_status must be one of {valid_statuses}')
        return v
        
    @validator('industry')
    def validate_industry(cls, v):
        valid_industries = ['AI', 'Cybersecurity', 'E-Commerce', 'EdTech', 
                           'FinTech', 'Gaming', 'HealthTech', 'IoT', 'SaaS']
        
        if v.lower() == 'e-commerce' or v.lower() == 'ecommerce':
            return 'E-Commerce'
        if v.lower() == 'fintech':
            return 'FinTech'
        if v.lower() == 'healthtech':
            return 'HealthTech'
        if v.lower() == 'edtech':
            return 'EdTech'
        

        if v not in valid_industries and v.upper() not in valid_industries:
            raise ValueError(f'industry should be one of {valid_industries}')
        return v
    
    @validator('region')
    def validate_region(cls, v):
        valid_regions = ['Asia', 'Australia', 'Europe', 'North America', 'South America']
        if v not in valid_regions:
            raise ValueError(f'region should be one of {valid_regions}')
        return v

class StartupPredictionResponse(BaseModel):
    cluster: int = Field(..., description="Predicted cluster")
    confidence: float = Field(..., description="Confidence score of the prediction")
    cluster_description: str = Field(..., description="Description of the cluster")

class StartupBatchInput(BaseModel):
    startups: List[StartupInput] = Field(..., description="List of startups to predict")

class StartupPrediction(BaseModel):
    startup_index: int = Field(..., description="Index of the startup in the batch")
    cluster: int = Field(..., description="Predicted cluster")
    confidence: float = Field(..., description="Confidence score of the prediction")
    cluster_description: str = Field(..., description="Description of the cluster")

class StartupBatchPredictionResponse(BaseModel):
    predictions: List[StartupPrediction] = Field(..., description="Predictions for each startup")

class ClusterInfoResponse(BaseModel):
    cluster_id: int = Field(..., description="Cluster identifier")
    size: int = Field(..., description="Number of startups in the cluster")
    description: str = Field(..., description="Description of the cluster")
    avg_funding: float = Field(..., description="Average funding amount")
    avg_valuation: float = Field(..., description="Average valuation")
    avg_revenue: float = Field(..., description="Average revenue")
    profitability_rate: float = Field(..., description="Percentage of profitable startups")
    common_industries: List[str] = Field(..., description="Most common industries in the cluster")

class ExampleRequest(BaseModel):
    funding_rounds: int = Field(3, example=3)
    funding_amount: float = Field(50.0, example=50.0)
    valuation: float = Field(200.0, example=200.0)
    revenue: float = Field(30.0, example=30.0)
    employees: int = Field(150, example=150)
    market_share: float = Field(2.5, example=2.5)
    profitable: int = Field(1, example=1)
    age: int = Field(5, example=5)
    industry: str = Field("FinTech", example="FinTech")
    region: str = Field("North America", example="North America")
    exit_status: str = Field("Private", example="Private")
