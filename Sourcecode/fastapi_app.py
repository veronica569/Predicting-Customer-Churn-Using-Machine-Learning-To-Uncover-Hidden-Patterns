from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the entire pipeline
with open('churn_model_pipeline.pkl', 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

class PredictionRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predict(data: PredictionRequest):
    try:
        # Convert input data to DataFrame
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])
        
        # Use the pipeline for prediction (it handles all preprocessing internally)
        prediction_result = pipeline.predict(input_df)[0]
        probability = float(pipeline.predict_proba(input_df)[0, 1])
        
        prediction = "Churn" if prediction_result == 1 else "No Churn"
        
        return {
            "prediction": prediction,
            "probability": probability,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "prediction": None,
            "probability": None,
            "status": "error",
            "message": str(e)
        }