from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any

app = FastAPI(title="Quantilligence AI Data Training Platform")

class DataTrainingRequest(BaseModel):
    dataset: List[Dict[str, Any]]

class ModelTrainingRequest(BaseModel):
    features: List[List[float]]
    labels: List[int]

@app.post("/preprocess-data")
async def preprocess_data(request: DataTrainingRequest):
    try:
        df = pd.DataFrame(request.dataset)
        
        # Basic data cleaning
        df.dropna(inplace=True)
        
        # Bias detection
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            skewness = df[column].skew()
            if abs(skewness) > 1:
                print(f"Potential bias detected in {column}: Skewness = {skewness}")
        
        # Normalization
        df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()
        
        return {
            "status": "success", 
            "processed_rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train-model")
async def train_model(request: ModelTrainingRequest):
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        X_train, X_test, y_train, y_test = train_test_split(
            request.features, request.labels, test_size=0.2
        )
        
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        performance = classification_report(y_test, predictions, output_dict=True)
        
        return {
            "status": "success",
            "model_type": "RandomForest",
            "performance": performance,
            "feature_importance": model.feature_importances_.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
