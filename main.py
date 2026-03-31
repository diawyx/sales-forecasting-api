# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

# Load model & features 
MODEL_PATH    = "xgboost_sales_model.pkl"
FEATURES_PATH = "features_list.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# App 
app = FastAPI(
    title="Sales Forecasting API",
    description="""
## 🛒 Sales Forecasting API

Predicts daily sales for a single retail product using an XGBoost model
trained on historical Brazilian retail data (2014–2016).

**Model performance:** MAE ~43.8 units/day (↓57% from ARIMA baseline)

**Built by:** [Your Name] — MLOps Portfolio Project
    """,
    version="1.0.0",
)

# Request / Response schemas 
class PredictRequest(BaseModel):
    price: float = Field(..., gt=0, example=1.49,
                         description="Current product price (must be > 0)")
    stock: int   = Field(..., ge=0, example=500,
                         description="Current stock level")
    lag_1:  float = Field(..., ge=0, example=85,
                          description="Sales yesterday (1 day ago)")
    lag_7:  float = Field(..., ge=0, example=90,
                          description="Sales 7 days ago")
    lag_14: float = Field(..., ge=0, example=78,
                          description="Sales 14 days ago")
    lag_30: float = Field(..., ge=0, example=82,
                          description="Sales 30 days ago")
    rolling_mean_7:  float = Field(..., ge=0, example=88.5,
                                   description="Average sales over last 7 days")
    rolling_std_7:   float = Field(..., ge=0, example=15.2,
                                   description="Std deviation of sales over last 7 days")
    rolling_mean_14: float = Field(..., ge=0, example=86.0,
                                   description="Average sales over last 14 days")
    target_date: date = Field(default_factory=lambda: date.today() + timedelta(days=1),
                              example="2024-07-15",
                              description="Date to forecast (default: tomorrow)")

class PredictResponse(BaseModel):
    predicted_sales: int
    target_date: str
    model_version: str
    inputs_used: dict

# Helper 
def build_input_row(req: PredictRequest) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature order."""
    d = req.target_date
    row = {
        "day":             d.day,
        "month":           d.month,
        "day_of_week":     d.weekday(),       # 0=Monday … 6=Sunday
        "week_of_year":    d.isocalendar()[1],
        "price":           req.price,
        "stock":           req.stock,
        "price_stock":     req.price * req.stock,
        "price_lag1":      req.price,         # assume price unchanged
        "lag_1":           req.lag_1,
        "lag_7":           req.lag_7,
        "lag_14":          req.lag_14,
        "lag_30":          req.lag_30,
        "rolling_mean_7":  req.rolling_mean_7,
        "rolling_std_7":   req.rolling_std_7,
        "rolling_mean_14": req.rolling_mean_14,
    }
    # Ensure column order matches training
    return pd.DataFrame([row])[features]

# Endpoints
@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Sales Forecasting API is running 🚀",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "model": "XGBoost",
        "features_count": len(features),
        "features": features,
    }

@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
def predict(req: PredictRequest):
    try:
        X = build_input_row(req)
        pred = model.predict(X)[0]
        predicted_sales = max(0, int(round(float(pred))))  # no negative sales
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictResponse(
        predicted_sales=predicted_sales,
        target_date=str(req.target_date),
        model_version="xgboost-v1.0",
        inputs_used={
            "price": req.price,
            "stock": req.stock,
            "lag_1": req.lag_1,
            "rolling_mean_7": req.rolling_mean_7,
        },
    )

# Run locally 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
