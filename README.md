# 🛒 Sales Forecasting System with MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple?logo=railway)

An end-to-end MLOps project for predicting daily retail sales using historical transaction data.
The system covers the full pipeline — from raw data to a deployed REST API and Business Intelligence dashboard.

**Live API:** [sales-forecasting-api-production.up.railway.app/docs](https://sales-forecasting-api-production.up.railway.app/docs)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Tech Stack](#tech-stack)
- [Business Value](#business-value)

---

## Overview

This project builds a production-ready sales forecasting system for a single-product retail store.
The goal is to predict next-day sales volume to support inventory and pricing decisions.

Key highlights:
- **57% MAE reduction** from ARIMA baseline (101 → 43) using XGBoost + feature engineering
- **Automated hyperparameter tuning** with Optuna (50 trials)
- **REST API** with single and batch (CSV) prediction endpoints
- **Power BI dashboard** integrating historical data and live API predictions

---

## Demo

### API — Swagger UI
> Interactive documentation available at `/docs`

```
GET  /                  → API status
GET  /health            → model info & feature list
GET  /predict-today     → live prediction (Power BI compatible)
POST /predict           → single prediction via JSON
POST /predict-batch     → batch prediction via CSV upload
```

### Example Request
```bash
curl -X POST https://sales-forecasting-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price": 1.49,
    "stock": 500,
    "lag_1": 85,
    "lag_7": 90,
    "lag_14": 78,
    "lag_30": 82,
    "rolling_mean_7": 88.5,
    "rolling_std_7": 15.2,
    "rolling_mean_14": 86.0
  }'
```

### Example Response
```json
{
  "predicted_sales": 91,
  "target_date": "2024-07-15",
  "model_version": "xgboost-v1.0",
  "inputs_used": {
    "price": 1.49,
    "stock": 500,
    "lag_1": 85,
    "rolling_mean_7": 88.5
  }
}
```

---

## Dataset

Brazilian retail dataset with daily transaction records from **January 2014 – July 2016**.

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Transaction date |
| `sales` | int | Units sold per day |
| `stock` | int | Available stock |
| `price` | float | Product price |

**Key statistics:**
- 937 daily records (927 after cleaning)
- Mean sales: 90 units/day · Median: 76 · Std: 81 · Max: 542
- 10 rows removed where `price = 0` (anomaly / recording error)

---

## Pipeline

```
Raw CSV
   │
   ▼
Data Cleaning ──── Remove price=0 anomalies, parse datetime
   │
   ▼
EDA ──────────────  Sales trend, monthly seasonality, price vs sales
   │
   ▼
Feature Engineering ── Lag features, rolling stats, time features, interactions
   │
   ▼
Baseline Model ───── ARIMA(1,1,1) → MAE 101
   │
   ▼
XGBoost + Optuna ─── 50-trial hyperparameter search → MAE 43
   │
   ▼
Model Serialization ── joblib → .pkl
   │
   ▼
FastAPI ──────────── REST API with single + batch endpoints
   │
   ▼
Railway Deployment ── Public URL, auto-redeploy on push
   │
   ▼
Power BI Dashboard ── Historical trends + live forecast
```

---

## Model Performance

| Model | MAE | Notes |
|-------|-----|-------|
| ARIMA(5,1,0) — manual | 101.87 | Flat prediction, unusable |
| ARIMA(1,1,1) — auto_arima | ~100 | Best ARIMA, still flat |
| XGBoost — default params | 53.3 | After fixing feature engineering |
| **XGBoost — Optuna tuned** | **43.8** | **Final model ✅** |

**Total improvement: ↓57% from baseline**

### Features Used (15 total)

| Category | Features |
|----------|----------|
| Time | `day`, `month`, `day_of_week`, `week_of_year` |
| Lag | `lag_1`, `lag_7`, `lag_14`, `lag_30` |
| Rolling | `rolling_mean_7`, `rolling_std_7`, `rolling_mean_14` |
| Price & Stock | `price`, `stock`, `price_stock`, `price_lag1` |

### Best Hyperparameters (Optuna)

```python
{
    'n_estimators':     628,
    'max_depth':        3,
    'learning_rate':    0.0113,
    'subsample':        0.654,
    'colsample_bytree': 0.729,
    'min_child_weight': 6,
    'reg_alpha':        0.0001,
    'reg_lambda':       1.539,
    'gamma':            3.831,
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status |
| `GET` | `/health` | Model info & feature list |
| `GET` | `/predict-today` | Live prediction with default inputs (Power BI compatible) |
| `POST` | `/predict` | Single prediction via JSON body |
| `POST` | `/predict-batch` | Batch prediction via CSV file upload |

### Batch CSV Format
Upload a `.csv` file with these columns:
```
date, price, stock, lag_1, lag_7, lag_14, lag_30,
rolling_mean_7, rolling_std_7, rolling_mean_14
```

---

## Project Structure

```
sales-forecasting-api/
│
├── main.py                       # FastAPI application
├── requirements.txt              # Python dependencies
├── xgboost_sales_model.pkl       # Trained XGBoost model
├── features_list.pkl             # Feature list (preserves column order)
│
├── notebooks/
│   └── retailprocessing.ipynb    # Full ML pipeline notebook
│
└── README.md
```

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/sales-forecasting-api.git
cd sales-forecasting-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
python main.py
```

### 4. Open docs
```
http://localhost:8000/docs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML Model | XGBoost |
| Hyperparameter Tuning | Optuna |
| Data Processing | Pandas, NumPy |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Deployment | Railway |
| BI Dashboard | Power BI |
| Model Serialization | Joblib |

---

## Business Value

This system enables data-driven decision making for retail operations:

- **📦 Inventory optimization** — anticipate demand to avoid overstock and stockout
- **📈 Demand forecasting** — identify high-demand periods (e.g., June spike) in advance
- **💰 Pricing analysis** — understand the relationship between price changes and sales volume
- **📊 Performance monitoring** — track forecast accuracy over time via Power BI dashboard

> *"Sales are predicted to increase next month — prepare 20% additional stock."*

---

## License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with love as an MLOps portfolio project</p>
