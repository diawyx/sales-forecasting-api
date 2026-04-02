import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

# Load data & model 
df = pd.read_csv('brazilian-retail.csv')
df.rename(columns={'data':'date','venda':'sales','estoque':'stock','preco':'price'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df[df['price'] > 0].sort_values('date').reset_index(drop=True)

model    = joblib.load('xgboost_sales_model.pkl')
features = joblib.load('features_list.pkl')

# Feature engineering
df['day']          = df['date'].dt.day
df['month']        = df['date'].dt.month
df['day_of_week']  = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['lag_1']        = df['sales'].shift(1)
df['lag_7']        = df['sales'].shift(7)
df['lag_14']       = df['sales'].shift(14)
df['lag_30']       = df['sales'].shift(30)
df['rolling_mean_7']  = df['sales'].rolling(7).mean()
df['rolling_std_7']   = df['sales'].rolling(7).std()
df['rolling_mean_14'] = df['sales'].rolling(14).mean()
df['price_stock']  = df['price'] * df['stock']
df['price_lag1']   = df['price'].shift(1)
df = df.dropna().reset_index(drop=True)

# Generate predictions
X = df[features]
df['predicted_sales'] = model.predict(X).round().astype(int)
df['predicted_sales']  = df['predicted_sales'].clip(lower=0)

# Tambah kolom analitik untuk Power BI
df['error']      = df['sales'] - df['predicted_sales']
df['abs_error']  = df['error'].abs()
df['month_name'] = df['date'].dt.strftime('%b')
df['year']       = df['date'].dt.year
df['year_month'] = df['date'].dt.to_period('M').astype(str)

# Rolling 30-day actual vs predicted
df['rolling_actual_30']    = df['sales'].rolling(30).mean().round(1)
df['rolling_predicted_30'] = df['predicted_sales'].rolling(30).mean().round(1)

# Export
output_cols = [
    'date', 'year', 'month', 'month_name', 'year_month',
    'day_of_week', 'week_of_year',
    'sales', 'predicted_sales', 'error', 'abs_error',
    'price', 'stock', 'price_stock',
    'rolling_actual_30', 'rolling_predicted_30',
]

df[output_cols].to_csv('dashboard_data.csv', index=False)

print("✅ dashboard_data.csv exported!")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {output_cols}")
print(f"   MAE     : {df['abs_error'].mean():.2f}")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
