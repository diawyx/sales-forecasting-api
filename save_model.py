
import joblib
# Save model
joblib.dump(model, 'xgboost_sales_model.pkl')

# Save feature list
joblib.dump(features, 'features_list.pkl')

print("✅ Model saved → xgboost_sales_model.pkl")
print("✅ Features saved → features_list.pkl")
print(f"   Features ({len(features)}): {features}")
