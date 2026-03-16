import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os

def train_models():
    """
    Train Random Forest and XGBoost models for EcoPackAI
    Run this during Week 3-4 of your project timeline
    """
    
    # Load your prepared dataset (Module 3)
    # Columns: material_safety, strength, shipping_category, weight, fragility, volume, 
    #          target_cost, target_co2
    df = pd.read_csv('packaging_dataset.csv')
    
    # Features
    X = df[['material_safety', 'strength', 'shipping_category_encoded', 
            'weight', 'fragility', 'volume', 'co2_impact_index', 
            'cost_efficiency_index', 'category_encoded']]
    
    # Targets
    y_cost = df['target_cost']
    y_co2 = df['target_co2']
    
    # Split data
    X_train, X_test, y_cost_train, y_cost_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
    _, _, y_co2_train, y_co2_test = train_test_split(X, y_co2, test_size=0.2, random_state=42)
    
    # Train Random Forest for Cost Prediction
    print("Training Random Forest Regressor for Cost Prediction...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_cost_train)
    
    # Evaluate
    cost_pred = rf_model.predict(X_test)
    print(f"Cost Model - RMSE: {np.sqrt(mean_squared_error(y_cost_test, cost_pred)):.4f}")
    print(f"Cost Model - R2 Score: {r2_score(y_cost_test, cost_pred):.4f}")
    
    # Train XGBoost for CO2 Prediction
    print("\nTraining XGBoost Regressor for CO2 Prediction...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_co2_train, 
                  eval_set=[(X_test, y_co2_test)],
                  early_stopping_rounds=20,
                  verbose=False)
    
    # Evaluate
    co2_pred = xgb_model.predict(X_test)
    print(f"CO2 Model - RMSE: {np.sqrt(mean_squared_error(y_co2_test, co2_pred)):.4f}")
    print(f"CO2 Model - R2 Score: {r2_score(y_co2_test, co2_pred):.4f}")
    
    # Save models
    os.makedirs('ml_models', exist_ok=True)
    joblib.dump(rf_model, 'ml_models/rf_cost_model.pkl')
    joblib.dump(xgb_model, 'ml_models/xgb_co2_model.pkl')
    
    print("\nModels saved to ml_models/ directory")
    print("Ready for Flask integration!")

if __name__ == '__main__':
    train_models()