#!/usr/bin/env python
# coding: utf-8

# # Section 2: Model Training and Evaluation
# 
# ## Bike Sharing Dataset - XGBoost Regression Model
# 
# This script trains and evaluates XGBoost regression model on the Bike Sharing Dataset.
# 
# **Model Evaluated:**
# - **XGBoost**: Gradient boosting with tree-based learners for regression
# 
# **Evaluation Components:**
# - Comprehensive regression performance metrics
# - Actual vs Predicted plots
# - Residual analysis
# - Feature importance analysis
# - Cross-validation evaluation
# - Computational efficiency analysis

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
import xgboost as xgb

# Import our custom evaluation framework
from enhanced_evaluation import ComprehensiveEvaluator

# Set random seeds for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Section 2: XGBoost Model Training and Evaluation")
print("Dataset: Bike Sharing Dataset")
print("Task: Regression - Predicting bike rental count")

# ## 2.1 Load Preprocessed Data

# Load preprocessed data from Section 1
import pickle

try:
    with open('./bike_sharing_preprocessed_data.pkl', 'rb') as f:
        preprocessing_data = pickle.load(f)

    # Extract variables
    X_train_scaled = preprocessing_data['X_train_scaled']
    X_val_scaled = preprocessing_data['X_val_scaled']
    X_test_scaled = preprocessing_data['X_test_scaled']
    y_train = preprocessing_data['y_train']
    y_val = preprocessing_data['y_val']
    y_test = preprocessing_data['y_test']
    feature_names = preprocessing_data['feature_names']
    scaler = preprocessing_data['scaler']
    data_summary = preprocessing_data['data_summary']

    print("✅ Preprocessed data loaded successfully!")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Task: Regression")
    print(f"Target range: [{y_train.min()}, {y_train.max()}]")

except FileNotFoundError:
    print("❌ Preprocessed data not found!")
    print("Please run Section 1 (Data Preprocessing) script first.")
    raise

# ## 2.2 Initialize Evaluator

# Initialize comprehensive evaluator
evaluator = ComprehensiveEvaluator()

# Store all models for later analysis
models = {}

print("🔧 Evaluator initialized")
print(f"Ready to train XGBoost model on {len(feature_names)} features")
print(f"Training samples: {len(X_train_scaled):,}")
print(f"Test samples: {len(X_test_scaled):,}")
print(f"Task: Regression (Bike Count Prediction)")

# Check target distribution
print(f"\nTarget distribution:")
print(f"Training:   Mean={y_train.mean():.2f}, Std={y_train.std():.2f}, Range=[{y_train.min()}, {y_train.max()}]")
print(f"Validation: Mean={y_val.mean():.2f}, Std={y_val.std():.2f}, Range=[{y_val.min()}, {y_val.max()}]")
print(f"Test:       Mean={y_test.mean():.2f}, Std={y_test.std():.2f}, Range=[{y_test.min()}, {y_test.max()}]")

# ## 2.3 XGBoost Training

print("🌳 Training XGBoost for Regression...")

# XGBoost with optimized parameters for regression
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',  # Regression objective
    eval_metric='rmse'  # Root Mean Square Error
)

# Evaluate with comprehensive metrics
xgb_results = evaluator.evaluate_model(
    xgb_model, "XGBoost", 
    X_train_scaled, X_test_scaled, y_train, y_test,
    X_val_scaled, y_val
)

models['XGBoost'] = xgb_model
print("✅ XGBoost training completed")

# Display XGBoost results
print(f"\n📊 XGBoost Results:")
print(f"   R² Score: {xgb_results['r2_score']:.4f}")
print(f"   RMSE: {xgb_results['rmse']:.4f}")
print(f"   MAE: {xgb_results['mae']:.4f}")
print(f"   MAPE: {xgb_results['mape']:.4f}%")
print(f"   Explained Variance: {xgb_results['explained_variance']:.4f}")
print(f"   Training Time: {xgb_results['train_time']:.2f}s")

# ## 2.4 Feature Importance Analysis

print("\n🔍 Feature Importance Analysis")
print("=" * 50)

# Get feature importance from XGBoost
feature_importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance - Top 15 Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('XGBoost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ## 2.5 Model Performance Analysis

print("\n📈 Model Performance Analysis")
print("=" * 50)

# Make predictions for detailed analysis
y_pred = xgb_model.predict(X_test_scaled)

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Detailed Performance Metrics:")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"  Explained Variance Score: {explained_variance_score(y_test, y_pred):.4f}")

# Performance interpretation
print(f"\n🎯 Performance Interpretation:")
if r2 > 0.8:
    print(f"   ✅ Excellent performance (R² = {r2:.4f})")
elif r2 > 0.6:
    print(f"   ✅ Good performance (R² = {r2:.4f})")
elif r2 > 0.4:
    print(f"   ⚠️ Moderate performance (R² = {r2:.4f})")
else:
    print(f"   ❌ Poor performance (R² = {r2:.4f})")

print(f"   Average prediction error: ±{mae:.0f} bikes")
print(f"   Percentage error: {mape:.1f}%")

# ## 2.6 Cross-Validation Analysis

print("\n🔄 Cross-Validation Analysis")
print("=" * 50)

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=kfold, scoring='r2')

print(f"5-Fold Cross-Validation Results:")
print(f"  R² Scores: {cv_scores}")
print(f"  Mean R² Score: {cv_scores.mean():.4f}")
print(f"  Standard Deviation: {cv_scores.std():.4f}")
print(f"  95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# ## 2.7 Prediction Analysis

print("\n🎯 Prediction Analysis")
print("=" * 50)

# Analyze prediction accuracy by ranges
def analyze_predictions_by_range(y_true, y_pred):
    """Analyze prediction accuracy across different target value ranges"""
    
    # Define ranges based on quartiles
    q1, q2, q3 = np.percentile(y_true, [25, 50, 75])
    
    ranges = [
        (y_true.min(), q1, "Low (0-25%)"),
        (q1, q2, "Medium-Low (25-50%)"),
        (q2, q3, "Medium-High (50-75%)"),
        (q3, y_true.max(), "High (75-100%)")
    ]
    
    print("Performance by Target Value Range:")
    print("-" * 60)
    
    for min_val, max_val, label in ranges:
        mask = (y_true >= min_val) & (y_true <= max_val)
        if mask.sum() > 0:
            range_y_true = y_true[mask]
            range_y_pred = y_pred[mask]
            
            range_r2 = r2_score(range_y_true, range_y_pred)
            range_mae = mean_absolute_error(range_y_true, range_y_pred)
            range_mape = mean_absolute_percentage_error(range_y_true, range_y_pred)
            
            print(f"{label:20} | Samples: {mask.sum():4d} | R²: {range_r2:.3f} | MAE: {range_mae:.1f} | MAPE: {range_mape:.1f}%")

analyze_predictions_by_range(y_test, y_pred)

# ## 2.8 Error Analysis

print("\n🔍 Error Analysis")
print("=" * 50)

# Calculate residuals
residuals = y_test - y_pred
abs_residuals = np.abs(residuals)

# Find worst predictions
worst_predictions_idx = np.argsort(abs_residuals)[-10:]
best_predictions_idx = np.argsort(abs_residuals)[:10]

print("Top 10 Worst Predictions:")
print("Actual | Predicted | Error | Abs Error")
print("-" * 40)
for idx in worst_predictions_idx[::-1]:
    actual = y_test[idx]
    predicted = y_pred[idx]
    error = residuals[idx]
    abs_error = abs_residuals[idx]
    print(f"{actual:6.0f} | {predicted:9.0f} | {error:5.0f} | {abs_error:9.0f}")

print(f"\nError Statistics:")
print(f"  Mean Absolute Error: {np.mean(abs_residuals):.2f}")
print(f"  Median Absolute Error: {np.median(abs_residuals):.2f}")
print(f"  90th Percentile Error: {np.percentile(abs_residuals, 90):.2f}")
print(f"  95th Percentile Error: {np.percentile(abs_residuals, 95):.2f}")
print(f"  Max Error: {np.max(abs_residuals):.2f}")

# ## 2.9 Business Insights

print("\n💼 Business Insights for Bike Sharing")
print("=" * 50)

# Analyze feature importance for business insights
top_5_features = feature_importance_df.head(5)
print("Top 5 Most Important Features for Bike Demand:")
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.3f}")

# Seasonal analysis if season/month features are important
if 'hr' in feature_names:
    hr_idx = feature_names.index('hr')
    print(f"\n🕐 Hour is {'very important' if feature_importance[hr_idx] > 0.1 else 'moderately important' if feature_importance[hr_idx] > 0.05 else 'less important'} for bike demand prediction")

if 'temp' in feature_names:
    temp_idx = feature_names.index('temp')
    print(f"🌡️ Temperature is {'very important' if feature_importance[temp_idx] > 0.1 else 'moderately important' if feature_importance[temp_idx] > 0.05 else 'less important'} for bike demand prediction")

if 'weathersit' in feature_names:
    weather_idx = feature_names.index('weathersit')
    print(f"🌤️ Weather situation is {'very important' if feature_importance[weather_idx] > 0.1 else 'moderately important' if feature_importance[weather_idx] > 0.05 else 'less important'} for bike demand prediction")

print(f"\n📊 Model Reliability:")
print(f"  The model explains {r2*100:.1f}% of the variance in bike demand")
print(f"  Average prediction error is ±{mae:.0f} bikes ({mape:.1f}%)")
print(f"  Model is most reliable for {'low' if cv_scores.std() < 0.05 else 'moderate' if cv_scores.std() < 0.1 else 'high'} variance predictions")

# ## 2.10 Save Results

print("\n💾 Saving Results")
print("=" * 50)

# Create output directory
os.makedirs('./Section2_Model_Training', exist_ok=True)

# Save model results
results_df = pd.DataFrame([xgb_results]).T
results_df.columns = ['XGBoost']
results_df.to_csv('./Section2_Model_Training/xgboost_evaluation_results.csv')
print("✅ Model results saved to './Section2_Model_Training/xgboost_evaluation_results.csv'")

# Save feature importance
feature_importance_df.to_csv('./Section2_Model_Training/xgboost_feature_importance.csv', index=False)
print("✅ Feature importance saved to './Section2_Model_Training/xgboost_feature_importance.csv'")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'residual': residuals,
    'abs_residual': abs_residuals
})
predictions_df.to_csv('./Section2_Model_Training/xgboost_predictions.csv', index=False)
print("✅ Predictions saved to './Section2_Model_Training/xgboost_predictions.csv'")

# Save model for later use
import joblib
joblib.dump(xgb_model, './Section2_Model_Training/xgboost_model.pkl')
print("✅ Model saved to './Section2_Model_Training/xgboost_model.pkl'")

# ## 2.11 Summary

print("\n" + "="*80)
print("XGBOOST TRAINING COMPLETION SUMMARY")
print("="*80)
print(f"✅ Model Successfully Trained: XGBoost Regressor")
print(f"\n📊 Performance Metrics:")
print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   MAPE: {mape:.4f}%")
print(f"   Training Time: {xgb_results['train_time']:.2f}s")
print(f"\n🔄 Cross-Validation:")
print(f"   Mean R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n🎯 Business Impact:")
print(f"   Average prediction error: ±{mae:.0f} bikes")
print(f"   Model explains {r2*100:.1f}% of demand variance")
print(f"   Most important feature: {feature_importance_df.iloc[0]['feature']}")
print(f"\n🎯 Dataset: Bike Sharing Dataset")
print(f"📊 Task: Regression (Bike Rental Count Prediction)")
print(f"🔢 Features: {len(feature_names)}")
print(f"📈 Training Samples: {len(X_train_scaled):,}")
print(f"🧪 Test Samples: {len(X_test_scaled):,}")
print(f"\n🚀 XGBoost model ready for deployment and comparison with other models!")

print("\n📁 Generated files:")
print("   - xgboost_evaluation_results.csv")
print("   - xgboost_feature_importance.csv")
print("   - xgboost_predictions.csv")
print("   - xgboost_model.pkl")
print("   - XGBoost_feature_importance.png")
print("   - XGBoost_regression_results.png")
print("   - XGBoost_residual_analysis.png")
