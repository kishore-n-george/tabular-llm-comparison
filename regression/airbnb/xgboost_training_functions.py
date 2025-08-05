#!/usr/bin/env python
# coding: utf-8

"""
XGBoost Training Functions for Airbnb Dataset

This module contains all the functions needed to train and evaluate XGBoost
regression model on the Airbnb Dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
import pickle
import joblib
import logging
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

def setup_logging(log_file='Section2_Model_Training.log'):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = './Section2_Model_Training'
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_preprocessed_data(logger):
    """Load preprocessed data from Section 1"""
    logger.info("🚀 Section 2: XGBoost Model Training and Evaluation")
    logger.info("Dataset: Airbnb Dataset")
    logger.info("Task: Regression - Predicting Airbnb listing prices")
    
    try:
        with open('./Section1_Data_PreProcessing/airbnb_preprocessed_data.pkl', 'rb') as f:
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

        logger.info("✅ Preprocessed data loaded successfully!")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Validation set: {X_val_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        logger.info(f"Features: {len(feature_names)}")
        logger.info(f"Task: Regression")
        logger.info(f"Target range: [{y_train.min()}, {y_train.max()}]")

        return {
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler,
            'data_summary': data_summary
        }

    except FileNotFoundError:
        logger.error("❌ Preprocessed data not found!")
        logger.error("Please run Section 1 (Data Preprocessing) script first.")
        raise

def initialize_evaluator(data, logger):
    """Initialize evaluator and display data information"""
    evaluator = ComprehensiveEvaluator()
    models = {}

    logger.info("🔧 Evaluator initialized")
    logger.info(f"Ready to train XGBoost model on {len(data['feature_names'])} features")
    logger.info(f"Training samples: {len(data['X_train_scaled']):,}")
    logger.info(f"Test samples: {len(data['X_test_scaled']):,}")
    logger.info(f"Task: Regression (Airbnb Price Prediction)")

    # Check target distribution
    logger.info(f"\nTarget distribution:")
    logger.info(f"Training:   Mean=${data['y_train'].mean():.2f}, Std=${data['y_train'].std():.2f}, Range=[${data['y_train'].min():.2f}, ${data['y_train'].max():.2f}]")
    logger.info(f"Validation: Mean=${data['y_val'].mean():.2f}, Std=${data['y_val'].std():.2f}, Range=[${data['y_val'].min():.2f}, ${data['y_val'].max():.2f}]")
    logger.info(f"Test:       Mean=${data['y_test'].mean():.2f}, Std=${data['y_test'].std():.2f}, Range=[${data['y_test'].min():.2f}, ${data['y_test'].max():.2f}]")

    return evaluator, models

def train_xgboost_model(data, evaluator, logger):
    """Train XGBoost regression model with optimized parameters for Airbnb pricing"""
    logger.info("🌳 Training XGBoost for Regression...")
    
    # Analyze target distribution for better parameter selection
    y_train = data['y_train']
    logger.info(f"\n📊 Target Analysis for Parameter Optimization:")
    logger.info(f"   Price Range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    logger.info(f"   Price Std: ${y_train.std():.2f}")
    logger.info(f"   Price Skewness: {pd.Series(y_train).skew():.3f}")
    
    # XGBoost with optimized parameters for Airbnb price prediction
    # Increased complexity to capture more patterns in pricing data
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,           # More trees for complex patterns
        learning_rate=0.05,         # Lower learning rate for better generalization
        max_depth=8,                # Deeper trees for complex interactions
        min_child_weight=3,         # Prevent overfitting on small samples
        subsample=0.8,              # Row sampling
        colsample_bytree=0.8,       # Feature sampling
        colsample_bylevel=0.8,      # Feature sampling per level
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        gamma=0.1,                  # Minimum split loss
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse',
        n_jobs=-1                   # Use all cores
    )

    # Evaluate with comprehensive metrics
    xgb_results = evaluator.evaluate_model(
        xgb_model, "XGBoost", 
        data['X_train_scaled'], data['X_test_scaled'], data['y_train'], data['y_test'],
        data['X_val_scaled'], data['y_val']
    )

    logger.info("✅ XGBoost training completed")

    # Display XGBoost results with enhanced analysis
    logger.info(f"\n📊 XGBoost Results:")
    logger.info(f"   R² Score: {xgb_results['r2_score']:.4f}")
    logger.info(f"   RMSE: ${xgb_results['rmse']:.2f}")
    logger.info(f"   MAE: ${xgb_results['mae']:.2f}")
    logger.info(f"   MAPE: {xgb_results['mape']:.2f}%")
    logger.info(f"   Explained Variance: {xgb_results['explained_variance']:.4f}")
    logger.info(f"   Training Time: {xgb_results['train_time']:.2f}s")
    
    # Performance context for Airbnb pricing
    logger.info(f"\n🎯 Airbnb Pricing Context:")
    if xgb_results['r2_score'] < 0.3:
        logger.info(f"   ⚠️ Low R² is common for Airbnb pricing due to:")
        logger.info(f"      • High price variability across listings")
        logger.info(f"      • Host subjective pricing strategies")
        logger.info(f"      • Missing features (amenities, photos, reviews quality)")
        logger.info(f"      • Seasonal and event-based demand fluctuations")
    elif xgb_results['r2_score'] < 0.5:
        logger.info(f"   ✅ Moderate performance - typical for real estate pricing")
    else:
        logger.info(f"   🎉 Good performance for Airbnb price prediction!")
    
    # Practical interpretation
    avg_price = y_train.mean()
    logger.info(f"\n💡 Practical Interpretation:")
    logger.info(f"   • Average listing price: ${avg_price:.2f}")
    logger.info(f"   • Average prediction error: ${xgb_results['mae']:.2f} ({xgb_results['mae']/avg_price*100:.1f}% of avg price)")
    logger.info(f"   • Model captures {xgb_results['r2_score']*100:.1f}% of price variance")

    return xgb_model, xgb_results

def analyze_feature_importance(xgb_model, feature_names, logger, save_dir='./Section2_Model_Training'):
    """Analyze and plot feature importance"""
    logger.info("\n🔍 Feature Importance Analysis")
    logger.info("=" * 50)

    # Get feature importance from XGBoost
    feature_importance = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logger.info("Top 10 Most Important Features:")
    logger.info(feature_importance_df.head(10).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance - Top 15 Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'XGBoost_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return feature_importance_df

def analyze_model_performance(xgb_model, data, logger):
    """Analyze model performance in detail"""
    logger.info("\n📈 Model Performance Analysis")
    logger.info("=" * 50)

    # Make predictions for detailed analysis
    y_pred = xgb_model.predict(data['X_test_scaled'])

    # Calculate additional metrics
    mse = mean_squared_error(data['y_test'], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data['y_test'], y_pred)
    r2 = r2_score(data['y_test'], y_pred)
    mape = mean_absolute_percentage_error(data['y_test'], y_pred)

    logger.info(f"Detailed Performance Metrics:")
    logger.info(f"  Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    logger.info(f"  Explained Variance Score: {explained_variance_score(data['y_test'], y_pred):.4f}")

    # Performance interpretation
    logger.info(f"\n🎯 Performance Interpretation:")
    if r2 > 0.8:
        logger.info(f"   ✅ Excellent performance (R² = {r2:.4f})")
    elif r2 > 0.6:
        logger.info(f"   ✅ Good performance (R² = {r2:.4f})")
    elif r2 > 0.4:
        logger.info(f"   ⚠️ Moderate performance (R² = {r2:.4f})")
    else:
        logger.info(f"   ❌ Poor performance (R² = {r2:.4f})")

    logger.info(f"   Average prediction error: ±${mae:.2f}")
    logger.info(f"   Percentage error: {mape:.1f}%")

    return y_pred, {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape
    }

def perform_cross_validation(xgb_model, data, logger):
    """Perform cross-validation analysis"""
    logger.info("\n🔄 Cross-Validation Analysis")
    logger.info("=" * 50)

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, data['X_train_scaled'], data['y_train'], cv=kfold, scoring='r2')

    logger.info(f"5-Fold Cross-Validation Results:")
    logger.info(f"  R² Scores: {cv_scores}")
    logger.info(f"  Mean R² Score: {cv_scores.mean():.4f}")
    logger.info(f"  Standard Deviation: {cv_scores.std():.4f}")
    logger.info(f"  95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

    return cv_scores

def analyze_predictions_by_range(y_true, y_pred, logger):
    """Analyze prediction accuracy across different target value ranges"""
    logger.info("\n🎯 Prediction Analysis")
    logger.info("=" * 50)
    
    # Define ranges based on quartiles
    q1, q2, q3 = np.percentile(y_true, [25, 50, 75])
    
    ranges = [
        (y_true.min(), q1, "Low (0-25%)"),
        (q1, q2, "Medium-Low (25-50%)"),
        (q2, q3, "Medium-High (50-75%)"),
        (q3, y_true.max(), "High (75-100%)")
    ]
    
    logger.info("Performance by Target Value Range:")
    logger.info("-" * 60)
    
    for min_val, max_val, label in ranges:
        mask = (y_true >= min_val) & (y_true <= max_val)
        if mask.sum() > 0:
            range_y_true = y_true[mask]
            range_y_pred = y_pred[mask]
            
            range_r2 = r2_score(range_y_true, range_y_pred)
            range_mae = mean_absolute_error(range_y_true, range_y_pred)
            range_mape = mean_absolute_percentage_error(range_y_true, range_y_pred)
            
            logger.info(f"{label:20} | Samples: {mask.sum():4d} | R²: {range_r2:.3f} | MAE: {range_mae:.1f} | MAPE: {range_mape:.1f}%")

def perform_error_analysis(y_test, y_pred, logger):
    """Perform detailed error analysis"""
    logger.info("\n🔍 Error Analysis")
    logger.info("=" * 50)

    # Calculate residuals
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)

    # Find worst predictions
    worst_predictions_idx = np.argsort(abs_residuals)[-10:]

    logger.info("Top 10 Worst Predictions:")
    logger.info("Actual | Predicted | Error | Abs Error")
    logger.info("-" * 40)
    for idx in worst_predictions_idx[::-1]:
        actual = y_test[idx]
        predicted = y_pred[idx]
        error = residuals[idx]
        abs_error = abs_residuals[idx]
        logger.info(f"{actual:6.0f} | {predicted:9.0f} | {error:5.0f} | {abs_error:9.0f}")

    logger.info(f"\nError Statistics:")
    logger.info(f"  Mean Absolute Error: {np.mean(abs_residuals):.2f}")
    logger.info(f"  Median Absolute Error: {np.median(abs_residuals):.2f}")
    logger.info(f"  90th Percentile Error: {np.percentile(abs_residuals, 90):.2f}")
    logger.info(f"  95th Percentile Error: {np.percentile(abs_residuals, 95):.2f}")
    logger.info(f"  Max Error: {np.max(abs_residuals):.2f}")

    return residuals, abs_residuals

def analyze_data_challenges(data, logger):
    """Analyze potential challenges in the Airbnb dataset that might affect R² score"""
    logger.info("\n🔍 Data Challenge Analysis")
    logger.info("=" * 50)
    
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Price distribution analysis
    logger.info("📊 Price Distribution Challenges:")
    
    # Calculate coefficient of variation (std/mean)
    cv_train = y_train.std() / y_train.mean()
    cv_test = y_test.std() / y_test.mean()
    
    logger.info(f"   • Coefficient of Variation (Train): {cv_train:.3f}")
    logger.info(f"   • Coefficient of Variation (Test): {cv_test:.3f}")
    
    if cv_train > 1.0:
        logger.info(f"   ⚠️ Very high price variability (CV > 1.0) makes prediction challenging")
    elif cv_train > 0.5:
        logger.info(f"   ⚠️ High price variability (CV > 0.5) - typical for real estate")
    else:
        logger.info(f"   ✅ Moderate price variability")
    
    # Outlier analysis
    q1, q3 = np.percentile(y_train, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = ((y_train < lower_bound) | (y_train > upper_bound)).sum()
    outlier_pct = outliers / len(y_train) * 100
    
    logger.info(f"   • Outliers: {outliers} ({outlier_pct:.1f}% of data)")
    
    if outlier_pct > 10:
        logger.info(f"   ⚠️ High outlier percentage may affect model performance")
    
    # Price range analysis
    price_ranges = [
        (0, 50, "Budget"),
        (50, 100, "Mid-range"),
        (100, 200, "Premium"),
        (200, 500, "Luxury"),
        (500, float('inf'), "Ultra-luxury")
    ]
    
    logger.info(f"\n💰 Price Range Distribution:")
    for min_price, max_price, label in price_ranges:
        mask = (y_train >= min_price) & (y_train < max_price)
        count = mask.sum()
        pct = count / len(y_train) * 100
        if count > 0:
            logger.info(f"   • {label} (${min_price}-${max_price if max_price != float('inf') else '∞'}): {count} ({pct:.1f}%)")
    
    # Feature analysis
    logger.info(f"\n🔧 Feature Analysis:")
    logger.info(f"   • Number of features: {len(data['feature_names'])}")
    logger.info(f"   • Training samples: {len(data['X_train_scaled']):,}")
    logger.info(f"   • Features per sample ratio: {len(data['feature_names']) / len(data['X_train_scaled']):.4f}")
    
    if len(data['feature_names']) / len(data['X_train_scaled']) > 0.1:
        logger.info(f"   ⚠️ High feature-to-sample ratio may cause overfitting")

def generate_business_insights(feature_importance_df, feature_names, logger):
    """Generate business insights for Airbnb pricing"""
    logger.info("\n💼 Business Insights for Airbnb Pricing")
    logger.info("=" * 50)

    # Analyze feature importance for business insights
    top_5_features = feature_importance_df.head(5)
    logger.info("Top 5 Most Important Features for Airbnb Price Prediction:")
    for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
        logger.info(f"  {i}. {row['feature']}: {row['importance']:.3f}")

    # Get feature importance array
    feature_importance = feature_importance_df.set_index('feature')['importance']

    # Location analysis if location features are important
    location_features = ['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude']
    for location_feature in location_features:
        if location_feature in feature_names:
            location_importance = feature_importance.get(location_feature, 0)
            logger.info(f"\n🗺️ {location_feature.replace('_', ' ').title()} is {'very important' if location_importance > 0.1 else 'moderately important' if location_importance > 0.05 else 'less important'} for price prediction")

    # Property type analysis
    property_features = ['room_type', 'property_type']
    for property_feature in property_features:
        if property_feature in feature_names:
            property_importance = feature_importance.get(property_feature, 0)
            logger.info(f"🏠 {property_feature.replace('_', ' ').title()} is {'very important' if property_importance > 0.1 else 'moderately important' if property_importance > 0.05 else 'less important'} for price prediction")

    # Reviews analysis
    review_features = ['number_of_reviews', 'reviews_per_month', 'last_review']
    for review_feature in review_features:
        if review_feature in feature_names:
            review_importance = feature_importance.get(review_feature, 0)
            logger.info(f"⭐ {review_feature.replace('_', ' ').title()} is {'very important' if review_importance > 0.1 else 'moderately important' if review_importance > 0.05 else 'less important'} for price prediction")

    # Availability analysis
    if 'availability_365' in feature_names:
        avail_importance = feature_importance.get('availability_365', 0)
        logger.info(f"📅 Availability is {'very important' if avail_importance > 0.1 else 'moderately important' if avail_importance > 0.05 else 'less important'} for price prediction")
    
    # Recommendations for improvement
    logger.info(f"\n🚀 Recommendations to Improve R² Score:")
    logger.info(f"   1. Feature Engineering:")
    logger.info(f"      • Create interaction features (location × room_type)")
    logger.info(f"      • Add polynomial features for numerical variables")
    logger.info(f"      • Create price-per-area ratios if size data available")
    logger.info(f"   2. Data Enhancement:")
    logger.info(f"      • Include amenities data (WiFi, kitchen, etc.)")
    logger.info(f"      • Add seasonal/temporal features")
    logger.info(f"      • Include neighborhood quality metrics")
    logger.info(f"   3. Model Improvements:")
    logger.info(f"      • Try ensemble methods (Random Forest + XGBoost)")
    logger.info(f"      • Hyperparameter tuning with GridSearch/Bayesian optimization")
    logger.info(f"      • Consider neural networks for complex patterns")

def save_results(xgb_model, xgb_results, feature_importance_df, y_test, y_pred, residuals, abs_residuals, logger, save_dir='./Section2_Model_Training'):
    """Save all results and models"""
    logger.info("\n💾 Saving Results")
    logger.info("=" * 50)

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Save model results
    results_df = pd.DataFrame([xgb_results]).T
    results_df.columns = ['XGBoost']
    results_df.to_csv(os.path.join(save_dir, 'xgboost_evaluation_results.csv'))
    logger.info("✅ Model results saved to './Section2_Model_Training/xgboost_evaluation_results.csv'")

    # Save feature importance
    feature_importance_df.to_csv(os.path.join(save_dir, 'xgboost_feature_importance.csv'), index=False)
    logger.info("✅ Feature importance saved to './Section2_Model_Training/xgboost_feature_importance.csv'")

    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'residual': residuals,
        'abs_residual': abs_residuals
    })
    predictions_df.to_csv(os.path.join(save_dir, 'xgboost_predictions.csv'), index=False)
    logger.info("✅ Predictions saved to './Section2_Model_Training/xgboost_predictions.csv'")

    # Save model for later use
    joblib.dump(xgb_model, os.path.join(save_dir, 'xgboost_model.pkl'))
    logger.info("✅ Model saved to './Section2_Model_Training/xgboost_model.pkl'")

def print_final_summary(performance_metrics, cv_scores, feature_importance_df, data, logger):
    """Print final comprehensive summary"""
    logger.info("\n" + "="*80)
    logger.info("XGBOOST TRAINING COMPLETION SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Model Successfully Trained: XGBoost Regressor")
    logger.info(f"\n📊 Performance Metrics:")
    logger.info(f"   R² Score: {performance_metrics['r2']:.4f}")
    logger.info(f"   RMSE: {performance_metrics['rmse']:.4f}")
    logger.info(f"   MAE: {performance_metrics['mae']:.4f}")
    logger.info(f"   MAPE: {performance_metrics['mape']:.4f}%")
    logger.info(f"\n🔄 Cross-Validation:")
    logger.info(f"   Mean R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"\n🎯 Business Impact:")
    logger.info(f"   Average prediction error: ±${performance_metrics['mae']:.2f}")
    logger.info(f"   Model explains {performance_metrics['r2']*100:.1f}% of price variance")
    logger.info(f"   Most important feature: {feature_importance_df.iloc[0]['feature']}")
    logger.info(f"\n🎯 Dataset: Airbnb NYC Dataset")
    logger.info(f"📊 Task: Regression (Airbnb Price Prediction)")
    logger.info(f"🔢 Features: {len(data['feature_names'])}")
    logger.info(f"📈 Training Samples: {len(data['X_train_scaled']):,}")
    logger.info(f"🧪 Test Samples: {len(data['X_test_scaled']):,}")
    logger.info(f"\n🚀 XGBoost model ready for deployment and comparison with other models!")

    logger.info("\n📁 Generated files:")
    logger.info("   - xgboost_evaluation_results.csv")
    logger.info("   - xgboost_feature_importance.csv")
    logger.info("   - xgboost_predictions.csv")
    logger.info("   - xgboost_model.pkl")
    logger.info("   - XGBoost_feature_importance.png")
    logger.info("   - XGBoost_regression_results.png")
    logger.info("   - XGBoost_residual_analysis.png")

def run_complete_xgboost_training():
    """Run the complete XGBoost training pipeline"""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load data
        data = load_preprocessed_data(logger)
        
        # Initialize evaluator
        evaluator, models = initialize_evaluator(data, logger)
        
        # Train XGBoost model
        xgb_model, xgb_results = train_xgboost_model(data, evaluator, logger)
        models['XGBoost'] = xgb_model
        
        # Analyze feature importance
        feature_importance_df = analyze_feature_importance(xgb_model, data['feature_names'], logger)
        
        # Analyze model performance
        y_pred, performance_metrics = analyze_model_performance(xgb_model, data, logger)
        
        # Perform cross-validation
        cv_scores = perform_cross_validation(xgb_model, data, logger)
        
        # Analyze data challenges
        analyze_data_challenges(data, logger)
        
        # Analyze predictions by range
        analyze_predictions_by_range(data['y_test'], y_pred, logger)
        
        # Perform error analysis
        residuals, abs_residuals = perform_error_analysis(data['y_test'], y_pred, logger)
        
        # Generate business insights
        generate_business_insights(feature_importance_df, data['feature_names'], logger)
        
        # Save results
        save_results(xgb_model, xgb_results, feature_importance_df, data['y_test'], y_pred, residuals, abs_residuals, logger)
        
        # Print final summary
        print_final_summary(performance_metrics, cv_scores, feature_importance_df, data, logger)
        
        return {
            'model': xgb_model,
            'results': xgb_results,
            'feature_importance': feature_importance_df,
            'predictions': y_pred,
            'data': data,
            'evaluator': evaluator,
            'models': models
        }
        
    except Exception as e:
        logger.error(f"❌ Error in XGBoost training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_xgboost_training()
