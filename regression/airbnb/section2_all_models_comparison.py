"""
Comprehensive Model Comparison for Airbnb Regression

This script loads and compares all trained models using enhanced data:
- XGBoost (Enhanced)
- FT-Transformer (Enhanced)
- SAINT (Enhanced)

It generates comprehensive comparisons, plots, and saves results for Section 3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import torch
import time
import warnings
from pathlib import Path
import logging
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup logging
def setup_logging(save_dir='./Section2_Model_Training'):
    """Setup logging configuration"""
    Path(save_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{save_dir}/model_comparison.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

print("🏠 COMPREHENSIVE MODEL COMPARISON FOR AIRBNB REGRESSION")
print("=" * 80)
logger.info("Starting comprehensive model comparison for Airbnb dataset")

# Load enhanced data
print("📊 Loading enhanced data...")
try:
    enhanced_data = joblib.load('./Section1_Data_PreProcessing/enhanced_data.pkl')

    X_train_scaled = enhanced_data['X_train']
    X_val_scaled = enhanced_data['X_val']
    X_test_scaled = enhanced_data['X_test']
    y_train = enhanced_data['y_train']
    y_val = enhanced_data['y_val']
    y_test = enhanced_data['y_test']
    feature_names = enhanced_data['feature_names']
    transform_info = enhanced_data['transform_info']
    
    print("✅ Enhanced data loaded successfully!")
    print(f"   Test samples: {len(X_test_scaled):,}")
    print(f"   Features: {len(feature_names)} (after feature engineering & selection)")
    print(f"   Target transformation: {transform_info['method']}")
    print(f"   Outliers removed: {enhanced_data['outlier_mask'].sum()}")
    logger.info(f"Enhanced data loaded: {len(X_test_scaled)} test samples, {len(feature_names)} features")
    
    # Create data_summary for compatibility
    data_summary = {
        'dataset': 'Airbnb Enhanced',
        'task': 'Regression',
        'n_samples': len(X_train_scaled),
        'n_features': len(feature_names),
        'target_name': 'price',
        'preprocessing': 'Enhanced (outlier removal, feature engineering, scaling)'
    }
    
except Exception as e:
    print(f"❌ Error loading enhanced data: {e}")
    print("Falling back to basic preprocessed data...")
    logger.error(f"Error loading enhanced data: {e}")
    
    try:
        with open('./Section1_Data_PreProcessing/airbnb_preprocessed_data.pkl', 'rb') as f:
            preprocessing_data = pickle.load(f)

        X_train_scaled = preprocessing_data['X_train_scaled']
        X_val_scaled = preprocessing_data['X_val_scaled']
        X_test_scaled = preprocessing_data['X_test_scaled']
        y_train = preprocessing_data['y_train']
        y_val = preprocessing_data['y_val']
        y_test = preprocessing_data['y_test']
        feature_names = preprocessing_data['feature_names']
        data_summary = preprocessing_data['data_summary']
        transform_info = {'method': 'none', 'params': {}}
        
        print("✅ Basic preprocessed data loaded successfully!")
        print(f"   Test samples: {len(X_test_scaled):,}")
        print(f"   Features: {len(feature_names)}")
        logger.info(f"Basic data loaded: {len(X_test_scaled)} test samples, {len(feature_names)} features")
        
    except Exception as e2:
        print(f"❌ Error loading basic data: {e2}")
        logger.error(f"Error loading basic data: {e2}")
        raise

# Initialize models dictionary and results
models = {}
model_results = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n🖥️ Using device: {device}")
logger.info(f"Using device: {device}")

# Load Enhanced XGBoost model
print("\n📦 Loading Enhanced XGBoost model...")
try:
    # Check if enhanced data was loaded successfully by checking feature count
    if len(feature_names) < 40:
        print(f"   ⚠️ Enhanced data not available (only {len(feature_names)} features), skipping enhanced XGBoost model")
        raise Exception("Enhanced data not available")
    
    # Try to load tuned enhanced model first
    try:
        xgb_model = joblib.load('./Section2_Model_Training/xgboost_tuned_enhanced.pkl')
        model_type = 'XGBoost Enhanced (Tuned)'
    except:
        # Fall back to baseline enhanced model
        xgb_model = joblib.load('./Section2_Model_Training/xgboost_baseline_enhanced.pkl')
        model_type = 'XGBoost Enhanced (Baseline)'
    
    # Check feature compatibility
    expected_features = getattr(xgb_model, 'n_features_in_', len(feature_names))
    actual_features = len(feature_names)
    
    print(f"   Model expects {expected_features} features, data has {actual_features} features")
    
    if expected_features != actual_features:
        print(f"   ⚠️ Feature mismatch detected! Model trained on {expected_features} features, but data has {actual_features}")
        if actual_features < expected_features:
            print("   This suggests the enhanced data wasn't loaded properly.")
            raise Exception(f"Feature mismatch: expected {expected_features}, got {actual_features}")
    
    models['XGBoost Enhanced'] = xgb_model
    
    # Load metrics from comparison CSV
    try:
        comparison_df_saved = pd.read_csv('./Section2_Model_Training/enhanced_xgboost_comparison.csv', index_col=0)
        # Get the tuned model metrics (last row)
        xgb_metrics = comparison_df_saved.iloc[-1].to_dict()
        training_time = xgb_metrics.get('training_time', 0)
    except:
        xgb_metrics = {}
        training_time = 0
    
    model_results['XGBoost Enhanced'] = {
        'metrics': xgb_metrics,
        'training_time': training_time,
        'model_size': model_type,
        'model_type': model_type
    }
    
    print(f"✅ {model_type} loaded successfully!")
    print(f"   Features match: {expected_features} expected, {actual_features} available")
    logger.info(f"{model_type} loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading Enhanced XGBoost model: {e}")
    logger.error(f"Error loading Enhanced XGBoost model: {e}")
    print("   Continuing without XGBoost Enhanced model...")

# Load FT-Transformer model
print("\n📦 Loading FT-Transformer model...")
try:
    import rtdl
    
    # Load model checkpoint with weights_only=False for compatibility
    checkpoint = torch.load('./Section2_Model_Training/improved_ft_transformer_model.pth', 
                          map_location=device, weights_only=False)
    
    # Check what feature count the model was trained on by examining the state dict
    model_state = checkpoint['model_state_dict']
    if 'feature_tokenizer.num_tokenizer.weight' in model_state:
        model_n_features = model_state['feature_tokenizer.num_tokenizer.weight'].shape[0]
        print(f"   Model was trained on {model_n_features} features")
        print(f"   Current data has {len(feature_names)} features")
        
        # Use appropriate data based on model's expected feature count
        if model_n_features == len(feature_names):
            # Model matches current data
            X_test_for_ft = X_test_scaled
            model_label = 'FT-Transformer Enhanced'
        elif model_n_features < len(feature_names) and 'data_summary' in locals():
            # Model was trained on basic data, need to load basic data
            print("   Model was trained on basic data, loading basic preprocessed data for FT-Transformer...")
            try:
                with open('./Section1_Data_PreProcessing/airbnb_preprocessed_data.pkl', 'rb') as f:
                    basic_data = pickle.load(f)
                X_test_for_ft = basic_data['X_test_scaled']
                model_label = 'FT-Transformer (Basic Data)'
                print(f"   Using basic data with {X_test_for_ft.shape[1]} features for FT-Transformer")
            except:
                print("   Could not load basic data, skipping FT-Transformer")
                raise Exception("Feature mismatch and cannot load basic data")
        else:
            raise Exception(f"Feature mismatch: model expects {model_n_features}, data has {len(feature_names)}")
    else:
        # Fallback: assume model matches current data
        model_n_features = len(feature_names)
        X_test_for_ft = X_test_scaled
        model_label = 'FT-Transformer'
    
    # Create model architecture with correct feature count
    feature_info = {
        'n_num_features': model_n_features,
        'n_cat_features': 0,
        'cat_cardinalities': []
    }
    
    ft_model = rtdl.FTTransformer.make_baseline(
        n_num_features=feature_info['n_num_features'],
        cat_cardinalities=feature_info['cat_cardinalities'],
        d_out=1,
        d_token=64,
        n_blocks=2,
        attention_dropout=0.2,
        ffn_d_hidden=128,
        ffn_dropout=0.2,
        residual_dropout=0.1,
    ).to(device)
    
    # Load state dict
    ft_model.load_state_dict(checkpoint['model_state_dict'])
    ft_model.eval()
    
    models[model_label] = ft_model
    model_results[model_label] = {
        'metrics': checkpoint['metrics'],
        'training_time': checkpoint['training_time'],
        'model_size': f"{checkpoint['total_params']:,} parameters",
        'target_scaler': checkpoint.get('target_scaler', None),
        'test_data': X_test_for_ft  # Store the appropriate test data
    }
    print(f"✅ {model_label} loaded successfully!")
    logger.info(f"{model_label} loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading FT-Transformer model: {e}")
    logger.error(f"Error loading FT-Transformer model: {e}")

# Load Improved SAINT model
print("\n📦 Loading Improved SAINT model...")
try:
    # Try to load improved SAINT model first
    try:
        # Import improved SAINT model class
        from improved_saint_training import ImprovedSAINTModel
        
        # Load improved model data
        with open('./Section2_Model_Training/improved_saint_model.pkl', 'rb') as f:
            saint_data = pickle.load(f)
        
        # Create improved model architecture
        saint_model = ImprovedSAINTModel(
            n_features=saint_data['model_architecture']['n_features'],
            d_model=saint_data['model_architecture']['d_model'],
            n_heads=saint_data['model_architecture']['n_heads'],
            n_layers=saint_data['model_architecture']['n_layers']
        ).to(device)
        
        model_label = 'SAINT Enhanced'
        print("   Using improved SAINT model with enhanced architecture")
        
    except:
        # Fall back to original SAINT model
        from regression.airbnb.old.saint_training_functions import SAINTModel
        
        # Try original SAINT model
        with open('./Section2_Model_Training/saint_model.pkl', 'rb') as f:
            saint_data = pickle.load(f)
        
        saint_model = SAINTModel(
            n_features=saint_data['model_architecture']['n_features'],
            d_model=saint_data['model_architecture']['d_model'],
            n_heads=saint_data['model_architecture']['n_heads'],
            n_layers=saint_data['model_architecture']['n_layers']
        ).to(device)
        
        model_label = 'SAINT (Original)'
        print("   Using original SAINT model (improved model not found)")
    
    # Load state dict
    saint_model.load_state_dict(saint_data['model_state_dict'])
    saint_model.eval()
    
    models[model_label] = saint_model
    model_results[model_label] = {
        'metrics': saint_data['metrics'],
        'training_time': saint_data['training_time'],
        'model_size': f"{saint_data['total_params']:,} parameters",
        'target_scaler': saint_data.get('target_scaler', None)  # For improved SAINT
    }
    print(f"✅ {model_label} loaded successfully!")
    logger.info(f"{model_label} loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading SAINT model: {e}")
    logger.error(f"Error loading SAINT model: {e}")

print(f"\n📊 Successfully loaded {len(models)} models: {list(models.keys())}")
logger.info(f"Successfully loaded {len(models)} models: {list(models.keys())}")

# Generate predictions for all models
print("\n🔮 Generating predictions for all models...")
predictions = {}
inference_times = {}

for model_name, model in models.items():
    print(f"   Predicting with {model_name}...")
    start_time = time.time()
    
    try:
        if 'XGBoost' in model_name:
            pred = model.predict(X_test_scaled)
        else:  # Neural network models
            model.eval()
            with torch.no_grad():
                # Use the appropriate test data for each model
                if 'test_data' in model_results[model_name]:
                    X_test_for_model = model_results[model_name]['test_data']
                else:
                    X_test_for_model = X_test_scaled
                
                X_test_tensor = torch.FloatTensor(X_test_for_model).to(device)
                
                if 'FT-Transformer' in model_name:
                    pred = model(X_test_tensor, None).squeeze().cpu().numpy()
                    
                    # Handle target scaling for improved FT-Transformer
                    if 'target_scaler' in model_results[model_name]:
                        target_scaler = model_results[model_name]['target_scaler']
                        if target_scaler is not None:
                            pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                        
                else:  # SAINT
                    pred = model(X_test_tensor).squeeze().cpu().numpy()
                    
                    # Handle target scaling for improved SAINT
                    if 'target_scaler' in model_results[model_name]:
                        target_scaler = model_results[model_name]['target_scaler']
                        if target_scaler is not None:
                            pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        
        predictions[model_name] = pred
        inference_times[model_name] = time.time() - start_time
        print(f"   ✅ {model_name} predictions completed in {inference_times[model_name]:.3f}s")
        
    except Exception as e:
        print(f"   ❌ Error with {model_name}: {e}")
        logger.error(f"Error generating predictions for {model_name}: {e}")

# Calculate comprehensive metrics for all models
print("\n📈 Calculating comprehensive metrics...")
comparison_results = []

for model_name in models.keys():
    if model_name in predictions:
        pred = predictions[model_name]
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        try:
            mape = mean_absolute_percentage_error(y_test, pred)
        except:
            mape = np.mean(np.abs((y_test - pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
        
        explained_var = explained_variance_score(y_test, pred)
        
        # Calculate additional metrics
        residuals = y_test - pred
        residual_std = np.std(residuals)
        max_error = np.max(np.abs(residuals))
        
        # Predictions per second
        pred_per_sec = len(X_test_scaled) / inference_times[model_name]
        
        comparison_results.append({
            'Model': model_name,
            'R²_Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape,
            'Explained_Variance': explained_var,
            'Residual_Std': residual_std,
            'Max_Error': max_error,
            'Training_Time_s': model_results[model_name]['training_time'],
            'Inference_Time_s': inference_times[model_name],
            'Predictions_per_Second': pred_per_sec,
            'Model_Size': model_results[model_name]['model_size']
        })

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_results)

print("\n" + "="*100)
print("COMPREHENSIVE MODEL COMPARISON RESULTS")
print("="*100)
print(comparison_df.round(4).to_string(index=False))

# Save comparison results
comparison_df.to_csv('./Section2_Model_Training/airbnb_model_comparison.csv', index=False)
print(f"\n💾 Comparison results saved to './Section2_Model_Training/airbnb_model_comparison.csv'")
logger.info("Comparison results saved")

# Create comprehensive visualization
print("\n📊 Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 15))

# 1. Model Performance Comparison (R² Score)
plt.subplot(3, 4, 1)
r2_scores = comparison_df['R²_Score'].values
model_names = comparison_df['Model'].values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = plt.bar(model_names, r2_scores, color=colors)
plt.title('R² Score Comparison', fontsize=14, fontweight='bold')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# 2. RMSE Comparison
plt.subplot(3, 4, 2)
rmse_scores = comparison_df['RMSE'].values
bars = plt.bar(model_names, rmse_scores, color=colors)
plt.title('RMSE Comparison', fontsize=14, fontweight='bold')
plt.ylabel('RMSE')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(rmse_scores)*0.01,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. MAE Comparison
plt.subplot(3, 4, 3)
mae_scores = comparison_df['MAE'].values
bars = plt.bar(model_names, mae_scores, color=colors)
plt.title('MAE Comparison', fontsize=14, fontweight='bold')
plt.ylabel('MAE')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(mae_scores)*0.01,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Training Time Comparison
plt.subplot(3, 4, 4)
train_times = comparison_df['Training_Time_s'].values.astype(float)
bars = plt.bar(range(len(model_names)), train_times, color=colors[:len(model_names)])
plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Training Time (seconds)')
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.01,
             f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# 5-7. Actual vs Predicted for each model
for i, (model_name, pred) in enumerate(predictions.items()):
    plt.subplot(3, 4, 5 + i)
    plt.scatter(y_test, pred, alpha=0.6, color=colors[i])
    min_val = min(y_test.min(), pred.min())
    max_val = max(y_test.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Airbnb Price ($)')
    plt.ylabel('Predicted Airbnb Price ($)')
    plt.title(f'{model_name}: Actual vs Predicted')
    
    # Add R² annotation
    r2 = r2_score(y_test, pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.grid(True, alpha=0.3)

# 8. Residuals comparison
plt.subplot(3, 4, 8)
for i, (model_name, pred) in enumerate(predictions.items()):
    residuals = y_test - pred
    plt.hist(residuals, bins=30, alpha=0.7, label=model_name, color=colors[i])
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Performance Radar Chart
plt.subplot(3, 4, 9)
# Normalize metrics for radar chart (higher is better)
metrics_for_radar = []
metric_names = ['R²', 'RMSE_inv', 'MAE_inv', 'Speed']

for _, row in comparison_df.iterrows():
    # Invert RMSE and MAE so higher is better
    rmse_inv = 1 / (1 + row['RMSE'] / 100)  # Normalize
    mae_inv = 1 / (1 + row['MAE'] / 100)    # Normalize
    speed_norm = row['Predictions_per_Second'] / comparison_df['Predictions_per_Second'].max()
    
    metrics_for_radar.append([
        row['R²_Score'],
        rmse_inv,
        mae_inv,
        speed_norm
    ])

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

ax = plt.subplot(3, 4, 9, projection='polar')
for i, (model_name, metrics) in enumerate(zip(model_names, metrics_for_radar)):
    metrics += metrics[:1]  # Complete the circle
    ax.plot(angles, metrics, 'o-', linewidth=2, label=model_name, color=colors[i])
    ax.fill(angles, metrics, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1)
plt.title('Performance Radar Chart')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 10. Error Analysis
plt.subplot(3, 4, 10)
for i, (model_name, pred) in enumerate(predictions.items()):
    errors = np.abs(y_test - pred)
    plt.scatter(y_test, errors, alpha=0.6, label=model_name, color=colors[i])
plt.xlabel('Actual Airbnb Price ($)')
plt.ylabel('Absolute Error')
plt.title('Error vs Actual Values')
plt.legend()
plt.grid(True, alpha=0.3)

# 11. Model Complexity vs Performance
plt.subplot(3, 4, 11)
# Extract parameter counts for neural networks
param_counts = []
r2_scores_for_complexity = []
model_labels = []

for _, row in comparison_df.iterrows():
    if 'parameters' in str(row['Model_Size']):
        param_count = int(row['Model_Size'].split()[0].replace(',', ''))
        param_counts.append(param_count)
        r2_scores_for_complexity.append(row['R²_Score'])
        model_labels.append(row['Model'])

if param_counts:
    plt.scatter(param_counts, r2_scores_for_complexity, s=100, alpha=0.7)
    for i, label in enumerate(model_labels):
        plt.annotate(label, (param_counts[i], r2_scores_for_complexity[i]),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Model Parameters')
    plt.ylabel('R² Score')
    plt.title('Model Complexity vs Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

# 12. Inference Speed Comparison
plt.subplot(3, 4, 12)
pred_speeds = comparison_df['Predictions_per_Second'].values
bars = plt.bar(model_names, pred_speeds, color=colors)
plt.title('Inference Speed Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Predictions per Second')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(pred_speeds)*0.01,
             f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./Section2_Model_Training/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comprehensive visualization saved to './Section2_Model_Training/comprehensive_model_comparison.png'")
logger.info("Comprehensive visualization created and saved")

# Feature importance comparison (for models that support it)
print("\n🔍 Analyzing feature importance...")

feature_importance_data = {}

# XGBoost feature importance
if 'XGBoost' in models:
    feature_importance_data['XGBoost'] = models['XGBoost'].feature_importances_

# For neural networks, we'll use a simple permutation importance approximation
for model_name in ['FT-Transformer', 'SAINT']:
    if model_name in models:
        print(f"   Calculating permutation importance for {model_name}...")
        model = models[model_name]
        
        # Get baseline performance
        if model_name == 'FT-Transformer':
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                baseline_pred = model(X_test_tensor, None).squeeze().cpu().numpy()
        else:  # SAINT
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                baseline_pred = model(X_test_tensor).squeeze().cpu().numpy()
        
        baseline_r2 = r2_score(y_test, baseline_pred)
        
        # Calculate permutation importance
        importances = []
        for i in range(len(feature_names)):
            X_test_permuted = X_test_scaled.copy()
            np.random.shuffle(X_test_permuted[:, i])
            
            if model_name == 'FT-Transformer':
                with torch.no_grad():
                    X_permuted_tensor = torch.FloatTensor(X_test_permuted).to(device)
                    permuted_pred = model(X_permuted_tensor, None).squeeze().cpu().numpy()
            else:  # SAINT
                with torch.no_grad():
                    X_permuted_tensor = torch.FloatTensor(X_test_permuted).to(device)
                    permuted_pred = model(X_permuted_tensor).squeeze().cpu().numpy()
            
            permuted_r2 = r2_score(y_test, permuted_pred)
            importance = baseline_r2 - permuted_r2
            importances.append(max(0, importance))  # Ensure non-negative
        
        feature_importance_data[model_name] = np.array(importances)

# Create feature importance comparison plot
if feature_importance_data:
    plt.figure(figsize=(15, 10))
    
    # Normalize importances for comparison
    normalized_importances = {}
    for model_name, importances in feature_importance_data.items():
        normalized_importances[model_name] = importances / np.sum(importances)
    
    # Create comparison plot
    x = np.arange(len(feature_names))
    width = 0.25
    
    for i, (model_name, importances) in enumerate(normalized_importances.items()):
        plt.bar(x + i * width, importances, width, label=model_name, alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance Comparison Across Models')
    plt.xticks(x + width, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./Section2_Model_Training/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save feature importance data
    importance_df = pd.DataFrame(normalized_importances, index=feature_names)
    importance_df.to_csv('./Section2_Model_Training/feature_importance_comparison.csv')
    print("✅ Feature importance comparison saved")
    logger.info("Feature importance analysis completed")

# Model summary and recommendations
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY AND RECOMMENDATIONS")
print("="*80)

# Find best model for each metric
best_r2_model = comparison_df.loc[comparison_df['R²_Score'].idxmax(), 'Model']
best_rmse_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
best_mae_model = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
best_speed_model = comparison_df.loc[comparison_df['Predictions_per_Second'].idxmax(), 'Model']

print(f"🏆 Best R² Score: {best_r2_model} ({comparison_df.loc[comparison_df['Model'] == best_r2_model, 'R²_Score'].values[0]:.4f})")
print(f"🏆 Best RMSE: {best_rmse_model} ({comparison_df.loc[comparison_df['Model'] == best_rmse_model, 'RMSE'].values[0]:.4f})")
print(f"🏆 Best MAE: {best_mae_model} ({comparison_df.loc[comparison_df['Model'] == best_mae_model, 'MAE'].values[0]:.4f})")
print(f"🏆 Fastest Inference: {best_speed_model} ({comparison_df.loc[comparison_df['Model'] == best_speed_model, 'Predictions_per_Second'].values[0]:.0f} pred/s)")

# Overall best model (weighted score)
print(f"\n📊 Overall Model Ranking:")
# Normalize metrics and calculate weighted score
comparison_df_norm = comparison_df.copy()
comparison_df_norm['R²_Score_norm'] = comparison_df_norm['R²_Score']  # Already 0-1
comparison_df_norm['RMSE_norm'] = 1 - (comparison_df_norm['RMSE'] / comparison_df_norm['RMSE'].max())  # Invert
comparison_df_norm['MAE_norm'] = 1 - (comparison_df_norm['MAE'] / comparison_df_norm['MAE'].max())    # Invert
comparison_df_norm['Speed_norm'] = comparison_df_norm['Predictions_per_Second'] / comparison_df_norm['Predictions_per_Second'].max()

# Weighted score (R² 40%, RMSE 30%, MAE 20%, Speed 10%)
comparison_df_norm['Overall_Score'] = (
    0.4 * comparison_df_norm['R²_Score_norm'] +
    0.3 * comparison_df_norm['RMSE_norm'] +
    0.2 * comparison_df_norm['MAE_norm'] +
    0.1 * comparison_df_norm['Speed_norm']
)

# Sort by overall score
ranking = comparison_df_norm.sort_values('Overall_Score', ascending=False)
for i, (_, row) in enumerate(ranking.iterrows()):
    print(f"   {i+1}. {row['Model']} (Score: {row['Overall_Score']:.4f})")

best_overall_model = ranking.iloc[0]['Model']
print(f"\n🥇 Overall Best Model: {best_overall_model}")

# Business recommendations
print(f"\n💼 Business Recommendations:")
print(f"   🎯 For highest accuracy: Use {best_r2_model}")
print(f"   ⚡ For fastest predictions: Use {best_speed_model}")
print(f"   ⚖️ For balanced performance: Use {best_overall_model}")

# Save detailed results for Section 3
print(f"\n💾 Preparing data for Section 3...")

section2_results = {
    'models': models,
    'predictions': predictions,
    'comparison_df': comparison_df,
    'feature_importance_data': feature_importance_data if 'feature_importance_data' in locals() else {},
    'X_train_scaled': X_train_scaled,
    'X_val_scaled': X_val_scaled,
    'X_test_scaled': X_test_scaled,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'feature_names': feature_names,
    'data_summary': data_summary,
    'model_results': model_results,
    'inference_times': inference_times,
    'best_models': {
        'best_r2': best_r2_model,
        'best_rmse': best_rmse_model,
        'best_mae': best_mae_model,
        'best_speed': best_speed_model,
        'best_overall': best_overall_model
    }
}

# Save to pickle file
with open('./airbnb_section2_results.pkl', 'wb') as f:
    pickle.dump(section2_results, f)

print("✅ Section 2 results saved to './airbnb_section2_results.pkl'")
logger.info("Section 2 results saved for Section 3")

# Final summary
print("\n" + "="*80)
print("SECTION 2 COMPLETION SUMMARY")
print("="*80)
print(f"✅ Models Successfully Compared: {len(models)}")
for model_name in models.keys():
    print(f"   - {model_name}")

print(f"\n📁 Files Generated:")
print(f"   - Model comparison CSV: ./Section2_Model_Training/airbnb_model_comparison.csv")
print(f"   - Comprehensive visualization: ./Section2_Model_Training/comprehensive_model_comparison.png")
print(f"   - Feature importance comparison: ./Section2_Model_Training/feature_importance_comparison.csv")
print(f"   - Feature importance plot: ./Section2_Model_Training/feature_importance_comparison.png")
print(f"   - Section 2 results pickle: ./airbnb_section2_results.pkl")
print(f"   - Comparison log: ./Section2_Model_Training/model_comparison.log")

print(f"\n🎯 Dataset: Airbnb Price Prediction")
print(f"📊 Task: Regression")
print(f"🔢 Features: {len(feature_names)}")
print(f"📈 Training Samples: {len(X_train_scaled):,}")
print(f"🧪 Test Samples: {len(X_test_scaled):,}")
print(f"\n🏆 Best Overall Model: {best_overall_model}")
print(f"📊 Best R² Score: {comparison_df['R²_Score'].max():.4f}")
print(f"📉 Best RMSE: {comparison_df['RMSE'].min():.4f}")

print(f"\n🚀 Ready for Section 3: Model Explainability and Analysis!")
logger.info("Section 2 model comparison completed successfully")
