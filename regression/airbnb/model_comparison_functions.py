"""
Model Comparison Functions for Airbnb Price Regression

This module contains functions for loading, comparing, and analyzing all trained models:
- XGBoost
- FT-Transformer (Original and Improved)
- SAINT

It provides comprehensive comparisons, visualizations, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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

def load_preprocessed_data(data_path='./airbnb_preprocessed_data.pkl'):
    """Load preprocessed Airbnb data (enhanced or basic)"""
    print("📊 Loading preprocessed data...")
    
    try:
        # Try to load enhanced data first
        import joblib
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
            
            # Create data_summary for compatibility
            data_summary = {
                'dataset': 'Airbnb Enhanced',
                'task': 'Regression',
                'n_samples': len(X_train_scaled),
                'n_features': len(feature_names),
                'target_name': 'price',
                'preprocessing': 'Enhanced (outlier removal, feature engineering, scaling)'
            }
            
            print("✅ Enhanced data loaded successfully!")
            print(f"   Test samples: {len(X_test_scaled):,}")
            print(f"   Features: {len(feature_names)} (after feature engineering & selection)")
            print(f"   Target transformation: {transform_info['method']}")
            print(f"   Outliers removed: {enhanced_data['outlier_mask'].sum()}")
            
            return (X_train_scaled, X_val_scaled, X_test_scaled, 
                    y_train, y_val, y_test, feature_names, data_summary)
            
        except Exception as e:
            print(f"Enhanced data not available: {e}")
            print("Falling back to basic preprocessed data...")
            
            # Fallback to basic preprocessed data
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
            
            print("✅ Basic preprocessed data loaded successfully!")
            print(f"   Test samples: {len(X_test_scaled):,}")
            print(f"   Features: {len(feature_names)}")
            
            return (X_train_scaled, X_val_scaled, X_test_scaled, 
                    y_train, y_val, y_test, feature_names, data_summary)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def load_xgboost_model(model_dir='./Section2_Model_Training'):
    """Load Enhanced XGBoost model and results"""
    print("📦 Loading Enhanced XGBoost model...")
    
    try:
        import joblib
        
        # Try to load tuned enhanced model first, then baseline enhanced, then regular
        xgb_file_paths = [
            f'{model_dir}/xgboost_tuned_enhanced.pkl'
            # f'{model_dir}/xgboost_baseline_enhanced.pkl',
            # f'{model_dir}/xgboost_model.pkl'
        ]
        
        xgb_model = None
        model_type = None
        
        for file_path in xgb_file_paths:
            try:
                xgb_model = joblib.load(file_path)
                if 'tuned_enhanced' in file_path:
                    model_type = 'XGBoost Enhanced (Tuned)'
                elif 'baseline_enhanced' in file_path:
                    model_type = 'XGBoost Enhanced (Baseline)'
                else:
                    model_type = 'XGBoost (Regular)'
                print(f"   ✅ Loaded {model_type} from {file_path}")
                break
            except FileNotFoundError:
                continue
        
        if xgb_model is None:
            print("   ❌ No XGBoost model file found")
            return None
        
        # Load metrics from comparison CSV
        try:
            comparison_df_saved = pd.read_csv(f'{model_dir}/enhanced_xgboost_comparison.csv', index_col=0)
            # Get the tuned model metrics (last row)
            xgb_metrics = comparison_df_saved.iloc[-1].to_dict()
            training_time = xgb_metrics.get('training_time', 0)
        except:
            # Fallback to regular evaluation results
            try:
                xgb_metrics_df = pd.read_csv(f'{model_dir}/xgboost_evaluation_results.csv', index_col=0)
                xgb_metrics = xgb_metrics_df['XGBoost'].to_dict()
                training_time = xgb_metrics.get('train_time', 0)
            except:
                xgb_metrics = {}
                training_time = 0
        
        model_info = {
            'model': xgb_model,
            'metrics': xgb_metrics,
            'training_time': training_time,
            'model_size': model_type,
            'model_type': model_type
        }
        
        print(f"✅ {model_type} loaded successfully!")
        return model_info
        
    except Exception as e:
        print(f"❌ Error loading XGBoost model: {e}")
        return None


def load_improved_ft_transformer_model(model_dir='./Section2_Model_Training', feature_names=None, device='cpu'):
    """Load Improved FT-Transformer model and results"""
    print("📦 Loading Improved FT-Transformer model...")
    
    try:
        import rtdl
        
        # Load model checkpoint
        checkpoint = torch.load(f'{model_dir}/improved_ft_transformer_model.pth', 
                              map_location=device, weights_only=False)
        
        # Create model architecture (improved version)
        feature_info = {
            'n_num_features': len(feature_names),
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
        
        model_info = {
            'model': ft_model,
            'metrics': checkpoint['metrics'],
            'training_time': checkpoint['training_time'],
            'model_size': f"{checkpoint['total_params']:,} parameters",
            'target_scaler': checkpoint.get('target_scaler', None)
        }
        
        print("✅ Improved FT-Transformer loaded successfully!")
        return model_info
        
    except Exception as e:
        print(f"❌ Error loading Improved FT-Transformer model: {e}")
        return None

def load_saint_model(model_dir='./Section2_Model_Training', device='cpu'):
    """Load Enhanced SAINT model and results"""
    print("📦 Loading Enhanced SAINT model...")
    
    try:
        # Try importing from improved_saint_training first, then fallback
        try:
            from improved_saint_training import ImprovedSAINTModel as SAINTModel
            model_label = 'SAINT Enhanced'  # Use consistent label that matches predictions #'XGBoost Enhanced', 'FT-Transformer Enhanced', 'SAINT Enhanced'
            print("   ✅ Using improved SAINT training module")
        except ImportError:
                print("   ❌ No SAINT training module found")
                return None
        
        # Try to load enhanced SAINT model first
        saint_file_paths = [
            f'{model_dir}/improved_saint_model.pkl',
        ]
        
        saint_data = None
        for file_path in saint_file_paths:
            try:
                with open(file_path, 'rb') as f:
                    saint_data = pickle.load(f)
                print(f"   ✅ Loaded SAINT model from {file_path}")
                break
            except FileNotFoundError:
                continue
        
        if saint_data is None:
            print("   ❌ No SAINT model file found")
            return None
        
        # Create model architecture
        saint_model = SAINTModel(
            n_features=saint_data['model_architecture']['n_features'],
            d_model=saint_data['model_architecture']['d_model'],
            n_heads=saint_data['model_architecture']['n_heads'],
            n_layers=saint_data['model_architecture']['n_layers']
        ).to(device)
        
        # Load state dict
        saint_model.load_state_dict(saint_data['model_state_dict'])
        saint_model.eval()
        
        model_info = {
            'model': saint_model,
            'metrics': saint_data['metrics'],
            'training_time': saint_data['training_time'],
            'model_size': f"{saint_data['total_params']:,} parameters",
            'target_scaler': saint_data.get('target_scaler', None),  # For improved SAINT
            'model_label': model_label  # Add model label for consistency
        }
        
        print(f"✅ Enhanced SAINT loaded successfully!")
        return model_info
        
    except Exception as e:
        print(f"❌ Error loading SAINT model: {e}")
        return None

def load_all_models(model_dir='./Section2_Model_Training', feature_names=None, device='cpu'):
    """Load all available models"""
    print("🔄 Loading all available models...")
    
    models = {}
    model_results = {}
    
    # Load XGBoost
    xgb_info = load_xgboost_model(model_dir)
    if xgb_info:
        models['XGBoost Enhanced'] = xgb_info['model']
        model_results['XGBoost Enhanced'] = {
            'metrics': xgb_info['metrics'],
            'training_time': xgb_info['training_time'],
            'model_size': xgb_info['model_size']
        }
    
    # Load Original FT-Transformer
    # ft_info = load_ft_transformer_model(model_dir, feature_names, device)
    # if ft_info:
    #     models['FT-Transformer'] = ft_info['model']
    #     model_results['FT-Transformer'] = {
    #         'metrics': ft_info['metrics'],
    #         'training_time': ft_info['training_time'],
    #         'model_size': ft_info['model_size']
    #     }
    
    # Load Improved FT-Transformer
    improved_ft_info = load_improved_ft_transformer_model(model_dir, feature_names, device)
    if improved_ft_info:
        models['FT-Transformer Enhanced'] = improved_ft_info['model']
        model_results['FT-Transformer Enhanced'] = {
            'metrics': improved_ft_info['metrics'],
            'training_time': improved_ft_info['training_time'],
            'model_size': improved_ft_info['model_size'],
            'target_scaler': improved_ft_info['target_scaler']
        }
    
    # Load SAINT
    saint_info = load_saint_model(model_dir, device)
    if saint_info:
        models['SAINT Enhanced'] = saint_info['model']
        model_results['SAINT Enhanced'] = {
            'metrics': saint_info['metrics'],
            'training_time': saint_info['training_time'],
            'model_size': saint_info['model_size']
        }
    
    print(f"✅ Successfully loaded {len(models)} models: {list(models.keys())}")
    return models, model_results

def generate_predictions(models, model_results, X_test_scaled, device='cpu'):
    """Generate predictions for all models"""
    print("🔮 Generating predictions for all models...")
    
    predictions = {}
    inference_times = {}
    
    # Load basic data if needed for FT-Transformer
    basic_test_data = None
    
    for model_name, model in models.items():
        print(f"   Predicting with {model_name}...")
        start_time = time.time()
        
        try:
            if model_name == 'XGBoost':
                pred = model.predict(X_test_scaled)
            elif 'FT-Transformer' in model_name:
                model.eval()
                with torch.no_grad():
                    # Check if model needs basic data
                    if ('test_data_key' in model_results[model_name] and 
                        model_results[model_name]['test_data_key'] == 'basic'):
                        # Load basic data if not already loaded
                        if basic_test_data is None:
                            try:
                                with open('./Section1_Data_PreProcessing/airbnb_preprocessed_data.pkl', 'rb') as f:
                                    basic_data = pickle.load(f)
                                basic_test_data = basic_data['X_test_scaled']
                                print(f"   Loaded basic data with {basic_test_data.shape[1]} features for FT-Transformer")
                            except:
                                print("   Could not load basic data, using current data")
                                basic_test_data = X_test_scaled
                        
                        X_test_for_model = basic_test_data
                    else:
                        X_test_for_model = X_test_scaled
                    
                    X_test_tensor = torch.FloatTensor(X_test_for_model).to(device)
                    pred = model(X_test_tensor, None).squeeze().cpu().numpy()
                    
                    # Handle improved FT-Transformer with target scaling
                    if model_name == 'Improved FT-Transformer' and 'target_scaler' in model_results[model_name]:
                        target_scaler = model_results[model_name]['target_scaler']
                        if target_scaler is not None:
                            pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                        
            else:  # SAINT
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
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
    
    return predictions, inference_times

def calculate_comprehensive_metrics(models, predictions, inference_times, model_results, y_test, X_test_scaled):
    """Calculate comprehensive metrics for all models"""
    print("📈 Calculating comprehensive metrics...")
    
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
    
    return pd.DataFrame(comparison_results)

def create_comprehensive_visualization(comparison_df, predictions, y_test, save_dir='./Section2_Model_Training'):
    """Create comprehensive visualization comparing all models"""
    print("📊 Creating comprehensive visualizations...")
    
    fig = plt.figure(figsize=(20, 15))
    
    model_names = comparison_df['Model'].values
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
    
    # 1. Model Performance Comparison (R² Score)
    plt.subplot(3, 4, 1)
    r2_scores = comparison_df['R²_Score'].values
    bars = plt.bar(range(len(model_names)), r2_scores, color=colors)
    plt.title('R² Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 2. RMSE Comparison
    plt.subplot(3, 4, 2)
    rmse_scores = comparison_df['RMSE'].values
    bars = plt.bar(range(len(model_names)), rmse_scores, color=colors)
    plt.title('RMSE Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(rmse_scores)*0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 3. MAE Comparison
    plt.subplot(3, 4, 3)
    mae_scores = comparison_df['MAE'].values
    bars = plt.bar(range(len(model_names)), mae_scores, color=colors)
    plt.title('MAE Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('MAE')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(mae_scores)*0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 4. Training Time Comparison
    plt.subplot(3, 4, 4)
    train_times = comparison_df['Training_Time_s'].values.astype(float)
    bars = plt.bar(range(len(model_names)), train_times, color=colors)
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.01,
                 f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 5-8. Actual vs Predicted for each model
    plot_idx = 5
    for i, (model_name, pred) in enumerate(predictions.items()):
        if plot_idx <= 8:
            plt.subplot(3, 4, plot_idx)
            plt.scatter(y_test, pred, alpha=0.6, color=colors[i], s=20)
            min_val = min(y_test.min(), pred.min())
            max_val = max(y_test.max(), pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('Actual Airbnb Price')
            plt.ylabel('Predicted Airbnb Price')
            plt.title(f'{model_name}: Actual vs Predicted', fontsize=10)
            
            # Add R² annotation
            r2 = r2_score(y_test, pred)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
            plt.grid(True, alpha=0.3)
            plot_idx += 1
    
    # 9. Residuals comparison
    plt.subplot(3, 4, 9)
    for i, (model_name, pred) in enumerate(predictions.items()):
        residuals = y_test - pred
        plt.hist(residuals, bins=30, alpha=0.7, label=model_name, color=colors[i])
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution Comparison')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 10. Performance Radar Chart
    plt.subplot(3, 4, 10)
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
    
    ax = plt.subplot(3, 4, 10, projection='polar')
    for i, (model_name, metrics) in enumerate(zip(model_names, metrics_for_radar)):
        metrics += metrics[:1]  # Complete the circle
        ax.plot(angles, metrics, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, metrics, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    plt.title('Performance Radar Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # 11. Error Analysis
    plt.subplot(3, 4, 11)
    for i, (model_name, pred) in enumerate(predictions.items()):
        errors = np.abs(y_test - pred)
        plt.scatter(y_test, errors, alpha=0.6, label=model_name, color=colors[i], s=20)
    plt.xlabel('Actual Airbnb Price')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Actual Values')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 12. Inference Speed Comparison
    plt.subplot(3, 4, 12)
    pred_speeds = comparison_df['Predictions_per_Second'].values
    bars = plt.bar(range(len(model_names)), pred_speeds, color=colors)
    plt.title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Predictions per Second')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(pred_speeds)*0.01,
                 f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comprehensive visualization saved to '{save_dir}/comprehensive_model_comparison.png'")

def analyze_feature_importance(models, X_test_scaled, y_test, feature_names, device='cpu', save_dir='./Section2_Model_Training'):
    """Analyze and compare feature importance across models"""
    print("🔍 Analyzing feature importance...")
    
    feature_importance_data = {}
    
    # XGBoost feature importance
    if 'XGBoost' in models:
        feature_importance_data['XGBoost'] = models['XGBoost'].feature_importances_
    
    # For neural networks, use permutation importance approximation
    for model_name in models.keys():
        if 'Transformer' in model_name or model_name == 'SAINT':
            print(f"   Calculating permutation importance for {model_name}...")
            model = models[model_name]
            
            # Get baseline performance
            if 'FT-Transformer' in model_name:
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
                
                if 'FT-Transformer' in model_name:
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
            if np.sum(importances) > 0:
                normalized_importances[model_name] = importances / np.sum(importances)
            else:
                normalized_importances[model_name] = importances
        
        # Create comparison plot
        x = np.arange(len(feature_names))
        width = 0.8 / len(normalized_importances)
        
        for i, (model_name, importances) in enumerate(normalized_importances.items()):
            plt.bar(x + i * width, importances, width, label=model_name, alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance Comparison Across Models')
        plt.xticks(x + width * (len(normalized_importances) - 1) / 2, feature_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance data
        importance_df = pd.DataFrame(normalized_importances, index=feature_names)
        importance_df.to_csv(f'{save_dir}/feature_importance_comparison.csv')
        print("✅ Feature importance comparison saved")
        
        return feature_importance_data
    
    return {}

def generate_model_summary(comparison_df):
    """Generate model performance summary and recommendations"""
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
    
    return {
        'best_r2': best_r2_model,
        'best_rmse': best_rmse_model,
        'best_mae': best_mae_model,
        'best_speed': best_speed_model,
        'best_overall': best_overall_model,
        'ranking': ranking
    }

def save_comparison_results(comparison_df, predictions, feature_importance_data, 
                          data_summary, model_results, inference_times, 
                          best_models, save_dir='./Section2_Model_Training'):
    """Save all comparison results and prepare data for Section 3"""
    print("💾 Saving comparison results...")
    
    # Save comparison DataFrame
    comparison_df.to_csv(f'{save_dir}/airbnb_model_comparison.csv', index=False)
    
    # Save feature importance data if available
    if feature_importance_data:
        importance_df = pd.DataFrame(feature_importance_data)
        importance_df.to_csv(f'{save_dir}/feature_importance_comparison.csv')
    
    # Prepare comprehensive results for Section 3
    section2_results = {
        'comparison_df': comparison_df,
        'predictions': predictions,
        'feature_importance_data': feature_importance_data,
        'data_summary': data_summary,
        'model_results': model_results,
        'inference_times': inference_times,
        'best_models': best_models
    }
    
    # Save to pickle file
    with open('./airbnb_section2_results.pkl', 'wb') as f:
        pickle.dump(section2_results, f)
    
    print("✅ All results saved successfully!")
    return section2_results

def run_complete_model_comparison(data_path='./enhanced_data.pkl',
                                model_dir='./Section2_Model_Training',
                                device=None):
    """Run complete model comparison pipeline"""
    print("🏠 COMPREHENSIVE MODEL COMPARISON FOR AIRBNB PRICE REGRESSION")
    print("=" * 80)
    
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logging(model_dir)
    logger.info("Starting comprehensive model comparison")
    
    try:
        # Step 1: Load preprocessed data
        (X_train_scaled, X_val_scaled, X_test_scaled, 
         y_train, y_val, y_test, feature_names, data_summary) = load_preprocessed_data(data_path)
        
        # Step 2: Load all models
        models, model_results = load_all_models(model_dir, feature_names, device)
        
        # Step 3: Generate predictions
        predictions, inference_times = generate_predictions(models, model_results, X_test_scaled, device)
        
        # Step 4: Calculate comprehensive metrics
        comparison_df = calculate_comprehensive_metrics(models, predictions, inference_times, 
                                                     model_results, y_test, X_test_scaled)
        
        # Step 5: Create visualizations
        create_comprehensive_visualization(comparison_df, predictions, y_test, model_dir)
        
        # Step 6: Analyze feature importance
        feature_importance_data = analyze_feature_importance(models, X_test_scaled, y_test, 
                                                           feature_names, device, model_dir)
        
        # Step 7: Generate summary and recommendations
        best_models = generate_model_summary(comparison_df)
        
        # Step 8: Save all results
        section2_results = save_comparison_results(comparison_df, predictions, feature_importance_data,
                                                 data_summary, model_results, inference_times,
                                                 best_models, model_dir)
        
        # Final summary
        print("\n" + "="*80)
        print("SECTION 2 COMPLETION SUMMARY")
        print("="*80)
        print(f"✅ Models Successfully Compared: {len(models)}")
        for model_name in models.keys():
            print(f"   - {model_name}")
        
        print(f"\n🏆 Best Overall Model: {best_models['best_overall']}")
        print(f"📊 Best R² Score: {comparison_df['R²_Score'].max():.4f}")
        print(f"📉 Best RMSE: {comparison_df['RMSE'].min():.4f}")
        
        print(f"\n🚀 Ready for Section 3: Model Explainability and Analysis!")
        logger.info("Section 2 model comparison completed successfully")
        
        return section2_results
        
    except Exception as e:
        print(f"❌ Error in model comparison pipeline: {e}")
        logger.error(f"Error in model comparison pipeline: {e}")
        raise

if __name__ == "__main__":
    # Run the complete comparison pipeline
    results = run_complete_model_comparison()
