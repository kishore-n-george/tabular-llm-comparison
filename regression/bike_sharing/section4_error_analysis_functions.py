"""
Section 4: Error Analysis Functions for Regression Models
========================================================

This module provides comprehensive error analysis functions for regression models
including XGBoost, FT-Transformer, and SAINT on the bike sharing dataset.

Functions:
- setup_error_analysis_environment: Setup analysis environment
- load_regression_models_and_data: Load trained models and preprocessed data
- generate_model_predictions: Generate predictions from all models
- calculate_regression_errors: Calculate various error metrics
- analyze_residuals: Analyze residual patterns
- perform_cross_model_error_comparison: Compare errors across models
- analyze_feature_based_errors: Feature-specific error analysis
- analyze_prediction_intervals: Confidence/prediction interval analysis
- generate_error_visualizations: Create comprehensive error visualizations
- analyze_model_specific_errors: Deep dive into individual model errors
- generate_business_insights: Generate actionable business insights
- save_error_analysis_results: Save all results and visualizations
- run_complete_error_analysis: Run the complete error analysis pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats
from scipy.stats import ttest_ind, pearsonr, spearmanr
import pickle
import os
import gc
import torch

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_error_analysis_environment():
    """Setup the error analysis environment."""
    print("🔍 Section 4: Regression Error Analysis")
    print("Dataset: Bike Sharing Demand Prediction")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('./Section4_ErrorAnalysis', exist_ok=True)
    
    # Memory management for PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("✅ Error analysis environment setup complete!")
    return True

def load_regression_models_and_data():
    """Load trained regression models and preprocessed data using existing infrastructure."""
    print("\n📂 Loading trained models and data...")
    
    try:
        # Import the robust model loading functions
        from model_comparison_functions import load_all_models, load_preprocessed_data
        
        # Load preprocessed data using existing function
        (X_train_scaled, X_val_scaled, X_test_scaled, 
         y_train, y_val, y_test, feature_names, data_summary) = load_preprocessed_data()
        
        # Load models using existing robust function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models, model_results = load_all_models('./Section2_Model_Training', feature_names, device)
        
        # Extract just the models (not the full model_results structure)
        clean_models = {}
        for model_name, model in models.items():
            clean_models[model_name] = model
        
        # Load Section 2 results if available
        try:
            with open('./bike_sharing_section2_results.pkl', 'rb') as f:
                section2_data = pickle.load(f)
            predictions = section2_data.get('predictions', {})
            comparison_df = section2_data.get('comparison_df', pd.DataFrame())
        except:
            predictions = {}
            comparison_df = pd.DataFrame()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Models available: {list(clean_models.keys())}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Training samples: {len(X_train_scaled):,}")
        print(f"   Test samples: {len(X_test_scaled):,}")
        
        return {
            'models': clean_models,
            'predictions': predictions,
            'comparison_df': comparison_df,
            'feature_names': feature_names,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'device': device
        }
        
    except FileNotFoundError as e:
        print(f"❌ Required data files not found: {str(e)}")
        print("Please run Section 2 (Model Training) first.")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def generate_model_predictions(models, X_test_scaled, device):
    """Generate predictions from all available models."""
    print("\n🔮 Generating predictions from all models...")
    
    predictions = {}
    prediction_details = {}
    
    for model_name, model in models.items():
        print(f"   Processing {model_name}...")
        
        try:
            if 'XGBoost' in model_name:
                # XGBoost predictions
                y_pred = model.predict(X_test_scaled)
                predictions[model_name] = y_pred
                prediction_details[model_name] = {
                    'type': 'sklearn',
                    'has_uncertainty': False
                }
                
            elif 'SAINT' in model_name:
                # SAINT predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test_scaled)
                else:
                    # PyTorch SAINT model
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
                        y_pred = model(X_tensor).cpu().numpy().flatten()
                
                predictions[model_name] = y_pred
                prediction_details[model_name] = {
                    'type': 'pytorch',
                    'has_uncertainty': False
                }
                
            elif 'FT-Transformer' in model_name:
                # FT-Transformer predictions - handle the x_cat requirement AND target scaling
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
                    
                    # FT-Transformer requires both x_num and x_cat arguments
                    # Use the same approach as in enhanced_ablation_studies.py
                    try:
                        # Try with None for x_cat first (improved training approach)
                        outputs = model(X_tensor, None)
                    except TypeError as e:
                        if "missing 1 required positional argument: 'x_cat'" in str(e):
                            # Create empty categorical tensor on correct device
                            model_device = next(model.parameters()).device if hasattr(model, 'parameters') else device
                            if X_tensor.device != model_device:
                                X_tensor = X_tensor.to(model_device)
                            empty_cat = torch.empty(X_tensor.shape[0], 0, dtype=torch.long, device=model_device)
                            outputs = model(X_tensor, empty_cat)
                        else:
                            raise e
                    
                    # Handle output format
                    if isinstance(outputs, tuple):
                        y_pred_scaled = outputs[0].cpu().numpy().flatten()
                    else:
                        y_pred_scaled = outputs.cpu().numpy().flatten()
                    
                    # CRITICAL FIX: Check if model was trained with target scaling
                    # Load the target scaler from the model checkpoint
                    try:
                        model_path = './Section2_Model_Training/improved_ft_transformer_model.pth'
                        if os.path.exists(model_path):
                            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                            if 'target_scaler' in checkpoint:
                                target_scaler = checkpoint['target_scaler']
                                # Unscale predictions to original scale
                                y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                                print(f"     🔧 Applied target inverse scaling for {model_name}")
                            else:
                                y_pred = y_pred_scaled
                                print(f"     ⚠️ No target scaler found for {model_name}")
                        else:
                            y_pred = y_pred_scaled
                            print(f"     ⚠️ Model checkpoint not found for {model_name}")
                    except Exception as scaling_error:
                        print(f"     ⚠️ Error loading target scaler for {model_name}: {scaling_error}")
                        y_pred = y_pred_scaled
                
                predictions[model_name] = y_pred
                prediction_details[model_name] = {
                    'type': 'pytorch',
                    'has_uncertainty': False,
                    'target_scaled': 'target_scaler' in locals() and target_scaler is not None
                }
            
            print(f"     ✅ Generated {len(predictions[model_name])} predictions")
            
        except Exception as e:
            print(f"     ❌ Failed to generate predictions: {str(e)}")
            continue
    
    print(f"\n✅ Predictions generated for {len(predictions)} models")
    return predictions, prediction_details

def calculate_regression_errors(predictions, y_test):
    """Calculate comprehensive regression error metrics."""
    print("\n📊 Calculating regression error metrics...")
    
    error_metrics = {}
    
    for model_name, y_pred in predictions.items():
        # Basic regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Residuals
        residuals = y_test - y_pred
        
        # Error statistics
        error_metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'residuals': residuals,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skew': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals),
            'max_error': np.max(np.abs(residuals)),
            'q95_error': np.percentile(np.abs(residuals), 95),
            'q75_error': np.percentile(np.abs(residuals), 75)
        }
        
        print(f"   {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    return error_metrics

def analyze_residuals(error_metrics, predictions, y_test):
    """Analyze residual patterns for all models."""
    print("\n🔍 Analyzing residual patterns...")
    
    residual_analysis = {}
    
    for model_name in predictions.keys():
        residuals = error_metrics[model_name]['residuals']
        y_pred = predictions[model_name]
        
        # Normality test
        _, normality_p = stats.shapiro(residuals[:1000] if len(residuals) > 1000 else residuals)
        
        # Homoscedasticity analysis
        # Correlation between absolute residuals and predictions
        abs_residuals = np.abs(residuals)
        homo_corr, homo_p = pearsonr(abs_residuals, y_pred)
        
        # Autocorrelation (if applicable)
        # For time series, check if residuals are autocorrelated
        
        residual_analysis[model_name] = {
            'normality_p_value': normality_p,
            'is_normal': normality_p > 0.05,
            'homoscedasticity_corr': homo_corr,
            'homoscedasticity_p': homo_p,
            'is_homoscedastic': abs(homo_corr) < 0.1,
            'outlier_threshold': np.percentile(abs_residuals, 95),
            'outlier_count': np.sum(abs_residuals > np.percentile(abs_residuals, 95)),
            'outlier_percentage': np.sum(abs_residuals > np.percentile(abs_residuals, 95)) / len(residuals) * 100
        }
        
        print(f"   {model_name}: Normal={residual_analysis[model_name]['is_normal']}, "
              f"Homoscedastic={residual_analysis[model_name]['is_homoscedastic']}, "
              f"Outliers={residual_analysis[model_name]['outlier_percentage']:.1f}%")
    
    return residual_analysis

def perform_cross_model_error_comparison(error_metrics):
    """Compare errors across all models."""
    print("\n📈 Performing cross-model error comparison...")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, metrics in error_metrics.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],
            'MAPE': metrics['MAPE'],
            'Max_Error': metrics['max_error'],
            'Q95_Error': metrics['q95_error'],
            'Residual_Std': metrics['residual_std'],
            'Residual_Skew': abs(metrics['residual_skew']),
            'Residual_Kurtosis': abs(metrics['residual_kurtosis'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Rank models
    comparison_df['RMSE_Rank'] = comparison_df['RMSE'].rank()
    comparison_df['MAE_Rank'] = comparison_df['MAE'].rank()
    comparison_df['R2_Rank'] = comparison_df['R2'].rank(ascending=False)
    comparison_df['Overall_Rank'] = (comparison_df['RMSE_Rank'] + 
                                   comparison_df['MAE_Rank'] + 
                                   comparison_df['R2_Rank']) / 3
    
    comparison_df = comparison_df.sort_values('Overall_Rank')
    
    print("📊 CROSS-MODEL ERROR COMPARISON")
    print("=" * 80)
    print(comparison_df.round(4).to_string(index=False))
    
    # Save results
    comparison_df.to_csv('./Section4_ErrorAnalysis/cross_model_error_comparison.csv', index=False)
    print("\n💾 Results saved to 'cross_model_error_comparison.csv'")
    
    return comparison_df

def analyze_feature_based_errors(models, predictions, error_metrics, X_test_scaled, y_test, feature_names):
    """Analyze which features are associated with larger errors."""
    print("\n🎯 Analyzing feature-based error patterns...")
    
    feature_error_analysis = {}
    
    for model_name in predictions.keys():
        print(f"\n   Analyzing {model_name}...")
        
        residuals = error_metrics[model_name]['residuals']
        abs_residuals = np.abs(residuals)
        
        # Find high-error samples (top 20%)
        error_threshold = np.percentile(abs_residuals, 80)
        high_error_mask = abs_residuals > error_threshold
        
        high_error_features = X_test_scaled[high_error_mask]
        low_error_features = X_test_scaled[~high_error_mask]
        
        # Calculate feature differences
        feature_diff = np.mean(high_error_features, axis=0) - np.mean(low_error_features, axis=0)
        
        # Statistical significance tests
        p_values = []
        effect_sizes = []
        
        for i in range(len(feature_names)):
            if len(high_error_features) > 1 and len(low_error_features) > 1:
                _, p_val = ttest_ind(high_error_features[:, i], low_error_features[:, i])
                p_values.append(p_val)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(high_error_features) - 1) * np.var(high_error_features[:, i]) + 
                                     (len(low_error_features) - 1) * np.var(low_error_features[:, i])) / 
                                    (len(high_error_features) + len(low_error_features) - 2))
                cohens_d = feature_diff[i] / pooled_std if pooled_std > 0 else 0
                effect_sizes.append(abs(cohens_d))
            else:
                p_values.append(1.0)
                effect_sizes.append(0.0)
        
        # Find significant features
        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)
        significant_indices = np.where(p_values < 0.05)[0]
        
        feature_error_analysis[model_name] = {
            'feature_differences': feature_diff,
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'significant_features': significant_indices,
            'high_error_count': np.sum(high_error_mask),
            'error_threshold': error_threshold
        }
        
        print(f"     High-error samples: {np.sum(high_error_mask)} ({np.sum(high_error_mask)/len(residuals)*100:.1f}%)")
        print(f"     Significant features: {len(significant_indices)}/{len(feature_names)}")
        
        if len(significant_indices) > 0:
            top_features = significant_indices[np.argsort(effect_sizes[significant_indices])[::-1]][:5]
            print(f"     Top problematic features:")
            for i, idx in enumerate(top_features):
                print(f"       {i+1}. {feature_names[idx]}: effect_size={effect_sizes[idx]:.3f}")
    
    return feature_error_analysis

def generate_error_visualizations(error_metrics, predictions, y_test, feature_error_analysis, feature_names):
    """Generate comprehensive error visualizations."""
    print("\n📊 Generating error visualizations...")
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    # 1. Overall Error Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE Comparison
    rmse_values = [error_metrics[name]['RMSE'] for name in model_names]
    axes[0, 0].bar(model_names, rmse_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Root Mean Square Error (RMSE)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² Comparison
    r2_values = [error_metrics[name]['R2'] for name in model_names]
    axes[0, 1].bar(model_names, r2_values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('R² Score')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE Comparison
    mae_values = [error_metrics[name]['MAE'] for name in model_names]
    axes[1, 0].bar(model_names, mae_values, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Mean Absolute Error (MAE)')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAPE Comparison
    mape_values = [error_metrics[name]['MAPE'] for name in model_names]
    axes[1, 1].bar(model_names, mape_values, color='gold', alpha=0.7)
    axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./Section4_ErrorAnalysis/overall_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Residual Analysis for each model
    for model_name in model_names:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        residuals = error_metrics[model_name]['residuals']
        y_pred = predictions[model_name]
        
        # Residuals vs Predictions
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title(f'{model_name}: Residuals vs Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[0, 1].scatter(y_test, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'{model_name}: Residuals vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual Distribution
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='skyblue', density=True)
        axes[0, 2].axvline(x=0, color='red', linestyle='--')
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title(f'{model_name}: Residual Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{model_name}: Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Predictions vs Actual
        axes[1, 1].scatter(y_test, y_pred, alpha=0.6, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title(f'{model_name}: Predictions vs Actual')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Error Distribution by Magnitude
        abs_residuals = np.abs(residuals)
        axes[1, 2].hist(abs_residuals, bins=30, alpha=0.7, color='orange')
        axes[1, 2].axvline(x=np.percentile(abs_residuals, 95), color='red', linestyle='--', label='95th Percentile')
        axes[1, 2].set_xlabel('Absolute Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title(f'{model_name}: Absolute Error Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'./Section4_ErrorAnalysis/{model_name}_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Feature-based Error Analysis Visualization
    for model_name in model_names:
        if model_name in feature_error_analysis:
            analysis = feature_error_analysis[model_name]
            significant_indices = analysis['significant_features']
            
            if len(significant_indices) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Top problematic features
                top_indices = significant_indices[np.argsort(analysis['effect_sizes'][significant_indices])[::-1]][:10]
                feature_diff = analysis['feature_differences']
                effect_sizes = analysis['effect_sizes']
                p_values = analysis['p_values']
                
                colors = ['red' if p_values[i] < 0.01 else 'orange' for i in top_indices]
                
                axes[0, 0].bar(range(len(top_indices)), feature_diff[top_indices], color=colors)
                axes[0, 0].set_xlabel('Features')
                axes[0, 0].set_ylabel('Mean Difference (High Error - Low Error)')
                axes[0, 0].set_title(f'{model_name}: Top Problematic Features\n(Red: p<0.01, Orange: p<0.05)')
                axes[0, 0].set_xticks(range(len(top_indices)))
                axes[0, 0].set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Effect sizes
                axes[0, 1].bar(range(len(top_indices)), effect_sizes[top_indices], color='purple', alpha=0.7)
                axes[0, 1].set_xlabel('Features')
                axes[0, 1].set_ylabel('Effect Size (Cohen\'s d)')
                axes[0, 1].set_title(f'{model_name}: Effect Sizes')
                axes[0, 1].set_xticks(range(len(top_indices)))
                axes[0, 1].set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right')
                axes[0, 1].grid(True, alpha=0.3)
                
                # P-value distribution
                axes[1, 0].hist(p_values, bins=20, alpha=0.7, color='blue')
                axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
                axes[1, 0].axvline(x=0.01, color='darkred', linestyle='--', label='p=0.01')
                axes[1, 0].set_xlabel('P-value')
                axes[1, 0].set_ylabel('Number of Features')
                axes[1, 0].set_title(f'{model_name}: P-value Distribution')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Feature importance vs error correlation
                if len(top_indices) > 0:
                    axes[1, 1].scatter(effect_sizes[top_indices], np.abs(feature_diff[top_indices]), alpha=0.7)
                    axes[1, 1].set_xlabel('Effect Size')
                    axes[1, 1].set_ylabel('Absolute Feature Difference')
                    axes[1, 1].set_title(f'{model_name}: Effect Size vs Feature Difference')
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'./Section4_ErrorAnalysis/{model_name}_feature_error_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()

def analyze_model_specific_errors(models, predictions, error_metrics, y_test, feature_names):
    """Analyze model-specific error patterns and characteristics."""
    print("\n🔍 MODEL-SPECIFIC ERROR ANALYSIS")
    print("=" * 80)
    
    model_insights = {}
    
    for model_name in predictions.keys():
        print(f"\n📊 {model_name.upper()} ERROR ANALYSIS")
        print("-" * 60)
        
        metrics = error_metrics[model_name]
        residuals = metrics['residuals']
        y_pred = predictions[model_name]
        
        # Basic error statistics
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"R²: {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Max Error: {metrics['max_error']:.4f}")
        print(f"95th Percentile Error: {metrics['q95_error']:.4f}")
        
        # Residual characteristics
        print(f"\nResidual Analysis:")
        print(f"  Mean: {metrics['residual_mean']:.4f}")
        print(f"  Std: {metrics['residual_std']:.4f}")
        print(f"  Skewness: {metrics['residual_skew']:.4f}")
        print(f"  Kurtosis: {metrics['residual_kurtosis']:.4f}")
        
        # Model-specific insights
        insights = []
        
        if 'XGBoost' in model_name:
            insights.extend([
                "🌳 XGBoost-Specific Insights:",
                "  - Tree-based model captures non-linear relationships well",
                "  - Large errors may indicate insufficient trees or depth",
                "  - Consider feature engineering for better performance",
                "  - Residual patterns suggest potential overfitting if highly skewed"
            ])
            
        elif 'FT-Transformer' in model_name:
            insights.extend([
                "🤖 FT-Transformer-Specific Insights:",
                "  - Attention mechanism captures feature interactions",
                "  - Large errors may indicate insufficient training or poor tokenization",
                "  - Consider analyzing attention weights for problematic samples",
                "  - Feature embedding quality affects prediction accuracy",
                "  - Deep architecture may overfit with limited data"
            ])
            
        elif 'SAINT' in model_name:
            insights.extend([
                "🎯 SAINT-Specific Insights:",
                "  - Self-attention captures inter-feature relationships",
                "  - Large errors may indicate poor attention patterns",
                "  - Consider adjusting attention heads or layers",
                "  - Regularization may help reduce overfitting",
                "  - Feature normalization affects attention computation"
            ])
        
        # Print insights
        for insight in insights:
            print(insight)
        
        model_insights[model_name] = {
            'metrics': metrics,
            'insights': insights,
            'error_characteristics': {
                'high_bias': abs(metrics['residual_mean']) > 0.1,
                'high_variance': metrics['residual_std'] > np.std(y_test) * 0.5,
                'skewed_errors': abs(metrics['residual_skew']) > 1.0,
                'heavy_tails': abs(metrics['residual_kurtosis']) > 3.0
            }
        }
    
    return model_insights

def generate_business_insights(error_metrics, predictions, feature_error_analysis, feature_names):
    """Generate actionable business insights from error analysis."""
    print("\n💼 BUSINESS INSIGHTS FROM ERROR ANALYSIS")
    print("=" * 80)
    
    # Overall model performance ranking
    model_performance = []
    for model_name, metrics in error_metrics.items():
        model_performance.append({
            'model': model_name,
            'rmse': metrics['RMSE'],
            'mae': metrics['MAE'],
            'r2': metrics['R2']
        })
    
    # Sort by R² score (descending)
    model_performance.sort(key=lambda x: x['r2'], reverse=True)
    
    print("\n🏆 Model Performance Ranking:")
    for i, model in enumerate(model_performance):
        print(f"   {i+1}. {model['model']}: R²={model['r2']:.4f}, RMSE={model['rmse']:.4f}")
    
    # Best performing model
    best_model = model_performance[0]['model']
    best_r2 = model_performance[0]['r2']
    best_rmse = model_performance[0]['rmse']
    
    print(f"\n🎯 Recommended Model: {best_model}")
    print(f"   - Explains {best_r2*100:.1f}% of variance in bike sharing demand")
    print(f"   - Average prediction error: ±{best_rmse:.2f} bikes")
    
    # Feature insights across models
    print(f"\n📊 Feature-Based Error Patterns:")
    
    # Aggregate feature importance across models
    feature_importance_scores = {}
    for model_name, analysis in feature_error_analysis.items():
        if len(analysis['significant_features']) > 0:
            for idx in analysis['significant_features']:
                feature_name = feature_names[idx]
                effect_size = analysis['effect_sizes'][idx]
                
                if feature_name not in feature_importance_scores:
                    feature_importance_scores[feature_name] = []
                feature_importance_scores[feature_name].append(effect_size)
    
    # Average effect sizes
    avg_feature_effects = {}
    for feature, effects in feature_importance_scores.items():
        avg_feature_effects[feature] = np.mean(effects)
    
    # Sort by average effect size
    sorted_features = sorted(avg_feature_effects.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_features:
        print(f"   Most problematic features (causing high errors):")
        for i, (feature, effect) in enumerate(sorted_features[:5]):
            print(f"     {i+1}. {feature}: Average effect size = {effect:.3f}")
    
    # Business recommendations
    print(f"\n📋 Business Recommendations:")
    print(f"   1. Deploy {best_model} for production bike sharing demand prediction")
    print(f"   2. Monitor prediction accuracy especially during extreme weather conditions")
    print(f"   3. Focus data collection efforts on improving problematic features")
    print(f"   4. Consider ensemble methods combining top 2-3 models for robustness")
    print(f"   5. Implement real-time model monitoring to detect performance degradation")
    
    if sorted_features:
        print(f"   6. Pay special attention to {sorted_features[0][0]} as it's most error-prone")
    
    # Operational insights
    print(f"\n🚴 Operational Insights:")
    worst_model = model_performance[-1]['model']
    worst_rmse = model_performance[-1]['rmse']
    improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
    
    print(f"   - Using {best_model} instead of {worst_model} reduces prediction error by {improvement:.1f}%")
    print(f"   - Better predictions lead to improved bike distribution and customer satisfaction")
    print(f"   - Reduced operational costs through optimized bike placement")
    
    return {
        'best_model': best_model,
        'model_ranking': model_performance,
        'problematic_features': sorted_features,
        'improvement_potential': improvement
    }

def save_error_analysis_results(error_metrics, predictions, feature_error_analysis, comparison_df, business_insights):
    """Save all error analysis results to files."""
    print("\n💾 Saving error analysis results...")
    
    # Create results directory
    os.makedirs('./Section4_ErrorAnalysis', exist_ok=True)
    
    # Save error metrics
    error_summary = []
    for model_name, metrics in error_metrics.items():
        error_summary.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],
            'MAPE': metrics['MAPE'],
            'Max_Error': metrics['max_error'],
            'Q95_Error': metrics['q95_error'],
            'Residual_Mean': metrics['residual_mean'],
            'Residual_Std': metrics['residual_std'],
            'Residual_Skew': metrics['residual_skew'],
            'Residual_Kurtosis': metrics['residual_kurtosis']
        })
    
    error_summary_df = pd.DataFrame(error_summary)
    error_summary_df.to_csv('./Section4_ErrorAnalysis/error_metrics_summary.csv', index=False)
    
    # Save feature error analysis
    for model_name, analysis in feature_error_analysis.items():
        if len(analysis['significant_features']) > 0:
            feature_analysis_df = pd.DataFrame({
                'Feature': [f"Feature_{i}" for i in range(len(analysis['feature_differences']))],
                'Feature_Name': [f"Unknown_{i}" for i in range(len(analysis['feature_differences']))],  # Will be updated if feature_names available
                'Mean_Difference': analysis['feature_differences'],
                'P_Value': analysis['p_values'],
                'Effect_Size': analysis['effect_sizes'],
                'Significant': analysis['p_values'] < 0.05
            })
            
            feature_analysis_df = feature_analysis_df.sort_values('Effect_Size', ascending=False)
            feature_analysis_df.to_csv(f'./Section4_ErrorAnalysis/{model_name}_feature_error_analysis.csv', index=False)
    
    # Save business insights
    with open('./Section4_ErrorAnalysis/business_insights.txt', 'w') as f:
        f.write("BIKE SHARING DEMAND PREDICTION - ERROR ANALYSIS INSIGHTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Model: {business_insights['best_model']}\n")
        f.write(f"Improvement over worst model: {business_insights['improvement_potential']:.1f}%\n\n")
        
        f.write("Model Ranking:\n")
        for i, model in enumerate(business_insights['model_ranking']):
            f.write(f"  {i+1}. {model['model']}: R²={model['r2']:.4f}\n")
        
        if business_insights['problematic_features']:
            f.write("\nMost Problematic Features:\n")
            for i, (feature, effect) in enumerate(business_insights['problematic_features'][:5]):
                f.write(f"  {i+1}. {feature}: {effect:.3f}\n")
    
    print("   ✅ Error metrics summary saved")
    print("   ✅ Feature error analysis saved")
    print("   ✅ Business insights saved")
    print("   ✅ Cross-model comparison saved")
    
    return './Section4_ErrorAnalysis/'

def run_complete_error_analysis(section2_results_file='./bike_sharing_section2_results.pkl',
                               preprocessed_data_file='./bike_sharing_preprocessed_data.pkl'):
    """Run the complete error analysis pipeline."""
    print("🚀 STARTING COMPLETE ERROR ANALYSIS PIPELINE")
    print("=" * 80)
    
    try:
        # Step 1: Setup environment
        setup_error_analysis_environment()
        
        # Step 2: Load data and models
        data = load_regression_models_and_data()
        
        if not data['models']:
            print("❌ No models available for analysis")
            return None
        
        # Step 3: Generate predictions
        predictions, prediction_details = generate_model_predictions(
            data['models'], data['X_test_scaled'], data['device']
        )
        
        if not predictions:
            print("❌ No predictions generated")
            return None
        
        # Step 4: Calculate error metrics
        error_metrics = calculate_regression_errors(predictions, data['y_test'])
        
        # Step 5: Analyze residuals
        residual_analysis = analyze_residuals(error_metrics, predictions, data['y_test'])
        
        # Step 6: Cross-model comparison
        comparison_df = perform_cross_model_error_comparison(error_metrics)
        
        # Step 7: Feature-based error analysis
        feature_error_analysis = analyze_feature_based_errors(
            data['models'], predictions, error_metrics, 
            data['X_test_scaled'], data['y_test'], data['feature_names']
        )
        
        # Step 8: Generate visualizations
        generate_error_visualizations(
            error_metrics, predictions, data['y_test'], 
            feature_error_analysis, data['feature_names']
        )
        
        # Step 9: Model-specific analysis
        model_insights = analyze_model_specific_errors(
            data['models'], predictions, error_metrics, 
            data['y_test'], data['feature_names']
        )
        
        # Step 10: Business insights
        business_insights = generate_business_insights(
            error_metrics, predictions, feature_error_analysis, data['feature_names']
        )
        
        # Step 11: Save results
        results_dir = save_error_analysis_results(
            error_metrics, predictions, feature_error_analysis, 
            comparison_df, business_insights
        )
        
        print("\n🎉 COMPLETE ERROR ANALYSIS FINISHED!")
        print("=" * 80)
        print(f"📊 Models analyzed: {list(predictions.keys())}")
        print(f"📁 Results saved to: {results_dir}")
        print(f"🏆 Best model: {business_insights['best_model']}")
        
        return {
            'error_metrics': error_metrics,
            'predictions': predictions,
            'feature_error_analysis': feature_error_analysis,
            'comparison_df': comparison_df,
            'business_insights': business_insights,
            'model_insights': model_insights,
            'results_dir': results_dir,
            'models_analyzed': list(predictions.keys())
        }
        
    except Exception as e:
        print(f"❌ Error analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Additional utility functions for specific analyses

def compare_error_patterns_across_models(error_metrics, predictions, y_test):
    """Compare specific error patterns across models."""
    print("\n🔄 Comparing error patterns across models...")
    
    pattern_analysis = {}
    
    for model_name in predictions.keys():
        residuals = error_metrics[model_name]['residuals']
        y_pred = predictions[model_name]
        
        # Error patterns
        patterns = {
            'underestimation_bias': np.mean(residuals > 0),  # Fraction of positive residuals
            'overestimation_bias': np.mean(residuals < 0),   # Fraction of negative residuals
            'high_demand_error': np.mean(np.abs(residuals[y_test > np.percentile(y_test, 75)])),
            'low_demand_error': np.mean(np.abs(residuals[y_test < np.percentile(y_test, 25)])),
            'medium_demand_error': np.mean(np.abs(residuals[
                (y_test >= np.percentile(y_test, 25)) & 
                (y_test <= np.percentile(y_test, 75))
            ]))
        }
        
        pattern_analysis[model_name] = patterns
        
        print(f"   {model_name}:")
        print(f"     Underestimation tendency: {patterns['underestimation_bias']*100:.1f}%")
        print(f"     High demand error: {patterns['high_demand_error']:.4f}")
        print(f"     Low demand error: {patterns['low_demand_error']:.4f}")
    
    return pattern_analysis

def analyze_temporal_error_patterns(error_metrics, predictions, y_test, time_features=None):
    """Analyze error patterns over time (if temporal features available)."""
    print("\n⏰ Analyzing temporal error patterns...")
    
    # This would require temporal features to be identified
    # For now, we'll analyze error patterns by prediction magnitude
    
    temporal_analysis = {}
    
    for model_name in predictions.keys():
        residuals = error_metrics[model_name]['residuals']
        abs_residuals = np.abs(residuals)
        
        # Analyze errors by prediction ranges
        y_pred = predictions[model_name]
        
        # Define prediction ranges
        low_pred = y_pred < np.percentile(y_pred, 33)
        med_pred = (y_pred >= np.percentile(y_pred, 33)) & (y_pred < np.percentile(y_pred, 67))
        high_pred = y_pred >= np.percentile(y_pred, 67)
        
        temporal_analysis[model_name] = {
            'low_prediction_error': np.mean(abs_residuals[low_pred]),
            'medium_prediction_error': np.mean(abs_residuals[med_pred]),
            'high_prediction_error': np.mean(abs_residuals[high_pred])
        }
        
        print(f"   {model_name}:")
        print(f"     Low prediction range error: {temporal_analysis[model_name]['low_prediction_error']:.4f}")
        print(f"     Medium prediction range error: {temporal_analysis[model_name]['medium_prediction_error']:.4f}")
        print(f"     High prediction range error: {temporal_analysis[model_name]['high_prediction_error']:.4f}")
    
    return temporal_analysis
