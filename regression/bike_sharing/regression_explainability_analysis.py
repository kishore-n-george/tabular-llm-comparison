"""
Regression Explainability Analysis for Bike Sharing Dataset

This module provides comprehensive explainability analysis for regression models
including feature importance, SHAP analysis, LIME analysis, and ablation studies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from copy import deepcopy

# ML libraries
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import torch

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not available. SHAP analysis will be skipped.")

# LIME for explainability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME library not available. LIME analysis will be skipped.")

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_logging(save_dir='./Section3_Explainability'):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{save_dir}/explainability_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clear_memory():
    """Gently clear memory and run garbage collection"""
    import gc
    
    # Single garbage collection
    gc.collect()
    
    # Clean up any existing GPU memory gently
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check GPU memory
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

def save_intermediate_results(data: Dict, filename: str):
    """Save intermediate results to pickle file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Intermediate results saved to '{filename}'")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save intermediate results: {e}")

def load_intermediate_results(filename: str):
    """Load intermediate results from pickle file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"⚠️ Warning: File '{filename}' not found")
            return None
    except Exception as e:
        print(f"⚠️ Warning: Failed to load intermediate results: {e}")
        return None

class RegressionExplainabilityAnalyzer:
    """
    Comprehensive explainability analyzer for regression models
    """
    
    def __init__(self, feature_names: List[str], save_dir: str = './Section3_Explainability'):
        self.feature_names = feature_names
        self.save_dir = save_dir
        self.explanations = {}
        self.logger = setup_logging(save_dir)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Initialized RegressionExplainabilityAnalyzer with {len(feature_names)} features")
    
    def analyze_model_explainability(self, model, model_name: str, 
                                   X_train, X_test, y_train, y_test,
                                   max_samples: int = 200):
        """
        Comprehensive explainability analysis for a regression model
        """
        self.logger.info(f"Starting explainability analysis for {model_name}")
        print(f"\n🔍 Analyzing {model_name} explainability...")
        
        # Initialize results dictionary
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': {},
            'feature_importance': {},
            'permutation_importance': {},
            'shap_analysis': {},
            'lime_analysis': {}
        }
        
        try:
            # 1. Baseline performance
            results['baseline_metrics'] = self._get_baseline_metrics(
                model, X_train, X_test, y_train, y_test
            )
            
            # 2. Built-in feature importance (if available)
            results['feature_importance'] = self._get_feature_importance(
                model, model_name, X_train, y_train
            )
            
            # 3. Permutation importance
            results['permutation_importance'] = self._get_permutation_importance(
                model, X_test, y_test, max_samples
            )
            
            # 4. SHAP analysis
            if SHAP_AVAILABLE:
                results['shap_analysis'] = self._get_shap_analysis(
                    model, model_name, X_train, X_test, max_samples
                )
            
            # 5. LIME analysis
            if LIME_AVAILABLE:
                results['lime_analysis'] = self._get_lime_analysis(
                    model, X_train, X_test, max_samples
                )
            
            # Store results
            self.explanations[model_name] = results
            
            # Save individual model results
            self._save_model_results(model_name, results)
            
            self.logger.info(f"Completed explainability analysis for {model_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in explainability analysis for {model_name}: {e}")
            print(f"❌ Error analyzing {model_name}: {e}")
            return results
    
    def _get_baseline_metrics(self, model, X_train, X_test, y_train, y_test):
        """Get baseline regression metrics"""
        try:
            # Fit model if needed
            if hasattr(model, 'fit'):
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
            else:
                train_time = 0
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Calculate regression metrics
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'explained_variance': explained_variance_score(y_test, y_pred),
                'train_time': train_time,
                'inference_time': inference_time,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
            # Calculate MAPE safely
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
            except:
                metrics['mape'] = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
            
            self.logger.info(f"Baseline metrics calculated - R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating baseline metrics: {e}")
            return {}
    
    def _get_feature_importance(self, model, model_name: str, X_train, y_train):
        """Get built-in feature importance if available"""
        try:
            importance_data = {}
            
            # XGBoost feature importance
            if hasattr(model, 'feature_importances_'):
                importance_data['importances'] = model.feature_importances_
                importance_data['method'] = 'built_in'
                
                # Create feature importance DataFrame
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_data['feature_ranking'] = feature_importance_df.to_dict('records')
                
                # Plot feature importance
                self._plot_feature_importance(
                    feature_importance_df, model_name, 'Built-in Feature Importance'
                )
                
                self.logger.info(f"Built-in feature importance extracted for {model_name}")
                
            elif 'XGBoost' in model_name and hasattr(model, 'get_booster'):
                # XGBoost specific importance types
                importance_types = ['weight', 'gain', 'cover']
                for imp_type in importance_types:
                    try:
                        importance_scores = model.get_booster().get_score(importance_type=imp_type)
                        
                        # Convert to array format
                        importances = np.zeros(len(self.feature_names))
                        for i, feature_name in enumerate(self.feature_names):
                            feature_key = f'f{i}'
                            importances[i] = importance_scores.get(feature_key, 0.0)
                        
                        importance_data[f'{imp_type}_importance'] = importances
                        
                        # Create DataFrame and plot
                        feature_importance_df = pd.DataFrame({
                            'feature': self.feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        self._plot_feature_importance(
                            feature_importance_df, model_name, f'XGBoost {imp_type.title()} Importance'
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"Could not extract {imp_type} importance: {e}")
            
            return importance_data
            
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {e}")
            return {}
    
    def _get_permutation_importance(self, model, X_test, y_test, max_samples: int):
        """Calculate permutation importance"""
        try:
            print("      📊 Calculating permutation importance...")
            
            # Limit samples for efficiency
            if len(X_test) > max_samples:
                indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_test_sample = X_test[indices]
                y_test_sample = y_test[indices]
            else:
                X_test_sample = X_test
                y_test_sample = y_test
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test_sample, y_test_sample,
                n_repeats=5, random_state=42, scoring='r2'
            )
            
            importance_data = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std,
                'method': 'permutation',
                'n_samples_used': len(X_test_sample)
            }
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            importance_data['feature_ranking'] = feature_importance_df.to_dict('records')
            
            # Plot permutation importance
            self._plot_permutation_importance(feature_importance_df)
            
            self.logger.info(f"Permutation importance calculated using {len(X_test_sample)} samples")
            return importance_data
            
        except Exception as e:
            self.logger.error(f"Error calculating permutation importance: {e}")
            return {}
    
    def _get_shap_analysis(self, model, model_name: str, X_train, X_test, max_samples: int):
        """Perform memory-aware SHAP analysis"""
        try:
            print("      🎯 Performing SHAP analysis...")
            
            # Check available GPU memory and adjust strategy
            use_cpu_for_shap = self._should_use_cpu_for_shap(model_name)
            adaptive_samples = self._get_adaptive_sample_size(model_name, max_samples)
            
            print(f"         Using {'CPU' if use_cpu_for_shap else 'GPU'} for SHAP with {adaptive_samples} samples")
            
            # Limit samples for memory efficiency
            if len(X_train) > adaptive_samples:
                train_indices = np.random.choice(len(X_train), adaptive_samples, replace=False)
                X_train_sample = X_train[train_indices]
            else:
                X_train_sample = X_train
            
            # Use even smaller test sample for memory-intensive models
            test_sample_size = min(adaptive_samples // 2, len(X_test)) if 'Transformer' in model_name else min(adaptive_samples, len(X_test))
            if len(X_test) > test_sample_size:
                test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
                X_test_sample = X_test[test_indices]
            else:
                X_test_sample = X_test
            
            shap_data = {}
            
            # Clear GPU memory before SHAP analysis
            self._clear_gpu_memory()
            
            # Choose appropriate SHAP explainer with memory management
            if 'XGBoost' in model_name:
                # XGBoost TreeExplainer is memory efficient
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                shap_data['explainer_type'] = 'TreeExplainer'
                n_analyzed = len(X_test_sample)
            else:
                # For neural networks, use CPU-based KernelExplainer with small samples
                print("         Using CPU-based KernelExplainer for neural network")
                
                # Move data to CPU for SHAP analysis
                if hasattr(X_train_sample, 'cpu'):
                    X_train_cpu = X_train_sample.cpu().numpy()
                    X_test_cpu = X_test_sample.cpu().numpy()
                else:
                    X_train_cpu = X_train_sample
                    X_test_cpu = X_test_sample
                
                # Create CPU-based model wrapper
                cpu_model_wrapper = self._create_cpu_model_wrapper(model, model_name)
                
                # Use very small sample for KernelExplainer to avoid memory issues
                kernel_samples = min(20, len(X_test_cpu))  # Very conservative
                explainer = shap.KernelExplainer(cpu_model_wrapper.predict, X_train_cpu[:50])  # Small background
                shap_values = explainer.shap_values(X_test_cpu[:kernel_samples])
                shap_data['explainer_type'] = 'KernelExplainer_CPU'
                n_analyzed = kernel_samples
            
            shap_data['shap_values'] = shap_values
            shap_data['base_value'] = explainer.expected_value
            shap_data['n_samples_analyzed'] = n_analyzed
            
            # Calculate feature importance from SHAP values
            feature_importance = np.abs(shap_values).mean(axis=0)
            shap_data['feature_importance'] = feature_importance
            
            # Create SHAP plots with memory management
            try:
                self._create_shap_plots_safe(shap_values, X_test_sample[:n_analyzed], model_name, explainer.expected_value)
            except Exception as plot_error:
                self.logger.warning(f"SHAP plotting failed: {plot_error}, continuing without plots")
            
            # Clear memory after SHAP analysis
            self._clear_gpu_memory()
            
            self.logger.info(f"SHAP analysis completed for {model_name} with {n_analyzed} samples")
            return shap_data
            
        except Exception as e:
            self.logger.error(f"Error in SHAP analysis: {e}")
            print(f"         ⚠️ SHAP analysis failed: {e}")
            print(f"         Continuing with other explainability methods...")
            
            # Clear memory on error
            self._clear_gpu_memory()
            return {}
    
    def _get_lime_analysis(self, model, X_train, X_test, max_samples: int):
        """Perform LIME analysis"""
        try:
            print("      🍋 Performing LIME analysis...")
            
            # Limit samples for efficiency
            if len(X_train) > max_samples:
                train_indices = np.random.choice(len(X_train), max_samples, replace=False)
                X_train_sample = X_train[train_indices]
            else:
                X_train_sample = X_train
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_sample,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True
            )
            
            lime_data = {
                'explainer_type': 'LimeTabular',
                'explanations': [],
                'feature_importance_summary': {}
            }
            
            # Explain a few test instances
            n_explain = min(5, len(X_test))
            all_feature_importance = {feature: [] for feature in self.feature_names}
            
            for i in range(n_explain):
                try:
                    explanation = explainer.explain_instance(
                        X_test[i], model.predict, num_features=len(self.feature_names)
                    )
                    
                    # Extract feature importance
                    exp_data = {
                        'instance_idx': i,
                        'prediction': model.predict(X_test[i:i+1])[0],
                        'feature_importance': {}
                    }
                    
                    # explanation.as_list() returns list of tuples (feature_name, importance_value)
                    # For regression, LIME returns feature names directly, not indices
                    for feature_name_or_idx, importance in explanation.as_list():
                        # Handle both cases: feature name (string) or feature index (int)
                        if isinstance(feature_name_or_idx, str):
                            feature_name = feature_name_or_idx
                        else:
                            # If it's an index, get the feature name
                            feature_name = self.feature_names[feature_name_or_idx]
                        
                        exp_data['feature_importance'][feature_name] = importance
                        all_feature_importance[feature_name].append(abs(importance))
                    
                    lime_data['explanations'].append(exp_data)
                    
                    # Save LIME plot
                    fig = explanation.as_pyplot_figure()
                    fig.savefig(f'{self.save_dir}/LIME_explanation_{i+1}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    self.logger.warning(f"Error explaining instance {i}: {e}")
            
            # Calculate average feature importance
            for feature, importances in all_feature_importance.items():
                if importances:
                    lime_data['feature_importance_summary'][feature] = {
                        'mean_importance': np.mean(importances),
                        'std_importance': np.std(importances)
                    }
            
            self.logger.info(f"LIME analysis completed for {n_explain} instances")
            return lime_data
            
        except Exception as e:
            self.logger.error(f"Error in LIME analysis: {e}")
            return {}
    
    def _plot_feature_importance(self, feature_df: pd.DataFrame, model_name: str, title: str):
        """Plot feature importance"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Select top 15 features
            top_features = feature_df.head(15)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'{model_name} - {title}')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (idx, row) in enumerate(top_features.iterrows()):
                plt.text(row['importance'] + max(top_features['importance']) * 0.01, i,
                        f'{row["importance"]:.4f}', va='center')
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/{model_name}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
    
    def _plot_permutation_importance(self, feature_df: pd.DataFrame):
        """Plot permutation importance with error bars"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Select top 15 features
            top_features = feature_df.head(15)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'],
                           xerr=top_features['std'], capsize=5)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Permutation Importance (R² decrease)')
            plt.title('Permutation Feature Importance')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (idx, row) in enumerate(top_features.iterrows()):
                plt.text(row['importance'] + max(top_features['importance']) * 0.01, i,
                        f'{row["importance"]:.4f}', va='center')
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/permutation_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting permutation importance: {e}")
    
    def _create_shap_plots(self, shap_values, X_test_sample, model_name: str, expected_value):
        """Create SHAP visualization plots"""
        try:
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_sample, feature_names=self.feature_names, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/{model_name}_shap_summary.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_sample, feature_names=self.feature_names, 
                             plot_type="bar", show=False)
            plt.title(f'{model_name} - SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/{model_name}_shap_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP plots: {e}")
    
    def _save_model_results(self, model_name: str, results: Dict):
        """Save individual model results"""
        try:
            filename = f'{self.save_dir}/{model_name}_explainability_results.pkl'
            save_intermediate_results(results, filename)
            
            # Also save as CSV for easy inspection
            if 'permutation_importance' in results and 'feature_ranking' in results['permutation_importance']:
                perm_df = pd.DataFrame(results['permutation_importance']['feature_ranking'])
                perm_df.to_csv(f'{self.save_dir}/{model_name}_permutation_importance.csv', index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving model results: {e}")
    
    def compare_feature_importance(self, explanations: Dict = None):
        """Compare feature importance across models"""
        try:
            if explanations is None:
                explanations = self.explanations
            
            if not explanations:
                self.logger.warning("No explanations available for comparison")
                return None
            
            print("\n🔄 Comparing feature importance across models...")
            
            # Collect feature importance from all models
            importance_data = {}
            
            for model_name, results in explanations.items():
                # Try permutation importance first
                if ('permutation_importance' in results and 
                    'importances_mean' in results['permutation_importance']):
                    importance_data[f'{model_name}_permutation'] = results['permutation_importance']['importances_mean']
                
                # Try built-in feature importance
                if ('feature_importance' in results and 
                    'importances' in results['feature_importance']):
                    importance_data[f'{model_name}_builtin'] = results['feature_importance']['importances']
                
                # Try SHAP importance
                if ('shap_analysis' in results and 
                    'feature_importance' in results['shap_analysis']):
                    importance_data[f'{model_name}_shap'] = results['shap_analysis']['feature_importance']
            
            if not importance_data:
                self.logger.warning("No feature importance data found for comparison")
                return None
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(importance_data, index=self.feature_names)
            
            # Normalize importances for comparison
            for col in comparison_df.columns:
                if comparison_df[col].sum() > 0:
                    comparison_df[col] = comparison_df[col] / comparison_df[col].sum()
            
            # Save comparison
            comparison_df.to_csv(f'{self.save_dir}/feature_importance_comparison.csv')
            
            # Create comparison visualization
            self._plot_feature_importance_comparison(comparison_df)
            
            self.logger.info("Feature importance comparison completed")
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing feature importance: {e}")
            return None
    
    def _plot_feature_importance_comparison(self, comparison_df: pd.DataFrame):
        """Plot feature importance comparison"""
        try:
            # Select top features based on average importance
            avg_importance = comparison_df.mean(axis=1).sort_values(ascending=False)
            top_features = avg_importance.head(15).index
            
            plt.figure(figsize=(15, 10))
            comparison_subset = comparison_df.loc[top_features]
            
            # Create grouped bar plot
            comparison_subset.plot(kind='bar', figsize=(15, 8))
            plt.title('Top 15 Feature Importance Comparison Across Models')
            plt.xlabel('Features')
            plt.ylabel('Normalized Importance Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/feature_importance_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            correlation_matrix = comparison_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
            plt.title('Feature Importance Method Correlation')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/feature_importance_correlation.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance comparison: {e}")
    
    def generate_explanation_report(self, model_name: str):
        """Generate detailed explanation report for a model"""
        try:
            if model_name not in self.explanations:
                print(f"❌ No explanations available for {model_name}")
                return
            
            results = self.explanations[model_name]
            
            print(f"\n📋 EXPLANATION REPORT: {model_name}")
            print("=" * 60)
            
            # Baseline metrics
            if 'baseline_metrics' in results:
                metrics = results['baseline_metrics']
                print(f"\n📊 Baseline Performance:")
                print(f"   R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
                print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                print(f"   MAE: {metrics.get('mae', 'N/A'):.4f}")
                print(f"   MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                print(f"   Training Time: {metrics.get('train_time', 'N/A'):.3f}s")
                print(f"   Inference Time: {metrics.get('inference_time', 'N/A'):.3f}s")
            
            # Feature importance
            if ('permutation_importance' in results and 
                'feature_ranking' in results['permutation_importance']):
                
                print(f"\n🎯 Top 10 Most Important Features (Permutation):")
                for i, feature_data in enumerate(results['permutation_importance']['feature_ranking'][:10]):
                    print(f"   {i+1:2d}. {feature_data['feature']}: {feature_data['importance']:.4f}")
            
            # Built-in importance
            if ('feature_importance' in results and 
                'feature_ranking' in results['feature_importance']):
                
                print(f"\n🏗️ Top 10 Most Important Features (Built-in):")
                for i, feature_data in enumerate(results['feature_importance']['feature_ranking'][:10]):
                    print(f"   {i+1:2d}. {feature_data['feature']}: {feature_data['importance']:.4f}")
            
            # SHAP analysis
            if 'shap_analysis' in results:
                shap_data = results['shap_analysis']
                print(f"\n🎯 SHAP Analysis:")
                print(f"   Explainer Type: {shap_data.get('explainer_type', 'N/A')}")
                print(f"   Samples Analyzed: {shap_data.get('n_samples_analyzed', 'N/A')}")
                if 'feature_importance' in shap_data:
                    # Show top SHAP features
                    shap_importance = shap_data['feature_importance']
                    top_indices = np.argsort(shap_importance)[::-1][:5]
                    print(f"   Top 5 SHAP Features:")
                    for i, idx in enumerate(top_indices):
                        print(f"      {i+1}. {self.feature_names[idx]}: {shap_importance[idx]:.4f}")
            
            # LIME analysis
            if 'lime_analysis' in results:
                lime_data = results['lime_analysis']
                print(f"\n🍋 LIME Analysis:")
                print(f"   Instances Explained: {len(lime_data.get('explanations', []))}")
                if 'feature_importance_summary' in lime_data:
                    # Show top LIME features
                    lime_summary = lime_data['feature_importance_summary']
                    sorted_features = sorted(lime_summary.items(), 
                                           key=lambda x: x[1]['mean_importance'], reverse=True)
                    print(f"   Top 5 LIME Features:")
                    for i, (feature, data) in enumerate(sorted_features[:5]):
                        print(f"      {i+1}. {feature}: {data['mean_importance']:.4f} ± {data['std_importance']:.4f}")
            
            print(f"\n✅ Report completed for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error generating explanation report: {e}")
    
    def _should_use_cpu_for_shap(self, model_name: str) -> bool:
        """Determine if SHAP analysis should use CPU to avoid memory issues"""
        # Use CPU for transformer models to avoid GPU memory issues
        if 'Transformer' in model_name or 'SAINT' in model_name:
            return True
        
        # Check GPU memory availability
        if torch.cuda.is_available():
            try:
                gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                # If less than 2GB free, use CPU
                if gpu_memory_free < 2 * 1024**3:  # 2GB in bytes
                    return True
            except:
                return True
        
        return False
    
    def _get_adaptive_sample_size(self, model_name: str, base_samples: int) -> int:
        """Get adaptive sample size based on model type and available memory"""
        if 'XGBoost' in model_name:
            # XGBoost is memory efficient
            return min(base_samples, 200)
        elif 'Transformer' in model_name or 'SAINT' in model_name:
            # Transformer models need smaller samples
            return min(base_samples // 4, 50)
        else:
            return min(base_samples // 2, 100)
    
    def _clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
    
    def _create_cpu_model_wrapper(self, model, model_name: str):
        """Create a CPU-based wrapper for the model"""
        class CPUModelWrapper:
            def __init__(self, model, model_name):
                self.model = model
                self.model_name = model_name
                
            def predict(self, X):
                try:
                    # Ensure model is in eval mode for PyTorch models
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    
                    # Handle different model types
                    if hasattr(self.model, 'predict'):
                        # Standard sklearn-like interface
                        predictions = self.model.predict(X)
                        # Ensure predictions are always 1D array
                        if isinstance(predictions, np.ndarray):
                            return predictions.flatten()
                        return np.array([predictions]).flatten()
                    else:
                        # PyTorch model - move to CPU for prediction
                        with torch.no_grad():
                            if isinstance(X, np.ndarray):
                                X_tensor = torch.FloatTensor(X).cpu()
                            else:
                                X_tensor = X.cpu()
                            
                            # Move model to CPU temporarily
                            device_backup = next(self.model.parameters()).device
                            self.model.cpu()
                            
                            try:
                                if 'Transformer' in self.model_name:
                                    predictions = self.model(X_tensor, None)
                                else:  # SAINT
                                    predictions = self.model(X_tensor)
                                
                                # Handle different prediction shapes
                                if isinstance(predictions, torch.Tensor):
                                    predictions = predictions.cpu().numpy()
                                
                                # Ensure predictions are properly shaped
                                if predictions.ndim == 0:
                                    # Single scalar prediction
                                    predictions = np.array([predictions])
                                elif predictions.ndim > 1:
                                    # Multi-dimensional - squeeze to 1D
                                    predictions = predictions.squeeze()
                                    if predictions.ndim == 0:
                                        predictions = np.array([predictions])
                                
                                # Ensure we have the right number of predictions
                                if len(predictions) != len(X):
                                    if len(predictions) == 1 and len(X) > 1:
                                        # Single prediction for multiple inputs - replicate
                                        predictions = np.full(len(X), predictions[0])
                                    else:
                                        # Fallback: return zeros
                                        predictions = np.zeros(len(X))
                                
                            finally:
                                # Move model back to original device
                                self.model.to(device_backup)
                            
                            return predictions.flatten()
                            
                except Exception as e:
                    print(f"Error in CPU model wrapper for {self.model_name}: {e}")
                    # Fallback: return zeros with correct shape
                    return np.zeros(len(X))
        
        return CPUModelWrapper(model, model_name)
    
    def _create_shap_plots_safe(self, shap_values, X_test_sample, model_name: str, expected_value):
        """Create SHAP visualization plots with memory management"""
        try:
            print(f"         Creating SHAP plots for {model_name}...")
            
            # Limit data size for plotting
            max_plot_samples = min(50, len(X_test_sample))
            if len(X_test_sample) > max_plot_samples:
                plot_indices = np.random.choice(len(X_test_sample), max_plot_samples, replace=False)
                X_plot = X_test_sample[plot_indices]
                shap_plot = shap_values[plot_indices]
            else:
                X_plot = X_test_sample
                shap_plot = shap_values
            
            # Summary plot with error handling
            try:
                plt.figure(figsize=(10, 6))  # Smaller figure to save memory
                shap.summary_plot(shap_plot, X_plot, feature_names=self.feature_names, 
                                show=False, max_display=10)  # Limit features displayed
                plt.title(f'{model_name} - SHAP Summary Plot (Top 10 Features)')
                plt.tight_layout()
                plt.savefig(f'{self.save_dir}/{model_name}_shap_summary.png', 
                           dpi=150, bbox_inches='tight')  # Lower DPI to save memory
                plt.close()  # Close immediately to free memory
                print(f"         ✓ SHAP summary plot saved")
            except Exception as e:
                print(f"         ⚠️ SHAP summary plot failed: {e}")
            
            # Feature importance bar plot
            try:
                plt.figure(figsize=(8, 6))
                feature_importance = np.abs(shap_plot).mean(axis=0)
                
                # Create simple bar plot instead of SHAP's summary_plot
                top_indices = np.argsort(feature_importance)[::-1][:10]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importance = feature_importance[top_indices]
                
                plt.barh(range(len(top_features)), top_importance)
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('Mean |SHAP Value|')
                plt.title(f'{model_name} - SHAP Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'{self.save_dir}/{model_name}_shap_importance.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                print(f"         ✓ SHAP importance plot saved")
            except Exception as e:
                print(f"         ⚠️ SHAP importance plot failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP plots: {e}")
            print(f"         ⚠️ SHAP plotting failed: {e}")
    
    def save_comprehensive_results(self, filename: str = None):
        """Save all explainability results"""
        try:
            if filename is None:
                filename = f'{self.save_dir}/comprehensive_explainability_results.pkl'
            
            comprehensive_data = {
                'explanations': self.explanations,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': {
                    'total_models': len(self.explanations),
                    'models_analyzed': list(self.explanations.keys()),
                    'features_analyzed': len(self.feature_names)
                }
            }
            
            save_intermediate_results(comprehensive_data, filename)
            self.logger.info(f"Comprehensive results saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {e}")
            return None

# Utility functions for running the analysis
def run_regression_explainability_analysis(models_dict, model_names, X_train, X_test, y_train, y_test, 
                                         feature_names, save_dir='./Section3_Explainability'):
    """
    Run comprehensive explainability analysis for regression models
    """
    analyzer = RegressionExplainabilityAnalyzer(feature_names, save_dir)
    
    for model_name in model_names:
        if model_name in models_dict:
            analyzer.analyze_model_explainability(
                models_dict[model_name], model_name,
                X_train, X_test, y_train, y_test
            )
    
    # Compare feature importance across models
    comparison_df = analyzer.compare_feature_importance()
    
    # Generate reports for each model
    for model_name in model_names:
        if model_name in models_dict:
            analyzer.generate_explanation_report(model_name)
    
    # Save comprehensive results
    analyzer.save_comprehensive_results()
    
    return analyzer, comparison_df

if __name__ == "__main__":
    print("Regression Explainability Analysis module loaded successfully!")
    print("Use run_regression_explainability_analysis() to perform comprehensive analysis.")
