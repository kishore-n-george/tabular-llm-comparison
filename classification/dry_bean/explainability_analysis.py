import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
import gc
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Memory management utilities
def clear_memory():
    """Clear memory and run garbage collection"""
    gc.collect()
    print("🧹 Memory cleared")

def save_intermediate_results(data, filename):
    """Save intermediate results to prevent data loss"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Intermediate results saved to {filename}")

def load_intermediate_results(filename):
    """Load intermediate results if they exist"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

class ExplainabilityAnalyzer:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.explanations = {}
    
    def analyze_model_explainability(self, model, model_name, X_train, X_test, 
                                   y_train, y_test, max_samples=100):
        """Comprehensive explainability analysis for a model"""
        
        print(f"\n🔍 Explainability Analysis for {model_name}")
        print("=" * 50)
        
        # 1. Feature Importance (if available)
        self._analyze_feature_importance(model, model_name)
        
        # 2. Permutation Importance
        self._analyze_permutation_importance(model, model_name, X_test, y_test)
        
        # 3. SHAP Analysis
        # run only for xg-boost.
        if (model_name.lower() == "xgboost"):
            self._analyze_shap(model, model_name, X_train, X_test, max_samples)
        
        # 4. LIME Analysis
        self._analyze_lime(model, model_name, X_train, X_test, max_samples)
        
        return self.explanations.get(model_name, {})
    
    def _analyze_feature_importance(self, model, model_name):
        """Analyze built-in feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create feature importance plot
                plt.figure(figsize=(10, 6))
                feature_names = self.feature_names or [f'Feature_{i}' for i in range(len(importances))]
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(importances)), importances[indices])
                plt.title(f'{model_name} - Feature Importance')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(range(len(importances)), 
                          [feature_names[i] for i in indices], rotation=45)
                plt.tight_layout()
                plt.show()
                plt.savefig(f'{model_name}_feature_importance.png')
                
                # Store results
                if model_name not in self.explanations:
                    self.explanations[model_name] = {}
                self.explanations[model_name]['feature_importance'] = {
                    'importances': importances,
                    'feature_names': feature_names
                }
                
                print(f"✅ Feature importance analysis completed for {model_name}")
                
            else:
                print(f"⚠️  {model_name} doesn't have built-in feature importance")
                
        except Exception as e:
            print(f"❌ Error in feature importance analysis: {str(e)}")
    
    def _analyze_permutation_importance(self, model, model_name, X_test, y_test):
        """Analyze permutation importance"""
        try:
            print(f"🔄 Computing permutation importance for {model_name}...")
            
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
            )
            
            # Create permutation importance plot
            plt.figure(figsize=(10, 6))
            feature_names = self.feature_names or [f'Feature_{i}' for i in range(len(perm_importance.importances_mean))]
            
            # Sort features by importance
            indices = np.argsort(perm_importance.importances_mean)[::-1]
            
            plt.bar(range(len(perm_importance.importances_mean)), 
                   perm_importance.importances_mean[indices])
            plt.errorbar(range(len(perm_importance.importances_mean)), 
                        perm_importance.importances_mean[indices],
                        yerr=perm_importance.importances_std[indices], 
                        fmt='none', color='black', capsize=3)
            
            plt.title(f'{model_name} - Permutation Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(perm_importance.importances_mean)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'{model_name}_permutation_importance.png')
            
            # Store results
            if model_name not in self.explanations:
                self.explanations[model_name] = {}
            self.explanations[model_name]['permutation_importance'] = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std,
                'feature_names': feature_names
            }
            
            print(f"✅ Permutation importance analysis completed for {model_name}")
            
        except Exception as e:
            print(f"❌ Error in permutation importance analysis: {str(e)}")
    
    def _analyze_shap(self, model, model_name, X_train, X_test, max_samples):
        """Analyze SHAP values"""
        try:
            print(f"🔄 Computing SHAP values for {model_name}...")
            
            # Limit samples for computational efficiency
            X_train_sample = X_train[:min(max_samples, len(X_train))]
            X_test_sample = X_test[:min(max_samples, len(X_test))]
            
            # Clear memory before SHAP computation
            clear_memory()

            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    # For other models, use KernelExplainer
                    explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
            else:
                explainer = shap.KernelExplainer(model.predict, X_train_sample)
            
            # Calculate SHAP values in smaller batches for memory safety
            batch_size = min(10, len(X_test_sample))
            all_shap_values = []
            
            for i in range(0, len(X_test_sample), batch_size):
                batch_end = min(i + batch_size, len(X_test_sample))
                batch_data = X_test_sample[i:batch_end]
                
                print(f"   Processing batch {i//batch_size + 1}/{(len(X_test_sample)-1)//batch_size + 1}")
                
                batch_shap_values = explainer.shap_values(batch_data)
                all_shap_values.append(batch_shap_values)
                
                # Clear memory after each batch
                clear_memory()
            
            # Combine batch results
            if isinstance(all_shap_values[0], list):
                # Multi-class case
                shap_values = []
                for class_idx in range(len(all_shap_values[0])):
                    class_shap = np.vstack([batch[class_idx] for batch in all_shap_values])
                    shap_values.append(class_shap)
            else:
                # Single output case
                shap_values = np.vstack(all_shap_values)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # For multi-class, use the first class or aggregate
                if len(shap_values) > 2:  # Multi-class
                    # Use mean absolute SHAP values across all classes
                    shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                else:  # Binary classification
                    shap_values = shap_values[1]  # Use positive class
            
            # SHAP Summary Plot
            plt.figure(figsize=(10, 6))
            feature_names = self.feature_names or [f'Feature_{i}' for i in range(X_test_sample.shape[1])]
            shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.show()
            
            # SHAP Bar Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                            plot_type="bar", show=False)
            plt.title(f'{model_name} - SHAP Feature Importance')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'{model_name}_shap_summary.png')
            
            # Store results
            if model_name not in self.explanations:
                self.explanations[model_name] = {}
            self.explanations[model_name]['shap'] = {
                'shap_values': shap_values,
                'feature_names': feature_names,
                'X_test_sample': X_test_sample
            }
            
            print(f"✅ SHAP analysis completed for {model_name}")
            
        except Exception as e:
            print(f"❌ Error in SHAP analysis: {str(e)}")
            print("Consider installing shap: pip install shap")
    
    def _analyze_lime(self, model, model_name, X_train, X_test, max_samples):
        """Analyze LIME explanations"""
        try:
            print(f"🔄 Computing LIME explanations for {model_name}...")
            
            # Limit samples for computational efficiency
            X_train_sample = X_train[:min(max_samples, len(X_train))]
            X_test_sample = X_test[:min(10, len(X_test))]  # Just a few samples for LIME
            
            feature_names = self.feature_names or [f'Feature_{i}' for i in range(X_train_sample.shape[1])]
            
            # Create LIME explainer - determine class names dynamically
            # For multi-class, we need to determine the number of classes
            if hasattr(model, 'classes_'):
                class_names = [f'Class_{i}' for i in range(len(model.classes_))]
            else:
                # Default to binary classification names
                class_names = ['Class_0', 'Class_1']
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_sample,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification'
            )
            
            # Generate explanations for a few instances
            explanations = []
            for i in range(min(3, len(X_test_sample))):
                if hasattr(model, 'predict_proba'):
                    exp = explainer.explain_instance(
                        X_test_sample[i], 
                        model.predict_proba,
                        num_features=len(feature_names)
                    )
                else:
                    exp = explainer.explain_instance(
                        X_test_sample[i], 
                        model.predict,
                        num_features=len(feature_names)
                    )
                explanations.append(exp)
                
                # Show explanation plot
                fig = exp.as_pyplot_figure()
                fig.suptitle(f'{model_name} - LIME Explanation (Instance {i+1})')
                plt.tight_layout()
                plt.show()
                plt.savefig(f'{model_name}_lime_explanation_{i+1}.png')
            
            # Store results
            if model_name not in self.explanations:
                self.explanations[model_name] = {}
            self.explanations[model_name]['lime'] = {
                'explanations': explanations,
                'feature_names': feature_names
            }
            
            print(f"✅ LIME analysis completed for {model_name}")
            
        except Exception as e:
            print(f"❌ Error in LIME analysis: {str(e)}")
            print("Consider installing lime: pip install lime")
    
    def compare_feature_importance(self, explanations=None):
        """Compare feature importance across models"""
        
            # Check if self.explanations is null or empty, if so assign explanations to self.explanations
        if not self.explanations and explanations is not None:
            self.explanations = explanations

        if len(self.explanations) < 2:
            print("Need at least 2 models for comparison")
            return
        
        print("\n🔍 Feature Importance Comparison")
        print("=" * 50)
        
        # Collect feature importances
        importance_data = {}
        
        for model_name, explanations in self.explanations.items():
            if 'feature_importance' in explanations:
                importance_data[f'{model_name}_builtin'] = explanations['feature_importance']['importances']
            if 'permutation_importance' in explanations:
                importance_data[f'{model_name}_permutation'] = explanations['permutation_importance']['importances_mean']
        
        if not importance_data:
            print("No feature importance data available")
            return
        
        # Create comparison DataFrame
        feature_names = self.feature_names or [f'Feature_{i}' for i in range(len(list(importance_data.values())[0]))]
        df_importance = pd.DataFrame(importance_data, index=feature_names)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        df_importance.plot(kind='bar', ax=plt.gca())
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig('feature_importance_comparison.png')
        
        return df_importance
    
    def generate_explanation_report(self, model_name):
        """Generate a comprehensive explanation report for a model"""
        if model_name not in self.explanations:
            print(f"No explanations available for {model_name}")
            return
        
        explanations = self.explanations[model_name]
        
        print(f"\n📋 Explainability Report for {model_name}")
        print("=" * 60)
        
        # Feature Importance Summary
        if 'feature_importance' in explanations:
            importances = explanations['feature_importance']['importances']
            feature_names = explanations['feature_importance']['feature_names']
            
            print("\n🎯 Top 5 Most Important Features (Built-in):")
            indices = np.argsort(importances)[::-1][:5]
            for i, idx in enumerate(indices):
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)  # Ensure idx is a scalar integer
                print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Permutation Importance Summary
        if 'permutation_importance' in explanations:
            importances = explanations['permutation_importance']['importances_mean']
            feature_names = explanations['permutation_importance']['feature_names']
            
            print("\n🔄 Top 5 Most Important Features (Permutation):")
            indices = np.argsort(importances)[::-1][:5]
            for i, idx in enumerate(indices):
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)  # Ensure idx is a scalar integer
                print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # SHAP Summary
        if 'shap' in explanations:
            shap_values = explanations['shap']['shap_values']
            feature_names = explanations['shap']['feature_names']
            
            # Handle different SHAP value dimensions properly
            if shap_values.ndim == 3:
                # 3D case: (samples, features, classes) - aggregate across samples and classes
                mean_abs_shap = np.mean(np.mean(np.abs(shap_values), axis=0), axis=-1)
            elif shap_values.ndim == 2:
                # 2D case: (samples, features) - aggregate across samples
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            else:
                # 1D case: just use absolute values
                mean_abs_shap = np.abs(shap_values)
            
            # Ensure mean_abs_shap is 1D
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.flatten()
            
            print("\n🎭 Top 5 Most Important Features (SHAP):")
            indices = np.argsort(mean_abs_shap)[::-1][:5]
            for i, idx in enumerate(indices):
                idx = int(idx)  # Convert to scalar integer
                print(f"   {i+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")
        
        print(f"\n✅ Explainability analysis complete for {model_name}")

# Usage example
def run_explainability_analysis():
    """Example of how to use the explainability analyzer"""
    
    # Example feature names for online shoppers dataset
    feature_names = [
        'Administrative', 'Administrative_Duration', 'Informational', 
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]
    
    analyzer = ExplainabilityAnalyzer(feature_names=feature_names)
    
    # Example usage:
    # analyzer.analyze_model_explainability(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)
    # analyzer.analyze_model_explainability(tabpfn_model, "TabPFN", X_train, X_test, y_train, y_test)
    # analyzer.compare_feature_importance()
    
    print("Explainability analysis framework ready!")
    return analyzer
