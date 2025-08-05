#!/usr/bin/env python
# coding: utf-8

# # Section 3: Explainability Analysis and Ablation Studies (Memory Safe Version)
# ## Dry Bean Dataset - Model Interpretability
# 
# This notebook provides comprehensive explainability analysis with memory management
# to prevent kernel crashes when running SHAP for memory-intensive models like TabICL.
# 
# **Key Features:**
# - Memory-safe SHAP analysis with garbage collection
# - Intermediate result saving after each model
# - Reduced sample sizes for memory-intensive operations
# - Clear memory management between models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import pickle
import os
warnings.filterwarnings('ignore')

# Import our custom analysis frameworks
from explainability_analysis import ExplainabilityAnalyzer

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔍 Section 3: Explainability Analysis and Ablation Studies (Memory Safe)")
print("Dataset: Dry Bean Classification")

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

# ## 3.1 Load Trained Models and Data

# Load trained models and results from Section 2
try:
    with open('dry_bean_section2_results.pkl', 'rb') as f:
        section2_data = pickle.load(f)

    # Extract variables
    models = section2_data['models']
    evaluator = section2_data['evaluator']
    X_train_scaled = section2_data['X_train_scaled']
    X_val_scaled = section2_data['X_val_scaled']
    X_test_scaled = section2_data['X_test_scaled']
    y_train = section2_data['y_train']
    y_val = section2_data['y_val']
    y_test = section2_data['y_test']
    feature_names = section2_data['feature_names']
    class_mapping = section2_data['class_mapping']
    class_names = section2_data['class_names']
    label_encoder = section2_data['label_encoder']
    comparison_df = section2_data['comparison_df']

    print("✅ Section 2 results loaded successfully!")
    print(f"Models available: {list(models.keys())}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {len(class_names)}")
    print(f"Test samples: {len(X_test_scaled):,}")

except FileNotFoundError:
    print("❌ Section 2 results not found!")
    print("Please run Section 2 (Model Training) notebook first.")
    raise

# ## 3.2 Initialize Memory-Safe Explainability Analyzer

class MemorySafeExplainabilityAnalyzer(ExplainabilityAnalyzer):
    """Memory-safe version of ExplainabilityAnalyzer with reduced memory footprint"""
    
    def _analyze_shap(self, model, model_name, X_train, X_test, max_samples):
        """Memory-safe SHAP analysis with reduced sample sizes"""
        try:
            print(f"🔄 Computing SHAP values for {model_name} (memory-safe mode)...")
            
            # Reduce sample sizes for memory-intensive models
            if model_name in ['TabICL', 'TabPFN v2']:
                # Use smaller samples for memory-intensive models
                train_sample_size = min(50, len(X_train))
                test_sample_size = min(25, len(X_test))
                print(f"   Using reduced samples: train={train_sample_size}, test={test_sample_size}")
            else:
                train_sample_size = min(max_samples, len(X_train))
                test_sample_size = min(max_samples//2, len(X_test))
            
            X_train_sample = X_train[:train_sample_size]
            X_test_sample = X_test[:test_sample_size]
            
            # Clear memory before SHAP computation
            clear_memory()
            
            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For tree-based models, use TreeExplainer if available
                if hasattr(model, 'feature_importances_'):
                    import shap
                    explainer = shap.TreeExplainer(model)
                    print(f"   Using TreeExplainer for {model_name}")
                else:
                    # For other models, use KernelExplainer with reduced background
                    import shap
                    background_size = min(20, len(X_train_sample))
                    explainer = shap.KernelExplainer(
                        model.predict_proba, 
                        X_train_sample[:background_size]
                    )
                    print(f"   Using KernelExplainer with background size {background_size}")
            else:
                import shap
                background_size = min(20, len(X_train_sample))
                explainer = shap.KernelExplainer(
                    model.predict, 
                    X_train_sample[:background_size]
                )
                print(f"   Using KernelExplainer for predictions")
            
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
            
            # Handle multi-class case for visualization
            if isinstance(shap_values, list):
                if len(shap_values) > 2:  # Multi-class
                    # Use mean absolute SHAP values across all classes
                    shap_values_viz = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                else:  # Binary classification
                    shap_values_viz = shap_values[1]  # Use positive class
            else:
                shap_values_viz = shap_values
            
            # Create visualizations with memory management
            try:
                import shap
                feature_names = self.feature_names or [f'Feature_{i}' for i in range(X_test_sample.shape[1])]
                
                # SHAP Summary Plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_viz, X_test_sample, feature_names=feature_names, show=False)
                plt.title(f'{model_name} - SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig(f'{model_name}_shap_summary.png', dpi=150, bbox_inches='tight')
                plt.show()
                plt.close()  # Close figure to free memory
                
                # SHAP Bar Plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_viz, X_test_sample, feature_names=feature_names, 
                                plot_type="bar", show=False)
                plt.title(f'{model_name} - SHAP Feature Importance')
                plt.tight_layout()
                plt.savefig(f'{model_name}_shap_bar.png', dpi=150, bbox_inches='tight')
                plt.show()
                plt.close()  # Close figure to free memory
                
            except Exception as viz_error:
                print(f"⚠️ SHAP visualization error: {str(viz_error)}")
            
            # Store results
            if model_name not in self.explanations:
                self.explanations[model_name] = {}
            self.explanations[model_name]['shap'] = {
                'shap_values': shap_values_viz,  # Store the visualization version
                'feature_names': feature_names,
                'sample_size': len(X_test_sample)
            }
            
            print(f"✅ SHAP analysis completed for {model_name}")
            
            # Clear memory after SHAP analysis
            clear_memory()
            
        except Exception as e:
            print(f"❌ Error in SHAP analysis for {model_name}: {str(e)}")
            print("Continuing with other analyses...")

# Initialize memory-safe explainability analyzer
explainer = MemorySafeExplainabilityAnalyzer(feature_names=feature_names)

print("🔧 Memory-safe explainability analyzer initialized")
print(f"Feature names: {feature_names[:5]}...")
print(f"Total features: {len(feature_names)}")
print(f"Class names: {class_names}")

# ## 3.3 XGBoost Explainability Analysis

print("\n" + "="*60)
print("XGBOOST EXPLAINABILITY ANALYSIS")
print("="*60)

if 'XGBoost' in models:
    xgb_explanations = explainer.analyze_model_explainability(
        models['XGBoost'], "XGBoost", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=100
    )
    
    # Save intermediate results
    save_intermediate_results({
        'xgb_explanations': xgb_explanations,
        'explainer_state': explainer.explanations
    }, 'dry_bean_xgb_explanations.pkl')
    
    print("✅ XGBoost explainability analysis completed")
    clear_memory()
else:
    print("⚠️ XGBoost model not available")

# ## 3.4 TabPFN v2 Explainability Analysis

print("\n" + "="*60)
print("TABPFN V2 EXPLAINABILITY ANALYSIS")
print("="*60)

if 'TabPFN v2' in models:
    tabpfn_explanations = explainer.analyze_model_explainability(
        models['TabPFN v2'], "TabPFN v2", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=75  # Reduced sample size
    )
    
    # Save intermediate results after TabPFN
    intermediate_data = {
        'explainer_state': explainer.explanations,
        'tabpfn_explanations': tabpfn_explanations,
        'models_completed': ['XGBoost', 'TabPFN v2']
    }
    save_intermediate_results(intermediate_data, 'dry_bean_tabpfn_explanations.pkl')
    
    print("✅ TabPFN v2 explainability analysis completed")
    print("💾 Results saved before proceeding to TabICL")
    clear_memory()
else:
    print("⚠️ TabPFN v2 model not available")

# ## 3.5 Memory Management Before TabICL

print("\n" + "="*60)
print("MEMORY MANAGEMENT BEFORE TABICL")
print("="*60)

# Clear all unnecessary variables and run aggressive garbage collection
print("🧹 Clearing memory before TabICL analysis...")

# Clear large data structures temporarily
temp_X_train = X_train_scaled.copy()
temp_X_test = X_test_scaled.copy()
temp_y_train = y_train.copy()
temp_y_test = y_test.copy()

# Delete large variables temporarily
del X_train_scaled, X_test_scaled, y_train, y_test
if 'section2_data' in locals():
    del section2_data

# Run aggressive garbage collection
import gc
gc.collect()
gc.collect()  # Run twice for better cleanup

print("🧹 Memory cleared successfully")
print("📊 Proceeding with TabICL analysis using minimal memory footprint")

# ## 3.6 TabICL Explainability Analysis (Memory Safe)

print("\n" + "="*60)
print("TABICL EXPLAINABILITY ANALYSIS (MEMORY SAFE)")
print("="*60)

if 'TabICL' in models:
    try:
        # Use very conservative settings for TabICL
        print("⚠️ Using ultra-conservative memory settings for TabICL")
        
        tabicl_explanations = explainer.analyze_model_explainability(
            models['TabICL'], "TabICL", 
            temp_X_train, temp_X_test, temp_y_train, temp_y_test,
            max_samples=50  # Very small sample size
        )
        
        print("✅ TabICL explainability analysis completed")
        
    except Exception as e:
        print(f"❌ TabICL analysis failed: {str(e)}")
        print("💡 This is likely due to memory constraints")
        print("📊 Continuing with available results...")
        
        # Create minimal explanation entry
        explainer.explanations['TabICL'] = {
            'status': 'failed_memory_constraint',
            'error': str(e)
        }
    
    # Save results after TabICL attempt
    save_intermediate_results({
        'explainer_state': explainer.explanations,
        'models_completed': ['XGBoost', 'TabPFN v2', 'TabICL']
    }, 'dry_bean_tabicl_explanations.pkl')
    
    clear_memory()
else:
    print("⚠️ TabICL model not available")

# Restore data for remaining analyses
X_train_scaled = temp_X_train
X_test_scaled = temp_X_test
y_train = temp_y_train
y_test = temp_y_test

# ## 3.7 FT-Transformer Explainability Analysis

print("\n" + "="*60)
print("FT-TRANSFORMER EXPLAINABILITY ANALYSIS")
print("="*60)

# Create PyTorch model wrapper for the FT-Transformer
if 'FT-Transformer' in models:
    try:
        import torch
        fttransformer_model = models['FT-Transformer']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ft_wrapper = evaluator.create_pytorch_wrapper(
            model=fttransformer_model,
            device=device,
            batch_size=256
        )
        models['FT-Transformer-Wrapper'] = ft_wrapper
        
        ft_transformer_explanations = explainer.analyze_model_explainability(
            models['FT-Transformer-Wrapper'], "FT-Transformer", 
            X_train_scaled, X_test_scaled, y_train, y_test,
            max_samples=75
        )
        
        print("✅ FT-Transformer explainability analysis completed")
        clear_memory()
        
    except Exception as e:
        print(f"❌ FT-Transformer analysis failed: {str(e)}")
else:
    print("⚠️ FT-Transformer model not available")

# ## 3.8 Cross-Model Feature Importance Comparison

print("\n" + "="*60)
print("CROSS-MODEL FEATURE IMPORTANCE COMPARISON")
print("="*60)

try:
    importance_comparison = explainer.compare_feature_importance()
    
    if importance_comparison is not None:
        print("\n📊 Feature Importance Comparison Table:")
        print(importance_comparison.round(4).to_string())
        
        # Save comparison results
        importance_comparison.to_csv('dry_bean_feature_importance_comparison.csv')
        print("\n💾 Feature importance comparison saved")
        
        # Identify consensus features
        print("\n🎯 FEATURE IMPORTANCE CONSENSUS:")
        avg_importance = importance_comparison.mean(axis=1).sort_values(ascending=False)
        
        print("\nTop 10 Most Important Features (Average Across All Methods):")
        for i, (feature, importance) in enumerate(avg_importance.head(10).items()):
            print(f"   {i+1:2d}. {feature}: {importance:.4f}")
        
        # Feature importance correlation
        print("\n🔗 Feature Importance Correlation Between Methods:")
        correlation_matrix = importance_comparison.corr()
        print(correlation_matrix.round(3).to_string())
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.3f')
        plt.title('Feature Importance Method Correlation')
        plt.tight_layout()
        plt.savefig('feature_importance_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    else:
        print("❌ No feature importance data available for comparison")
        
except Exception as e:
    print(f"❌ Error in feature importance comparison: {str(e)}")

# ## 3.9 Generate Explanation Reports

print("\n" + "="*60)
print("DETAILED EXPLANATION REPORTS")
print("="*60)

for model_name in explainer.explanations.keys():
    try:
        explainer.generate_explanation_report(model_name)
        print("\n" + "-"*40)
    except Exception as e:
        print(f"❌ Error generating report for {model_name}: {str(e)}")

# ## 3.10 Final Results and Summary

print("\n" + "="*80)
print("EXPLAINABILITY ANALYSIS SUMMARY (MEMORY SAFE)")
print("="*80)

# Model performance recap
print("\n🏆 MODEL PERFORMANCE RECAP:")
if 'comparison_df' in locals() and comparison_df is not None:
    print(comparison_df[['accuracy', 'f1', 'precision', 'recall']].round(4).to_string())
else:
    print("Model comparison data not available")

# Feature importance insights
print("\n🎯 KEY FEATURE IMPORTANCE INSIGHTS:")
if 'importance_comparison' in locals() and importance_comparison is not None:
    avg_importance = importance_comparison.mean(axis=1).sort_values(ascending=False)
    print("\nMost Important Features for Dry Bean Classification:")
    for i, (feature, importance) in enumerate(avg_importance.head(5).items()):
        print(f"   {i+1}. {feature}: {importance:.4f}")
else:
    print("Feature importance data not available")

# Analysis status summary
print("\n📊 ANALYSIS STATUS SUMMARY:")
for model_name, explanations in explainer.explanations.items():
    if isinstance(explanations, dict) and 'status' in explanations:
        print(f"   {model_name}: {explanations['status']}")
    else:
        analyses = []
        if 'feature_importance' in explanations:
            analyses.append('Feature Importance')
        if 'permutation_importance' in explanations:
            analyses.append('Permutation')
        if 'shap' in explanations:
            analyses.append('SHAP')
        if 'lime' in explanations:
            analyses.append('LIME')
        print(f"   {model_name}: {', '.join(analyses) if analyses else 'No analyses'}")

print("\n💡 MEMORY MANAGEMENT INSIGHTS:")
print("• Used reduced sample sizes for memory-intensive models")
print("• Implemented batch processing for SHAP calculations")
print("• Saved intermediate results to prevent data loss")
print("• Applied aggressive garbage collection between models")

print("\n✅ Section 3 completed successfully with memory management!")
print("📊 Explainability analysis finished with memory safety measures")

# ## 3.11 Save Final Results

# Save comprehensive final results
section3_data = {
    'explainer': explainer,
    'importance_comparison': importance_comparison if 'importance_comparison' in locals() else None,
    'models': models,
    'feature_names': feature_names,
    'class_names': class_names,
    'explanations': explainer.explanations,
    'memory_safe_version': True,
    'analysis_summary': {
        'models_analyzed': list(explainer.explanations.keys()),
        'memory_management_applied': True,
        'intermediate_saves': True
    }
}

# Save to pickle file
with open('dry_bean_section3_explainability_memory_safe.pkl', 'wb') as f:
    pickle.dump(section3_data, f)

print("\n💾 Section 3 memory-safe results saved to 'dry_bean_section3_explainability_memory_safe.pkl'")
print("📋 This file contains all explainability analysis results with memory management")
print("🔧 Intermediate results also saved for recovery purposes")

# Final memory cleanup
clear_memory()
print("\n🧹 Final memory cleanup completed")
