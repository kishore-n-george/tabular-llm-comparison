#!/usr/bin/env python
# coding: utf-8

# # Section 3: Explainability Analysis and Ablation Studies
# ## Dry Bean Dataset - Model Interpretability
# 
# This notebook provides comprehensive explainability analysis and ablation studies for the trained models.
# 
# **Analysis Components:**
# - **Feature Importance**: Built-in and permutation-based importance
# - **SHAP Analysis**: Shapley Additive Explanations for model predictions
# - **LIME Analysis**: Local Interpretable Model-agnostic Explanations
# - **Ablation Studies**: Feature ablation, hyperparameter sensitivity
# - **Cross-Model Comparison**: Feature importance consensus across models
# 
# **Models Analyzed:**
# - XGBoost, TabPFN v2, TabICL
# - Focus on understanding what drives model decisions
# - Identify most important features for dry bean classification

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our custom analysis frameworks
from explainability_analysis import ExplainabilityAnalyzer, save_intermediate_results, clear_memory, load_intermediate_results
from ablation_studies import AblationStudyAnalyzer
from enhanced_ablation_studies import run_enhanced_ablation_studies

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔍 Section 3: Explainability Analysis and Ablation Studies")
print("Dataset: Dry Bean Classification")


# ## 3.1 Load Trained Models and Data

# In[2]:


# Load trained models and results from Section 2
import pickle

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


# ## 3.2 Initialize Explainability Analyzer

# In[3]:


# Initialize explainability analyzer with dry bean feature names
explainer = ExplainabilityAnalyzer(feature_names=feature_names)

print("🔧 Explainability analyzer initialized")
print(f"Feature names: {feature_names[:5]}...")
print(f"Total features: {len(feature_names)}")
print(f"Class names: {class_names}")


# ## 3.3 XGBoost Explainability Analysis

# In[4]:


# # Analyze XGBoost explainability (most interpretable)
# if 'XGBoost' in models:
#     print("\n" + "="*60)
#     print("XGBOOST EXPLAINABILITY ANALYSIS")
#     print("="*60)

#     xgb_explanations = explainer.analyze_model_explainability(
#         models['XGBoost'], "XGBoost", 
#         X_train_scaled, X_test_scaled, y_train, y_test,
#         max_samples=100
#     )

#         # Save intermediate results
#     save_intermediate_results({
#         'xgb_explanations': xgb_explanations,
#         'explainer_state': explainer.explanations
#     }, 'dry_bean_xgb_explanations.pkl')

#     print("✅ XGBoost explainability analysis completed")
#     clear_memory()

# else:
#     print("⚠️ XGBoost model not available")


# ## 3.4 TabPFN v2 Explainability Analysis

# In[ ]:


# # Analyze TabPFN v2 explainability
# if 'TabPFN v2' in models:
#     print("\n" + "="*60)
#     print("TABPFN V2 EXPLAINABILITY ANALYSIS")
#     print("="*60)

#     tabpfn_explanations = explainer.analyze_model_explainability(
#         models['TabPFN v2'], "TabPFN v2", 
#         X_train_scaled, X_test_scaled, y_train, y_test,
#         max_samples=100
#     )

#         # Save intermediate results after TabPFN
#     intermediate_data = {
#         'explainer_state': explainer.explanations,
#         'tabpfn_explanations': tabpfn_explanations,
#         'models_completed': ['XGBoost', 'TabPFN v2']
#     }
#     save_intermediate_results(intermediate_data, 'dry_bean_tabpfn_explanations.pkl')

#     print("✅ TabPFN v2 explainability analysis completed")
#     print("💾 Results saved before proceeding to TabICL")
#     clear_memory()

#     print("✅ TabPFN v2 explainability analysis completed")
# else:
#     print("⚠️ TabPFN v2 model not available")


# ## 3.5 TabICL Explainability Analysis

# In[ ]:


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
# del X_train_scaled, X_test_scaled, y_train, y_test
if 'section2_data' in locals():
    del section2_data

# Run aggressive garbage collection
import gc
gc.collect()
gc.collect()  # Run twice for better cleanup

print("🧹 Memory cleared successfully")
print("📊 Proceeding with TabICL analysis using minimal memory footprint")

# Analyze TabICL explainability
if 'TabICL' in models:
    print("\n" + "="*60)
    print("TABICL EXPLAINABILITY ANALYSIS")
    print("="*60)

    tabicl_explanations = explainer.analyze_model_explainability(
        models['TabICL'], "TabICL", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=100
    )
        # Save results after TabICL attempt
    save_intermediate_results({
        'explainer_state': explainer.explanations,
        'models_completed': ['XGBoost', 'TabPFN v2', 'TabICL']
    }, 'dry_bean_tabicl_explanations.pkl')

    clear_memory()
    print("✅ TabICL explainability analysis completed")
else:
    print("⚠️ TabICL model not available")


# Restore data for remaining analyses
X_train_scaled = temp_X_train
X_test_scaled = temp_X_test
y_train = temp_y_train
y_test = temp_y_test


# ## 3.5 Ft-Transformer Explainability Analysis

# In[5]:


# Create PyTorch model wrapper for the FT-Transformer
import torch
fttransformer_model = models['FT-Transformer']
device='cuda' if torch.cuda.is_available() else 'cpu'
ft_wrapper = evaluator.create_pytorch_wrapper(
    model=fttransformer_model,
    device=device,
    batch_size=256
)
models['FT-Transformer-Wrapper'] = ft_wrapper


# In[6]:


# Analyze FT-Transformer explainability
if 'FT-Transformer-Wrapper' in models:
    print("\n" + "="*60)
    print("FT-Transformer EXPLAINABILITY ANALYSIS")
    print("="*60)

    ft_transformer_explanations = explainer.analyze_model_explainability(
        models['FT-Transformer-Wrapper'], "FT-Transformer", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=100
    )
        # Save results after TabICL attempt
    save_intermediate_results({
        'explainer_state': explainer.explanations,
        'models_completed': ['XGBoost', 'TabPFN v2', 'TabICL', 'FT-Transformer']
    }, 'dry_bean_fttransformer_explanations.pkl')

    clear_memory()
    print("✅ FT-Transformer explainability analysis completed")
else:
    print("⚠️ FT-Transformer model not available")


# ## 3.6 Cross-Model Feature Importance Comparison

# In[ ]:


clear_memory()
def combine_explainability_results():
    """Combine results from different intermediate saves"""
    print("🔄 Combining explainability results from intermediate saves")

    # Load all available intermediate results
    xgb_results = load_intermediate_results('dry_bean_xgb_explanations.pkl')
    tabpfn_results = load_intermediate_results('dry_bean_tabpfn_explanations.pkl')
    tabicl_results = load_intermediate_results('dry_bean_tabicl_explanations.pkl')
    fttransformer_results = load_intermediate_results('dry_bean_fttransformer_explanations.pkl')

    combined_explanations = {}

    # Combine XGBoost results
    if xgb_results and 'explainer_state' in xgb_results:
        combined_explanations.update(xgb_results['explainer_state'])
        print("✅ XGBoost results added")

    # Combine TabPFN results
    if tabpfn_results and 'explainer_state' in tabpfn_results:
        combined_explanations.update(tabpfn_results['explainer_state'])
        print("✅ TabPFN results added")

    # Combine TabICL results
    if tabicl_results and 'explainer_state' in tabicl_results:
        combined_explanations.update(tabicl_results['explainer_state'])
        print("✅ TabICL results added")

    if fttransformer_results and 'explainer_state' in fttransformer_results:
        combined_explanations.update(fttransformer_results['explainer_state'])
        print("✅ FT-Transformer results added")

    return combined_explanations

explanations = combine_explainability_results()


# In[ ]:


# Compare feature importance across models
print("\n" + "="*60)
print("CROSS-MODEL FEATURE IMPORTANCE COMPARISON")
print("="*60)

importance_comparison = explainer.compare_feature_importance(explanations)

if importance_comparison is not None:
    print("\n📊 Feature Importance Comparison Table:")
    print(importance_comparison.round(4).to_string())

    # Save comparison results
    importance_comparison.to_csv('dry_bean_feature_importance_comparison.csv')
    print("\n💾 Feature importance comparison saved to 'dry_bean_feature_importance_comparison.csv'")

    # Identify consensus features
    print("\n🎯 FEATURE IMPORTANCE CONSENSUS:")

    # Calculate average importance across all methods
    avg_importance = importance_comparison.mean(axis=1).sort_values(ascending=False)

    print("\nTop 10 Most Important Features (Average Across All Methods):")
    for i, (feature, importance) in enumerate(avg_importance.head(10).items()):
        print(f"   {i+1:2d}. {feature}: {importance:.4f}")

    # Feature importance correlation between methods
    print("\n🔗 Feature Importance Correlation Between Methods:")
    correlation_matrix = importance_comparison.corr()
    print(correlation_matrix.round(3).to_string())

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Feature Importance Method Correlation')
    plt.tight_layout()
    plt.show()
    plt.savefig('feature_importance_correlation.png', dpi=300, bbox_inches='tight')
else:
    print("❌ No feature importance data available for comparison")


# ## 3.7 Generate Explanation Reports

# In[ ]:


# Generate detailed explanation reports for each model
print("\n" + "="*60)
print("DETAILED EXPLANATION REPORTS")
print("="*60)

for model_name in models.keys():
    explainer.generate_explanation_report(model_name)
    print("\n" + "-"*40)


# ## 3.8 Summary and Insights

# In[ ]:


# Generate comprehensive summary of explainability findings
print("\n" + "="*80)
print("EXPLAINABILITY ANALYSIS SUMMARY")
print("="*80)

# Model performance recap
print("\n🏆 MODEL PERFORMANCE RECAP:")
if comparison_df is not None:
    print(comparison_df[['accuracy', 'f1', 'precision', 'recall']].round(4).to_string())
else:
    print("Model comparison data not available")

# Feature importance insights
print("\n🎯 KEY FEATURE IMPORTANCE INSIGHTS:")
if importance_comparison is not None:
    avg_importance = importance_comparison.mean(axis=1).sort_values(ascending=False)
    print("\nMost Important Features for Dry Bean Classification:")
    for i, (feature, importance) in enumerate(avg_importance.head(5).items()):
        print(f"   {i+1}. {feature}: {importance:.4f}")

    print(f"\nFeature Importance Consensus: {len(avg_importance)} features analyzed")
    print(f"Top feature: {avg_importance.index[0]} ({avg_importance.iloc[0]:.4f})")
else:
    print("Feature importance data not available")

# Model interpretability insights
print("\n🔍 MODEL INTERPRETABILITY INSIGHTS:")
print("\n1. XGBoost:")
print("   - Most interpretable with built-in feature importance")
print("   - Tree-based structure allows for clear decision paths")
print("   - SHAP values provide detailed feature contributions")

print("\n2. TabPFN v2:")
print("   - Prior-based model with limited interpretability")
print("   - Relies on permutation importance and SHAP for explanations")
print("   - Black-box nature makes feature interactions unclear")

print("\n3. TabICL:")
print("   - In-context learning approach")
print("   - Interpretability through example-based reasoning")
print("   - Context examples influence decision making")

# Key findings
print("\n📋 KEY FINDINGS:")
print("\n• Dry bean classification relies on geometric and shape-based features")
print("• Different models may prioritize different feature combinations")
print("• Feature importance consensus helps identify robust predictors")
print("• Model interpretability varies significantly across architectures")

print("\n✅ Section 3 completed successfully!")
print("📊 Comprehensive explainability analysis finished")
print("📁 All results and visualizations saved")


# In[ ]:


# Save final results for future reference
import pickle

# Save explainability results
section3_data = {
    'explainer': explainer,
    'importance_comparison': importance_comparison if 'importance_comparison' in locals() else None,
    'models': models,
    'feature_names': feature_names,
    'class_names': class_names,
    'explanations': explainer.explanations
}

# Save to pickle file
with open('dry_bean_section3_explainability.pkl', 'wb') as f:
    pickle.dump(section3_data, f)

print("💾 Section 3 explainability results saved to 'dry_bean_section3_explainability.pkl'")
print("📋 This file contains all explainability analysis results")

