#!/usr/bin/env python
# coding: utf-8

# # Section 3: Explainability Analysis and Ablation Studies (Simplified)
# 
# ## Bike Sharing Dataset - Model Interpretability for Regression
# 
# This notebook provides comprehensive explainability analysis and ablation studies for the trained regression models using wrapped functions for better modularity.
# 
# **Analysis Components:**
# - **Feature Importance**: Built-in and permutation-based importance
# - **SHAP Analysis**: Shapley Additive Explanations for model predictions
# - **LIME Analysis**: Local Interpretable Model-agnostic Explanations
# - **Ablation Studies**: Feature ablation, hyperparameter sensitivity
# - **Cross-Model Comparison**: Feature importance consensus across models
# 
# **Models Analyzed:**
# - XGBoost, Improved FT-Transformer, Original FT-Transformer, SAINT
# - Focus on understanding what drives bike sharing demand prediction
# - Identify most important features for bike rental count prediction

# ## 1. Import Required Libraries and Functions

# In[ ]:


# Import the wrapped functions from our custom module
from section3_explainability_functions import (
    setup_analysis_environment,
    load_section2_results,
    load_preprocessed_data,
    load_and_filter_models,
    initialize_explainability_analyzer,
    analyze_xgboost_explainability,
    analyze_improved_ft_transformer_explainability,
    analyze_saint_explainability,
    perform_cross_model_comparison,
    generate_explanation_reports,
    generate_business_insights,
    run_ablation_studies,
    generate_analysis_summary,
    save_final_results,
    run_complete_explainability_analysis
)

# Import additional libraries for display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Memory management utilities
import gc
import pickle
import os


# ## 2. Setup Analysis Environment

# In[ ]:


# Setup the analysis environment
setup_analysis_environment()


# ## 3. Option 1: Run Complete Analysis Pipeline (Recommended)
# 
# This single function call will run the entire explainability analysis pipeline:

# In[ ]:


# Run the complete explainability analysis pipeline
results = run_complete_explainability_analysis(
    section2_results_file='./bike_sharing_section2_results.pkl',
    preprocessed_data_file='./bike_sharing_preprocessed_data.pkl',
    model_dir='./Section2_Model_Training',
    save_dir='./Section3_Explainability',
    min_r2_threshold=0.5,
    max_samples=300
)

print("\n🎉 Complete explainability analysis finished!")
print(f"📊 Models analyzed: {results['models_analyzed']}")
print(f"📁 Results saved to: {results['results_file']}")


# ## 4. Option 2: Step-by-Step Analysis (For Detailed Control)
# 
# If you prefer to run the analysis step-by-step for more control, use the following cells:

# ### 4.1 Load Data and Models

# In[ ]:


# Load Section 2 results
section2_data = load_section2_results('./bike_sharing_section2_results.pkl')

# Load preprocessed data
preprocessing_data = load_preprocessed_data('./bike_sharing_preprocessed_data.pkl')

# Extract variables
comparison_df = section2_data['comparison_df']
predictions = section2_data['predictions']
feature_names = preprocessing_data['feature_names']
X_train_scaled = preprocessing_data['X_train_scaled']
X_test_scaled = preprocessing_data['X_test_scaled']
y_train = preprocessing_data['y_train']
y_test = preprocessing_data['y_test']

print(f"\n📊 Data loaded successfully!")
print(f"Features: {len(feature_names)}")
print(f"Training samples: {len(X_train_scaled):,}")
print(f"Test samples: {len(X_test_scaled):,}")


# ### 4.2 Load and Filter Models

# In[ ]:


# Load and filter models based on performance
models_to_analyze, device, model_results_detailed = load_and_filter_models(
    model_dir='./Section2_Model_Training',
    feature_names=feature_names,
    predictions=predictions,
    comparison_df=comparison_df,
    min_r2_threshold=0.5
)

print(f"\n🔍 Models selected for analysis: {list(models_to_analyze.keys())}")


# ### 4.3 Initialize Explainability Analyzer

# In[ ]:


# Initialize the explainability analyzer
explainer = initialize_explainability_analyzer(
    feature_names=feature_names,
    save_dir='./Section3_Explainability'
)

# Store intermediate results
intermediate_results = {}


# ### 4.4 Run Model-Specific Explainability Analysis

# In[ ]:


# Analyze XGBoost explainability
xgb_results = analyze_xgboost_explainability(
    models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
    y_train, y_test, feature_names, max_samples=300
)
if xgb_results:
    intermediate_results['xgb_explanations'] = xgb_results


# In[ ]:


# Analyze Improved FT-Transformer explainability
ft_results = analyze_improved_ft_transformer_explainability(
    models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
    y_train, y_test, feature_names, model_results_detailed, device, max_samples=300
)
if ft_results:
    intermediate_results['improved_ft_explanations'] = ft_results


# In[ ]:


# Analyze SAINT explainability
saint_results = analyze_saint_explainability(
    models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
    y_train, y_test, feature_names, device, max_samples=300
)
if saint_results:
    intermediate_results['saint_explanations'] = saint_results


# ### 4.5 Cross-Model Feature Importance Comparison

# In[ ]:


# Compare feature importance across models
importance_comparison, avg_importance = perform_cross_model_comparison(explainer)

if importance_comparison is not None:
    print("\n📊 Feature importance comparison completed successfully!")
    print(f"Top 3 features: {list(avg_importance.head(3).index)}")
else:
    print("⚠️ No feature importance data available for comparison")


# ### 4.6 Generate Detailed Reports

# In[ ]:


# Generate detailed explanation reports for each model
generate_explanation_reports(explainer)


# ### 4.7 Business Insights

# In[ ]:


# Generate business insights
available_models = list(explainer.explanations.keys())
generate_business_insights(importance_comparison, avg_importance, available_models)


# ### 4.8 Ablation Studies

# In[ ]:


# Run enhanced ablation studies
ablation_analyzer, ablation_results, ablation_summary_df = run_ablation_studies(
    models_to_analyze, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
)

if ablation_results:
    print("\n✅ Ablation studies completed successfully!")
    print(f"Models analyzed: {list(ablation_results.keys())}")
else:
    print("⚠️ Ablation studies encountered issues")


# ### 4.9 Generate Analysis Summary

# In[ ]:


# Generate comprehensive analysis summary
generate_analysis_summary(comparison_df, importance_comparison, avg_importance, ablation_results)


# ### 4.10 Save Final Results

# In[ ]:


# Save final results
results_file = save_final_results(
    explainer, importance_comparison, avg_importance, feature_names, 
    models_to_analyze, ablation_results
)

print(f"\n🎉 Analysis complete! Results saved to: {results_file}")


# ## 5. Results Summary
# 
# Display key results and insights from the analysis:

# In[ ]:


# Display key results summary
if 'results' in locals() and results is not None:
    print("🎯 ANALYSIS RESULTS SUMMARY")
    print("=" * 50)
    print(f"Models Analyzed: {results['models_analyzed']}")
    print(f"Total Features: {len(results['feature_names'])}")

    if results['avg_importance'] is not None:
        print(f"\nTop 5 Most Important Features:")
        for i, (feature, importance) in enumerate(results['avg_importance'].head(5).items()):
            print(f"   {i+1}. {feature}: {importance:.4f}")

    print(f"\nResults saved to: {results['results_file']}")
    print("\n📊 Generated visualizations and reports in ./Section3_Explainability/")

elif 'explainer' in locals():
    print("🎯 STEP-BY-STEP ANALYSIS COMPLETED")
    print("=" * 50)
    print(f"Models Analyzed: {list(explainer.explanations.keys())}")
    print(f"Total Features: {len(feature_names)}")

    if 'avg_importance' in locals() and avg_importance is not None:
        print(f"\nTop 5 Most Important Features:")
        for i, (feature, importance) in enumerate(avg_importance.head(5).items()):
            print(f"   {i+1}. {feature}: {importance:.4f}")

    print("\n📊 Generated visualizations and reports in ./Section3_Explainability/")

else:
    print("⚠️ No analysis results available. Please run the analysis first.")


# ## 6. Next Steps
# 
# After completing this explainability analysis, you can:
# 
# 1. **Review Generated Files**: Check the `./Section3_Explainability/` directory for:
#    - Feature importance comparison CSV
#    - SHAP and LIME explanation plots
#    - Ablation study results
#    - Comprehensive analysis logs
# 
# 2. **Business Implementation**: Use the business insights to:
#    - Optimize bike distribution strategies
#    - Improve demand forecasting models
#    - Develop weather-based operational plans
# 
# 3. **Model Deployment**: Choose the best model based on:
#    - Performance metrics (R², RMSE, MAE)
#    - Interpretability requirements
#    - Computational efficiency needs
# 
# 4. **Further Analysis**: Consider:
#    - Time-series specific explainability
#    - Seasonal pattern analysis
#    - Real-time prediction monitoring
