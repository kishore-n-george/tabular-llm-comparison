#!/usr/bin/env python
# coding: utf-8

# # Section 3: Explainability Analysis and Ablation Studies (Simplified)
# 
# ## Airbnb Dataset - Model Interpretability for Regression
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
# - Focus on understanding what drives Airbnb price prediction
# - Identify most important features for rental price prediction

# ## 1. Import Required Libraries and Functions

# In[ ]:


# Import the wrapped functions from our custom module
# Import the wrapped functions from our custom module
from section3_explainability_functions import (
    setup_analysis_environment,
    load_section2_results,
    load_preprocessed_data,
    load_and_filter_models,
    initialize_explainability_analyzer,
    analyze_xgboost_explainability,
    analyze_ft_transformer_enhanced_explainability,
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
    section2_results_file='./airbnb_section2_results.pkl',
    preprocessed_data_file='./Section1_Data_PreProcessing/enhanced_data.pkl',
    model_dir='./Section2_Model_Training',
    save_dir='./Section3_Explainability',
    min_r2_threshold=0.5,
    max_samples=300
)

print("\n🎉 Complete explainability analysis finished!")
print(f"📊 Models analyzed: {results['models_analyzed']}")
print(f"📁 Results saved to: {results['results_file']}")

