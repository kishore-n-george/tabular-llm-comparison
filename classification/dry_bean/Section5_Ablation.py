#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our custom analysis frameworks
from explainability_analysis import ExplainabilityAnalyzer, clear_memory, save_intermediate_results,load_intermediate_results
from enhanced_evaluation import ComprehensiveEvaluator
from ablation_studies import AblationStudyAnalyzer
from enhanced_ablation_studies import run_enhanced_ablation_studies, load_model_ablation_results

# Memory management utilities
import gc
import pickle
import os
import xgboost as xgb
import logging
import sys

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Ablation.log'),
        logging.StreamHandler(sys.stdout)  # Also print to console
    ]
)

# Usage
logging.info("This will be written to Ablation.log")
logging.error("Error messages too")

print("🔍 Section 5: Ablation Studies")
print("Dataset: Dry Bean Classification")


# In[2]:


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('Section5_Ablation', exist_ok=True)

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


# In[3]:


# Create FT-Transformer PyTorch wrapper for ablation studies
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Check if FT-Transformer exists in models
if 'FT-Transformer' in models:
    print("🔧 Creating PyTorch wrapper for FT-Transformer...")
    original_ft_transformer = models['FT-Transformer']

    try:
        ft_wrapper = evaluator.create_pytorch_wrapper(
            model=original_ft_transformer,
            device=device,
            batch_size=256
        )

        # Replace the original model with the wrapper for ablation studies
        models['FT-Transformer'] = ft_wrapper
        print("✅ FT-Transformer wrapper created successfully!")

    except Exception as e:
        print(f"⚠️ Warning: Failed to create FT-Transformer wrapper: {e}")
        print("Continuing with original FT-Transformer model...")
else:
    print("ℹ️ FT-Transformer not found in models, skipping wrapper creation.")


# 
# 
# ## Advanced XGBoost, FT-transformer, TabICL and TabPFN Ablation Analysis
# 
# This notebook integrates enhanced ablation studies for TabICL and TabPFN models, building upon the comprehensive comparison analysis. It assumes that the following are already available from the comprehensive comparison notebook:
# 
# - **Preprocessed Data**: `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`, `feature_names`
# - **Trained Models**: `models` dictionary containing XGBoost, TabPFN v2, and TabICL
# - **Model Names**: `model_names` list
# 
# ### Enhanced Ablation Studies Include:
# - **TabPFN-Specific**: Context size optimization, device performance, memory efficiency
# - **TabICL-Specific**: In-context learning examples, example selection strategies, context window utilization
# - **Cross-Model Analysis**: Feature importance comparison, robustness analysis
# - **Production Insights**: Performance recommendations and deployment considerations

# ## Import Enhanced Ablation Framework

# In[4]:


# Import enhanced ablation studies framework
from enhanced_ablation_studies import (
    EnhancedAblationStudyAnalyzer, 
    run_enhanced_ablation_studies,
    create_ablation_summary_dataframe,
    plot_ablation_dashboard
)

print("🔬 Enhanced Ablation Studies Framework Loaded")
print("Ready to perform advanced ablation analysis on TabICL and TabPFN models")


# ## Verify Available Resources
# 
# Let's confirm that all required resources from the comprehensive comparison are available.

# In[5]:


# Verify that required variables are available from comprehensive comparison notebook
required_vars = [
    'X_train_scaled', 'X_test_scaled', 'y_train', 'y_test', 
    'feature_names', 'models', 'model_names'
]

model_names= models.keys()
import os
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
print("🔍 Verifying Required Resources:")
for var_name in required_vars:
    if var_name in globals():
        if var_name == 'models':
            print(f"   ✅ {var_name}: {list(models.keys())}")
        elif var_name in ['X_train_scaled', 'X_test_scaled']:
            print(f"   ✅ {var_name}: shape {globals()[var_name].shape}")
        elif var_name in ['y_train', 'y_test']:
            print(f"   ✅ {var_name}: length {len(globals()[var_name])}")
        elif var_name == 'feature_names':
            print(f"   ✅ {var_name}: {len(feature_names)} features")
        else:
            print(f"   ✅ {var_name}: available")
    else:
        print(f"   ❌ {var_name}: NOT FOUND - Please run comprehensive comparison notebook first")

print(f"\n📊 Models available for enhanced ablation studies: {model_names}")


# ## Initialize Enhanced Ablation Studies
# 
# Now we'll run the comprehensive enhanced ablation studies specifically designed for TabICL and TabPFN.

# In[ ]:


# Initialize enhanced ablation study analyzer
print("🚀 INITIALIZING ENHANCED ABLATION STUDIES")
print("=" * 80)

enhanced_analyzer = EnhancedAblationStudyAnalyzer()


# In[ ]:


# Run comprehensive ablation studies
print("\n🔬 Starting enhanced ablation analysis...")
print("This will perform model-specific ablations for TabICL and TabPFN")
print("Expected duration: 5-15 minutes depending on hardware")

enhanced_ablation_results = enhanced_analyzer.comprehensive_ablation_study(
    models_dict=models,
    model_names=model_names,
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names
)


# ## Enhanced Ablation Results Summary

# In[ ]:


# Create comprehensive summary
ablation_summary_df = create_ablation_summary_dataframe(enhanced_ablation_results)

print("\n📊 ENHANCED ABLATION STUDY SUMMARY")
print("=" * 80)
print(ablation_summary_df.round(4).to_string())

# Save results for further analysis
ablation_summary_df.to_csv('enhanced_ablation_summary.csv', index=False)
print("\n💾 Results saved to 'enhanced_ablation_summary.csv'")


# ## Comprehensive Ablation Dashboard

# In[ ]:


#temp = load_model_ablation_results('XGBoost')

#enhanced_analyzer.results[temp['model_name']] = temp['results']

#temp = load_model_ablation_results('FT-Transformer')

#enhanced_analyzer.results[temp['model_name']] = temp['results']

#temp = load_model_ablation_results('TabICL')

#enhanced_analyzer.results[temp['model_name']] = temp['results']

#temp = load_model_ablation_results('TabPFN v2')

#enhanced_analyzer.results[temp['model_name']] = temp['results']
#print(enhanced_analyzer.results)


# In[ ]:


# enhanced_analyzer.results['XGBoost']
# Generate comprehensive dashboard
print("📈 Generating Enhanced Ablation Dashboard...")
plot_ablation_dashboard(enhanced_analyzer, model_names)


# ## TabPFN-Specific Enhanced Ablations
# 
# Deep dive into TabPFN-specific ablation studies including context size optimization, device performance analysis, and memory efficiency testing.

# In[ ]:


# TabPFN-specific analysis
# enhanced_ablation_results = enhanced_analyzer.results
if 'TabPFN v2' in enhanced_ablation_results:
    tabpfn_results = enhanced_ablation_results['TabPFN v2']

    print("\n🔬 TABPFN ENHANCED ABLATION ANALYSIS")
    print("=" * 80)

    # Context Size Optimization
    if 'context_size_ablation' in tabpfn_results:
        context_results = tabpfn_results['context_size_ablation']

        print("\n📏 Context Size Optimization Results:")
        context_df = pd.DataFrame(context_results)
        print(context_df.round(4).to_string())

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Performance vs Context Size
        ax1.plot(context_df['context_size'], context_df['f1_score'], 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Context Size')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('TabPFN: Context Size vs Performance')
        ax1.grid(True, alpha=0.3)

        # Training Time vs Context Size
        ax2.plot(context_df['context_size'], context_df['train_time'], 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Context Size')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('TabPFN: Context Size vs Training Time')
        ax2.grid(True, alpha=0.3)

        # Accuracy vs Context Size
        ax3.plot(context_df['context_size'], context_df['accuracy'], 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Context Size')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('TabPFN: Context Size vs Accuracy')
        ax3.grid(True, alpha=0.3)

        # Efficiency (Performance/Time)
        efficiency = context_df['f1_score'] / context_df['train_time']
        ax4.plot(context_df['context_size'], efficiency, 'o-', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Context Size')
        ax4.set_ylabel('Efficiency (F1/Time)')
        ax4.set_title('TabPFN: Context Size vs Efficiency')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig('TabPFN_Ablation_Results.png')

        # Optimal context size analysis
        best_f1_idx = context_df['f1_score'].idxmax()
        best_efficiency_idx = efficiency.idxmax()

        print(f"\n🎯 TabPFN Context Size Insights:")
        print(f"   Optimal for F1: Context size {context_df.loc[best_f1_idx, 'context_size']} (F1: {context_df.loc[best_f1_idx, 'f1_score']:.4f})")
        print(f"   Optimal for Efficiency: Context size {context_df.loc[best_efficiency_idx, 'context_size']} (Efficiency: {efficiency.iloc[best_efficiency_idx]:.4f})")
        print(f"   Performance Range: {context_df['f1_score'].min():.4f} - {context_df['f1_score'].max():.4f}")
        print(f"   Time Range: {context_df['train_time'].min():.2f}s - {context_df['train_time'].max():.2f}s")


# ## TabICL-Specific Enhanced Ablations
# 
# Comprehensive analysis of TabICL's in-context learning capabilities, including context examples optimization, example selection strategies, and context window utilization.

# In[ ]:


# TabICL-specific analysis
if 'TabICL' in enhanced_ablation_results:
    tabicl_results = enhanced_ablation_results['TabICL']

    print("\n🎯 TABICL ENHANCED ABLATION ANALYSIS")
    print("=" * 80)

    # Context Examples Optimization
    if 'context_examples_ablation' in tabicl_results:
        examples_results = tabicl_results['context_examples_ablation']

        print("\n📝 Context Examples Optimization Results:")
        examples_df = pd.DataFrame(examples_results)
        print(examples_df.round(4).to_string())

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Performance vs Context Examples
        ax1.plot(examples_df['context_examples'], examples_df['f1_score'], 'o-', 
                linewidth=2, markersize=8, color='purple')
        ax1.set_xlabel('Number of Context Examples')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('TabICL: Context Examples vs Performance')
        ax1.grid(True, alpha=0.3)

        # Training Time vs Context Examples
        ax2.plot(examples_df['context_examples'], examples_df['train_time'], 'o-', 
                linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Number of Context Examples')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('TabICL: Context Examples vs Training Time')
        ax2.grid(True, alpha=0.3)

        # Accuracy vs Context Examples
        ax3.plot(examples_df['context_examples'], examples_df['accuracy'], 'o-', 
                linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Number of Context Examples')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('TabICL: Context Examples vs Accuracy')
        ax3.grid(True, alpha=0.3)

        # Learning Efficiency
        learning_efficiency = examples_df['f1_score'] / examples_df['context_examples']
        ax4.plot(examples_df['context_examples'], learning_efficiency, 'o-', 
                linewidth=2, markersize=8, color='orange')
        ax4.set_xlabel('Number of Context Examples')
        ax4.set_ylabel('Learning Efficiency (F1/Examples)')
        ax4.set_title('TabICL: Context Examples vs Learning Efficiency')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig('TabICL_Ablation_Results.png')

        # Optimal context examples analysis
        best_f1_idx = examples_df['f1_score'].idxmax()
        best_efficiency_idx = learning_efficiency.idxmax()

        print(f"\n🎯 TabICL Context Examples Insights:")
        print(f"   Optimal for F1: {examples_df.loc[best_f1_idx, 'context_examples']} examples (F1: {examples_df.loc[best_f1_idx, 'f1_score']:.4f})")
        print(f"   Most Efficient: {examples_df.loc[best_efficiency_idx, 'context_examples']} examples (Efficiency: {learning_efficiency.iloc[best_efficiency_idx]:.4f})")
        print(f"   Performance Range: {examples_df['f1_score'].min():.4f} - {examples_df['f1_score'].max():.4f}")
        print(f"   Time Range: {examples_df['train_time'].min():.2f}s - {examples_df['train_time'].max():.2f}s")


# ## Cross-Model Feature Importance Analysis

# In[ ]:


# Cross-model feature importance comparison
#temp = {}
#filename='online_shoppers_section5_ablation_results.pkl'
#with open(filename, 'rb') as f:
#        temp = pickle.load(f)

#print(temp)
#enhanced_ablation_results = temp['all_results']
if 'comparative_analysis' in enhanced_ablation_results:
    comparative_results = enhanced_ablation_results['comparative_analysis']

    print("\n🎯 CROSS-MODEL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    if 'feature_importance_comparison' in comparative_results:
        importance_comparison = comparative_results['feature_importance_comparison']

        # Create feature importance comparison table
        all_features = set()
        for model_importance in importance_comparison.values():
            all_features.update(model_importance.keys())
        all_features = sorted(list(all_features))

        importance_df = pd.DataFrame(index=all_features)
        for model_name, feature_importance in importance_comparison.items():
            importance_df[model_name] = [feature_importance.get(feature, 0) for feature in all_features]

        print("\nFeature Importance Comparison (Relative Importance):")
        print(importance_df.round(4).to_string())

        # Plot feature importance heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(importance_df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Relative Importance'})
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Features')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        plt.savefig('Ablation_Feature_Imp_Across_Models_Results.png')

        # Top features consensus
        print("\n🏆 TOP FEATURES CONSENSUS:")

        # Calculate average importance across models
        avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)

        print("\nTop 5 Most Important Features (Average Across Models):")
        for i, (feature, importance) in enumerate(avg_importance.head().items()):
            print(f"   {i+1}. {feature}: {importance:.4f}")

        # Model agreement analysis
        print("\nModel Agreement on Top Features:")
        for model in importance_df.columns:
            top_features = importance_df[model].nlargest(3).index.tolist()
            print(f"   {model}: {', '.join(top_features)}")

