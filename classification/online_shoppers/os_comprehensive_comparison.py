# %% [markdown]
# # Comprehensive Tabular Model Comparison
# ## XGBoost vs TabPFNv2 vs TabICL vs FT-Transformer
# 
# This notebook provides a comprehensive comparison of tabular machine learning models with:
# - **Enhanced Evaluation Metrics**: ROC curves, calibration plots, statistical significance
# - **Explainability Analysis**: SHAP, LIME, feature importance
# - **Ablation Studies**: Feature importance, hyperparameter sensitivity
# - **Class Imbalance Analysis**: Performance on minority class
# - **Computational Efficiency**: Training time, inference speed, memory usage
# Configure TabPFN for large datasets
import os
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'


# %%
# Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from rtdl import FTTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# UCI ML Repository
from ucimlrepo import fetch_ucirepo

# Our custom analysis frameworks
from enhanced_evaluation import ComprehensiveEvaluator
from explainability_analysis import ExplainabilityAnalyzer
from ablation_studies import AblationStudyAnalyzer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

import logging

logging.basicConfig(
    filename='online_shoppers.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)

print = logging.info  # Redirect print

print("This goes to output.log")


# %% [markdown]
# ## 1. Data Loading and Preprocessing

# %%
# Load the Online Shoppers Purchasing Intention Dataset
print("Loading Online Shoppers Dataset...")
online_original = fetch_ucirepo(id=468)
online_shoppers = online_original.data.original

print(f"Dataset shape: {online_shoppers.shape}")
print(f"\nDataset info:")
print(online_shoppers.info())

# Display class distribution
print(f"\nClass distribution:")
print(online_shoppers['Revenue'].value_counts())
print(f"Class imbalance ratio: {online_shoppers['Revenue'].value_counts()[False] / online_shoppers['Revenue'].value_counts()[True]:.2f}:1")

# %%
# Preprocessing
df = online_shoppers.copy()

# Encode categorical features
label_encoder = LabelEncoder()
df['Month'] = label_encoder.fit_transform(df['Month'])
df['VisitorType'] = label_encoder.fit_transform(df['VisitorType'])
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Define feature names
feature_names = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
    'VisitorType', 'Weekend'
]

# Separate features and target
X = df.drop(columns=['Revenue']).values
y = df['Revenue'].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {feature_names}")

# %%
# Split the data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Check class distribution in splits
print(f"\nClass distribution in train: {np.bincount(y_train)}")
print(f"Class distribution in val: {np.bincount(y_val)}")
print(f"Class distribution in test: {np.bincount(y_test)}")

# %% [markdown]
# ## 2. Model Training and Evaluation

# %%
# Initialize comprehensive evaluator
evaluator = ComprehensiveEvaluator()

# Store all models for later analysis
models = {}

# %% [markdown]
# ### 2.1 XGBoost

# %%
print("Training XGBoost...")

# XGBoost with optimized parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Evaluate with comprehensive metrics
xgb_results = evaluator.evaluate_model(
    xgb_model, "XGBoost", 
    X_train_scaled, X_test_scaled, y_train, y_test,
    X_val_scaled, y_val
)

models['XGBoost'] = xgb_model

# %% [markdown]
# ### 2.2 TabPFN v2

# %%
print("Training TabPFN v2...")

# TabPFN v2
tabpfn_model = TabPFNClassifier(device=\'cuda\' if torch.cuda.is_available() else \'cpu\', ignore_pretraining_limits=True)

# Evaluate with comprehensive metrics
tabpfn_results = evaluator.evaluate_model(
    tabpfn_model, "TabPFN v2", 
    X_train_scaled, X_test_scaled, y_train, y_test,
    X_val_scaled, y_val
)

models['TabPFN v2'] = tabpfn_model

# %% [markdown]
# ### 2.3 TabICL

# %%
print("Training TabICL...")

# from sklearn.base import BaseEstimator
# import numpy as np

# # Monkey patch for compatibility
# def _validate_data_fallback(self, X, y=None, reset=True, validate_separately=False, **check_params):
#     from sklearn.utils.validation import check_X_y, check_array
#     if y is not None:
#         return check_X_y(X, y, **check_params)
#     else:
#         return check_array(X, **check_params)

# # Apply the patch if needed
# if not hasattr(BaseEstimator, '_validate_data'):
#     BaseEstimator._validate_data = _validate_data_fallback


# TabICL
tabicl_model = TabICLClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate with comprehensive metrics
tabicl_results = evaluator.evaluate_model(
    tabicl_model, "TabICL", 
    X_train_scaled, X_test_scaled, y_train, y_test,
    X_val_scaled, y_val
)

models['TabICL'] = tabicl_model

# %% [markdown]
# ### 2.4 FT-Transformer

# %%
print("Training FT-Transformer...")

# Note: This is a simplified version. For full implementation, see the individual FT-Transformer notebook
print("FT-Transformer training would be implemented here with proper categorical/numerical feature separation")
print("For now, we'll skip this model to avoid complexity in the comprehensive comparison")

# %% [markdown]
# ## 3. Comprehensive Model Comparison

# %%
# Generate comprehensive comparison
comparison_df = evaluator.compare_models()

# Display detailed comparison table
print("\n" + "="*100)
print("DETAILED PERFORMANCE COMPARISON")
print("="*100)
print(comparison_df.round(4).to_string())

# %% [markdown]
# ## 4. Explainability Analysis

# %%
# Initialize explainability analyzer
explainer = ExplainabilityAnalyzer(feature_names=feature_names)

# Analyze XGBoost (most interpretable)
print("\n" + "="*60)
print("EXPLAINABILITY ANALYSIS")
print("="*60)

xgb_explanations = explainer.analyze_model_explainability(
    models['XGBoost'], "XGBoost", 
    X_train_scaled, X_test_scaled, y_train, y_test,
    max_samples=200
)

# %%
# Analyze TabPFN v2 explainability
tabpfn_explanations = explainer.analyze_model_explainability(
    models['TabPFN v2'], "TabPFN v2", 
    X_train_scaled, X_test_scaled, y_train, y_test,
    max_samples=100
)

# %%
# Compare feature importance across models
importance_comparison = explainer.compare_feature_importance()

# Generate explanation reports
explainer.generate_explanation_report("XGBoost")
explainer.generate_explanation_report("TabPFN v2")

# %% [markdown]
# ## 5. Ablation Studies

# %%
# Initialize ablation study analyzer
ablation_analyzer = AblationStudyAnalyzer()

print("\n" + "="*60)
print("ABLATION STUDIES")
print("="*60)

# %%
# Feature ablation study for XGBoost
xgb_ablation = ablation_analyzer.feature_ablation_study(
    models['XGBoost'], "XGBoost",
    X_train_scaled, X_test_scaled, y_train, y_test,
    feature_names=feature_names,
    max_features_to_remove=3
)

# %%
# Hyperparameter ablation study for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 6, 9]
}

xgb_hyperparam_ablation = ablation_analyzer.hyperparameter_ablation_study(
    xgb.XGBClassifier, "XGBoost",
    X_train_scaled, X_test_scaled, y_train, y_test,
    xgb_param_grid, cv_folds=3
)

# %%
# Data size ablation study
xgb_data_ablation = ablation_analyzer.data_size_ablation_study(
    models['XGBoost'], "XGBoost",
    X_train_scaled, X_test_scaled, y_train, y_test,
    size_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
)

# %%
# Generate ablation report
ablation_analyzer.generate_ablation_report("XGBoost")

# %% [markdown]
# ## 6. Class Imbalance Analysis

# %%
# Analyze performance on minority class
print("\n" + "="*60)
print("CLASS IMBALANCE ANALYSIS")
print("="*60)

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

for model_name, model in models.items():
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test_scaled)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        print(f"\n{model_name} - Per-Class Performance:")
        print(f"  Class 0 (No Purchase): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}")
        print(f"  Class 1 (Purchase): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}")
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"  True Negatives: {tn}, False Positives: {fp}")
        print(f"  False Negatives: {fn}, True Positives: {tp}")
        print(f"  Sensitivity (TPR): {tp/(tp+fn):.4f}")
        print(f"  Specificity (TNR): {tn/(tn+fp):.4f}")
        print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
        print(f"  False Negative Rate: {fn/(fn+tp):.4f}")

# %% [markdown]
# ## 7. Error Analysis

# %%
# Analyze misclassified samples
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Focus on XGBoost for detailed error analysis
xgb_pred = models['XGBoost'].predict(X_test_scaled)
xgb_proba = models['XGBoost'].predict_proba(X_test_scaled)[:, 1]

# Find misclassified samples
misclassified_mask = (xgb_pred != y_test)
misclassified_indices = np.where(misclassified_mask)[0]

print(f"Total misclassified samples: {len(misclassified_indices)} out of {len(y_test)}")
print(f"Misclassification rate: {len(misclassified_indices)/len(y_test)*100:.2f}%")

# Analyze confidence of misclassified samples
misclassified_proba = xgb_proba[misclassified_mask]
correct_proba = xgb_proba[~misclassified_mask]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(misclassified_proba, bins=20, alpha=0.7, label='Misclassified', color='red')
plt.hist(correct_proba, bins=20, alpha=0.7, label='Correct', color='green')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Prediction Confidence Distribution')
plt.legend()

plt.subplot(1, 2, 2)
# Feature importance for misclassified samples
misclassified_features = X_test_scaled[misclassified_mask]
correct_features = X_test_scaled[~misclassified_mask]

feature_diff = np.mean(misclassified_features, axis=0) - np.mean(correct_features, axis=0)
sorted_indices = np.argsort(np.abs(feature_diff))[::-1][:10]

plt.bar(range(10), feature_diff[sorted_indices])
plt.xlabel('Features')
plt.ylabel('Mean Difference (Misclassified - Correct)')
plt.title('Feature Differences in Misclassified Samples')
plt.xticks(range(10), [feature_names[i] for i in sorted_indices], rotation=45)

plt.tight_layout()
plt.show()

print(f"\nTop 5 features with largest differences in misclassified samples:")
for i, idx in enumerate(sorted_indices[:5]):
    print(f"   {i+1}. {feature_names[idx]}: {feature_diff[idx]:.4f}")

# %% [markdown]
# ## 8. Summary and Recommendations

# %%
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("="*80)

print("\n🎯 Key Findings:")
print("1. Performance Comparison: [Results from comprehensive evaluation]")
print("2. Feature Importance: [Key features identified across models]")
print("3. Computational Efficiency: [Training/inference time analysis]")
print("4. Class Imbalance Impact: [Minority class performance analysis]")
print("5. Error Patterns: [Common misclassification patterns]")

print("\n📊 Recommendations:")
print("1. Best Overall Model: [Based on comprehensive metrics]")
print("2. Best for Interpretability: [Most explainable model]")
print("3. Best for Speed: [Fastest training/inference]")
print("4. Best for Accuracy: [Highest performance model]")
print("5. Production Considerations: [Deployment recommendations]")

print("\n🔬 Future Work:")
print("1. Ensemble methods combining best models")
print("2. Advanced hyperparameter optimization")
print("3. Feature engineering based on importance analysis")
print("4. Class balancing techniques")
print("5. Model calibration improvements")


