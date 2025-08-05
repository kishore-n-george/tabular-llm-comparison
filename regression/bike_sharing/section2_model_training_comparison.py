

# In[ ]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ML libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# Import rtdl library for FT-Transformer
try:
    import rtdl
    print("✅ rtdl library imported successfully")
    print(f"rtdl version: {rtdl.__version__ if hasattr(rtdl, '__version__') else 'unknown'}")
except ImportError:
    print("❌ rtdl library not available. Install with: pip install rtdl")
    print("This notebook requires the rtdl library to run.")
    raise ImportError("Please install rtdl: pip install rtdl")

# Import our custom evaluation framework
try:
    from enhanced_evaluation import ComprehensiveEvaluator
    print("✅ Enhanced evaluation imported successfully")
except ImportError:
    print("⚠️ Enhanced evaluation not available. Using basic evaluation.")
    ComprehensiveEvaluator = None

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("🤖 FT-Transformer Training for Online Shoppers Classification")
print("Dataset: Online Shoppers Purchasing Intention")

# ## 2.9 Model Performance Summary

# In[ ]:


# Create comprehensive summary
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

summary_data = []
for model_name, results in evaluator.results.items():
    summary_data.append({
        'Model': model_name,
        'Accuracy': f"{results['accuracy']:.4f}",
        'F1-Score': f"{results['f1']:.4f}",
        'Precision': f"{results['precision']:.4f}",
        'Recall': f"{results['recall']:.4f}",
        'ROC-AUC': f"{results.get('auc_roc', 'N/A')}",
        'Training_Time': f"{results['train_time']:.2f}s",
        'Inference_Speed': f"{results['predictions_per_second']:.0f} pred/s"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Identify best performing model
if evaluator.results:
    best_model = max(evaluator.results.items(), key=lambda x: x[1]['f1'])
    print(f"\n🏆 Best performing model: {best_model[0]} (F1-Score: {best_model[1]['f1']:.4f})")

    # Additional insights for binary classification
    best_results = best_model[1]
    print(f"\n📈 Best Model Insights:")
    print(f"   Model: {best_model[0]}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"   Precision: {best_results['precision']:.4f}")
    print(f"   Recall: {best_results['recall']:.4f}")
    print(f"   F1-Score: {best_results['f1']:.4f}")
    if 'auc_roc' in best_results:
        print(f"   ROC-AUC: {best_results['auc_roc']:.4f}")
    print(f"   Training Time: {best_results['train_time']:.2f}s")
    print(f"   Inference Speed: {best_results['predictions_per_second']:.0f} predictions/second")

    # Business insights
    print(f"\n💼 Business Insights:")
    if best_results['precision'] > 0.8:
        print(f"   ✅ High precision ({best_results['precision']:.4f}) - Low false positive rate")
        print(f"   📊 Good for targeted marketing campaigns")
    if best_results['recall'] > 0.8:
        print(f"   ✅ High recall ({best_results['recall']:.4f}) - Captures most potential buyers")
        print(f"   📊 Good for maximizing revenue opportunities")

    # Save summary
    summary_df.to_csv('./Section2_Model_Training/online_shoppers_model_summary.csv', index=False)
    print("\n💾 Model summary saved to 'online_shoppers/Section2_Model_Training/online_shoppers_model_summary.csv'")

else:
    print("\n⚠️ No models were successfully trained")

print(f"\n✅ Section 2 completed successfully!")
print(f"📁 Ready for explainability analysis in Section 3")
print(f"\n📊 Models trained: {list(models.keys())}")
print(f"📈 Total training samples: {len(X_train_scaled):,}")
print(f"🎯 Task: Binary Classification (Purchase Prediction)")


# In[ ]:


# Save trained models and results for Section 3
import pickle

# Save all necessary variables for Section 3
section2_data = {
    'models': models,
    'evaluator': evaluator,
    'X_train_scaled': X_train_scaled,
    'X_val_scaled': X_val_scaled,
    'X_test_scaled': X_test_scaled,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'feature_names': feature_names,
    'month_mapping': month_mapping,
    'visitor_mapping': visitor_mapping,
    'class_names': class_names,
    'label_encoder_month': label_encoder_month,
    'label_encoder_visitor': label_encoder_visitor,
    'scaler': scaler,
    'data_summary': data_summary,
    'comparison_df': comparison_df if 'comparison_df' in locals() else None
}

# Save to pickle file
with open('./online_shoppers_section2_results.pkl', 'wb') as f:
    pickle.dump(section2_data, f)

print("💾 Section 2 results saved to 'online_shoppers/online_shoppers_section2_results.pkl'")
print("📋 This file contains all trained models and results for Section 3")

# Display final summary
print("\n" + "="*80)
print("SECTION 2 COMPLETION SUMMARY")
print("="*80)
print(f"✅ Models Successfully Trained: {len(models)}")
for model_name in models.keys():
    print(f"   - {model_name}")
print(f"\n📁 Files Generated:")
print(f"   - Model comparison CSV")
print(f"   - Per-class results for each model")
print(f"   - Class imbalance analysis")
print(f"   - Error analysis dashboard (PNG)")
print(f"   - Model summary CSV")
print(f"   - Section 2 results pickle file")
print(f"\n🎯 Dataset: Online Shoppers Purchasing Intention")
print(f"📊 Task: Binary Classification")
print(f"🔢 Features: {len(feature_names)}")
print(f"📈 Training Samples: {len(X_train_scaled):,}")
print(f"🧪 Test Samples: {len(X_test_scaled):,}")


# In[ ]: