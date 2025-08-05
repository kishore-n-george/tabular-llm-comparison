# Dry Bean Extended Error Analysis

This module provides comprehensive error analysis functions for the dry bean classification models, adapted from the online shoppers extended error analysis framework. It offers detailed insights into model performance, error patterns, and actionable recommendations for improvement.

## Features

### 1. Cross-Model Error Comparison
- Compare error patterns across all trained models
- Detailed metrics including accuracy, precision, recall, F1-score
- Both weighted and macro averages for multi-class classification
- Performance variability analysis across classes

### 2. Model-Specific Confidence Analysis
- Prediction confidence distribution analysis
- Comparison of confidence between correct and misclassified predictions
- Low and high confidence error identification
- Statistical confidence metrics

### 3. Feature-based Error Analysis
- Identify features most associated with model errors
- Statistical significance testing (t-tests)
- Feature difference analysis between correct and incorrect predictions
- Visualization of problematic features

### 4. Error Overlap Analysis
- Common errors across all models
- Model-specific unique errors
- Pairwise error overlap analysis
- Error pattern visualization

### 5. Model-Specific Insights
- Tailored insights for each model type:
  - **XGBoost**: Tree-based error patterns
  - **TabPFN**: Prior-based prediction issues
  - **TabICL**: In-context learning problems
  - **FT-Transformer**: Attention mechanism insights
- Performance-based recommendations

### 6. Comprehensive Summary and Recommendations
- Performance ranking
- Key findings summary
- Actionable recommendations for:
  - Model selection
  - Error reduction strategies
  - Ensemble opportunities
  - Data quality improvements
  - Model-specific optimizations

## Installation and Requirements

### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from scipy.stats import ttest_ind
import pickle
import torch
```

### Data Requirements
- Trained models from Section 2 (saved in `dry_bean_section2_results.pkl`)
- Preprocessed data with scaled features
- Model predictions and probabilities

## Usage

### Method 1: Complete Analysis (Recommended)
```python
from dry_bean_extended_error_analysis import run_dry_bean_error_analysis

# Run complete analysis with all components
analyzer = run_dry_bean_error_analysis('dry_bean_section2_results.pkl')
```

### Method 2: Quick Error Comparison
```python
from dry_bean_extended_error_analysis import quick_error_comparison

# Get just the cross-model error comparison
error_df = quick_error_comparison('dry_bean_section2_results.pkl')
print(error_df)
```

### Method 3: Custom Step-by-Step Analysis
```python
from dry_bean_extended_error_analysis import DryBeanErrorAnalyzer

# Initialize analyzer
analyzer = DryBeanErrorAnalyzer()

# Load data
analyzer.load_data('dry_bean_section2_results.pkl')

# Run specific analyses
analyzer.generate_predictions()
analyzer.cross_model_error_comparison()
analyzer.confidence_analysis()
analyzer.feature_based_error_analysis()
analyzer.error_overlap_analysis()
analyzer.model_specific_insights()
analyzer.comprehensive_summary()

# Save results
analyzer.save_analysis_results('my_results')
```

### Method 4: Individual Analysis Components
```python
analyzer = DryBeanErrorAnalyzer()
analyzer.load_data('dry_bean_section2_results.pkl')
analyzer.generate_predictions()

# Run only specific analyses
error_df = analyzer.cross_model_error_comparison()
confidence_stats = analyzer.confidence_analysis()
feature_analysis = analyzer.feature_based_error_analysis()
overlap_analysis = analyzer.error_overlap_analysis()
```

## Output Files

### Generated Visualizations
- `dry_bean_cross_model_error_comparison.png`: Error rate and accuracy comparisons
- `dry_bean_model_confidence_analysis.png`: Confidence distribution plots
- `dry_bean_feature_error_analysis.png`: Feature importance in errors
- `dry_bean_error_overlap_analysis.png`: Error overlap patterns

### Generated CSV Files
- `dry_bean_cross_model_error_comparison.csv`: Detailed error metrics
- `dry_bean_model_confidence_statistics.csv`: Confidence analysis results
- `{ModelName}_feature_error_analysis.csv`: Per-model feature analysis
- `error_overlap_analysis.csv`: Error overlap statistics

### Results Directory Structure
```
dry_bean_error_analysis_results/
├── cross_model_error_comparison.csv
├── confidence_statistics.csv
├── XGBoost_feature_error_analysis.csv
├── TabPFN_v2_feature_error_analysis.csv
├── TabICL_feature_error_analysis.csv
├── FT-Transformer_feature_error_analysis.csv
└── error_overlap_analysis.csv
```

## Key Metrics Explained

### Error Rate Metrics
- **Total Errors**: Number of misclassified samples
- **Error Rate %**: Percentage of misclassified samples
- **Accuracy**: Percentage of correctly classified samples

### Performance Metrics
- **Precision (Weighted)**: Weighted average precision across classes
- **Recall (Weighted)**: Weighted average recall across classes
- **F1 (Weighted)**: Weighted average F1-score across classes
- **Precision/Recall/F1 (Macro)**: Unweighted average across classes
- **Standard Deviation**: Performance variability across classes

### Confidence Metrics
- **Avg Confidence (Correct)**: Average confidence for correct predictions
- **Avg Confidence (Misclassified)**: Average confidence for errors
- **Low Confidence Errors %**: Percentage of errors with confidence < 60%
- **High Confidence Errors %**: Percentage of errors with confidence > 80%

### Feature Analysis Metrics
- **Mean Difference**: Average feature value difference (misclassified - correct)
- **P-Value**: Statistical significance of the difference
- **Significant Features**: Features with p-value < 0.05

## Model-Specific Insights

### XGBoost
- Tree-based error patterns
- Feature interaction issues
- Overfitting indicators
- Hyperparameter optimization suggestions

### TabPFN v2
- Prior-based prediction challenges
- Out-of-distribution sample detection
- Context size optimization
- Distribution shift analysis

### TabICL
- In-context learning effectiveness
- Example selection quality
- Context window utilization
- Performance dependency on examples

### FT-Transformer
- Attention mechanism analysis
- Feature tokenization effectiveness
- Architecture optimization suggestions
- Training strategy recommendations

## Example Analysis Workflow

1. **Load and Prepare Data**
   ```python
   analyzer = DryBeanErrorAnalyzer()
   analyzer.load_data('dry_bean_section2_results.pkl')
   ```

2. **Generate Predictions**
   ```python
   analyzer.generate_predictions()
   ```

3. **Compare Models**
   ```python
   error_df = analyzer.cross_model_error_comparison()
   analyzer.visualize_error_comparison()
   ```

4. **Analyze Confidence**
   ```python
   confidence_stats = analyzer.confidence_analysis()
   ```

5. **Identify Problematic Features**
   ```python
   feature_analysis = analyzer.feature_based_error_analysis()
   analyzer.visualize_feature_error_analysis()
   ```

6. **Study Error Patterns**
   ```python
   overlap_analysis = analyzer.error_overlap_analysis()
   analyzer.visualize_error_overlap()
   ```

7. **Get Insights and Recommendations**
   ```python
   insights = analyzer.model_specific_insights()
   analyzer.comprehensive_summary()
   ```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `dry_bean_section2_results.pkl` exists
2. **Memory Issues**: Use smaller batch sizes for FT-Transformer
3. **Missing Probabilities**: Some models may not support `predict_proba`
4. **Empty Results**: Check if models were trained successfully

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data loading
analyzer = DryBeanErrorAnalyzer()
success = analyzer.load_data('dry_bean_section2_results.pkl')
print(f"Data loaded: {success}")
print(f"Models available: {analyzer.model_names}")
```

## Advanced Usage

### Custom Analysis for Specific Models
```python
def analyze_model_performance(model_name):
    analyzer = DryBeanErrorAnalyzer()
    analyzer.load_data('dry_bean_section2_results.pkl')
    analyzer.generate_predictions()
    
    if model_name in analyzer.predictions:
        # Custom analysis for specific model
        errors = len(analyzer.misclassified_indices[model_name])
        error_rate = errors / len(analyzer.y_test) * 100
        print(f"{model_name} Error Rate: {error_rate:.2f}%")
        
        # Feature analysis
        analyzer.feature_based_error_analysis()
        if analyzer.feature_error_analysis.get(model_name):
            feature_data = analyzer.feature_error_analysis[model_name]
            sig_features = np.sum(feature_data['significant_features'])
            print(f"Significant Features: {sig_features}")

# Usage
analyze_model_performance('XGBoost')
```

### Batch Analysis for Multiple Datasets
```python
def batch_error_analysis(data_files):
    results = {}
    for file_path in data_files:
        print(f"Analyzing {file_path}...")
        analyzer = run_dry_bean_error_analysis(file_path)
        if analyzer:
            results[file_path] = analyzer.error_df
    return results
```

## Contributing

To extend the error analysis framework:

1. Add new analysis methods to the `DryBeanErrorAnalyzer` class
2. Update visualization functions for new metrics
3. Add model-specific insights for new model types
4. Extend the summary and recommendations section

## References

- Original framework adapted from online shoppers extended error analysis
- Statistical methods based on scikit-learn metrics
- Visualization using matplotlib and seaborn
- Model-specific insights based on literature and best practices
