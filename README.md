# Tabular LLM Comparison: A Comprehensive Study

A systematic comparison of state-of-the-art tabular machine learning models including XGBoost, TabPFN v2, TabICL, FT-Transformer, and SAINT across classification and regression tasks.

## 🎯 Project Overview

This repository contains a comprehensive empirical study comparing traditional machine learning methods with modern tabular-specific deep learning models. The study evaluates model performance, explainability, computational efficiency, and robustness across multiple datasets and tasks.

### Models Compared
- **XGBoost**: Gradient boosting baseline
- **TabPFN v2**: Prior-Fitted Networks for tabular data
- **TabICL**: In-Context Learning for tabular data
- **FT-Transformer**: Feature Tokenizer + Transformer
- **SAINT**: Self-Attention and Intersample Attention Transformer

### Key Research Questions
1. How do modern tabular LLMs compare to traditional methods?
2. What are the trade-offs between performance and computational efficiency?
3. Which models provide better explainability and interpretability?
4. How robust are these models to different data characteristics?

## 📁 Repository Structure

```
tabular-llm-comparison/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── classification/             # Classification experiments
│   ├── dry_bean/              # Dry Bean Dataset (7-class classification)
│   └── online_shoppers/       # Online Shoppers Dataset (binary classification)
│
└── regression/                # Regression experiments
    ├── airbnb/               # Airbnb Price Prediction
    └── bike_sharing/         # Bike Sharing Demand Prediction
```

## 🔬 Experimental Framework

Each dataset follows a standardized 5-section experimental pipeline:

### Section 1: Data Preprocessing
- **Purpose**: Data loading, cleaning, and feature engineering
- **Key Files**: `section1_data_preprocessing.ipynb`
- **Outputs**: Preprocessed datasets, feature analysis visualizations
- **Analysis**: Distribution analysis, feature correlation, class balance

### Section 2: Model Training & Evaluation
- **Purpose**: Train all models and compare baseline performance
- **Key Files**: `section2_model_training.ipynb`, model-specific training scripts
- **Outputs**: Trained models, performance metrics, comparison tables
- **Metrics**: Accuracy, F1-score, Precision, Recall, ROC-AUC (classification) / R², RMSE, MAE (regression)

### Section 3: Explainability Analysis
- **Purpose**: Model interpretability and feature importance analysis
- **Key Files**: `Section3_explainability_ablation.ipynb`, `explainability_analysis.py`
- **Methods**: SHAP, LIME, feature importance, attention visualization
- **Outputs**: Feature importance rankings, explanation visualizations

### Section 4: Error Analysis
- **Purpose**: Deep dive into model failures and error patterns
- **Key Files**: `Section4_ErrorAnalysis.ipynb`, `*_error_analysis.py`
- **Analysis**: Confidence analysis, feature-based errors, error overlap
- **Outputs**: Error analysis reports, misclassification patterns

### Section 5: Ablation Studies
- **Purpose**: Systematic analysis of model components and hyperparameters
- **Key Files**: `Section5_Ablation.ipynb`, `ablation_studies.py`
- **Studies**: Feature ablation, hyperparameter sensitivity, data size effects
- **Outputs**: Ablation results, optimization recommendations

## 📊 Datasets & Results Summary

### Classification Tasks

#### Dry Bean Dataset (7-class classification)
- **Samples**: 13,611 instances
- **Features**: 16 morphological features
- **Classes**: 7 bean varieties
- **Best Model**: TabICL (92.99% accuracy)

| Model | Accuracy | F1-Score | Training Time | Inference Speed |
|-------|----------|----------|---------------|-----------------|
| XGBoost | 92.07% | 92.07% | 1.12s | 128,463 pred/s |
| TabPFN v2 | 92.95% | 92.94% | 0.45s | 333 pred/s |
| **TabICL** | **92.99%** | **92.97%** | 0.57s | 214 pred/s |
| FT-Transformer | 92.51% | 92.53% | 0.00s | 57,305 pred/s |

#### Online Shoppers Dataset (binary classification)
- **Samples**: 12,330 instances
- **Features**: 17 behavioral features
- **Classes**: Purchase intention (highly imbalanced)
- **Best Model**: TabPFN v2 (90.11% accuracy)

| Model | Accuracy | F1-Score | ROC-AUC | Training Time | Inference Speed |
|-------|----------|----------|---------|---------------|-----------------|
| XGBoost | 89.74% | 63.07% | 92.12% | 0.35s | 746,475 pred/s |
| **TabPFN v2** | **90.11%** | **64.94%** | **93.26%** | 0.54s | 315 pred/s |
| TabICL | 89.94% | 64.27% | 93.40% | 0.59s | 218 pred/s |
| FT-Transformer | 89.58% | 64.65% | 92.20% | 0.00s | 54,986 pred/s |

### Regression Tasks

#### Bike Sharing Dataset
- **Samples**: 17,379 instances
- **Target**: Hourly bike rental count
- **Features**: Weather, temporal, and situational features
- **Challenge**: Fixed FT-Transformer negative R² issue (detailed in `FT_Transformer_Issue_Analysis_and_Fix.md`)

#### Airbnb Price Prediction
- **Samples**: Variable (location-dependent)
- **Target**: Listing price prediction
- **Features**: Property characteristics, location, amenities

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for deep learning models)

### Installation
```bash
# Clone the repository
git clone https://github.com/kishore-n-george/tabular-llm-comparison.git
cd tabular-llm-comparison

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- **Core ML**: `scikit-learn`, `xgboost`, `torch`, `transformers`
- **Tabular Models**: `tabpfn`, `tabicl`, `rtdl`
- **Explainability**: `shap`, `lime`, `eli5`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Data**: `pandas`, `numpy`, `ucimlrepo`

## 🚀 Quick Start

### Running a Complete Experiment
```bash
# Navigate to a dataset directory
cd classification/dry_bean/

# Run the complete pipeline
jupyter notebook section1_data_preprocessing.ipynb
jupyter notebook section2_model_training.ipynb
jupyter notebook Section3_explainability_ablation.ipynb
jupyter notebook Section4_ErrorAnalysis.ipynb
jupyter notebook Section5_Ablation.ipynb
```


## 📈 Key Findings

### Performance Insights
1. **TabPFN v2** and **TabICL** consistently outperform traditional methods
2. **XGBoost** remains competitive with much faster inference
3. **FT-Transformer** shows promise but requires careful tuning
4. Performance gaps are dataset-dependent

### Computational Trade-offs
- **Training Speed**: XGBoost > TabPFN v2 > TabICL > FT-Transformer
- **Inference Speed**: XGBoost >> FT-Transformer >> TabPFN v2 > TabICL
- **Memory Usage**: XGBoost < FT-Transformer < TabPFN v2 < TabICL

### Explainability Analysis
- **XGBoost**: Native feature importance, highly interpretable
- **TabPFN v2**: Limited explainability, black-box nature
- **TabICL**: Context-based explanations possible
- **FT-Transformer**: Attention weights provide some interpretability

### Robustness Findings
- **Class Imbalance**: TabPFN v2 and TabICL handle imbalance better
- **Feature Scaling**: Deep models more sensitive to preprocessing
- **Dataset Size**: TabPFN v2 excels on smaller datasets
- **Noise Robustness**: XGBoost shows better noise tolerance

## 🔍 Advanced Analysis Features

### Error Analysis Framework
- **Cross-model error comparison**: Identify common failure patterns
- **Confidence analysis**: Understand model uncertainty
- **Feature-based error analysis**: Find problematic feature combinations
- **Error overlap analysis**: Discover complementary model strengths

### Ablation Study Framework
- **Feature ablation**: Systematic feature importance analysis
- **Hyperparameter sensitivity**: Robustness to parameter changes
- **Data size effects**: Performance scaling with dataset size
- **Architecture ablation**: Component importance in deep models

### Enhanced Evaluation Metrics
- **Statistical significance testing**: Robust performance comparisons
- **Calibration analysis**: Prediction confidence reliability
- **Fairness metrics**: Bias detection across subgroups
- **Efficiency metrics**: Performance per computational cost

## 📚 Documentation

### Dataset-Specific Documentation
- [`classification/dry_bean/README_error_analysis.md`](classification/dry_bean/README_error_analysis.md): Detailed error analysis framework
- [`regression/bike_sharing/FT_Transformer_Issue_Analysis_and_Fix.md`](regression/bike_sharing/FT_Transformer_Issue_Analysis_and_Fix.md): FT-Transformer debugging guide

### Analysis Frameworks
Each dataset includes modular analysis frameworks:
- `enhanced_evaluation.py`: Comprehensive evaluation metrics
- `explainability_analysis.py`: Model interpretability tools
- `ablation_studies.py`: Systematic ablation framework
- `*_error_analysis.py`: Deep error analysis tools

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- Additional datasets and benchmarks
- New tabular models integration
- Enhanced explainability methods
- Computational efficiency optimizations
- Fairness and bias analysis tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Kishore N George
- **Institution**: Liverpool John Moores University (LJMU)
- **Email**: [Contact through GitHub Issues]
- **Repository**: https://github.com/kishore-n-george/tabular-llm-comparison

## 🙏 Acknowledgments

- **Datasets**: UCI Machine Learning Repository
- **Models**: TabPFN, TabICL, FT-Transformer, XGBoost development teams
- **Frameworks**: PyTorch, scikit-learn, Hugging Face Transformers
- **Visualization**: Matplotlib, Seaborn, Plotly communities

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{george2024tabular,
  title={Tabular LLM Comparison: A Comprehensive Study},
  author={George, Kishore N},
  year={2024},
  institution={Liverpool John Moores University},
  url={https://github.com/kishore-n-george/tabular-llm-comparison}
}
```

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Active Development
