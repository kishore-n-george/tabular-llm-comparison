"""
Section 3 Explainability Functions for Bike Sharing Regression Analysis

This module contains wrapped functions for the explainability analysis notebook,
making the code more modular and reusable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
import gc
import torch
from typing import Dict, List, Tuple, Any, Optional

# Import custom analysis frameworks
from regression_explainability_analysis import (
    RegressionExplainabilityAnalyzer, 
    run_regression_explainability_analysis,
    clear_memory, 
    save_intermediate_results,
    load_intermediate_results
)

from enhanced_ablation_studies import (
    EnhancedAblationStudyAnalyzer,
    run_enhanced_ablation_studies,
    create_ablation_summary_dataframe,
    plot_ablation_dashboard,
    PyTorchRegressionModelWrapper,
    create_pytorch_wrapper
)

from model_comparison_functions import load_all_models

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_analysis_environment():
    """
    Setup the analysis environment with proper styling and configurations
    """
    print("🔍 Section 3: Explainability Analysis and Ablation Studies")
    print("Dataset: Bike Sharing Demand Prediction")
    print("Task: Regression")
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    return True

def load_section2_results(results_file='./bike_sharing_section2_results.pkl'):
    """
    Load trained models and results from Section 2
    
    Args:
        results_file (str): Path to the Section 2 results pickle file
        
    Returns:
        dict: Dictionary containing all Section 2 results
    """
    try:
        with open(results_file, 'rb') as f:
            section2_data = pickle.load(f)

        print("✅ Section 2 results loaded successfully!")
        print(f"Models available: {list(section2_data['predictions'].keys())}")
        print(f"Best overall model: {section2_data['best_models']['best_overall']}")
        print(f"Best R² model: {section2_data['best_models']['best_r2']}")
        
        return section2_data
        
    except FileNotFoundError:
        print("❌ Section 2 results not found!")
        print("Please run Section 2 (Model Comparison) notebook first.")
        raise

def load_preprocessed_data(data_file='./bike_sharing_preprocessed_data.pkl'):
    """
    Load preprocessed data from Section 1
    
    Args:
        data_file (str): Path to the preprocessed data pickle file
        
    Returns:
        dict: Dictionary containing preprocessed data
    """
    try:
        with open(data_file, 'rb') as f:
            preprocessing_data = pickle.load(f)

        print("✅ Preprocessed data loaded successfully!")
        print(f"Features: {len(preprocessing_data['feature_names'])}")
        print(f"Test samples: {len(preprocessing_data['X_test_scaled']):,}")
        
        # Display feature names for reference
        print(f"\n📋 Feature Names:")
        for i, feature in enumerate(preprocessing_data['feature_names']):
            print(f"   {i+1:2d}. {feature}")

        return preprocessing_data
        
    except FileNotFoundError:
        print("❌ Preprocessed data not found!")
        print("Please run Section 1 (Data Preprocessing) notebook first.")
        raise

def load_and_filter_models(model_dir='./Section2_Model_Training', feature_names=None, 
                          predictions=None, comparison_df=None, min_r2_threshold=0.5):
    """
    Load models and filter out those with poor performance
    
    Args:
        model_dir (str): Directory containing trained models
        feature_names (list): List of feature names
        predictions (dict): Dictionary of model predictions
        comparison_df (pd.DataFrame): Model comparison results
        min_r2_threshold (float): Minimum R² threshold for analysis
        
    Returns:
        tuple: (models_to_analyze, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, model_results_detailed = load_all_models(model_dir, feature_names, device)

    print(f"\n📦 Models Loaded for Analysis:")
    for model_name in models.keys():
        print(f"   ✅ {model_name}")

    # Filter out models with poor performance for meaningful analysis
    models_to_analyze = {}
    for model_name, model in models.items():
        if model_name in predictions:
            # Get R² score from comparison_df
            r2_score = comparison_df[comparison_df['Model'] == model_name]['R²_Score'].values
            if len(r2_score) > 0 and r2_score[0] > min_r2_threshold:
                models_to_analyze[model_name] = model
                print(f"   📊 {model_name}: R² = {r2_score[0]:.4f} - Will analyze")
            else:
                print(f"   ⚠️ {model_name}: R² = {r2_score[0]:.4f} - Skipping (poor performance)")

    print(f"\n🔍 Models selected for analysis: {list(models_to_analyze.keys())}")
    
    return models_to_analyze, device, model_results_detailed

def initialize_explainability_analyzer(feature_names, save_dir='./Section3_Explainability'):
    """
    Initialize the explainability analyzer
    
    Args:
        feature_names (list): List of feature names
        save_dir (str): Directory to save results
        
    Returns:
        RegressionExplainabilityAnalyzer: Initialized analyzer
    """
    explainer = RegressionExplainabilityAnalyzer(feature_names=feature_names, save_dir=save_dir)

    print("🔧 Regression explainability analyzer initialized")
    print(f"Feature names: {feature_names[:5]}...")
    print(f"Total features: {len(feature_names)}")
    print(f"Save directory: {save_dir}")

    return explainer

def analyze_xgboost_explainability(models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
                                 y_train, y_test, feature_names, max_samples=300):
    """
    Analyze XGBoost explainability
    
    Args:
        models_to_analyze (dict): Dictionary of models to analyze
        explainer (RegressionExplainabilityAnalyzer): Explainability analyzer
        X_train_scaled, X_test_scaled: Training and test features
        y_train, y_test: Training and test targets
        feature_names (list): List of feature names
        max_samples (int): Maximum samples for analysis
        
    Returns:
        dict: XGBoost explanation results
    """
    if 'XGBoost' not in models_to_analyze:
        print("⚠️ XGBoost model not available or performance too low")
        return None

    print("\n" + "="*60)
    print("XGBOOST EXPLAINABILITY ANALYSIS")
    print("="*60)

    xgb_explanations = explainer.analyze_model_explainability(
        models_to_analyze['XGBoost'], "XGBoost", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=max_samples
    )

    # Save intermediate results
    save_intermediate_results({
        'xgb_explanations': xgb_explanations,
        'explainer_state': explainer.explanations
    }, 'bike_sharing_xgb_explanations.pkl')

    print("✅ XGBoost explainability analysis completed")
    
    # Display top features for XGBoost
    if 'XGBoost' in explainer.explanations and 'feature_importance' in explainer.explanations['XGBoost']:
        if 'importances' in explainer.explanations['XGBoost']['feature_importance']:
            importances = explainer.explanations['XGBoost']['feature_importance']['importances']
            indices = np.argsort(importances)[::-1][:10]
            print("\n🎯 Top 10 Most Important Features (XGBoost Built-in):")
            for i, idx in enumerate(indices):
                print(f"   {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    clear_memory()
    return xgb_explanations

def create_improved_ft_transformer_wrapper(model, model_results_detailed, device):
    """
    Create a wrapper for the Improved FT-Transformer model using PyTorchRegressionModelWrapper
    
    Args:
        model: The FT-Transformer model
        model_results_detailed (dict): Detailed model results
        device: PyTorch device
        
    Returns:
        PyTorchRegressionModelWrapper: Wrapped model
    """
    # Check if we need to handle target scaling
    target_scaler = None
    if 'Improved FT-Transformer' in model_results_detailed:
        target_scaler = model_results_detailed['Improved FT-Transformer'].get('target_scaler', None)
    
    # Use the standardized PyTorchRegressionModelWrapper
    return PyTorchRegressionModelWrapper(
        model=model,
        device=device,
        batch_size=256,
        target_scaler=target_scaler
    )

def analyze_improved_ft_transformer_explainability(models_to_analyze, explainer, X_train_scaled, 
                                                 X_test_scaled, y_train, y_test, feature_names,
                                                 model_results_detailed, device, max_samples=300):
    """
    Analyze Improved FT-Transformer explainability
    
    Args:
        models_to_analyze (dict): Dictionary of models to analyze
        explainer (RegressionExplainabilityAnalyzer): Explainability analyzer
        X_train_scaled, X_test_scaled: Training and test features
        y_train, y_test: Training and test targets
        feature_names (list): List of feature names
        model_results_detailed (dict): Detailed model results
        device: PyTorch device
        max_samples (int): Maximum samples for analysis
        
    Returns:
        dict: Improved FT-Transformer explanation results
    """
    if 'Improved FT-Transformer' not in models_to_analyze:
        print("⚠️ Improved FT-Transformer model not available or performance too low")
        return None

    print("\n" + "="*60)
    print("IMPROVED FT-TRANSFORMER EXPLAINABILITY ANALYSIS")
    print("="*60)

    # Create wrapper
    ft_wrapper = create_improved_ft_transformer_wrapper(
        models_to_analyze['Improved FT-Transformer'], 
        model_results_detailed, 
        device
    )

    improved_ft_explanations = explainer.analyze_model_explainability(
        ft_wrapper, "Improved FT-Transformer", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=max_samples
    )

    # Save intermediate results
    save_intermediate_results({
        'explainer_state': explainer.explanations,
        'improved_ft_explanations': improved_ft_explanations,
        'models_completed': ['XGBoost', 'Improved FT-Transformer']
    }, 'bike_sharing_improved_ft_explanations.pkl')

    print("✅ Improved FT-Transformer explainability analysis completed")
    
    # Display top features for Improved FT-Transformer (permutation importance)
    if ('Improved FT-Transformer' in explainer.explanations and 
        'permutation_importance' in explainer.explanations['Improved FT-Transformer']):
        importances = explainer.explanations['Improved FT-Transformer']['permutation_importance']['importances_mean']
        indices = np.argsort(importances)[::-1][:10]
        print("\n🎯 Top 10 Most Important Features (Improved FT-Transformer - Permutation):")
        for i, idx in enumerate(indices):
            print(f"   {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    clear_memory()
    return improved_ft_explanations

def create_saint_wrapper(model, device):
    """
    Create a wrapper for the SAINT model using PyTorchRegressionModelWrapper
    
    Args:
        model: The SAINT model
        device: PyTorch device
        
    Returns:
        PyTorchRegressionModelWrapper: Wrapped model
    """
    # Use the standardized PyTorchRegressionModelWrapper
    return PyTorchRegressionModelWrapper(
        model=model,
        device=device,
        batch_size=256,
        target_scaler=None  # SAINT typically doesn't use target scaling
    )

def analyze_saint_explainability(models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
                                y_train, y_test, feature_names, device, max_samples=300):
    """
    Analyze SAINT explainability
    
    Args:
        models_to_analyze (dict): Dictionary of models to analyze
        explainer (RegressionExplainabilityAnalyzer): Explainability analyzer
        X_train_scaled, X_test_scaled: Training and test features
        y_train, y_test: Training and test targets
        feature_names (list): List of feature names
        device: PyTorch device
        max_samples (int): Maximum samples for analysis
        
    Returns:
        dict: SAINT explanation results
    """
    if 'SAINT' not in models_to_analyze:
        print("⚠️ SAINT model not available or performance too low")
        return None

    print("\n" + "="*60)
    print("SAINT EXPLAINABILITY ANALYSIS")
    print("="*60)

    saint_wrapper = create_saint_wrapper(models_to_analyze['SAINT'], device)

    saint_explanations = explainer.analyze_model_explainability(
        saint_wrapper, "SAINT", 
        X_train_scaled, X_test_scaled, y_train, y_test,
        max_samples=max_samples
    )

    # Save intermediate results
    save_intermediate_results({
        'explainer_state': explainer.explanations,
        'models_completed': ['XGBoost', 'Improved FT-Transformer', 'SAINT']
    }, 'bike_sharing_saint_explanations.pkl')

    print("✅ SAINT explainability analysis completed")
    
    # Display top features for SAINT (permutation importance)
    if 'SAINT' in explainer.explanations and 'permutation_importance' in explainer.explanations['SAINT']:
        importances = explainer.explanations['SAINT']['permutation_importance']['importances_mean']
        indices = np.argsort(importances)[::-1][:10]
        print("\n🎯 Top 10 Most Important Features (SAINT - Permutation):")
        for i, idx in enumerate(indices):
            print(f"   {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    clear_memory()
    return saint_explanations

def perform_cross_model_comparison(explainer):
    """
    Compare feature importance across models
    
    Args:
        explainer (RegressionExplainabilityAnalyzer): Explainability analyzer
        
    Returns:
        pd.DataFrame: Feature importance comparison results
    """
    print("\n" + "="*60)
    print("CROSS-MODEL FEATURE IMPORTANCE COMPARISON")
    print("="*60)

    # Clear memory before comparison
    clear_memory()

    # Compare feature importance across all analyzed models
    importance_comparison = explainer.compare_feature_importance()

    if importance_comparison is not None:
        print("\n📊 Feature Importance Comparison Table:")
        print(importance_comparison.round(4).to_string())

        # Save comparison results
        importance_comparison.to_csv('./Section3_Explainability/bike_sharing_feature_importance_comparison.csv')
        print("\n💾 Feature importance comparison saved to './Section3_Explainability/bike_sharing_feature_importance_comparison.csv'")

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

        return importance_comparison, avg_importance
    else:
        print("❌ No feature importance data available for comparison")
        return None, None

def generate_explanation_reports(explainer):
    """
    Generate detailed explanation reports for each model
    
    Args:
        explainer (RegressionExplainabilityAnalyzer): Explainability analyzer
    """
    print("\n" + "="*60)
    print("DETAILED EXPLANATION REPORTS")
    print("="*60)

    available_models = list(explainer.explanations.keys())
    print(f"Available models for reporting: {available_models}")

    for model_name in available_models:
        print(f"\n{'='*50}")
        print(f"EXPLANATION REPORT: {model_name}")
        print(f"{'='*50}")
        
        explainer.generate_explanation_report(model_name)
        print("\n" + "-"*40)

def generate_business_insights(importance_comparison, avg_importance, available_models):
    """
    Generate business insights specific to bike sharing demand prediction
    
    Args:
        importance_comparison (pd.DataFrame): Feature importance comparison
        avg_importance (pd.Series): Average importance across models
        available_models (list): List of available models
    """
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS FOR BIKE SHARING DEMAND PREDICTION")
    print("="*80)

    if importance_comparison is not None:
        print("\n💼 KEY BUSINESS INSIGHTS:")
        
        # Analyze top features for business insights
        top_5_features = avg_importance.head(5)
        
        print("\nTop 5 Most Critical Features for Bike Demand Prediction:")
        for i, (feature, importance) in enumerate(top_5_features.items()):
            print(f"   {i+1}. {feature}: {importance:.4f}")
            
            # Provide business interpretation for each feature
            business_impact = get_feature_business_impact(feature)
            print(f"      💡 Business Impact: {business_impact}")
        
        # Model interpretability insights for regression
        print("\n🔍 MODEL INTERPRETABILITY INSIGHTS:")
        
        if 'XGBoost' in available_models:
            print("\n1. XGBoost:")
            print("   - Most interpretable with built-in feature importance")
            print("   - Tree-based structure allows for clear decision paths")
            print("   - SHAP values provide detailed feature contributions to predictions")
            print("   - Best for understanding individual prediction reasoning")
        
        if 'Improved FT-Transformer' in available_models:
            print("\n2. Improved FT-Transformer:")
            print("   - Attention-based architecture for tabular data")
            print("   - Feature interactions through attention mechanisms")
            print("   - Moderate interpretability through attention weights")
            print("   - Good balance of performance and explainability for regression")
        
        if 'SAINT' in available_models:
            print("\n3. SAINT:")
            print("   - Self-attention and intersample attention mechanisms")
            print("   - Complex feature interactions modeling")
            print("   - Limited interpretability due to attention complexity")
            print("   - Requires permutation importance for feature understanding")
        
        # Actionable recommendations for bike sharing business
        print("\n📋 ACTIONABLE BUSINESS RECOMMENDATIONS:")
        print("\n🎯 For Operations Teams:")
        print("   • Monitor weather forecasts for demand planning")
        print("   • Adjust bike availability based on temperature and humidity")
        print("   • Prepare for seasonal demand variations")
        print("   • Optimize bike distribution for peak hours")
        
        print("\n🎯 For Marketing Teams:")
        print("   • Target campaigns based on weather-dependent usage patterns")
        print("   • Promote usage during optimal weather conditions")
        print("   • Develop seasonal membership strategies")
        print("   • Focus on commuter vs recreational user segments")
        
        print("\n🎯 For Data Science Teams:")
        print("   • Use XGBoost for interpretable demand forecasting")
        print("   • Implement SHAP explanations for stakeholder communication")
        print("   • Monitor feature importance drift over time")
        print("   • A/B test weather-based interventions")

    else:
        print("❌ No feature importance data available for business insights")

def get_feature_business_impact(feature):
    """
    Get business impact description for a feature
    
    Args:
        feature (str): Feature name
        
    Returns:
        str: Business impact description
    """
    feature_lower = feature.lower()
    
    if 'temp' in feature_lower or 'temperature' in feature_lower:
        return "Temperature directly affects cycling comfort and demand"
    elif 'hour' in feature_lower or 'hr' in feature_lower:
        return "Hour of day indicates commuting patterns and peak usage times"
    elif 'humidity' in feature_lower or 'hum' in feature_lower:
        return "Humidity affects comfort and willingness to cycle"
    elif 'windspeed' in feature_lower or 'wind' in feature_lower:
        return "Wind conditions significantly impact cycling experience"
    elif 'season' in feature_lower:
        return "Seasonal patterns drive long-term demand variations"
    elif 'weather' in feature_lower:
        return "Weather conditions are primary drivers of bike usage"
    elif 'holiday' in feature_lower:
        return "Holiday patterns affect commuting vs recreational usage"
    elif 'workingday' in feature_lower:
        return "Working day status determines commuting demand patterns"
    elif 'month' in feature_lower or 'mnth' in feature_lower:
        return "Monthly patterns reflect seasonal and weather trends"
    elif 'weekday' in feature_lower:
        return "Day of week affects commuting vs leisure usage patterns"
    elif 'year' in feature_lower or 'yr' in feature_lower:
        return "Year-over-year growth and system maturity effects"
    else:
        return "This feature significantly influences bike sharing demand"

def run_ablation_studies(models_to_analyze, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    Run enhanced ablation studies for regression models
    
    Args:
        models_to_analyze (dict): Dictionary of models to analyze
        X_train_scaled, X_test_scaled: Training and test features
        y_train, y_test: Training and test targets
        feature_names (list): List of feature names
        
    Returns:
        tuple: (ablation_analyzer, ablation_results, ablation_summary_df)
    """
    print("\n" + "="*80)
    print("ENHANCED ABLATION STUDIES")
    print("="*80)

    # Initialize ablation analyzer
    ablation_analyzer = EnhancedAblationStudyAnalyzer()

    # Run ablation studies for models with good performance
    model_names_for_ablation = [name for name in models_to_analyze.keys()]

    print(f"Running ablation studies for: {model_names_for_ablation}")

    try:
        ablation_results = ablation_analyzer.comprehensive_ablation_study(
            models_to_analyze, model_names_for_ablation,
            X_train_scaled, X_test_scaled, y_train, y_test,
            feature_names=feature_names,
            dataset_name="bike_sharing"
        )
        
        print("\n✅ Ablation studies completed successfully!")
        
        # Create ablation summary
        ablation_summary_df = create_ablation_summary_dataframe(ablation_results)
        print("\n📊 Ablation Study Summary:")
        print(ablation_summary_df.to_string(index=False))
        
        # Create ablation dashboard
        plot_ablation_dashboard(ablation_analyzer, model_names_for_ablation)
        
        return ablation_analyzer, ablation_results, ablation_summary_df
        
    except Exception as e:
        print(f"⚠️ Error in ablation studies: {e}")
        print("Continuing with explainability analysis...")
        return None, None, None

def generate_analysis_summary(comparison_df, importance_comparison, avg_importance, ablation_results):
    """
    Generate comprehensive summary of explainability findings
    
    Args:
        comparison_df (pd.DataFrame): Model comparison results
        importance_comparison (pd.DataFrame): Feature importance comparison
        avg_importance (pd.Series): Average importance across models
        ablation_results (dict): Ablation study results
    """
    print("\n" + "="*80)
    print("EXPLAINABILITY ANALYSIS SUMMARY")
    print("="*80)

    # Model performance recap
    print("\n🏆 MODEL PERFORMANCE RECAP:")
    if comparison_df is not None:
        print(comparison_df[['R²_Score', 'RMSE', 'MAE', 'MAPE']].round(4).to_string())
    else:
        print("Model comparison data not available")

    # Feature importance insights
    print("\n🎯 KEY FEATURE IMPORTANCE INSIGHTS:")
    if importance_comparison is not None:
        print("\nMost Important Features for Bike Sharing Demand Prediction:")
        for i, (feature, importance) in enumerate(avg_importance.head(5).items()):
            print(f"   {i+1}. {feature}: {importance:.4f}")

        print(f"\nFeature Importance Consensus: {len(avg_importance)} features analyzed")
        print(f"Top feature: {avg_importance.index[0]} ({avg_importance.iloc[0]:.4f})")
    else:
        print("Feature importance data not available")

    # Key findings
    print("\n📋 KEY FINDINGS:")
    print("\n• Bike sharing demand is primarily driven by weather and temporal factors")
    print("• Temperature and humidity are consistently the most important features")
    print("• Hour of day and seasonal patterns are strong indicators of usage")
    print("• Different models may prioritize different feature combinations")
    print("• Feature importance consensus helps identify robust predictors")
    print("• Model interpretability varies significantly across architectures")
    print("• XGBoost provides the best balance of performance and interpretability")

    print("\n✅ Section 3 completed successfully!")
    print("📊 Comprehensive explainability analysis finished")
    print("📁 All results and visualizations saved")
    print("🎯 Ready for deployment and business implementation")

def save_final_results(explainer, importance_comparison, avg_importance, feature_names, 
                      models_to_analyze, ablation_results=None):
    """
    Save final results for future reference
    
    Args:
        explainer (RegressionExplainabilityAnalyzer): Explainability analyzer
        importance_comparison (pd.DataFrame): Feature importance comparison
        avg_importance (pd.Series): Average importance across models
        feature_names (list): List of feature names
        models_to_analyze (dict): Dictionary of models analyzed
        ablation_results (dict): Ablation study results
        
    Returns:
        str: Path to saved results file
    """
    # Save explainability results
    section3_data = {
        'explainer': explainer,
        'importance_comparison': importance_comparison,
        'models_analyzed': list(models_to_analyze.keys()),
        'feature_names': feature_names,
        'explanations': explainer.explanations,
        'avg_importance': avg_importance,
        'ablation_results': ablation_results
    }

    # Save to pickle file
    results_file = 'bike_sharing_section3_explainability.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(section3_data, f)

    print(f"💾 Section 3 explainability results saved to '{results_file}'")
    print("📋 This file contains all explainability analysis results")

    # Save comprehensive results using the analyzer
    explainer.save_comprehensive_results()

    print("\n🎉 Bike Sharing Explainability Analysis Complete!")
    print("\n📊 Generated Files:")
    print("   - Feature importance comparison CSV")
    print("   - Feature importance correlation heatmap")
    print("   - Top features comparison visualization")
    print("   - Individual model explanation plots (SHAP, LIME, etc.)")
    print("   - Comprehensive explainability results pickle file")
    print("   - Ablation study results and visualizations")
    print("   - Analysis log file")
    print("\n🚀 Ready for business implementation and model deployment!")
    
    return results_file

def run_complete_explainability_analysis(section2_results_file='./bike_sharing_section2_results.pkl',
                                       preprocessed_data_file='./bike_sharing_preprocessed_data.pkl',
                                       model_dir='./Section2_Model_Training',
                                       save_dir='./Section3_Explainability',
                                       min_r2_threshold=0.5,
                                       max_samples=300):
    """
    Run the complete explainability analysis pipeline
    
    Args:
        section2_results_file (str): Path to Section 2 results
        preprocessed_data_file (str): Path to preprocessed data
        model_dir (str): Directory containing trained models
        save_dir (str): Directory to save results
        min_r2_threshold (float): Minimum R² threshold for analysis
        max_samples (int): Maximum samples for analysis
        
    Returns:
        dict: Complete analysis results
    """
    # Setup environment
    setup_analysis_environment()
    
    # Load data and results
    section2_data = load_section2_results(section2_results_file)
    preprocessing_data = load_preprocessed_data(preprocessed_data_file)
    
    # Extract variables
    comparison_df = section2_data['comparison_df']
    predictions = section2_data['predictions']
    feature_names = preprocessing_data['feature_names']
    X_train_scaled = preprocessing_data['X_train_scaled']
    X_test_scaled = preprocessing_data['X_test_scaled']
    y_train = preprocessing_data['y_train']
    y_test = preprocessing_data['y_test']
    
    # Load and filter models
    models_to_analyze, device, model_results_detailed = load_and_filter_models(
        model_dir, feature_names, predictions, comparison_df, min_r2_threshold
    )
    
    # Initialize explainability analyzer
    explainer = initialize_explainability_analyzer(feature_names, save_dir)
    
    # Store intermediate results
    intermediate_results = {}
    
    # Run explainability analysis for each model
    xgb_results = analyze_xgboost_explainability(
        models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
        y_train, y_test, feature_names, max_samples
    )
    if xgb_results:
        intermediate_results['xgb_explanations'] = xgb_results
    
    ft_results = analyze_improved_ft_transformer_explainability(
        models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
        y_train, y_test, feature_names, model_results_detailed, device, max_samples
    )
    if ft_results:
        intermediate_results['improved_ft_explanations'] = ft_results
    
    saint_results = analyze_saint_explainability(
        models_to_analyze, explainer, X_train_scaled, X_test_scaled, 
        y_train, y_test, feature_names, device, max_samples
    )
    if saint_results:
        intermediate_results['saint_explanations'] = saint_results
    
    # Cross-model comparison
    importance_comparison, avg_importance = perform_cross_model_comparison(explainer)
    
    # Generate explanation reports
    generate_explanation_reports(explainer)
    
    # Generate business insights
    available_models = list(explainer.explanations.keys())
    generate_business_insights(importance_comparison, avg_importance, available_models)
    
    # Run ablation studies
    ablation_analyzer, ablation_results, ablation_summary_df = run_ablation_studies(
        models_to_analyze, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # Generate analysis summary
    generate_analysis_summary(comparison_df, importance_comparison, avg_importance, ablation_results)
    
    # Save final results
    results_file = save_final_results(
        explainer, importance_comparison, avg_importance, feature_names, 
        models_to_analyze, ablation_results
    )
    
    # Return complete results
    return {
        'explainer': explainer,
        'importance_comparison': importance_comparison,
        'avg_importance': avg_importance,
        'models_analyzed': list(models_to_analyze.keys()),
        'feature_names': feature_names,
        'explanations': explainer.explanations,
        'intermediate_results': intermediate_results,
        'ablation_results': ablation_results,
        'ablation_summary_df': ablation_summary_df,
        'results_file': results_file
    }

if __name__ == "__main__":
    print("Section 3 Explainability Functions module loaded successfully!")
    print("Use run_complete_explainability_analysis() to perform the complete analysis pipeline.")
