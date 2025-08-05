"""
Section 1: Data Loading and Preprocessing

Airbnb Dataset Analysis

This script handles the data loading and preprocessing for the Airbnb Dataset from Dgomonov's collection.

Dataset Information:
- Source: Dgomonov's New York City Airbnb Open Data
- Task: Regression - predicting Airbnb listing prices
- Features: Various features including location, property type, reviews, etc.
- Target: price (listing price per night)
- Samples: ~49,000 instances (NYC Airbnb listings)

Analysis Components:
- Data loading and exploration
- Feature analysis and visualization
- Data preprocessing and encoding
- Train/validation/test splits
- Feature scaling
- Target distribution analysis
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import helper functions
from airbnb_preprocessing_helpers import (
    setup_logging, load_airbnb_data, identify_target_variable, 
    clean_price_column, analyze_target_variable, categorize_features,
    create_airbnb_visualizations, preprocess_airbnb_data, save_preprocessing_results
)

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def separate_features_target(df, target_column, feature_names, logger):
    """Separate features and target variable"""
    logger.info("\n" + "="*50)
    logger.info("1.5 FEATURE-TARGET SEPARATION")
    logger.info("="*50)
    
    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target type: Continuous (regression)")
    logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Verify feature statistics
    logger.info(f"\nFeature value ranges (first 5 features):")
    for i, feature_name in enumerate(feature_names[:5]):
        logger.info(f"  {feature_name}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    if len(feature_names) > 5:
        logger.info("  ...")
    
    return X, y

def split_data(X, y, logger):
    """Split data into train/validation/test sets"""
    logger.info("\n" + "="*50)
    logger.info("1.6 DATA SPLITTING")
    logger.info("="*50)
    
    # Split the data (no stratification needed for regression)
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: 80% train, 20% val (of the temp set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Check target distribution in splits
    logger.info(f"\nTarget distribution in splits:")
    logger.info(f"Training:   Mean={y_train.mean():.2f}, Std={y_train.std():.2f}, Range=[{y_train.min():.2f}, {y_train.max():.2f}]")
    logger.info(f"Validation: Mean={y_val.mean():.2f}, Std={y_val.std():.2f}, Range=[{y_val.min():.2f}, {y_val.max():.2f}]")
    logger.info(f"Test:       Mean={y_test.mean():.2f}, Std={y_test.std():.2f}, Range=[{y_test.min():.2f}, {y_test.max():.2f}]")
    
    # Verify proportions are maintained
    total_samples = len(X)
    logger.info(f"\nData split proportions:")
    logger.info(f"  Training:   {len(y_train)} samples ({len(y_train)/total_samples*100:.1f}%)")
    logger.info(f"  Validation: {len(y_val)} samples ({len(y_val)/total_samples*100:.1f}%)")
    logger.info(f"  Test:       {len(y_test)} samples ({len(y_test)/total_samples*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, feature_names, output_dir, logger):
    """Scale features using StandardScaler"""
    logger.info("\n" + "="*50)
    logger.info("1.7 FEATURE SCALING")
    logger.info("="*50)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Scaled training set: {X_train_scaled.shape}")
    logger.info(f"Scaled validation set: {X_val_scaled.shape}")
    logger.info(f"Scaled test set: {X_test_scaled.shape}")
    
    # Verify scaling worked correctly
    logger.info(f"\nScaling verification (training set):")
    logger.info(f"  Mean (should be ~0): {X_train_scaled.mean(axis=0)[:3]}...")
    logger.info(f"  Std (should be ~1):  {X_train_scaled.std(axis=0)[:3]}...")
    
    # Show scaling effect on sample features
    sample_features_idx = [0, 1, 2]  # First 3 features
    logger.info(f"\nScaling effect on sample features:")
    for idx in sample_features_idx:
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            logger.info(f"  {feature_name}:")
            logger.info(f"    Original range: [{X_train[:, idx].min():.2f}, {X_train[:, idx].max():.2f}]")
            logger.info(f"    Scaled range:   [{X_train_scaled[:, idx].min():.2f}, {X_train_scaled[:, idx].max():.2f}]")
    
    # Visualize scaling effect
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    # Select features to visualize scaling effect (first 6 features)
    features_to_plot = feature_names[:6]
    
    for i, feature_name in enumerate(features_to_plot):
        if i >= 6:
            break
        feature_idx = i
        
        # Original data
        axes[i].hist(X_train[:, feature_idx], bins=30, alpha=0.7, label='Original', color='lightcoral')
        # Scaled data
        axes[i].hist(X_train_scaled[:, feature_idx], bins=30, alpha=0.7, label='Scaled', color='lightblue')
        
        axes[i].set_title(f'Scaling Effect: {feature_name}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_scaling_effect.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def create_summary_and_export(airbnb_data, target_column, X_train, X_val, X_test, y_train, y_val, y_test, 
                             X_train_scaled, X_val_scaled, X_test_scaled, feature_names, 
                             feature_info, scaler, y, output_dir, logger):
    """Create data summary and export preprocessed data"""
    logger.info("\n" + "="*50)
    logger.info("1.8 DATA SUMMARY AND EXPORT")
    logger.info("="*50)
    
    # Create comprehensive data summary
    data_summary = {
        'dataset_info': {
            'name': 'Airbnb NYC Dataset',
            'source': 'Dgomonov New York City Airbnb Open Data',
            'total_samples': len(airbnb_data),
            'num_features': len(feature_names),
            'task_type': 'Regression',
            'target_variable': target_column,
            'target_range': [float(y.min()), float(y.max())],
            'target_mean': float(y.mean()),
            'target_std': float(y.std())
        },
        'data_splits': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_ratio': len(X_train) / len(airbnb_data),
            'val_ratio': len(X_val) / len(airbnb_data),
            'test_ratio': len(X_test) / len(airbnb_data)
        },
        'target_distribution': {
            'train_mean': float(y_train.mean()),
            'val_mean': float(y_val.mean()),
            'test_mean': float(y_test.mean()),
            'train_std': float(y_train.std()),
            'val_std': float(y_val.std()),
            'test_std': float(y_test.std())
        },
        'feature_info': {
            'feature_names': feature_names,
            'numerical_features': feature_info['numerical'],
            'categorical_features': feature_info['categorical']
        }
    }
    
    logger.info("📊 DATA PREPROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Dataset: {data_summary['dataset_info']['name']}")
    logger.info(f"Total Samples: {data_summary['dataset_info']['total_samples']:,}")
    logger.info(f"Features: {data_summary['dataset_info']['num_features']}")
    logger.info(f"Task Type: {data_summary['dataset_info']['task_type']}")
    logger.info(f"Target Variable: {data_summary['dataset_info']['target_variable']}")
    logger.info(f"Target Range: [{data_summary['dataset_info']['target_range'][0]:.2f}, {data_summary['dataset_info']['target_range'][1]:.2f}]")
    logger.info(f"Target Mean: {data_summary['dataset_info']['target_mean']:.2f}")
    logger.info(f"Target Std: {data_summary['dataset_info']['target_std']:.2f}")
    logger.info(f"\nData Splits:")
    logger.info(f"  Training:   {data_summary['data_splits']['train_size']:,} ({data_summary['data_splits']['train_ratio']:.1%})")
    logger.info(f"  Validation: {data_summary['data_splits']['val_size']:,} ({data_summary['data_splits']['val_ratio']:.1%})")
    logger.info(f"  Test:       {data_summary['data_splits']['test_size']:,} ({data_summary['data_splits']['test_ratio']:.1%})")
    logger.info(f"\nTarget Distribution:")
    logger.info(f"  Train Mean: {data_summary['target_distribution']['train_mean']:.2f}")
    logger.info(f"  Val Mean: {data_summary['target_distribution']['val_mean']:.2f}")
    logger.info(f"  Test Mean: {data_summary['target_distribution']['test_mean']:.2f}")
    
    logger.info(f"\n✅ Data preprocessing completed successfully!")
    logger.info(f"📁 Ready for model training in Section 2")
    
    # Save preprocessed data for use in other scripts
    preprocessing_data = {
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'data_summary': data_summary
    }
    
    # Save using helper function
    pickle_file, csv_file = save_preprocessing_results(
        preprocessing_data, output_dir, 'airbnb', logger
    )
    
    return data_summary, preprocessing_data

def main():
    """Main function to run the complete preprocessing pipeline"""
    # Setup logging and output directory
    output_dir = './Section1_Data_PreProcessing'
    logger = setup_logging(output_dir)
    
    logger.info("🏠 Starting Airbnb Dataset Preprocessing Pipeline")
    logger.info("="*60)
    
    try:
        # Step 1: Load data
        airbnb_data = load_airbnb_data(logger)
        
        # Step 2: Identify and clean target variable
        target_column = identify_target_variable(airbnb_data, logger)
        airbnb_data = clean_price_column(airbnb_data, target_column, logger)
        
        # Step 3: Analyze target variable
        target_data = analyze_target_variable(airbnb_data, target_column, logger)
        
        # Step 4: Categorize features
        feature_info = categorize_features(airbnb_data, target_column, logger)
        
        # Step 5: Create visualizations
        create_airbnb_visualizations(airbnb_data, target_column, feature_info, output_dir, logger)
        
        # Step 6: Preprocess data
        df, feature_names, label_encoders = preprocess_airbnb_data(airbnb_data, target_column, feature_info, logger)
        
        # Step 7: Separate features and target
        X, y = separate_features_target(df, target_column, feature_names, logger)
        
        # Step 8: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, logger)
        
        # Step 9: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test, feature_names, output_dir, logger
        )
        
        # Step 10: Create summary and export
        data_summary, preprocessing_data = create_summary_and_export(
            airbnb_data, target_column, X_train, X_val, X_test, y_train, y_val, y_test,
            X_train_scaled, X_val_scaled, X_test_scaled, feature_names,
            feature_info, scaler, y, output_dir, logger
        )
        
        logger.info("\n🎉 Preprocessing pipeline completed successfully!")
        logger.info("📊 Data is ready for XGBoost, FT-Transformer, SAINT, and TabTransformer model training")
        
        return preprocessing_data
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error("Please check the log file for detailed error information")
        raise

if __name__ == "__main__":
    preprocessing_data = main()
