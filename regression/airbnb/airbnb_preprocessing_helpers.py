"""
Airbnb Dataset Preprocessing Helper Functions

This module contains helper functions for loading, preprocessing, and analyzing
the Airbnb dataset from Dgomonov's collection.

Functions:
- Data loading and validation
- Feature analysis and categorization
- Data visualization utilities
- Preprocessing and scaling functions
- Export and logging utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
import logging
import requests
import zipfile
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

def setup_logging(log_dir='./Section1_Data_PreProcessing'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'airbnb_preprocessing.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def download_airbnb_dataset(logger):
    """Download Airbnb dataset from reliable source"""
    logger.info("Attempting to download Airbnb dataset...")
    
    # Try multiple sources for Airbnb dataset
    sources = [
        {
            'name': 'Kaggle API - Dgomonov Airbnb NYC',
            'url': 'https://www.kaggle.com/api/v1/datasets/download/dgomonov/new-york-city-airbnb-open-data',
            'description': 'New York City Airbnb Open Data from Kaggle',
            'type': 'kaggle_zip'
        },
        {
            'name': 'GitHub Raw - Dgomonov Airbnb NYC',
            'url': 'https://raw.githubusercontent.com/dgomonov/new-york-city-airbnb-open-data/master/AB_NYC_2019.csv',
            'description': 'New York City Airbnb Open Data 2019 from GitHub',
            'type': 'csv'
        }
    ]
    
    for source in sources:
        try:
            logger.info(f"Trying to download from: {source['name']}")
            response = requests.get(source['url'], timeout=30)
            response.raise_for_status()
            
            if source['type'] == 'kaggle_zip':
                # Handle Kaggle ZIP download
                logger.info("Processing Kaggle ZIP file...")
                
                # Save the ZIP content to a temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                    temp_zip.write(response.content)
                    temp_zip_path = temp_zip.name
                
                # Extract and read CSV from ZIP
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        # List files in ZIP
                        file_list = zip_ref.namelist()
                        logger.info(f"Files in ZIP: {file_list}")
                        
                        # Look for CSV file (usually AB_NYC_2019.csv)
                        csv_file = None
                        for file_name in file_list:
                            if file_name.endswith('.csv') and 'AB_NYC' in file_name:
                                csv_file = file_name
                                break
                        
                        if csv_file is None:
                            # If no AB_NYC file, take the first CSV
                            csv_files = [f for f in file_list if f.endswith('.csv')]
                            if csv_files:
                                csv_file = csv_files[0]
                        
                        if csv_file:
                            logger.info(f"Reading CSV file: {csv_file}")
                            with zip_ref.open(csv_file) as csv_data:
                                df = pd.read_csv(csv_data)
                                logger.info(f"Successfully downloaded dataset: {source['description']}")
                                logger.info(f"Dataset shape: {df.shape}")
                                
                                # Clean up temp file
                                os.unlink(temp_zip_path)
                                return df, source
                        else:
                            logger.warning("No CSV file found in ZIP archive")
                            os.unlink(temp_zip_path)
                            continue
                            
                except zipfile.BadZipFile:
                    logger.warning("Downloaded file is not a valid ZIP archive")
                    os.unlink(temp_zip_path)
                    continue
                finally:
                    # Ensure temp file is cleaned up
                    if os.path.exists(temp_zip_path):
                        os.unlink(temp_zip_path)
            
            elif source['type'] == 'csv':
                # Handle direct CSV download
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"Successfully downloaded dataset: {source['description']}")
                logger.info(f"Dataset shape: {df.shape}")
                return df, source
            
        except Exception as e:
            logger.warning(f"Failed to download from {source['name']}: {str(e)}")
            continue
    
    # If all sources fail, create a fallback message
    logger.error("All download sources failed. Please manually download the dataset.")
    logger.error("You can manually download from:")
    logger.error("1. https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data")
    logger.error("2. https://github.com/dgomonov/new-york-city-airbnb-open-data")
    raise Exception("Could not download Airbnb dataset from any source")

def load_airbnb_data(logger):
    """Load the Airbnb Dataset"""
    logger.info("="*50)
    logger.info("1.1 DATA LOADING")
    logger.info("="*50)
    
    # Try to download the dataset
    try:
        airbnb_data, source_info = download_airbnb_dataset(logger)
        
        logger.info(f"Dataset source: {source_info['description']}")
        logger.info(f"Dataset shape: {airbnb_data.shape}")
        logger.info(f"Dataset columns: {list(airbnb_data.columns)}")
        
        # Display basic info
        logger.info("\nDataset info:")
        buffer = StringIO()
        airbnb_data.info(buf=buffer)
        logger.info(buffer.getvalue())
        
        # Display first few rows
        logger.info("\nFirst 5 rows:")
        logger.info(str(airbnb_data.head()))
        
        return airbnb_data
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def identify_target_variable(airbnb_data, logger):
    """Identify and validate target variable for regression"""
    logger.info("Identifying target variable for regression task...")
    
    # Common target variables in Airbnb datasets
    potential_targets = ['price', 'Price', 'PRICE', 'cost', 'Cost', 'rate', 'Rate']
    
    target_column = None
    for col in potential_targets:
        if col in airbnb_data.columns:
            target_column = col
            break
    
    if target_column is None:
        # Look for numeric columns that could be price-related
        numeric_cols = airbnb_data.select_dtypes(include=[np.number]).columns
        logger.info(f"Available numeric columns: {list(numeric_cols)}")
        
        # If price column has $ signs, clean it
        for col in airbnb_data.columns:
            if 'price' in col.lower():
                target_column = col
                break
    
    if target_column is None:
        raise ValueError("Could not identify target variable. Please specify manually.")
    
    logger.info(f"Target variable identified: {target_column}")
    return target_column

def clean_price_column(airbnb_data, target_column, logger):
    """Clean price column by removing $ signs and converting to numeric"""
    logger.info(f"Cleaning target variable: {target_column}")
    
    if airbnb_data[target_column].dtype == 'object':
        # Remove $ signs and commas, convert to numeric
        airbnb_data[target_column] = airbnb_data[target_column].astype(str)
        airbnb_data[target_column] = airbnb_data[target_column].str.replace('$', '', regex=False)
        airbnb_data[target_column] = airbnb_data[target_column].str.replace(',', '', regex=False)
        airbnb_data[target_column] = pd.to_numeric(airbnb_data[target_column], errors='coerce')
    
    # Remove rows with invalid prices
    initial_count = len(airbnb_data)
    airbnb_data = airbnb_data.dropna(subset=[target_column])
    airbnb_data = airbnb_data[airbnb_data[target_column] > 0]  # Remove zero or negative prices
    
    final_count = len(airbnb_data)
    logger.info(f"Removed {initial_count - final_count} rows with invalid prices")
    logger.info(f"Final dataset shape: {airbnb_data.shape}")
    
    return airbnb_data

def analyze_target_variable(airbnb_data, target_column, logger):
    """Analyze target variable distribution"""
    logger.info("="*50)
    logger.info("TARGET VARIABLE ANALYSIS")
    logger.info("="*50)
    
    target_data = airbnb_data[target_column]
    
    logger.info(f"Target variable ({target_column}) statistics:")
    logger.info(str(target_data.describe()))
    
    logger.info(f"\nTarget variable range:")
    logger.info(f"  Minimum: {target_data.min()}")
    logger.info(f"  Maximum: {target_data.max()}")
    logger.info(f"  Mean: {target_data.mean():.2f}")
    logger.info(f"  Median: {target_data.median():.2f}")
    logger.info(f"  Standard deviation: {target_data.std():.2f}")
    
    # Check for outliers
    Q1 = target_data.quantile(0.25)
    Q3 = target_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = airbnb_data[(target_data < lower_bound) | (target_data > upper_bound)]
    logger.info(f"\nOutliers (using IQR method): {len(outliers)} samples ({len(outliers)/len(airbnb_data)*100:.2f}%)")
    
    return target_data

def categorize_features(airbnb_data, target_column, logger):
    """Categorize features into numerical, categorical, and text features"""
    logger.info("="*50)
    logger.info("1.2 FEATURE ANALYSIS")
    logger.info("="*50)
    
    # Get all columns except target
    feature_columns = [col for col in airbnb_data.columns if col != target_column]
    logger.info(f"Number of features: {len(feature_columns)}")
    
    # Categorize features
    numerical_features = []
    categorical_features = []
    text_features = []
    datetime_features = []
    id_features = []
    
    for col in feature_columns:
        col_data = airbnb_data[col]
        
        # Skip columns with too many missing values
        if col_data.isnull().sum() / len(col_data) > 0.5:
            logger.info(f"Skipping {col} due to high missing values ({col_data.isnull().sum() / len(col_data) * 100:.1f}%)")
            continue
        
        # ID columns
        if 'id' in col.lower() or col.lower() in ['host_id', 'listing_id']:
            id_features.append(col)
        # Datetime columns
        elif col_data.dtype == 'datetime64[ns]' or 'date' in col.lower():
            datetime_features.append(col)
        # Numerical columns
        elif col_data.dtype in ['int64', 'float64']:
            numerical_features.append(col)
        # Text columns (long text)
        elif col_data.dtype == 'object' and col_data.str.len().mean() > 50:
            text_features.append(col)
        # Categorical columns
        elif col_data.dtype == 'object' and col_data.nunique() < 100:
            categorical_features.append(col)
        else:
            # Default to categorical for remaining object columns
            if col_data.dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
    
    logger.info(f"\nNumerical features ({len(numerical_features)}):")
    for i, feature in enumerate(numerical_features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    logger.info(f"\nCategorical features ({len(categorical_features)}):")
    for i, feature in enumerate(categorical_features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    logger.info(f"\nText features ({len(text_features)}) - will be excluded:")
    for i, feature in enumerate(text_features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    logger.info(f"\nID features ({len(id_features)}) - will be excluded:")
    for i, feature in enumerate(id_features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    logger.info(f"\nDatetime features ({len(datetime_features)}) - will be processed separately:")
    for i, feature in enumerate(datetime_features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    return {
        'numerical': numerical_features,
        'categorical': categorical_features,
        'text': text_features,
        'datetime': datetime_features,
        'id': id_features,
        'all_features': feature_columns
    }

def create_airbnb_visualizations(airbnb_data, target_column, feature_info, output_dir, logger):
    """Create comprehensive visualizations for Airbnb data"""
    logger.info("="*50)
    logger.info("1.3 DATA VISUALIZATION")
    logger.info("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    numerical_features = feature_info['numerical']
    categorical_features = feature_info['categorical']
    
    # Main distribution plots
    plt.figure(figsize=(15, 10))
    
    # Target distribution
    plt.subplot(2, 3, 1)
    plt.hist(airbnb_data[target_column], bins=50, alpha=0.7, color='lightblue')
    plt.title(f'Distribution of {target_column} (Target)')
    plt.xlabel(target_column)
    plt.ylabel('Frequency')
    
    # Target distribution (log scale)
    plt.subplot(2, 3, 2)
    plt.hist(np.log1p(airbnb_data[target_column]), bins=50, alpha=0.7, color='lightgreen')
    plt.title(f'Log Distribution of {target_column}')
    plt.xlabel(f'Log({target_column} + 1)')
    plt.ylabel('Frequency')
    
    # Plot relationships with numerical features (if available)
    plot_idx = 3
    for i, feature in enumerate(numerical_features[:4]):  # Plot up to 4 numerical features
        if plot_idx > 6:
            break
        plt.subplot(2, 3, plot_idx)
        plt.scatter(airbnb_data[feature], airbnb_data[target_column], alpha=0.5)
        plt.title(f'{feature} vs {target_column}')
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_and_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis for numerical features
    if len(numerical_features) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Correlation heatmap
        corr_features = numerical_features + [target_column]
        corr_data = airbnb_data[corr_features].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=axes[0])
        axes[0].set_title('Correlation Matrix: Numerical Features')
        
        # Feature correlation with target
        target_corr = corr_data[target_column].drop(target_column).sort_values(key=abs, ascending=False)
        target_corr.plot(kind='barh', ax=axes[1], color='skyblue')
        axes[1].set_title(f'Feature Correlation with Target ({target_column})')
        axes[1].set_xlabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Categorical features analysis
    if len(categorical_features) > 0:
        n_plots = min(4, len(categorical_features))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(categorical_features[:n_plots]):
            try:
                # Group by categorical feature and calculate mean target
                grouped = airbnb_data.groupby(feature)[target_column].mean().sort_values(ascending=False)
                
                # Limit to top categories if too many
                if len(grouped) > 10:
                    grouped = grouped.head(10)
                
                grouped.plot(kind='bar', ax=axes[i], color=plt.cm.Set3(i))
                axes[i].set_title(f'Average {target_column} by {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(f'Average {target_column}')
                axes[i].tick_params(axis='x', rotation=45)
                
            except Exception as e:
                logger.warning(f"Could not plot {feature}: {str(e)}")
                axes[i].text(0.5, 0.5, f'Could not plot {feature}', 
                           transform=axes[i].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'categorical_features_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    logger.info("Visualizations saved successfully")

def preprocess_airbnb_data(airbnb_data, target_column, feature_info, logger):
    """Preprocess the Airbnb data"""
    logger.info("="*50)
    logger.info("1.4 DATA PREPROCESSING")
    logger.info("="*50)
    
    df = airbnb_data.copy()
    
    # Remove text, ID, and datetime features for now
    columns_to_drop = (feature_info['text'] + feature_info['id'] + 
                      feature_info['datetime'])
    
    logger.info(f"Dropping columns: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle missing values
    logger.info("Handling missing values...")
    initial_shape = df.shape
    
    # For numerical features, fill with median
    for col in feature_info['numerical']:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled {col} missing values with median: {median_val}")
    
    # For categorical features, fill with mode or 'Unknown'
    for col in feature_info['categorical']:
        if col in df.columns and df[col].isnull().sum() > 0:
            if df[col].mode().empty:
                df[col].fillna('Unknown', inplace=True)
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_val}")
    
    # Encode categorical features
    label_encoders = {}
    categorical_features_in_df = [col for col in feature_info['categorical'] if col in df.columns]
    
    for col in categorical_features_in_df:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Encoded categorical feature: {col}")
    
    logger.info(f"Dataset shape after preprocessing: {df.shape}")
    logger.info(f"Shape change: {initial_shape} -> {df.shape}")
    
    # Get final feature names
    feature_names = [col for col in df.columns if col != target_column]
    logger.info(f"Final feature names ({len(feature_names)} features):")
    for i, name in enumerate(feature_names, 1):
        logger.info(f"  {i:2d}. {name}")
    
    return df, feature_names, label_encoders

def save_preprocessing_results(output_data, output_dir, dataset_name, logger):
    """Save preprocessing results to pickle file"""
    logger.info("="*50)
    logger.info("1.7 DATA SUMMARY AND EXPORT")
    logger.info("="*50)
    
    # Save to pickle file
    pickle_file = os.path.join(output_dir, f'{dataset_name}_preprocessed_data.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    logger.info(f"💾 Preprocessed data saved to '{pickle_file}'")
    
    # Also save CSV for inspection
    csv_file = os.path.join(output_dir, f'{dataset_name}_train_scaled.csv')
    df_export = pd.DataFrame(output_data['X_train_scaled'], columns=output_data['feature_names'])
    df_export[output_data['data_summary']['dataset_info']['target_variable']] = output_data['y_train']
    df_export.to_csv(csv_file, index=False)
    logger.info(f"📄 Training data also saved as '{csv_file}'")
    
    # Print summary
    summary = output_data['data_summary']
    logger.info("📊 DATA PREPROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Dataset: {summary['dataset_info']['name']}")
    logger.info(f"Total Samples: {summary['dataset_info']['total_samples']:,}")
    logger.info(f"Features: {summary['dataset_info']['num_features']}")
    logger.info(f"Task Type: {summary['dataset_info']['task_type']}")
    logger.info(f"Target Variable: {summary['dataset_info']['target_variable']}")
    logger.info(f"Target Range: {summary['dataset_info']['target_range']}")
    logger.info(f"Target Mean: {summary['dataset_info']['target_mean']:.2f}")
    logger.info(f"Target Std: {summary['dataset_info']['target_std']:.2f}")
    
    logger.info("✅ Data preprocessing completed successfully!")
    logger.info("📁 Ready for model training in Section 2")
    
    return pickle_file, csv_file
