#!/usr/bin/env python
# coding: utf-8

"""
Enhanced XGBoost Training with Advanced Preprocessing and Hyperparameter Tuning

This module provides XGBoost-specific data cleaning, preprocessing, feature engineering,
hyperparameter tuning, and model comparison for the Airbnb dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
import pickle
import joblib
import logging
from scipy import stats
from scipy.stats import boxcox
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import (
    cross_val_score, KFold, GridSearchCV, RandomizedSearchCV,
    train_test_split, validation_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# Import our custom evaluation framework
from enhanced_evaluation import ComprehensiveEvaluator

class XGBoostEnhancedTrainer:
    """Enhanced XGBoost trainer with advanced preprocessing and hyperparameter tuning"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.logger = None
        self.scaler = None
        self.feature_selector = None
        self.outlier_detector = None
        self.poly_features = None
        self.power_transformer = None
        
    def setup_logging(self, log_file='Enhanced_XGBoost_Training.log'):
        """Setup logging configuration"""
        log_dir = './Section2_Model_Training'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, log_file)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        return self.logger
    
    def load_preprocessed_data(self):
        """Load preprocessed data from Section 1"""
        self.logger.info("🚀 Enhanced XGBoost Training with Advanced Preprocessing")
        self.logger.info("Dataset: Airbnb Dataset")
        self.logger.info("Task: Regression - Predicting Airbnb listing prices")
        
        try:
            with open('./Section1_Data_PreProcessing/airbnb_preprocessed_data.pkl', 'rb') as f:
                preprocessing_data = pickle.load(f)

            self.logger.info("✅ Base preprocessed data loaded successfully!")
            return preprocessing_data

        except FileNotFoundError:
            self.logger.error("❌ Preprocessed data not found!")
            self.logger.error("Please run Section 1 (Data Preprocessing) script first.")
            raise
    
    def xgboost_specific_preprocessing(self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
        """Apply XGBoost-specific preprocessing techniques"""
        self.logger.info("\n🔧 XGBoost-Specific Data Preprocessing")
        self.logger.info("=" * 60)
        
        # 1. Outlier Detection and Handling
        self.logger.info("1. Outlier Detection and Handling")
        X_train_clean, y_train_clean, outlier_mask = self._handle_outliers(X_train, y_train)
        self.logger.info(f"   • Removed {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(y_train)*100:.1f}%)")
        
        # 2. Target Variable Transformation
        self.logger.info("2. Target Variable Transformation")
        y_train_transformed, y_val_transformed, y_test_transformed, transform_info = self._transform_target(
            y_train_clean, y_val, y_test
        )
        
        # 3. Advanced Feature Scaling
        self.logger.info("3. Advanced Feature Scaling (Robust Scaler)")
        X_train_scaled, X_val_scaled, X_test_scaled = self._advanced_scaling(
            X_train_clean, X_val, X_test
        )
        
        # 4. Feature Engineering
        self.logger.info("4. Feature Engineering")
        X_train_engineered, X_val_engineered, X_test_engineered, new_feature_names = self._feature_engineering(
            X_train_scaled, X_val_scaled, X_test_scaled, feature_names
        )
        
        # 5. Feature Selection
        self.logger.info("5. Feature Selection")
        X_train_selected, X_val_selected, X_test_selected, selected_features = self._feature_selection(
            X_train_engineered, X_val_engineered, X_test_engineered, 
            y_train_transformed, new_feature_names
        )
        
        self.logger.info(f"✅ Enhanced preprocessing completed!")
        self.logger.info(f"   • Original features: {len(feature_names)}")
        self.logger.info(f"   • Engineered features: {len(new_feature_names)}")
        self.logger.info(f"   • Selected features: {len(selected_features)}")
        self.logger.info(f"   • Training samples: {len(y_train_clean)} (removed {len(y_train) - len(y_train_clean)} outliers)")
        
        return {
            'X_train': X_train_selected,
            'X_val': X_val_selected,
            'X_test': X_test_selected,
            'y_train': y_train_transformed,
            'y_val': y_val_transformed,
            'y_test': y_test_transformed,
            'feature_names': selected_features,
            'transform_info': transform_info,
            'outlier_mask': outlier_mask
        }
    
    def _handle_outliers(self, X_train, y_train):
        """Handle outliers using Isolation Forest and statistical methods"""
        # Combine features and target for outlier detection
        combined_data = np.column_stack([X_train, y_train])
        
        # Use Isolation Forest for multivariate outlier detection
        self.outlier_detector = IsolationForest(
            contamination=0.05,  # Remove top 5% outliers
            random_state=self.random_state,
            n_jobs=-1
        )
        
        outlier_labels = self.outlier_detector.fit_predict(combined_data)
        outlier_mask = outlier_labels == -1
        
        # Additional target-based outlier removal (extreme prices)
        q1, q3 = np.percentile(y_train, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # More conservative than 1.5*IQR
        upper_bound = q3 + 3 * iqr
        
        price_outliers = (y_train < lower_bound) | (y_train > upper_bound)
        
        # Combine both outlier detection methods
        final_outlier_mask = outlier_mask | price_outliers
        
        # Keep only non-outliers
        X_train_clean = X_train[~final_outlier_mask]
        y_train_clean = y_train[~final_outlier_mask]
        
        return X_train_clean, y_train_clean, final_outlier_mask
    
    def _transform_target(self, y_train, y_val, y_test):
        """Transform target variable for better model performance"""
        # Analyze target distribution
        skewness = stats.skew(y_train)
        self.logger.info(f"   • Original target skewness: {skewness:.3f}")
        
        transform_info = {'method': 'none', 'params': {}}
        
        if abs(skewness) > 1.0:  # Highly skewed
            # Try different transformations
            transformations = {
                'log': np.log1p,
                'sqrt': np.sqrt,
                'boxcox': None  # Special handling
            }
            
            best_transform = 'none'
            best_skewness = abs(skewness)
            transformed_data = y_train.copy()
            
            for name, transform_func in transformations.items():
                try:
                    if name == 'boxcox':
                        # Box-Cox transformation (requires positive values)
                        if np.all(y_train > 0):
                            transformed, lambda_param = boxcox(y_train)
                            current_skewness = abs(stats.skew(transformed))
                            if current_skewness < best_skewness:
                                best_transform = name
                                best_skewness = current_skewness
                                transformed_data = transformed
                                transform_info = {'method': 'boxcox', 'params': {'lambda': lambda_param}}
                    else:
                        if name == 'log' and np.any(y_train <= 0):
                            continue  # Skip log transform for non-positive values
                        if name == 'sqrt' and np.any(y_train < 0):
                            continue  # Skip sqrt transform for negative values
                        
                        transformed = transform_func(y_train)
                        current_skewness = abs(stats.skew(transformed))
                        if current_skewness < best_skewness:
                            best_transform = name
                            best_skewness = current_skewness
                            transformed_data = transformed
                            transform_info = {'method': name, 'params': {}}
                except:
                    continue
            
            if best_transform != 'none':
                self.logger.info(f"   • Applied {best_transform} transformation")
                self.logger.info(f"   • New target skewness: {best_skewness:.3f}")
                
                # Apply same transformation to validation and test sets
                if best_transform == 'log':
                    y_val_transformed = np.log1p(y_val)
                    y_test_transformed = np.log1p(y_test)
                elif best_transform == 'sqrt':
                    y_val_transformed = np.sqrt(y_val)
                    y_test_transformed = np.sqrt(y_test)
                elif best_transform == 'boxcox':
                    y_val_transformed = boxcox(y_val, lmbda=transform_info['params']['lambda'])
                    y_test_transformed = boxcox(y_test, lmbda=transform_info['params']['lambda'])
                
                return transformed_data, y_val_transformed, y_test_transformed, transform_info
        
        self.logger.info("   • No transformation applied (skewness acceptable)")
        return y_train, y_val, y_test, transform_info
    
    def _advanced_scaling(self, X_train, X_val, X_test):
        """Apply robust scaling for better handling of outliers"""
        # Use RobustScaler which is less sensitive to outliers
        self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"   • Applied RobustScaler (median-based scaling)")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _feature_engineering(self, X_train, X_val, X_test, feature_names):
        """Create new features for better model performance"""
        # Convert to DataFrame for easier manipulation
        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_val = pd.DataFrame(X_val, columns=feature_names)
        df_test = pd.DataFrame(X_test, columns=feature_names)
        
        # 1. Polynomial features (degree 2) for top important features
        # Select top 5 features for polynomial expansion to avoid curse of dimensionality
        important_features = feature_names[:5]  # Assume first 5 are most important
        
        self.logger.info(f"   • Creating polynomial features for top {len(important_features)} features")
        
        poly_train = df_train[important_features].copy()
        poly_val = df_val[important_features].copy()
        poly_test = df_test[important_features].copy()
        
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
        poly_train_transformed = self.poly_features.fit_transform(poly_train)
        poly_val_transformed = self.poly_features.transform(poly_val)
        poly_test_transformed = self.poly_features.transform(poly_test)
        
        # Get polynomial feature names
        poly_feature_names = self.poly_features.get_feature_names_out(important_features)
        
        # 2. Statistical features
        self.logger.info("   • Creating statistical features")
        
        # Add statistical features based on numerical columns
        numerical_cols = [col for col in feature_names if 'latitude' in col or 'longitude' in col or 
                         'availability' in col or 'reviews' in col or 'minimum_nights' in col]
        
        if len(numerical_cols) >= 2:
            # Create interaction features
            for i, col1 in enumerate(numerical_cols[:3]):  # Limit to avoid too many features
                for col2 in numerical_cols[i+1:4]:
                    if col1 in df_train.columns and col2 in df_train.columns:
                        # Interaction
                        df_train[f'{col1}_x_{col2}'] = df_train[col1] * df_train[col2]
                        df_val[f'{col1}_x_{col2}'] = df_val[col1] * df_val[col2]
                        df_test[f'{col1}_x_{col2}'] = df_test[col1] * df_test[col2]
                        
                        # Ratio (avoid division by zero)
                        df_train[f'{col1}_div_{col2}'] = df_train[col1] / (df_train[col2] + 1e-8)
                        df_val[f'{col1}_div_{col2}'] = df_val[col1] / (df_val[col2] + 1e-8)
                        df_test[f'{col1}_div_{col2}'] = df_test[col1] / (df_test[col2] + 1e-8)
        
        # 3. Binning features for categorical-like behavior
        self.logger.info("   • Creating binned features")
        
        for col in numerical_cols[:3]:  # Limit to avoid too many features
            if col in df_train.columns:
                # Create quantile-based bins
                _, bins = pd.qcut(df_train[col], q=5, retbins=True, duplicates='drop')
                
                df_train[f'{col}_binned'] = pd.cut(df_train[col], bins=bins, labels=False, include_lowest=True)
                df_val[f'{col}_binned'] = pd.cut(df_val[col], bins=bins, labels=False, include_lowest=True)
                df_test[f'{col}_binned'] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                
                # Fill NaN values with median bin
                median_bin = df_train[f'{col}_binned'].median()
                df_train[f'{col}_binned'].fillna(median_bin, inplace=True)
                df_val[f'{col}_binned'].fillna(median_bin, inplace=True)
                df_test[f'{col}_binned'].fillna(median_bin, inplace=True)
        
        # Combine original features with polynomial features
        X_train_engineered = np.column_stack([df_train.values, poly_train_transformed])
        X_val_engineered = np.column_stack([df_val.values, poly_val_transformed])
        X_test_engineered = np.column_stack([df_test.values, poly_test_transformed])
        
        # Create combined feature names
        new_feature_names = list(df_train.columns) + list(poly_feature_names)
        
        self.logger.info(f"   • Created {len(new_feature_names) - len(feature_names)} new features")
        
        return X_train_engineered, X_val_engineered, X_test_engineered, new_feature_names
    
    def _feature_selection(self, X_train, X_val, X_test, y_train, feature_names):
        """Select most important features using multiple methods"""
        # Use mutual information for feature selection (good for non-linear relationships)
        n_features_to_select = min(50, len(feature_names))  # Select top 50 features or all if less
        
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=n_features_to_select)
        
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        self.logger.info(f"   • Selected {len(selected_features)} features using mutual information")
        
        return X_train_selected, X_val_selected, X_test_selected, selected_features
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform comprehensive multi-stage hyperparameter tuning for maximum R² improvement"""
        self.logger.info("\n🎯 Advanced Multi-Stage Hyperparameter Tuning")
        self.logger.info("=" * 60)
        
        # Base model configuration
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist'  # Faster training
        )
        
        # Stage 1: Broad exploration with extended parameter ranges
        self.logger.info("🔍 Stage 1: Broad Parameter Space Exploration")
        stage1_params = {
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1500],
            'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12],
            'min_child_weight': [1, 2, 3, 4, 5, 6, 8, 10],
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            'colsample_bytree': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            'reg_alpha': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            'reg_lambda': [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        }
        
        stage1_search = RandomizedSearchCV(
            base_model,
            stage1_params,
            n_iter=200,  # More iterations for better exploration
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        start_time = time.time()
        stage1_search.fit(X_train, y_train)
        stage1_time = time.time() - start_time
        
        stage1_best_params = stage1_search.best_params_
        stage1_best_score = stage1_search.best_score_
        
        self.logger.info(f"   ✅ Stage 1 completed in {stage1_time:.2f}s")
        self.logger.info(f"   • Best CV R² Score: {stage1_best_score:.4f}")
        self.logger.info(f"   • Best Parameters:")
        for param, value in stage1_best_params.items():
            self.logger.info(f"     - {param}: {value}")
        
        # Stage 2: Focused refinement around best parameters
        self.logger.info("\n🎯 Stage 2: Focused Parameter Refinement")
        
        # Create refined ranges around the best parameters
        stage2_params = {}
        
        # n_estimators refinement
        n_est = stage1_best_params['n_estimators']
        stage2_params['n_estimators'] = [
            max(100, n_est - 200), max(150, n_est - 100), n_est, 
            n_est + 100, n_est + 200, min(2000, n_est + 300)
        ]
        
        # learning_rate refinement
        lr = stage1_best_params['learning_rate']
        lr_range = max(0.005, lr * 0.5), lr * 0.8, lr, lr * 1.2, min(0.3, lr * 1.5)
        stage2_params['learning_rate'] = [round(x, 4) for x in lr_range]
        
        # max_depth refinement
        depth = stage1_best_params['max_depth']
        stage2_params['max_depth'] = [
            max(3, depth - 2), max(3, depth - 1), depth, 
            min(15, depth + 1), min(15, depth + 2)
        ]
        
        # min_child_weight refinement
        mcw = stage1_best_params['min_child_weight']
        stage2_params['min_child_weight'] = [
            max(1, mcw - 2), max(1, mcw - 1), mcw, mcw + 1, mcw + 2
        ]
        
        # subsample refinement
        ss = stage1_best_params['subsample']
        stage2_params['subsample'] = [
            max(0.5, ss - 0.1), max(0.5, ss - 0.05), ss, 
            min(1.0, ss + 0.05), min(1.0, ss + 0.1)
        ]
        
        # colsample_bytree refinement
        cs = stage1_best_params['colsample_bytree']
        stage2_params['colsample_bytree'] = [
            max(0.5, cs - 0.1), max(0.5, cs - 0.05), cs, 
            min(1.0, cs + 0.05), min(1.0, cs + 0.1)
        ]
        
        # reg_alpha refinement
        alpha = stage1_best_params['reg_alpha']
        if alpha == 0:
            stage2_params['reg_alpha'] = [0, 0.01, 0.05, 0.1]
        else:
            stage2_params['reg_alpha'] = [
                max(0, alpha * 0.5), max(0, alpha * 0.8), alpha, alpha * 1.2, alpha * 1.5
            ]
        
        # reg_lambda refinement
        lam = stage1_best_params['reg_lambda']
        stage2_params['reg_lambda'] = [
            max(0.01, lam * 0.5), max(0.01, lam * 0.8), lam, lam * 1.2, lam * 1.5
        ]
        
        # gamma refinement
        gamma = stage1_best_params['gamma']
        if gamma == 0:
            stage2_params['gamma'] = [0, 0.01, 0.05, 0.1]
        else:
            stage2_params['gamma'] = [
                max(0, gamma * 0.5), max(0, gamma * 0.8), gamma, gamma * 1.2, gamma * 1.5
            ]
        
        stage2_search = RandomizedSearchCV(
            base_model,
            stage2_params,
            n_iter=100,  # Focused search
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state + 1,
            verbose=1
        )
        
        start_time = time.time()
        stage2_search.fit(X_train, y_train)
        stage2_time = time.time() - start_time
        
        stage2_best_params = stage2_search.best_params_
        stage2_best_score = stage2_search.best_score_
        
        self.logger.info(f"   ✅ Stage 2 completed in {stage2_time:.2f}s")
        self.logger.info(f"   • Best CV R² Score: {stage2_best_score:.4f}")
        self.logger.info(f"   • Improvement from Stage 1: {stage2_best_score - stage1_best_score:+.4f}")
        
        # Stage 3: Final precision tuning with early stopping
        self.logger.info("\n🎯 Stage 3: Final Precision Tuning with Early Stopping")
        
        # Use the best parameters from stage 2 with early stopping callback
        final_model = xgb.XGBRegressor(
            **stage2_best_params,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist',
            enable_categorical=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)]
        )
        
        # Train with early stopping using validation set
        start_time = time.time()
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        stage3_time = time.time() - start_time
        
        # Get the actual number of estimators used (after early stopping)
        try:
            actual_n_estimators = final_model.get_booster().num_boosted_rounds()
        except:
            # Fallback if method doesn't exist
            actual_n_estimators = final_model.n_estimators
        
        # Create a new model with the optimal number of estimators for CV evaluation
        final_model_for_cv = xgb.XGBRegressor(
            **{k: v for k, v in stage2_best_params.items() if k != 'n_estimators'},
            n_estimators=actual_n_estimators,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist',
            enable_categorical=False
        )
        
        # Evaluate final model
        final_cv_scores = cross_val_score(final_model_for_cv, X_train, y_train, cv=5, scoring='r2')
        final_cv_score = final_cv_scores.mean()
        final_val_score = final_model.score(X_val, y_val)
        
        self.logger.info(f"   ✅ Stage 3 completed in {stage3_time:.2f}s")
        self.logger.info(f"   • Early stopping at {actual_n_estimators} estimators")
        self.logger.info(f"   • Final CV R² Score: {final_cv_score:.4f}")
        self.logger.info(f"   • Validation R² Score: {final_val_score:.4f}")
        self.logger.info(f"   • Total improvement: {final_cv_score - stage1_best_score:+.4f}")
        
        # Update final parameters with actual n_estimators
        final_best_params = stage2_best_params.copy()
        final_best_params['n_estimators'] = actual_n_estimators
        
        # Stage 4: Advanced ensemble techniques (if improvement is still small)
        improvement = final_cv_score - stage1_best_score
        if improvement < 0.02:  # If improvement is less than 2%
            self.logger.info("\n🚀 Stage 4: Advanced Ensemble Techniques")
            
            # Try different boosting strategies
            advanced_params = [
                # More aggressive learning with higher regularization
                {**final_best_params, 'learning_rate': final_best_params['learning_rate'] * 0.5, 
                 'n_estimators': int(final_best_params['n_estimators'] * 1.5), 
                 'reg_alpha': final_best_params['reg_alpha'] * 2},
                
                # Deeper trees with more regularization
                {**final_best_params, 'max_depth': min(15, final_best_params['max_depth'] + 2), 
                 'min_child_weight': final_best_params['min_child_weight'] + 2,
                 'reg_lambda': final_best_params['reg_lambda'] * 1.5},
                
                # More feature sampling
                {**final_best_params, 'colsample_bytree': 0.6, 'subsample': 0.7,
                 'colsample_bylevel': 0.8, 'colsample_bynode': 0.8}
            ]
            
            best_advanced_score = final_cv_score
            best_advanced_model = final_model
            best_advanced_params = final_best_params
            
            for i, params in enumerate(advanced_params):
                self.logger.info(f"   Testing advanced configuration {i+1}/3...")
                
                advanced_model = xgb.XGBRegressor(
                    **{k: v for k, v in params.items() if k != 'early_stopping_rounds'},
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    random_state=self.random_state + i,
                    n_jobs=-1,
                    tree_method='hist'
                )
                
                # Quick CV evaluation
                cv_scores = cross_val_score(advanced_model, X_train, y_train, cv=3, scoring='r2')
                cv_score = cv_scores.mean()
                
                if cv_score > best_advanced_score:
                    best_advanced_score = cv_score
                    best_advanced_model = advanced_model
                    best_advanced_params = params
                    self.logger.info(f"     ✅ New best score: {cv_score:.4f}")
                else:
                    self.logger.info(f"     Score: {cv_score:.4f}")
            
            if best_advanced_score > final_cv_score:
                self.logger.info(f"   🎯 Advanced techniques improved score by {best_advanced_score - final_cv_score:+.4f}")
                final_model = best_advanced_model
                final_cv_score = best_advanced_score
                final_best_params = best_advanced_params
                final_val_score = final_model.score(X_val, y_val)
        
        # Final summary
        total_time = stage1_time + stage2_time + stage3_time
        total_improvement = final_cv_score - stage1_best_score
        
        self.logger.info(f"\n📊 Hyperparameter Tuning Summary:")
        self.logger.info(f"   • Total tuning time: {total_time:.2f}s")
        self.logger.info(f"   • Initial best score: {stage1_best_score:.4f}")
        self.logger.info(f"   • Final best score: {final_cv_score:.4f}")
        self.logger.info(f"   • Total improvement: {total_improvement:+.4f} ({total_improvement/stage1_best_score*100:+.2f}%)")
        self.logger.info(f"   • Final validation score: {final_val_score:.4f}")
        
        self.logger.info(f"\n🏆 Final Optimized Parameters:")
        for param, value in final_best_params.items():
            self.logger.info(f"     - {param}: {value}")
        
        return final_model, final_best_params, final_cv_score, final_val_score
    
    def train_baseline_model(self, X_train, y_train):
        """Train baseline XGBoost model with default parameters"""
        self.logger.info("\n🏗️ Training Baseline XGBoost Model")
        self.logger.info("=" * 50)
        
        baseline_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        start_time = time.time()
        baseline_model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        self.logger.info(f"✅ Baseline model trained in {train_time:.2f}s")
        
        return baseline_model
    
    def compare_models(self, baseline_model, tuned_model, X_test, y_test, transform_info):
        """Compare baseline and tuned models"""
        self.logger.info("\n📊 Model Comparison")
        self.logger.info("=" * 50)
        
        # Make predictions
        baseline_pred = baseline_model.predict(X_test)
        tuned_pred = tuned_model.predict(X_test)
        
        # Inverse transform predictions if target was transformed
        if transform_info['method'] != 'none':
            y_test_original, baseline_pred_original, tuned_pred_original = self._inverse_transform_predictions(
                y_test, baseline_pred, tuned_pred, transform_info
            )
        else:
            y_test_original = y_test
            baseline_pred_original = baseline_pred
            tuned_pred_original = tuned_pred
        
        # Calculate metrics for both models
        baseline_metrics = self._calculate_metrics(y_test_original, baseline_pred_original, "Baseline")
        tuned_metrics = self._calculate_metrics(y_test_original, tuned_pred_original, "Tuned")
        
        # Display comparison
        self.logger.info("\nModel Performance Comparison:")
        self.logger.info("-" * 60)
        self.logger.info(f"{'Metric':<25} {'Baseline':<15} {'Tuned':<15} {'Improvement':<15}")
        self.logger.info("-" * 60)
        
        metrics_to_compare = ['r2_score', 'rmse', 'mae', 'mape']
        
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            tuned_val = tuned_metrics[metric]
            
            if metric in ['rmse', 'mae', 'mape']:  # Lower is better
                improvement = ((baseline_val - tuned_val) / baseline_val) * 100
                improvement_str = f"{improvement:+.2f}%"
            else:  # Higher is better (r2_score)
                improvement = ((tuned_val - baseline_val) / abs(baseline_val)) * 100
                improvement_str = f"{improvement:+.2f}%"
            
            self.logger.info(f"{metric.upper():<25} {baseline_val:<15.4f} {tuned_val:<15.4f} {improvement_str:<15}")
        
        # Statistical significance test
        self._statistical_significance_test(y_test_original, baseline_pred_original, tuned_pred_original)
        
        # Create comparison plots
        self._plot_model_comparison(y_test_original, baseline_pred_original, tuned_pred_original)
        
        return baseline_metrics, tuned_metrics
    
    def _inverse_transform_predictions(self, y_test, baseline_pred, tuned_pred, transform_info):
        """Inverse transform predictions to original scale"""
        if transform_info['method'] == 'log':
            y_test_original = np.expm1(y_test)
            baseline_pred_original = np.expm1(baseline_pred)
            tuned_pred_original = np.expm1(tuned_pred)
        elif transform_info['method'] == 'sqrt':
            y_test_original = y_test ** 2
            baseline_pred_original = baseline_pred ** 2
            tuned_pred_original = tuned_pred ** 2
        elif transform_info['method'] == 'boxcox':
            from scipy.special import inv_boxcox
            lambda_param = transform_info['params']['lambda']
            y_test_original = inv_boxcox(y_test, lambda_param)
            baseline_pred_original = inv_boxcox(baseline_pred, lambda_param)
            tuned_pred_original = inv_boxcox(tuned_pred, lambda_param)
        else:
            y_test_original = y_test
            baseline_pred_original = baseline_pred
            tuned_pred_original = tuned_pred
        
        return y_test_original, baseline_pred_original, tuned_pred_original
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics"""
        metrics = {
            'model_name': model_name,
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred)
        }
        return metrics
    
    def _statistical_significance_test(self, y_true, pred1, pred2):
        """Perform statistical significance test between two models"""
        from scipy.stats import wilcoxon
        
        # Calculate absolute errors
        errors1 = np.abs(y_true - pred1)
        errors2 = np.abs(y_true - pred2)
        
        # Wilcoxon signed-rank test
        try:
            statistic, p_value = wilcoxon(errors1, errors2, alternative='two-sided')
            
            self.logger.info(f"\n🔬 Statistical Significance Test (Wilcoxon):")
            self.logger.info(f"   • Test Statistic: {statistic}")
            self.logger.info(f"   • P-value: {p_value:.6f}")
            
            if p_value < 0.05:
                self.logger.info(f"   • Result: Significant difference (p < 0.05)")
                if np.mean(errors2) < np.mean(errors1):
                    self.logger.info(f"   • Tuned model is significantly better")
                else:
                    self.logger.info(f"   • Baseline model is significantly better")
            else:
                self.logger.info(f"   • Result: No significant difference (p >= 0.05)")
        except Exception as e:
            self.logger.info(f"   • Statistical test failed: {str(e)}")
    
    def _plot_model_comparison(self, y_true, pred_baseline, pred_tuned):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Actual vs Predicted - Baseline
        axes[0, 0].scatter(y_true, pred_baseline, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Prices ($)')
        axes[0, 0].set_ylabel('Predicted Prices ($)')
        axes[0, 0].set_title(f'Baseline Model\nR² = {r2_score(y_true, pred_baseline):.4f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted - Tuned
        axes[0, 1].scatter(y_true, pred_tuned, alpha=0.6, color='green')
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Prices ($)')
        axes[0, 1].set_ylabel('Predicted Prices ($)')
        axes[0, 1].set_title(f'Tuned Model\nR² = {r2_score(y_true, pred_tuned):.4f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals - Baseline
        residuals_baseline = y_true - pred_baseline
        axes[0, 2].scatter(pred_baseline, residuals_baseline, alpha=0.6, color='blue')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[0, 2].set_xlabel('Predicted Prices ($)')
        axes[0, 2].set_ylabel('Residuals ($)')
        axes[0, 2].set_title('Baseline Model - Residuals')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residuals - Tuned
        residuals_tuned = y_true - pred_tuned
        axes[1, 0].scatter(pred_tuned, residuals_tuned, alpha=0.6, color='green')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Prices ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        axes[1, 0].set_title('Tuned Model - Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error Distribution Comparison
        errors_baseline = np.abs(residuals_baseline)
        errors_tuned = np.abs(residuals_tuned)
        
        axes[1, 1].hist(errors_baseline, bins=30, alpha=0.7, label='Baseline', color='blue', density=True)
        axes[1, 1].hist(errors_tuned, bins=30, alpha=0.7, label='Tuned', color='green', density=True)
        axes[1, 1].set_xlabel('Absolute Error ($)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Metrics Comparison
        metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
        baseline_values = [
            r2_score(y_true, pred_baseline),
            np.sqrt(mean_squared_error(y_true, pred_baseline)),
            mean_absolute_error(y_true, pred_baseline),
            mean_absolute_percentage_error(y_true, pred_baseline) * 100
        ]
        tuned_values = [
            r2_score(y_true, pred_tuned),
            np.sqrt(mean_squared_error(y_true, pred_tuned)),
            mean_absolute_error(y_true, pred_tuned),
            mean_absolute_percentage_error(y_true, pred_tuned) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Normalize values for better visualization (except R²)
        baseline_norm = baseline_values.copy()
        tuned_norm = tuned_values.copy()
        
        # For RMSE, MAE, MAPE - show relative improvement
        for i in range(1, 4):
            if baseline_values[i] > 0:
                baseline_norm[i] = 1.0  # Baseline as reference
                tuned_norm[i] = tuned_values[i] / baseline_values[i]
        
        axes[1, 2].bar(x - width/2, [baseline_norm[0]] + [1.0]*3, width, label='Baseline', color='blue', alpha=0.7)
        axes[1, 2].bar(x + width/2, [tuned_norm[0]] + tuned_norm[1:4], width, label='Tuned', color='green', alpha=0.7)
        
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Normalized Values')
        axes[1, 2].set_title('Performance Metrics Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./Section2_Model_Training/Enhanced_XGBoost_Model_Comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_enhanced_results(self, baseline_model, tuned_model, baseline_metrics, tuned_metrics, 
                            enhanced_data, transform_info):
        """Save enhanced training results including both baseline and tuned models"""
        self.logger.info("\n💾 Saving Enhanced Results")
        self.logger.info("=" * 50)
        
        save_dir = './Section2_Model_Training'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save both models with detailed logging
        self.logger.info("Saving trained models:")
        
        # Save baseline model (enhanced preprocessing + default hyperparameters)
        baseline_model_path = os.path.join(save_dir, 'xgboost_baseline_enhanced.pkl')
        joblib.dump(baseline_model, baseline_model_path)
        self.logger.info(f"   ✅ Baseline Enhanced Model: {baseline_model_path}")
        self.logger.info(f"      • R² Score: {baseline_metrics['r2_score']:.4f}")
        self.logger.info(f"      • RMSE: ${baseline_metrics['rmse']:.2f}")
        self.logger.info(f"      • MAE: ${baseline_metrics['mae']:.2f}")
        
        # Save tuned model (enhanced preprocessing + optimized hyperparameters)
        tuned_model_path = os.path.join(save_dir, 'xgboost_tuned_enhanced.pkl')
        joblib.dump(tuned_model, tuned_model_path)
        self.logger.info(f"   ✅ Tuned Enhanced Model: {tuned_model_path}")
        self.logger.info(f"      • R² Score: {tuned_metrics['r2_score']:.4f}")
        self.logger.info(f"      • RMSE: ${tuned_metrics['rmse']:.2f}")
        self.logger.info(f"      • MAE: ${tuned_metrics['mae']:.2f}")
        
        # Save model metadata
        model_metadata = {
            'baseline_model': {
                'file_path': baseline_model_path,
                'model_type': 'XGBoost Enhanced Baseline',
                'preprocessing': 'Advanced (outlier removal, feature engineering, scaling)',
                'hyperparameters': 'Default optimized',
                'performance': baseline_metrics,
                'training_date': pd.Timestamp.now().isoformat()
            },
            'tuned_model': {
                'file_path': tuned_model_path,
                'model_type': 'XGBoost Enhanced Tuned',
                'preprocessing': 'Advanced (outlier removal, feature engineering, scaling)',
                'hyperparameters': 'Optimized via RandomizedSearchCV + GridSearchCV',
                'performance': tuned_metrics,
                'training_date': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save model metadata as JSON
        import json
        metadata_path = os.path.join(save_dir, 'enhanced_models_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        self.logger.info(f"   ✅ Model Metadata: {metadata_path}")
        
        # Save preprocessing components (required for model deployment)
        preprocessing_components = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'outlier_detector': self.outlier_detector,
            'poly_features': self.poly_features,
            'transform_info': transform_info,
            'feature_names': enhanced_data['feature_names'],
            'preprocessing_steps': [
                'outlier_detection_isolation_forest',
                'target_transformation',
                'robust_scaling',
                'polynomial_feature_engineering',
                'interaction_features',
                'binned_features',
                'mutual_information_feature_selection'
            ]
        }
        preprocessing_path = os.path.join(save_dir, 'enhanced_preprocessing_components.pkl')
        joblib.dump(preprocessing_components, preprocessing_path)
        self.logger.info(f"   ✅ Preprocessing Components: {preprocessing_path}")
        
        # Save comparison results
        comparison_df = pd.DataFrame([baseline_metrics, tuned_metrics]).set_index('model_name')
        comparison_path = os.path.join(save_dir, 'enhanced_xgboost_comparison.csv')
        comparison_df.to_csv(comparison_path)
        self.logger.info(f"   ✅ Model Comparison: {comparison_path}")
        
        # Save feature names and importance
        feature_df = pd.DataFrame({'feature_names': enhanced_data['feature_names']})
        feature_path = os.path.join(save_dir, 'enhanced_feature_names.csv')
        feature_df.to_csv(feature_path, index=False)
        self.logger.info(f"   ✅ Feature Names: {feature_path}")
        
        # Save the complete enhanced_data dictionary to Section1 for consistency
        enhanced_data_path = './Section1_Data_PreProcessing/enhanced_data.pkl'
        os.makedirs('./Section1_Data_PreProcessing', exist_ok=True)
        joblib.dump(enhanced_data, enhanced_data_path)
        self.logger.info(f"   ✅ Enhanced Data: {enhanced_data_path}")
        
        # Also save a copy to Section2 for local reference
        enhanced_data_path_local = os.path.join(save_dir, 'enhanced_data.pkl')
        joblib.dump(enhanced_data, enhanced_data_path_local)
        self.logger.info(f"   ✅ Enhanced Data (local copy): {enhanced_data_path_local}")
        
        # Save model predictions for analysis
        baseline_pred = baseline_model.predict(enhanced_data['X_test'])
        tuned_pred = tuned_model.predict(enhanced_data['X_test'])
        
        # Inverse transform predictions if needed
        if transform_info['method'] != 'none':
            y_test_orig, baseline_pred_orig, tuned_pred_orig = self._inverse_transform_predictions(
                enhanced_data['y_test'], baseline_pred, tuned_pred, transform_info
            )
        else:
            y_test_orig = enhanced_data['y_test']
            baseline_pred_orig = baseline_pred
            tuned_pred_orig = tuned_pred
        
        predictions_df = pd.DataFrame({
            'actual': y_test_orig,
            'baseline_predicted': baseline_pred_orig,
            'tuned_predicted': tuned_pred_orig,
            'baseline_residual': y_test_orig - baseline_pred_orig,
            'tuned_residual': y_test_orig - tuned_pred_orig,
            'baseline_abs_error': np.abs(y_test_orig - baseline_pred_orig),
            'tuned_abs_error': np.abs(y_test_orig - tuned_pred_orig)
        })
        predictions_path = os.path.join(save_dir, 'enhanced_model_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"   ✅ Model Predictions: {predictions_path}")
        
        self.logger.info("\n📁 Complete Enhanced Training Results:")
        self.logger.info("   🤖 Models:")
        self.logger.info("      - xgboost_baseline_enhanced.pkl (Enhanced preprocessing + default params)")
        self.logger.info("      - xgboost_tuned_enhanced.pkl (Enhanced preprocessing + optimized params)")
        self.logger.info("   🔧 Preprocessing:")
        self.logger.info("      - enhanced_preprocessing_components.pkl (All preprocessing objects)")
        self.logger.info("   📊 Analysis:")
        self.logger.info("      - enhanced_xgboost_comparison.csv (Performance comparison)")
        self.logger.info("      - enhanced_model_predictions.csv (Test set predictions)")
        self.logger.info("      - enhanced_feature_names.csv (Selected features)")
        self.logger.info("      - enhanced_models_metadata.json (Model metadata)")
        self.logger.info("   📈 Visualizations:")
        self.logger.info("      - Enhanced_XGBoost_Model_Comparison.png (Comparison plots)")
        
        # Performance summary
        improvement = tuned_metrics['r2_score'] - baseline_metrics['r2_score']
        self.logger.info(f"\n🎯 Model Performance Summary:")
        self.logger.info(f"   • Baseline Enhanced Model R²: {baseline_metrics['r2_score']:.4f}")
        self.logger.info(f"   • Tuned Enhanced Model R²: {tuned_metrics['r2_score']:.4f}")
        self.logger.info(f"   • Performance Improvement: {improvement:+.4f} R² points")
        self.logger.info(f"   • Recommended Model: {'Tuned Enhanced' if improvement > 0.001 else 'Baseline Enhanced'}")
    
    def run_complete_enhanced_training(self):
        """Run the complete enhanced XGBoost training pipeline"""
        # Setup logging
        logger = self.setup_logging()
        
        try:
            # Load base preprocessed data
            base_data = self.load_preprocessed_data()
            
            # Apply XGBoost-specific preprocessing
            enhanced_data = self.xgboost_specific_preprocessing(
                base_data['X_train_scaled'], base_data['X_val_scaled'], base_data['X_test_scaled'],
                base_data['y_train'], base_data['y_val'], base_data['y_test'],
                base_data['feature_names']
            )
            
            # Train baseline model (with enhanced preprocessing)
            baseline_model = self.train_baseline_model(enhanced_data['X_train'], enhanced_data['y_train'])
            
            # Perform hyperparameter tuning
            tuned_model, best_params, best_cv_score, val_score = self.hyperparameter_tuning(
                enhanced_data['X_train'], enhanced_data['y_train'],
                enhanced_data['X_val'], enhanced_data['y_val']
            )
            
            # Compare models
            baseline_metrics, tuned_metrics = self.compare_models(
                baseline_model, tuned_model,
                enhanced_data['X_test'], enhanced_data['y_test'],
                enhanced_data['transform_info']
            )
            
            # Save results
            self.save_enhanced_results(
                baseline_model, tuned_model, baseline_metrics, tuned_metrics,
                enhanced_data, enhanced_data['transform_info']
            )
            
            # Final summary
            self.logger.info("\n" + "="*80)
            self.logger.info("ENHANCED XGBOOST TRAINING COMPLETION SUMMARY")
            self.logger.info("="*80)
            self.logger.info(f"✅ Enhanced XGBoost Training Completed Successfully!")
            self.logger.info(f"\n📊 Final Results Comparison:")
            self.logger.info(f"   Baseline R² Score: {baseline_metrics['r2_score']:.4f}")
            self.logger.info(f"   Tuned R² Score: {tuned_metrics['r2_score']:.4f}")
            self.logger.info(f"   R² Improvement: {tuned_metrics['r2_score'] - baseline_metrics['r2_score']:.4f}")
            self.logger.info(f"\n   Baseline RMSE: ${baseline_metrics['rmse']:.2f}")
            self.logger.info(f"   Tuned RMSE: ${tuned_metrics['rmse']:.2f}")
            self.logger.info(f"   RMSE Improvement: ${baseline_metrics['rmse'] - tuned_metrics['rmse']:.2f}")
            self.logger.info(f"\n🔧 Enhancements Applied:")
            self.logger.info(f"   • Outlier removal: {enhanced_data['outlier_mask'].sum()} samples")
            self.logger.info(f"   • Target transformation: {enhanced_data['transform_info']['method']}")
            self.logger.info(f"   • Feature engineering: {len(enhanced_data['feature_names'])} features")
            self.logger.info(f"   • Hyperparameter tuning: {len(best_params)} parameters optimized")
            self.logger.info(f"   • Advanced preprocessing: RobustScaler + feature selection")
            
            return {
                'baseline_model': baseline_model,
                'tuned_model': tuned_model,
                'baseline_metrics': baseline_metrics,
                'tuned_metrics': tuned_metrics,
                'enhanced_data': enhanced_data,
                'best_params': best_params,
                'transform_info': enhanced_data['transform_info']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced XGBoost training pipeline: {str(e)}")
            raise


def run_enhanced_xgboost_training():
    """Main function to run enhanced XGBoost training"""
    trainer = XGBoostEnhancedTrainer(random_state=42)
    return trainer.run_complete_enhanced_training()


if __name__ == "__main__":
    # Run the complete enhanced pipeline
    results = run_enhanced_xgboost_training()
    print("\n🎉 Enhanced XGBoost training completed successfully!")
    print(f"Baseline R² Score: {results['baseline_metrics']['r2_score']:.4f}")
    print(f"Tuned R² Score: {results['tuned_metrics']['r2_score']:.4f}")
    print(f"Improvement: {results['tuned_metrics']['r2_score'] - results['baseline_metrics']['r2_score']:.4f}")
