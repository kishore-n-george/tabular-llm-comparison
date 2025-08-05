import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import itertools
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from copy import deepcopy
import xgboost as xgb
import pickle
import os
from datetime import datetime

# Import rtdl for FT-Transformer
try:
    import rtdl
    RTDL_AVAILABLE = True
except ImportError:
    RTDL_AVAILABLE = False
    print("Warning: rtdl library not available. FT-Transformer functions will be limited.")

warnings.filterwarnings('ignore')

def save_model_ablation_results(model_name: str, results: Dict, dataset_name: str = "online_shoppers", 
                               feature_names: List[str] = None):
    """
    Save individual model ablation results to pickle file
    """
    try:
        # Clean model name for filename
        clean_model_name = model_name.replace(' ', '_').replace('-', '_')
        filename = f"{dataset_name}_ablation_{clean_model_name}.pkl"
        
        # Prepare data to save
        save_data = {
            'model_name': model_name,
            'results': results,
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 {model_name} ablation results saved to '{filename}'")
        return filename
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to save {model_name} ablation results: {e}")
        return None

def load_model_ablation_results(model_name: str, dataset_name: str = "online_shoppers"):
    """
    Load individual model ablation results from pickle file
    """
    try:
        clean_model_name = model_name.replace(' ', '_').replace('-', '_')
        filename = f"{dataset_name}_ablation_{clean_model_name}.pkl"
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"⚠️ Warning: Ablation results file '{filename}' not found")
            return None
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to load {model_name} ablation results: {e}")
        return None

def save_comprehensive_ablation_results(all_results: Dict, dataset_name: str = "online_shoppers"):
    """
    Save comprehensive ablation results to pickle file
    """
    try:
        filename = f"{dataset_name}_section5_ablation_results.pkl"
        
        # Prepare comprehensive data
        save_data = {
            'all_results': all_results,
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'summary': {
                'total_models': len([k for k in all_results.keys() if k != 'comparative_analysis']),
                'models_analyzed': [k for k in all_results.keys() if k != 'comparative_analysis'],
                'analysis_types': ['baseline', 'feature_ablation', 'data_size_ablation', 'noise_robustness', 'model_specific_ablations', 'comparative_analysis']
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Comprehensive ablation results saved to '{filename}'")
        print(f"📋 Contains results for {save_data['summary']['total_models']} models")
        return filename
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to save comprehensive ablation results: {e}")
        return None

class EnhancedAblationStudyAnalyzer:
    """
    Enhanced ablation study analyzer with specific support for TabICL and TabPFN models
    """
    
    def __init__(self):
        self.results = {}
        self.feature_names = None
        self.model_specific_configs = {
            'TabICL': {
                'context_sizes': [16, 32, 64, 128, 256],
                'example_selection_strategies': ['random', 'diverse', 'similar'],
                'prompt_formats': ['standard', 'detailed', 'minimal']
            },
            'TabPFN': {
                'context_sizes': [100, 500, 1000, 1500, 2000],
                'devices': ['cpu', 'cuda'],
                'batch_sizes': [1, 8, 16, 32, 64]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200, 300, 500],
                'learning_rates': [0.01, 0.05, 0.1, 0.2, 0.3],
                'max_depths': [3, 4, 5, 6, 8, 10],
                'subsamples': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytrees': [0.6, 0.7, 0.8, 0.9, 1.0]
            },
            'FT-Transformer': {
                'learning_rates': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                'batch_sizes': [64, 128, 256, 512],
                'n_epochs': [50, 100, 150, 200],
                'weight_decays': [1e-6, 1e-5, 1e-4, 1e-3],
                'devices': ['cpu', 'cuda']
            }
        }
    
    def comprehensive_ablation_study(self, models_dict: Dict, model_names: List[str], 
                                   X_train, X_test, y_train, y_test, 
                                   feature_names: List[str] = None, 
                                   dataset_name: str = "online_shoppers"):
        """
        Perform comprehensive ablation studies for all models with pickle saving
        """
        print("🔬 COMPREHENSIVE ABLATION STUDY")
        print("=" * 80)
        
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Store baseline results for all models
        baseline_results = {}
        
        for model_name in model_names:
            if model_name in models_dict:
                print(f"\n📊 Running ablation studies for {model_name}...")
                
                # Standard ablation studies
                ablation_results = self._run_standard_ablations(
                    models_dict[model_name], model_name, 
                    X_train, X_test, y_train, y_test
                )
                
                # Model-specific ablation studies
                if model_name in ['TabICL', 'TabPFN', 'TabPFN v2', 'XGBoost', 'FT-Transformer', 'FTTransformer']:
                    model_specific_results = self._run_model_specific_ablations(
                        models_dict[model_name], model_name,
                        X_train, X_test, y_train, y_test
                    )
                    ablation_results.update(model_specific_results)
                
                self.results[model_name] = ablation_results
                baseline_results[model_name] = ablation_results['baseline']
                
                # Save individual model results to pickle file
                save_model_ablation_results(
                    model_name=model_name,
                    results=ablation_results,
                    dataset_name=dataset_name,
                    feature_names=self.feature_names
                )

        
        # Comparative analysis
        self._run_comparative_analysis(model_names, baseline_results)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(model_names)
        
        # Save comprehensive results to pickle file
        save_comprehensive_ablation_results(
            all_results=self.results,
            dataset_name=dataset_name
        )
        
        return self.results
    
    def _run_standard_ablations(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """Run standard ablation studies (feature, data size, etc.)"""
        
        # Get baseline performance
        baseline_metrics = self._get_baseline_performance(model, X_train, X_test, y_train, y_test)
        
        results = {
            'baseline': baseline_metrics,
            'feature_ablation': self._feature_ablation_study(
                model, model_name, X_train, X_test, y_train, y_test, baseline_metrics
            ),
            'data_size_ablation': self._data_size_ablation_study(
                model, model_name, X_train, X_test, y_train, y_test
            ),
            'noise_robustness': self._noise_robustness_study(
                model, model_name, X_train, X_test, y_train, y_test, baseline_metrics
            )
        }
        
        return results
    
    def _run_model_specific_ablations(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """Run model-specific ablation studies"""
        
        results = {}
        
        if 'TabPFN' in model_name:
            results.update(self._tabpfn_specific_ablations(model, model_name, X_train, X_test, y_train, y_test))
        
        if 'TabICL' in model_name:
            results.update(self._tabicl_specific_ablations(model, model_name, X_train, X_test, y_train, y_test))
        
        if 'XGBoost' in model_name:
            results.update(self._xgboost_specific_ablations(model, model_name, X_train, X_test, y_train, y_test))
        
        if 'FT-Transformer' in model_name or 'FTTransformer' in model_name:
            results.update(self._ft_transformer_specific_ablations(model, model_name, X_train, X_test, y_train, y_test))
        
        return results
    
    def _tabpfn_specific_ablations(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """TabPFN-specific ablation studies"""
        print(f"   🧪 Running TabPFN-specific ablations...")
        
        results = {}
        
        # Context size ablation
        results['context_size_ablation'] = self._tabpfn_context_size_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Device performance ablation
        results['device_ablation'] = self._tabpfn_device_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Memory efficiency ablation
        results['memory_ablation'] = self._tabpfn_memory_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        return results
    
    def _tabicl_specific_ablations(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """TabICL-specific ablation studies"""
        print(f"   🧪 Running TabICL-specific ablations...")
        
        results = {}
        
        # In-context examples ablation
        results['context_examples_ablation'] = self._tabicl_context_examples_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Example selection strategy ablation
        results['example_selection_ablation'] = self._tabicl_example_selection_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Context window utilization ablation
        results['context_window_ablation'] = self._tabicl_context_window_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        return results
    
    def _get_baseline_performance(self, model, X_train, X_test, y_train, y_test):
        """Get baseline performance metrics"""
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'train_time': train_time,
            'inference_time': inference_time,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        return metrics
    
    def _feature_ablation_study(self, model, model_name: str, X_train, X_test, y_train, y_test, baseline_metrics):
        """Enhanced feature ablation study"""
        print(f"      🔍 Feature ablation study...")
        
        results = {
            'single_feature_removal': [],
            'feature_group_removal': [],
            'cumulative_removal': []
        }
        
        n_features = X_train.shape[1]
        
        # Single feature removal
        for i in range(n_features):
            feature_mask = np.ones(n_features, dtype=bool)
            feature_mask[i] = False
            
            X_train_reduced = X_train[:, feature_mask]
            X_test_reduced = X_test[:, feature_mask]
            
            try:
                model_copy = deepcopy(model)
                model_copy.fit(X_train_reduced, y_train)
                y_pred = model_copy.predict(X_test_reduced)
                
                f1_score_reduced = f1_score(y_test, y_pred)
                f1_drop = baseline_metrics['f1'] - f1_score_reduced
                
                results['single_feature_removal'].append({
                    'removed_feature': self.feature_names[i],
                    'removed_feature_idx': i,
                    'f1_score': f1_score_reduced,
                    'f1_drop': f1_drop,
                    'relative_importance': f1_drop / baseline_metrics['f1'] if baseline_metrics['f1'] > 0 else 0
                })
            except Exception as e:
                print(f"        Warning: Failed to evaluate without feature {self.feature_names[i]}: {e}")
        
        # Sort by importance
        results['single_feature_removal'].sort(key=lambda x: x['f1_drop'], reverse=True)
        
        # Feature group removal (domain-specific groups)
        feature_groups = self._define_feature_groups()
        for group_name, feature_indices in feature_groups.items():
            if max(feature_indices) < n_features:
                feature_mask = np.ones(n_features, dtype=bool)
                feature_mask[feature_indices] = False
                
                X_train_reduced = X_train[:, feature_mask]
                X_test_reduced = X_test[:, feature_mask]
                
                try:
                    model_copy = deepcopy(model)
                    model_copy.fit(X_train_reduced, y_train)
                    y_pred = model_copy.predict(X_test_reduced)
                    
                    f1_score_reduced = f1_score(y_test, y_pred)
                    f1_drop = baseline_metrics['f1'] - f1_score_reduced
                    
                    results['feature_group_removal'].append({
                        'group_name': group_name,
                        'removed_features': [self.feature_names[i] for i in feature_indices],
                        'f1_score': f1_score_reduced,
                        'f1_drop': f1_drop,
                        'relative_importance': f1_drop / baseline_metrics['f1'] if baseline_metrics['f1'] > 0 else 0
                    })
                except Exception as e:
                    print(f"        Warning: Failed to evaluate without feature group {group_name}: {e}")
        
        # Sort feature groups by importance
        results['feature_group_removal'].sort(key=lambda x: x['f1_drop'], reverse=True)
        
        # Cumulative feature removal (greedy approach)
        remaining_features = list(range(n_features))
        cumulative_results = []
        
        for step in range(min(5, n_features)):  # Remove up to 5 features
            if not remaining_features:
                break
                
            best_removal = None
            best_f1_drop = -1
            
            for feature_idx in remaining_features:
                temp_mask = np.ones(n_features, dtype=bool)
                temp_mask[[f for f in range(n_features) if f not in remaining_features or f == feature_idx]] = False
                
                X_train_temp = X_train[:, temp_mask]
                X_test_temp = X_test[:, temp_mask]
                
                try:
                    model_copy = deepcopy(model)
                    model_copy.fit(X_train_temp, y_train)
                    y_pred = model_copy.predict(X_test_temp)
                    f1_temp = f1_score(y_test, y_pred)
                    
                    # Calculate F1 drop from current state
                    current_f1 = cumulative_results[-1]['f1_score'] if cumulative_results else baseline_metrics['f1']
                    f1_drop = current_f1 - f1_temp
                    
                    if f1_drop > best_f1_drop:
                        best_f1_drop = f1_drop
                        best_removal = feature_idx
                        best_f1 = f1_temp
                except:
                    continue
            
            if best_removal is not None:
                remaining_features.remove(best_removal)
                cumulative_results.append({
                    'step': step + 1,
                    'removed_feature': self.feature_names[best_removal],
                    'remaining_features': len(remaining_features),
                    'f1_score': best_f1,
                    'f1_drop_step': best_f1_drop,
                    'cumulative_f1_drop': baseline_metrics['f1'] - best_f1
                })
        
        results['cumulative_removal'] = cumulative_results
        
        return results
    
    def _define_feature_groups(self):
        """Define domain-specific feature groups for online shoppers dataset"""
        return {
            'Administrative': [0, 1],  # Administrative, Administrative_Duration
            'Informational': [2, 3],   # Informational, Informational_Duration
            'ProductRelated': [4, 5],  # ProductRelated, ProductRelated_Duration
            'Behavior_Metrics': [6, 7, 8],  # BounceRates, ExitRates, PageValues
            'Temporal': [9, 10],       # SpecialDay, Month
            'Technical': [11, 12, 13, 14],  # OperatingSystems, Browser, Region, TrafficType
            'User_Type': [15, 16]      # VisitorType, Weekend
        }
    
    def _data_size_ablation_study(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """Study impact of training data size"""
        print(f"      📊 Data size ablation study...")
        
        size_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        results = []
        
        for fraction in size_fractions:
            n_samples = int(len(X_train) * fraction)
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
            
            try:
                start_time = time.time()
                model_copy = deepcopy(model)
                model_copy.fit(X_train_sample, y_train_sample)
                train_time = time.time() - start_time
                
                y_pred = model_copy.predict(X_test)
                
                results.append({
                    'data_fraction': fraction,
                    'n_samples': n_samples,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"        Warning: Failed with {fraction*100}% data: {e}")
        
        return results
    
    def _noise_robustness_study(self, model, model_name: str, X_train, X_test, y_train, y_test, baseline_metrics):
        """Study robustness to feature noise"""
        print(f"      🔊 Noise robustness study...")
        
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        results = []
        
        for noise_level in noise_levels:
            # Add Gaussian noise to features
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_test_noisy = X_test + noise
            
            try:
                model_copy = deepcopy(model)
                model_copy.fit(X_train, y_train)
                y_pred = model_copy.predict(X_test_noisy)
                
                f1_noisy = f1_score(y_test, y_pred)
                f1_drop = baseline_metrics['f1'] - f1_noisy
                
                results.append({
                    'noise_level': noise_level,
                    'f1_score': f1_noisy,
                    'f1_drop': f1_drop,
                    'robustness_score': 1 - (f1_drop / baseline_metrics['f1']) if baseline_metrics['f1'] > 0 else 0
                })
            except Exception as e:
                print(f"        Warning: Failed with noise level {noise_level}: {e}")
        
        return results
    
    def _tabpfn_context_size_ablation(self, model, X_train, X_test, y_train, y_test):
        """TabPFN context size ablation"""
        print(f"        📏 Context size ablation...")
        
        context_sizes = self.model_specific_configs['TabPFN']['context_sizes']
        results = []
        
        for context_size in context_sizes:
            if context_size > len(X_train):
                continue
                
            # Sample training data to match context size
            indices = np.random.choice(len(X_train), min(context_size, len(X_train)), replace=False)
            X_train_context = X_train[indices]
            y_train_context = y_train[indices]
            
            try:
                # Create new model instance with context size constraint
                from tabpfn import TabPFNClassifier
                model_context = TabPFNClassifier(
                    device=model.device if hasattr(model, 'device') else 'cpu',
                    ignore_pretraining_limits=True
                )
                
                start_time = time.time()
                model_context.fit(X_train_context, y_train_context)
                train_time = time.time() - start_time
                
                y_pred = model_context.predict(X_test)
                
                results.append({
                    'context_size': context_size,
                    'actual_samples': len(X_train_context),
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"          Warning: Failed with context size {context_size}: {e}")
        
        return results
    
    def _tabpfn_device_ablation(self, model, X_train, X_test, y_train, y_test):
        """TabPFN device performance ablation"""
        print(f"        💻 Device performance ablation...")
        
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        results = []
        
        for device in devices:
            try:
                from tabpfn import TabPFNClassifier
                model_device = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
                
                start_time = time.time()
                model_device.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = model_device.predict(X_test)
                inference_time = time.time() - start_time
                
                results.append({
                    'device': device,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time,
                    'inference_time': inference_time,
                    'speedup_factor': results[0]['train_time'] / train_time if results else 1.0
                })
            except Exception as e:
                print(f"          Warning: Failed with device {device}: {e}")
        
        return results
    
    def _tabpfn_memory_ablation(self, model, X_train, X_test, y_train, y_test):
        """TabPFN memory efficiency ablation - optimized for speed"""
        print(f"        🧠 Memory efficiency ablation (optimized)...")
        
        # Use smaller, more realistic batch sizes for faster testing
        batch_sizes = [1, 16, 64, 256]  # Reduced from original config
        results = []
        
        # Train model once and reuse for all batch size tests
        try:
            print(f"          Training model once for reuse...")
            model_trained = deepcopy(model)
            model_trained.fit(X_train, y_train)
            
            # Use smaller test set for faster evaluation (first 200 samples)
            X_test_sample = X_test[:min(200, len(X_test))]
            y_test_sample = y_test[:min(200, len(y_test))]
            
            for batch_size in batch_sizes:
                try:
                    # Calculate number of batches
                    n_batches = len(X_test_sample) // batch_size + (1 if len(X_test_sample) % batch_size > 0 else 0)
                    
                    start_time = time.time()
                    y_pred_batches = []
                    
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(X_test_sample))
                        X_batch = X_test_sample[start_idx:end_idx]
                        
                        y_pred_batch = model_trained.predict(X_batch)
                        y_pred_batches.extend(y_pred_batch)
                    
                    inference_time = time.time() - start_time
                    
                    results.append({
                        'batch_size': batch_size,
                        'n_batches': n_batches,
                        'n_samples_tested': len(X_test_sample),
                        'f1_score': f1_score(y_test_sample, y_pred_batches),
                        'inference_time': inference_time,
                        'time_per_sample': inference_time / len(X_test_sample),
                        'throughput_samples_per_sec': len(X_test_sample) / inference_time if inference_time > 0 else 0
                    })
                    
                    print(f"          ✓ Batch size {batch_size}: {inference_time:.3f}s ({len(X_test_sample)/inference_time:.1f} samples/sec)")
                    
                except Exception as e:
                    print(f"          Warning: Failed with batch size {batch_size}: {e}")
                    
        except Exception as e:
            print(f"          Warning: Failed to train model for memory ablation: {e}")
            return []
        
        return results
    
    def _tabicl_context_examples_ablation(self, model, X_train, X_test, y_train, y_test):
        """TabICL in-context examples ablation"""
        print(f"        📝 Context examples ablation...")
        
        context_sizes = self.model_specific_configs['TabICL']['context_sizes']
        results = []
        
        for context_size in context_sizes:
            if context_size > len(X_train):
                continue
                
            try:
                # Sample different numbers of in-context examples
                indices = np.random.choice(len(X_train), min(context_size, len(X_train)), replace=False)
                X_train_context = X_train[indices]
                y_train_context = y_train[indices]
                
                model_copy = deepcopy(model)
                
                start_time = time.time()
                model_copy.fit(X_train_context, y_train_context)
                train_time = time.time() - start_time
                
                y_pred = model_copy.predict(X_test)
                
                results.append({
                    'context_examples': context_size,
                    'actual_examples': len(X_train_context),
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"          Warning: Failed with {context_size} examples: {e}")
        
        return results
    
    def _tabicl_example_selection_ablation(self, model, X_train, X_test, y_train, y_test):
        """TabICL example selection strategy ablation"""
        print(f"        🎯 Example selection strategy ablation...")
        
        strategies = self.model_specific_configs['TabICL']['example_selection_strategies']
        results = []
        context_size = min(64, len(X_train))  # Fixed context size for fair comparison
        
        for strategy in strategies:
            try:
                if strategy == 'random':
                    indices = np.random.choice(len(X_train), context_size, replace=False)
                elif strategy == 'diverse':
                    # Select diverse examples (simplified approach)
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=context_size, random_state=42)
                    clusters = kmeans.fit_predict(X_train)
                    indices = []
                    for i in range(context_size):
                        cluster_indices = np.where(clusters == i)[0]
                        if len(cluster_indices) > 0:
                            indices.append(np.random.choice(cluster_indices))
                    indices = np.array(indices[:context_size])
                elif strategy == 'similar':
                    # Select similar examples to test set (simplified approach)
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(X_train, X_test.mean(axis=0).reshape(1, -1))
                    indices = np.argsort(similarities.flatten())[-context_size:]
                
                X_train_selected = X_train[indices]
                y_train_selected = y_train[indices]
                
                model_copy = deepcopy(model)
                model_copy.fit(X_train_selected, y_train_selected)
                y_pred = model_copy.predict(X_test)
                
                results.append({
                    'selection_strategy': strategy,
                    'context_size': len(indices),
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred)
                })
            except Exception as e:
                print(f"          Warning: Failed with strategy {strategy}: {e}")
        
        return results
    
    def _tabicl_context_window_ablation(self, model, X_train, X_test, y_train, y_test):
        """TabICL context window utilization ablation"""
        print(f"        🪟 Context window utilization ablation...")
        
        # Test different ways of utilizing the context window
        window_strategies = ['full', 'balanced', 'recent', 'diverse']
        results = []
        
        for strategy in window_strategies:
            try:
                if strategy == 'full':
                    # Use all available training data
                    X_train_window = X_train
                    y_train_window = y_train
                elif strategy == 'balanced':
                    # Balance classes in context
                    class_0_indices = np.where(y_train == 0)[0]
                    class_1_indices = np.where(y_train == 1)[0]
                    n_per_class = min(len(class_0_indices), len(class_1_indices), 32)
                    
                    selected_0 = np.random.choice(class_0_indices, n_per_class, replace=False)
                    selected_1 = np.random.choice(class_1_indices, n_per_class, replace=False)
                    indices = np.concatenate([selected_0, selected_1])
                    
                    X_train_window = X_train[indices]
                    y_train_window = y_train[indices]
                elif strategy == 'recent':
                    # Use most recent examples (last 64)
                    indices = np.arange(max(0, len(X_train) - 64), len(X_train))
                    X_train_window = X_train[indices]
                    y_train_window = y_train[indices]
                elif strategy == 'diverse':
                    # Use diverse examples
                    from sklearn.cluster import KMeans
                    n_clusters = min(64, len(X_train))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_train)
                    indices = []
                    for i in range(n_clusters):
                        cluster_indices = np.where(clusters == i)[0]
                        if len(cluster_indices) > 0:
                            indices.append(np.random.choice(cluster_indices))
                    indices = np.array(indices)
                    
                    X_train_window = X_train[indices]
                    y_train_window = y_train[indices]
                
                model_copy = deepcopy(model)
                model_copy.fit(X_train_window, y_train_window)
                y_pred = model_copy.predict(X_test)
                
                results.append({
                    'window_strategy': strategy,
                    'context_size': len(X_train_window),
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred)
                })
            except Exception as e:
                print(f"          Warning: Failed with strategy {strategy}: {e}")
        
        return results
    
    def _xgboost_specific_ablations(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """XGBoost-specific ablation studies"""
        print(f"   🧪 Running XGBoost-specific ablations...")
        
        results = {}
        
        # Hyperparameter ablation
        results['hyperparameter_ablation'] = self._xgboost_hyperparameter_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Tree complexity ablation
        results['tree_complexity_ablation'] = self._xgboost_tree_complexity_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Regularization ablation
        results['regularization_ablation'] = self._xgboost_regularization_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Feature importance analysis
        results['feature_importance_analysis'] = self._xgboost_feature_importance_analysis(
            model, X_train, X_test, y_train, y_test
        )
        
        return results
    
    def _ft_transformer_specific_ablations(self, model, model_name: str, X_train, X_test, y_train, y_test):
        """FT-Transformer-specific ablation studies"""
        print(f"   🧪 Running FT-Transformer-specific ablations...")
        
        results = {}
        
        # Architecture ablation
        results['architecture_ablation'] = self._ft_transformer_architecture_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Training hyperparameter ablation
        results['training_ablation'] = self._ft_transformer_training_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Device performance ablation
        results['device_ablation'] = self._ft_transformer_device_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        # Batch size efficiency ablation
        results['batch_size_ablation'] = self._ft_transformer_batch_size_ablation(
            model, X_train, X_test, y_train, y_test
        )
        
        return results
    
    def _xgboost_hyperparameter_ablation(self, model, X_train, X_test, y_train, y_test):
        """XGBoost hyperparameter ablation"""
        print(f"        🎛️ Hyperparameter ablation...")
        
        results = []
        base_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'objective': 'binary:logistic'
        }
        
        # Test different n_estimators
        for n_est in self.model_specific_configs['XGBoost']['n_estimators']:
            try:
                params = base_params.copy()
                params['n_estimators'] = n_est
                
                xgb_model = xgb.XGBClassifier(**params)
                
                start_time = time.time()
                xgb_model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = xgb_model.predict(X_test)
                
                results.append({
                    'parameter': 'n_estimators',
                    'value': n_est,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"          Warning: Failed with n_estimators={n_est}: {e}")
        
        # Test different learning rates
        for lr in self.model_specific_configs['XGBoost']['learning_rates']:
            try:
                params = base_params.copy()
                params['learning_rate'] = lr
                
                xgb_model = xgb.XGBClassifier(**params)
                
                start_time = time.time()
                xgb_model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = xgb_model.predict(X_test)
                
                results.append({
                    'parameter': 'learning_rate',
                    'value': lr,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time
                })
            except Exception as e:
                print(f"          Warning: Failed with learning_rate={lr}: {e}")
        
        return results
    
    def _xgboost_tree_complexity_ablation(self, model, X_train, X_test, y_train, y_test):
        """XGBoost tree complexity ablation"""
        print(f"        🌳 Tree complexity ablation...")
        
        results = []
        base_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'objective': 'binary:logistic'
        }
        
        # Test different max_depths
        for depth in self.model_specific_configs['XGBoost']['max_depths']:
            try:
                params = base_params.copy()
                params['max_depth'] = depth
                
                xgb_model = xgb.XGBClassifier(**params)
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                
                results.append({
                    'parameter': 'max_depth',
                    'value': depth,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'model_complexity': depth  # Proxy for complexity
                })
            except Exception as e:
                print(f"          Warning: Failed with max_depth={depth}: {e}")
        
        return results
    
    def _xgboost_regularization_ablation(self, model, X_train, X_test, y_train, y_test):
        """XGBoost regularization ablation"""
        print(f"        🛡️ Regularization ablation...")
        
        results = []
        base_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'objective': 'binary:logistic'
        }
        
        # Test different subsample rates
        for subsample in self.model_specific_configs['XGBoost']['subsamples']:
            try:
                params = base_params.copy()
                params['subsample'] = subsample
                
                xgb_model = xgb.XGBClassifier(**params)
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                
                results.append({
                    'parameter': 'subsample',
                    'value': subsample,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred)
                })
            except Exception as e:
                print(f"          Warning: Failed with subsample={subsample}: {e}")
        
        # Test different colsample_bytree rates
        for colsample in self.model_specific_configs['XGBoost']['colsample_bytrees']:
            try:
                params = base_params.copy()
                params['colsample_bytree'] = colsample
                
                xgb_model = xgb.XGBClassifier(**params)
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                
                results.append({
                    'parameter': 'colsample_bytree',
                    'value': colsample,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred)
                })
            except Exception as e:
                print(f"          Warning: Failed with colsample_bytree={colsample}: {e}")
        
        return results
    
    def _xgboost_feature_importance_analysis(self, model, X_train, X_test, y_train, y_test):
        """XGBoost feature importance analysis"""
        print(f"        🎯 Feature importance analysis...")
        
        try:
            # Train model to get feature importance
            model_copy = deepcopy(model)
            model_copy.fit(X_train, y_train)
            
            # Get different types of feature importance
            importance_types = ['weight', 'gain', 'cover']
            results = {}
            
            for imp_type in importance_types:
                try:
                    importance_scores = model_copy.get_booster().get_score(importance_type=imp_type)
                    
                    # Convert to list format
                    feature_importance = []
                    for i, feature_name in enumerate(self.feature_names):
                        feature_key = f'f{i}'  # XGBoost uses f0, f1, etc.
                        importance = importance_scores.get(feature_key, 0.0)
                        feature_importance.append({
                            'feature': feature_name,
                            'importance': importance,
                            'importance_type': imp_type
                        })
                    
                    # Sort by importance
                    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                    results[imp_type] = feature_importance
                    
                except Exception as e:
                    print(f"          Warning: Failed to get {imp_type} importance: {e}")
            
            return results
            
        except Exception as e:
            print(f"          Warning: Failed feature importance analysis: {e}")
            return {}
    
    def _ft_transformer_architecture_ablation(self, model, X_train, X_test, y_train, y_test):
        """FT-Transformer architecture ablation"""
        print(f"        🏗️ Architecture ablation...")
        
        if not RTDL_AVAILABLE:
            print("          Warning: rtdl not available, skipping architecture ablation")
            return []
        
        results = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test different model configurations
        configs = [
            {'d_token': 64, 'n_blocks': 2, 'd_ffn_factor': 2},
            {'d_token': 128, 'n_blocks': 3, 'd_ffn_factor': 2},
            {'d_token': 192, 'n_blocks': 4, 'd_ffn_factor': 2},
            {'d_token': 256, 'n_blocks': 3, 'd_ffn_factor': 4}
        ]
        
        for config in configs:
            try:
                # Create model with specific architecture
                ft_model = rtdl.FTTransformer.make_default(
                    n_num_features=X_train.shape[1],
                    cat_cardinalities=[],
                    d_out=2,
                    **config
                )
                ft_model = ft_model.to(device)
                
                # Convert data to tensors
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                
                # Simple training loop
                optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                start_time = time.time()
                
                # Train for a few epochs
                ft_model.train()
                for epoch in range(10):  # Quick training for ablation
                    optimizer.zero_grad()
                    output = ft_model(X_train_tensor, None)
                    loss = criterion(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                train_time = time.time() - start_time
                
                # Evaluate
                ft_model.eval()
                with torch.no_grad():
                    output = ft_model(X_test_tensor, None)
                    y_pred = output.argmax(dim=1).cpu().numpy()
                
                results.append({
                    'config': config,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time,
                    'n_parameters': sum(p.numel() for p in ft_model.parameters())
                })
                
            except Exception as e:
                print(f"          Warning: Failed with config {config}: {e}")
        
        return results
    
    def _ft_transformer_training_ablation(self, model, X_train, X_test, y_train, y_test):
        """FT-Transformer training hyperparameter ablation"""
        print(f"        📚 Training hyperparameter ablation...")
        
        if not RTDL_AVAILABLE:
            print("          Warning: rtdl not available, skipping training ablation")
            return []
        
        results = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test different learning rates
        for lr in self.model_specific_configs['FT-Transformer']['learning_rates']:
            try:
                ft_model = rtdl.FTTransformer.make_default(
                    n_num_features=X_train.shape[1],
                    cat_cardinalities=[],
                    d_out=2
                )
                ft_model = ft_model.to(device)
                
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                
                optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                
                start_time = time.time()
                
                # Train for a few epochs
                ft_model.train()
                for epoch in range(10):
                    optimizer.zero_grad()
                    output = ft_model(X_train_tensor, None)
                    loss = criterion(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                train_time = time.time() - start_time
                
                # Evaluate
                ft_model.eval()
                with torch.no_grad():
                    output = ft_model(X_test_tensor, None)
                    y_pred = output.argmax(dim=1).cpu().numpy()
                
                results.append({
                    'parameter': 'learning_rate',
                    'value': lr,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time
                })
                
            except Exception as e:
                print(f"          Warning: Failed with learning_rate={lr}: {e}")
        
        return results
    
    def _ft_transformer_device_ablation(self, model, X_train, X_test, y_train, y_test):
        """FT-Transformer device performance ablation"""
        print(f"        💻 Device performance ablation...")
        
        if not RTDL_AVAILABLE:
            print("          Warning: rtdl not available, skipping device ablation")
            return []
        
        results = []
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device_name in devices:
            try:
                device = torch.device(device_name)
                
                ft_model = rtdl.FTTransformer.make_default(
                    n_num_features=X_train.shape[1],
                    cat_cardinalities=[],
                    d_out=2
                )
                ft_model = ft_model.to(device)
                
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_train_tensor = torch.LongTensor(y_train).to(device)
                
                optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                # Training time
                start_time = time.time()
                ft_model.train()
                for epoch in range(5):  # Quick training for comparison
                    optimizer.zero_grad()
                    output = ft_model(X_train_tensor, None)
                    loss = criterion(output, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                train_time = time.time() - start_time
                
                # Inference time
                ft_model.eval()
                start_time = time.time()
                with torch.no_grad():
                    output = ft_model(X_test_tensor, None)
                    y_pred = output.argmax(dim=1).cpu().numpy()
                inference_time = time.time() - start_time
                
                results.append({
                    'device': device_name,
                    'f1_score': f1_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'train_time': train_time,
                    'inference_time': inference_time,
                    'speedup_factor': results[0]['train_time'] / train_time if results else 1.0
                })
                
            except Exception as e:
                print(f"          Warning: Failed with device {device_name}: {e}")
        
        return results
    
    def _ft_transformer_batch_size_ablation(self, model, X_train, X_test, y_train, y_test):
        """FT-Transformer batch size efficiency ablation"""
        print(f"        📦 Batch size efficiency ablation...")
        
        if not RTDL_AVAILABLE:
            print("          Warning: rtdl not available, skipping batch size ablation")
            return []
        
        results = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for batch_size in self.model_specific_configs['FT-Transformer']['batch_sizes']:
            try:
                ft_model = rtdl.FTTransformer.make_default(
                    n_num_features=X_train.shape[1],
                    cat_cardinalities=[],
                    d_out=2
                )
                ft_model = ft_model.to(device)
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                )
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test),
                    torch.LongTensor(y_test)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                # Training with batches
                start_time = time.time()
                ft_model.train()
                for epoch in range(3):  # Quick training
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        output = ft_model(batch_X, None)
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()
                train_time = time.time() - start_time
                
                # Evaluation with batches
                ft_model.eval()
                all_preds = []
                start_time = time.time()
                with torch.no_grad():
                    for batch_X, _ in test_loader:
                        batch_X = batch_X.to(device)
                        output = ft_model(batch_X, None)
                        preds = output.argmax(dim=1).cpu().numpy()
                        all_preds.extend(preds)
                inference_time = time.time() - start_time
                
                results.append({
                    'batch_size': batch_size,
                    'f1_score': f1_score(y_test, all_preds),
                    'accuracy': accuracy_score(y_test, all_preds),
                    'train_time': train_time,
                    'inference_time': inference_time,
                    'time_per_sample': inference_time / len(y_test)
                })
                
            except Exception as e:
                print(f"          Warning: Failed with batch_size={batch_size}: {e}")
        
        return results
    
    def _run_comparative_analysis(self, model_names: List[str], baseline_results: Dict):
        """Run comparative analysis across models"""
        print(f"\n🔄 Running comparative analysis...")
        
        # Feature importance comparison
        self._compare_feature_importance(model_names)
        
        # Robustness comparison
        self._compare_robustness(model_names)
        
        # Efficiency comparison
        self._compare_efficiency(model_names, baseline_results)
    
    def _compare_feature_importance(self, model_names: List[str]):
        """Compare feature importance across models"""
        print(f"      🎯 Comparing feature importance...")
        
        importance_comparison = {}
        
        for model_name in model_names:
            if model_name in self.results and 'feature_ablation' in self.results[model_name]:
                single_removal = self.results[model_name]['feature_ablation']['single_feature_removal']
                
                # Create importance ranking
                feature_importance = {}
                for result in single_removal:
                    feature_importance[result['removed_feature']] = result['relative_importance']
                
                importance_comparison[model_name] = feature_importance
        
        # Store comparison results
        self.results['comparative_analysis'] = {
            'feature_importance_comparison': importance_comparison
        }
        
        # Create visualization
        self._plot_feature_importance_comparison(importance_comparison)
    
    def _compare_robustness(self, model_names: List[str]):
        """Compare model robustness"""
        print(f"      🛡️ Comparing model robustness...")
        
        robustness_scores = {}
        
        for model_name in model_names:
            if model_name in self.results and 'noise_robustness' in self.results[model_name]:
                noise_results = self.results[model_name]['noise_robustness']
                
                # Calculate average robustness score
                avg_robustness = np.mean([r['robustness_score'] for r in noise_results])
                robustness_scores[model_name] = avg_robustness
        
        # Store robustness comparison
        if 'comparative_analysis' not in self.results:
            self.results['comparative_analysis'] = {}
        self.results['comparative_analysis']['robustness_comparison'] = robustness_scores
        
        # Create visualization
        self._plot_robustness_comparison(robustness_scores)
    
    def _compare_efficiency(self, model_names: List[str], baseline_results: Dict):
        """Compare computational efficiency"""
        print(f"      ⚡ Comparing computational efficiency...")
        
        efficiency_comparison = {}
        
        for model_name in model_names:
            if model_name in baseline_results:
                baseline = baseline_results[model_name]
                efficiency_comparison[model_name] = {
                    'train_time': baseline['train_time'],
                    'inference_time': baseline['inference_time'],
                    'time_per_sample': baseline['inference_time'] / baseline['n_test_samples']
                }
        
        # Store efficiency comparison
        if 'comparative_analysis' not in self.results:
            self.results['comparative_analysis'] = {}
        self.results['comparative_analysis']['efficiency_comparison'] = efficiency_comparison
        
        # Create visualization
        self._plot_efficiency_comparison(efficiency_comparison)
    
    def _plot_feature_importance_comparison(self, importance_comparison: Dict):
        """Plot feature importance comparison across models"""
        
        if not importance_comparison:
            return
        
        # Get all unique features
        all_features = set()
        for model_importance in importance_comparison.values():
            all_features.update(model_importance.keys())
        all_features = sorted(list(all_features))
        
        # Create comparison matrix
        comparison_matrix = []
        model_names = list(importance_comparison.keys())
        
        for feature in all_features:
            row = []
            for model_name in model_names:
                importance = importance_comparison[model_name].get(feature, 0)
                row.append(importance)
            comparison_matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            comparison_matrix,
            xticklabels=model_names,
            yticklabels=all_features,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd'
        )
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Models')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_robustness_comparison(self, robustness_scores: Dict):
        """Plot robustness comparison"""
        
        if not robustness_scores:
            return
        
        models = list(robustness_scores.keys())
        scores = list(robustness_scores.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(models)])
        plt.title('Model Robustness Comparison')
        plt.xlabel('Models')
        plt.ylabel('Average Robustness Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_efficiency_comparison(self, efficiency_comparison: Dict):
        """Plot efficiency comparison"""
        
        if not efficiency_comparison:
            return
        
        models = list(efficiency_comparison.keys())
        train_times = [efficiency_comparison[m]['train_time'] for m in models]
        inference_times = [efficiency_comparison[m]['inference_time'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training time comparison
        bars1 = ax1.bar(models, train_times, color='lightblue')
        ax1.set_title('Training Time Comparison')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars1, train_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_times)*0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # Inference time comparison
        bars2 = ax2.bar(models, inference_times, color='lightcoral')
        ax2.set_title('Inference Time Comparison')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Inference Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars2, inference_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(inference_times)*0.01,
                    f'{time:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_comprehensive_report(self, model_names: List[str]):
        """Generate comprehensive ablation study report"""
        print(f"\n📋 COMPREHENSIVE ABLATION STUDY REPORT")
        print("=" * 80)
        
        # Performance summary
        print(f"\n📊 BASELINE PERFORMANCE SUMMARY")
        print("-" * 50)
        
        for model_name in model_names:
            if model_name in self.results and 'baseline' in self.results[model_name]:
                baseline = self.results[model_name]['baseline']
                print(f"\n{model_name}:")
                print(f"  Accuracy: {baseline['accuracy']:.4f}")
                print(f"  F1-Score: {baseline['f1']:.4f}")
                print(f"  Precision: {baseline['precision']:.4f}")
                print(f"  Recall: {baseline['recall']:.4f}")
                if baseline['auc']:
                    print(f"  AUC: {baseline['auc']:.4f}")
                print(f"  Train Time: {baseline['train_time']:.3f}s")
                print(f"  Inference Time: {baseline['inference_time']:.3f}s")
        
        # Feature importance insights
        print(f"\n🎯 FEATURE IMPORTANCE INSIGHTS")
        print("-" * 50)
        
        for model_name in model_names:
            if (model_name in self.results and 
                'feature_ablation' in self.results[model_name] and
                self.results[model_name]['feature_ablation']['single_feature_removal']):
                
                top_features = self.results[model_name]['feature_ablation']['single_feature_removal'][:3]
                print(f"\n{model_name} - Top 3 Most Important Features:")
                for i, feature in enumerate(top_features):
                    print(f"  {i+1}. {feature['removed_feature']}: "
                          f"F1 drop = {feature['f1_drop']:.4f} "
                          f"({feature['relative_importance']*100:.1f}% relative importance)")
        
        # Model-specific insights
        print(f"\n🔬 MODEL-SPECIFIC INSIGHTS")
        print("-" * 50)
        
        for model_name in model_names:
            if model_name in self.results:
                print(f"\n{model_name}:")
                
                # TabPFN insights
                if 'context_size_ablation' in self.results[model_name]:
                    context_results = self.results[model_name]['context_size_ablation']
                    if context_results:
                        best_context = max(context_results, key=lambda x: x['f1_score'])
                        print(f"  Optimal context size: {best_context['context_size']} "
                              f"(F1: {best_context['f1_score']:.4f})")
                
                # TabICL insights
                if 'context_examples_ablation' in self.results[model_name]:
                    examples_results = self.results[model_name]['context_examples_ablation']
                    if examples_results:
                        best_examples = max(examples_results, key=lambda x: x['f1_score'])
                        print(f"  Optimal context examples: {best_examples['context_examples']} "
                              f"(F1: {best_examples['f1_score']:.4f})")
                
                if 'example_selection_ablation' in self.results[model_name]:
                    selection_results = self.results[model_name]['example_selection_ablation']
                    if selection_results:
                        best_strategy = max(selection_results, key=lambda x: x['f1_score'])
                        print(f"  Best example selection: {best_strategy['selection_strategy']} "
                              f"(F1: {best_strategy['f1_score']:.4f})")
        
        # Robustness analysis
        if 'comparative_analysis' in self.results and 'robustness_comparison' in self.results['comparative_analysis']:
            print(f"\n🛡️ ROBUSTNESS RANKING")
            print("-" * 50)
            
            robustness_scores = self.results['comparative_analysis']['robustness_comparison']
            sorted_models = sorted(robustness_scores.items(), key=lambda x: x[1], reverse=True)
            
            for i, (model_name, score) in enumerate(sorted_models):
                print(f"   {i+1}. {model_name}: {score:.3f}")
        
        # Efficiency analysis
        if 'comparative_analysis' in self.results and 'efficiency_comparison' in self.results['comparative_analysis']:
            print(f"\n⚡ EFFICIENCY RANKING")
            print("-" * 50)
            
            efficiency_scores = self.results['comparative_analysis']['efficiency_comparison']
            
            # Sort by training time (faster is better)
            sorted_by_train_time = sorted(efficiency_scores.items(), key=lambda x: x[1]['train_time'])
            print(f"\nFastest Training:")
            for i, (model_name, metrics) in enumerate(sorted_by_train_time[:3]):
                print(f"   {i+1}. {model_name}: {metrics['train_time']:.3f}s")
            
            # Sort by inference time (faster is better)
            sorted_by_inference_time = sorted(efficiency_scores.items(), key=lambda x: x[1]['inference_time'])
            print(f"\nFastest Inference:")
            for i, (model_name, metrics) in enumerate(sorted_by_inference_time[:3]):
                print(f"   {i+1}. {model_name}: {metrics['inference_time']:.3f}s")
        
        # Final recommendations
        print(f"\n🎯 FINAL RECOMMENDATIONS")
        print("-" * 50)
        
        print(f"\n1. Best Overall Performance: [Based on F1 scores and AUC]")
        print(f"2. Most Robust Model: [Based on noise robustness analysis]")
        print(f"3. Most Efficient Model: [Based on training/inference time]")
        print(f"4. Most Feature-Dependent: [Based on feature ablation results]")
        print(f"5. Best for Production: [Balanced performance, efficiency, and robustness]")
        
        print(f"\n✅ Comprehensive ablation study analysis complete!")
        
        return self.results

# Usage example and utility functions
def run_enhanced_ablation_studies(models_dict, model_names, X_train, X_test, y_train, y_test, 
                                 feature_names=None, dataset_name="online_shoppers"):
    """
    Convenience function to run enhanced ablation studies with pickle saving
    """
    analyzer = EnhancedAblationStudyAnalyzer()
    results = analyzer.comprehensive_ablation_study(
        models_dict, model_names, X_train, X_test, y_train, y_test, 
        feature_names, dataset_name
    )
    return analyzer, results

def create_ablation_summary_dataframe(results):
    """
    Create a summary DataFrame from ablation results
    """
    summary_data = []
    
    for model_name, model_results in results.items():
        if model_name == 'comparative_analysis':
            continue
            
        if 'baseline' in model_results:
            baseline = model_results['baseline']
            
            row = {
                'Model': model_name,
                'Accuracy': baseline['accuracy'],
                'F1_Score': baseline['f1'],
                'Precision': baseline['precision'],
                'Recall': baseline['recall'],
                'AUC': baseline.get('auc', None),
                'Train_Time': baseline['train_time'],
                'Inference_Time': baseline['inference_time']
            }
            
            # Add feature importance info
            if ('feature_ablation' in model_results and 
                model_results['feature_ablation']['single_feature_removal']):
                top_feature = model_results['feature_ablation']['single_feature_removal'][0]
                row['Most_Important_Feature'] = top_feature['removed_feature']
                row['Feature_Importance_Score'] = top_feature['relative_importance']
            
            # Add robustness info
            if 'noise_robustness' in model_results:
                avg_robustness = np.mean([r['robustness_score'] for r in model_results['noise_robustness']])
                row['Robustness_Score'] = avg_robustness
            
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def plot_ablation_dashboard(analyzer, model_names):
    """
    Create a comprehensive ablation study dashboard
    """
    # Convert model_names to list if it's dict_keys or other iterable
    if not isinstance(model_names, list):
        model_names = list(model_names)
    
    fig = plt.figure(figsize=(20, 15))
    
    # Create a 3x3 grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Baseline Performance Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    f1_scores = []
    model_labels = []
    
    for model_name in model_names:
        if model_name in analyzer.results and 'baseline' in analyzer.results[model_name]:
            f1_scores.append(analyzer.results[model_name]['baseline']['f1'])
            model_labels.append(model_name)
    
    bars = ax1.bar(model_labels, f1_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(model_labels)])
    ax1.set_title('Baseline F1 Score Comparison')
    ax1.set_ylabel('F1 Score')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Training Time Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    train_times = []
    
    for model_name in model_labels:
        train_times.append(analyzer.results[model_name]['baseline']['train_time'])
    print(model_labels)
    print(train_times)
    bars = ax2.bar(model_labels, train_times, color='lightblue')
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Robustness Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if 'comparative_analysis' in analyzer.results and 'robustness_comparison' in analyzer.results['comparative_analysis']:
        robustness_data = analyzer.results['comparative_analysis']['robustness_comparison']
        models = list(robustness_data.keys())
        scores = list(robustness_data.values())
        
        bars = ax3.bar(models, scores, color='lightgreen')
        ax3.set_title('Robustness Score Comparison')
        ax3.set_ylabel('Robustness Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
    
    # 4-6. Feature Importance for each model (top 3 models)
    for i, model_name in enumerate(model_names[:3]):
        ax = fig.add_subplot(gs[1, i])
        
        if (model_name in analyzer.results and 
            'feature_ablation' in analyzer.results[model_name] and
            analyzer.results[model_name]['feature_ablation']['single_feature_removal']):
            
            top_features = analyzer.results[model_name]['feature_ablation']['single_feature_removal'][:5]
            features = [f['removed_feature'] for f in top_features]
            importance = [f['relative_importance'] for f in top_features]
            
            bars = ax.barh(features, importance, color='orange')
            ax.set_title(f'{model_name} - Top Features')
            ax.set_xlabel('Relative Importance')
    
    # 7. Data Size Ablation (if available)
    ax7 = fig.add_subplot(gs[2, 0])
    for model_name in model_names:
        if (model_name in analyzer.results and 
            'data_size_ablation' in analyzer.results[model_name]):
            
            data_results = analyzer.results[model_name]['data_size_ablation']
            fractions = [r['data_fraction'] for r in data_results]
            f1_scores = [r['f1_score'] for r in data_results]
            
            ax7.plot(fractions, f1_scores, marker='o', label=model_name)
    
    ax7.set_title('Data Size Ablation')
    ax7.set_xlabel('Data Fraction')
    ax7.set_ylabel('F1 Score')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Model-specific insights (TabPFN context size)
    ax8 = fig.add_subplot(gs[2, 1])
    for model_name in model_names:
        if (model_name in analyzer.results and 
            'context_size_ablation' in analyzer.results[model_name]):
            
            context_results = analyzer.results[model_name]['context_size_ablation']
            context_sizes = [r['context_size'] for r in context_results]
            f1_scores = [r['f1_score'] for r in context_results]
            
            ax8.plot(context_sizes, f1_scores, marker='o', label=model_name)
    
    ax8.set_title('Context Size Ablation (TabPFN)')
    ax8.set_xlabel('Context Size')
    ax8.set_ylabel('F1 Score')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary metrics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary text
    summary_text = "ABLATION STUDY SUMMARY\n\n"
    
    for model_name in model_names:
        if model_name in analyzer.results and 'baseline' in analyzer.results[model_name]:
            baseline = analyzer.results[model_name]['baseline']
            summary_text += f"{model_name}:\n"
            summary_text += f"  F1: {baseline['f1']:.3f}\n"
            summary_text += f"  Time: {baseline['train_time']:.2f}s\n\n"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Comprehensive Ablation Study Dashboard', fontsize=16, fontweight='bold')
    plt.savefig('ablation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Enhanced Ablation Study Analyzer loaded successfully!")
    print("Use run_enhanced_ablation_studies() to perform comprehensive ablation studies.")
