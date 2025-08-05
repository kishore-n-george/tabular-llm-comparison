import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

class AblationStudyAnalyzer:
    def __init__(self):
        self.results = {}
        self.feature_names = None
    
    def feature_ablation_study(self, model, model_name, X_train, X_test, y_train, y_test, 
                             feature_names=None, max_features_to_remove=5):
        """
        Perform feature ablation study by systematically removing features
        and measuring performance impact
        """
        print(f"\n🔬 Feature Ablation Study for {model_name}")
        print("=" * 50)
        
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Baseline performance with all features
        model.fit(X_train, y_train)
        baseline_pred = model.predict(X_test)
        baseline_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        baseline_metrics = {
            'accuracy': accuracy_score(y_test, baseline_pred),
            'f1': f1_score(y_test, baseline_pred),
            'auc': roc_auc_score(y_test, baseline_proba) if baseline_proba is not None else None
        }
        
        print(f"📊 Baseline Performance (All Features):")
        print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"   F1-Score: {baseline_metrics['f1']:.4f}")
        if baseline_metrics['auc']:
            print(f"   AUC: {baseline_metrics['auc']:.4f}")
        
        # Single feature removal study
        single_removal_results = self._single_feature_removal(
            model, X_train, X_test, y_train, y_test, baseline_metrics
        )
        
        # Multiple feature removal study
        multiple_removal_results = self._multiple_feature_removal(
            model, X_train, X_test, y_train, y_test, baseline_metrics, max_features_to_remove
        )
        
        # Feature group ablation (if applicable)
        group_removal_results = self._feature_group_removal(
            model, X_train, X_test, y_train, y_test, baseline_metrics
        )
        
        # Store results
        self.results[f'{model_name}_ablation'] = {
            'baseline': baseline_metrics,
            'single_removal': single_removal_results,
            'multiple_removal': multiple_removal_results,
            'group_removal': group_removal_results
        }
        
        # Generate visualizations
        self._plot_ablation_results(model_name, single_removal_results, multiple_removal_results)
        
        return self.results[f'{model_name}_ablation']
    
    def _single_feature_removal(self, model, X_train, X_test, y_train, y_test, baseline_metrics):
        """Remove one feature at a time and measure performance impact"""
        print(f"\n🔍 Single Feature Removal Analysis...")
        
        results = []
        n_features = X_train.shape[1]
        
        for i in range(n_features):
            # Create dataset without feature i
            feature_mask = np.ones(n_features, dtype=bool)
            feature_mask[i] = False
            
            X_train_reduced = X_train[:, feature_mask]
            X_test_reduced = X_test[:, feature_mask]
            
            # Train and evaluate
            model.fit(X_train_reduced, y_train)
            pred = model.predict(X_test_reduced)
            proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'removed_feature': self.feature_names[i],
                'removed_feature_idx': i,
                'accuracy': accuracy_score(y_test, pred),
                'f1': f1_score(y_test, pred),
                'auc': roc_auc_score(y_test, proba) if proba is not None else None,
                'accuracy_drop': baseline_metrics['accuracy'] - accuracy_score(y_test, pred),
                'f1_drop': baseline_metrics['f1'] - f1_score(y_test, pred),
                'auc_drop': (baseline_metrics['auc'] - roc_auc_score(y_test, proba)) if baseline_metrics['auc'] and proba is not None else None
            }
            
            results.append(metrics)
        
        # Sort by performance drop (most important features first)
        results.sort(key=lambda x: x['f1_drop'], reverse=True)
        
        print(f"\n🎯 Top 5 Most Important Features (by F1 drop when removed):")
        for i, result in enumerate(results[:5]):
            print(f"   {i+1}. {result['removed_feature']}: F1 drop = {result['f1_drop']:.4f}")
        
        return results
    
    def _multiple_feature_removal(self, model, X_train, X_test, y_train, y_test, 
                                baseline_metrics, max_features_to_remove):
        """Remove multiple features simultaneously"""
        print(f"\n🔍 Multiple Feature Removal Analysis...")
        
        results = []
        n_features = X_train.shape[1]
        
        # Try removing 2 to max_features_to_remove features
        for n_remove in range(2, min(max_features_to_remove + 1, n_features)):
            print(f"   Testing removal of {n_remove} features...")
            
            # Try different combinations (sample to avoid combinatorial explosion)
            max_combinations = 50  # Limit combinations for computational efficiency
            feature_combinations = list(itertools.combinations(range(n_features), n_remove))
            
            if len(feature_combinations) > max_combinations:
                feature_combinations = np.random.choice(
                    len(feature_combinations), max_combinations, replace=False
                )
                feature_combinations = [list(itertools.combinations(range(n_features), n_remove))[i] 
                                      for i in feature_combinations]
            
            for removed_features in feature_combinations:
                # Create dataset without selected features
                feature_mask = np.ones(n_features, dtype=bool)
                feature_mask[list(removed_features)] = False
                
                X_train_reduced = X_train[:, feature_mask]
                X_test_reduced = X_test[:, feature_mask]
                
                # Train and evaluate
                model.fit(X_train_reduced, y_train)
                pred = model.predict(X_test_reduced)
                proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, 'predict_proba') else None
                
                metrics = {
                    'removed_features': [self.feature_names[i] for i in removed_features],
                    'removed_feature_indices': removed_features,
                    'n_removed': n_remove,
                    'accuracy': accuracy_score(y_test, pred),
                    'f1': f1_score(y_test, pred),
                    'auc': roc_auc_score(y_test, proba) if proba is not None else None,
                    'accuracy_drop': baseline_metrics['accuracy'] - accuracy_score(y_test, pred),
                    'f1_drop': baseline_metrics['f1'] - f1_score(y_test, pred),
                    'auc_drop': (baseline_metrics['auc'] - roc_auc_score(y_test, proba)) if baseline_metrics['auc'] and proba is not None else None
                }
                
                results.append(metrics)
        
        return results
    
    def _feature_group_removal(self, model, X_train, X_test, y_train, y_test, baseline_metrics):
        """Remove logical groups of features"""
        print(f"\n🔍 Feature Group Removal Analysis...")
        
        # Define feature groups for online shoppers dataset
        feature_groups = {
            'Administrative': [0, 1],  # Administrative, Administrative_Duration
            'Informational': [2, 3],   # Informational, Informational_Duration
            'ProductRelated': [4, 5],  # ProductRelated, ProductRelated_Duration
            'Behavior_Metrics': [6, 7, 8],  # BounceRates, ExitRates, PageValues
            'Temporal': [9, 10],       # SpecialDay, Month
            'Technical': [11, 12, 13, 14],  # OperatingSystems, Browser, Region, TrafficType
            'User_Type': [15, 16]      # VisitorType, Weekend
        }
        
        results = []
        n_features = X_train.shape[1]
        
        for group_name, feature_indices in feature_groups.items():
            # Skip if indices are out of range
            if max(feature_indices) >= n_features:
                continue
                
            # Create dataset without feature group
            feature_mask = np.ones(n_features, dtype=bool)
            feature_mask[feature_indices] = False
            
            X_train_reduced = X_train[:, feature_mask]
            X_test_reduced = X_test[:, feature_mask]
            
            # Train and evaluate
            model.fit(X_train_reduced, y_train)
            pred = model.predict(X_test_reduced)
            proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'group_name': group_name,
                'removed_features': [self.feature_names[i] for i in feature_indices if i < len(self.feature_names)],
                'accuracy': accuracy_score(y_test, pred),
                'f1': f1_score(y_test, pred),
                'auc': roc_auc_score(y_test, proba) if proba is not None else None,
                'accuracy_drop': baseline_metrics['accuracy'] - accuracy_score(y_test, pred),
                'f1_drop': baseline_metrics['f1'] - f1_score(y_test, pred),
                'auc_drop': (baseline_metrics['auc'] - roc_auc_score(y_test, proba)) if baseline_metrics['auc'] and proba is not None else None
            }
            
            results.append(metrics)
        
        # Sort by performance drop
        results.sort(key=lambda x: x['f1_drop'], reverse=True)
        
        print(f"\n🎯 Feature Group Importance (by F1 drop when removed):")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result['group_name']}: F1 drop = {result['f1_drop']:.4f}")
        
        return results
    
    def _plot_ablation_results(self, model_name, single_results, multiple_results):
        """Create visualizations for ablation study results"""
        
        # Plot 1: Single feature removal impact
        plt.figure(figsize=(12, 6))
        
        features = [r['removed_feature'] for r in single_results[:10]]  # Top 10
        f1_drops = [r['f1_drop'] for r in single_results[:10]]
        
        plt.bar(range(len(features)), f1_drops, color='skyblue')
        plt.title(f'{model_name} - Single Feature Removal Impact (Top 10)')
        plt.xlabel('Removed Feature')
        plt.ylabel('F1 Score Drop')
        plt.xticks(range(len(features)), features, rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{model_name}_single_feature_removal.png')
        
        # Plot 2: Multiple feature removal analysis
        if multiple_results:
            plt.figure(figsize=(10, 6))
            
            # Group by number of features removed
            n_removed_groups = {}
            for result in multiple_results:
                n = result['n_removed']
                if n not in n_removed_groups:
                    n_removed_groups[n] = []
                n_removed_groups[n].append(result['f1_drop'])
            
            # Plot distribution of performance drops
            for n, drops in n_removed_groups.items():
                plt.scatter([n] * len(drops), drops, alpha=0.6, label=f'{n} features')
            
            plt.title(f'{model_name} - Multiple Feature Removal Impact')
            plt.xlabel('Number of Features Removed')
            plt.ylabel('F1 Score Drop')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'{model_name}_multiple_feature_removal.png')
    
    def hyperparameter_ablation_study(self, model_class, model_name, X_train, X_test, 
                                    y_train, y_test, param_grid, cv_folds=3):
        """
        Perform hyperparameter ablation study to understand parameter importance
        """
        print(f"\n⚙️  Hyperparameter Ablation Study for {model_name}")
        print("=" * 50)
        
        # Baseline with default parameters
        baseline_model = model_class()
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_f1 = f1_score(y_test, baseline_pred)
        
        print(f"📊 Baseline F1 Score (default params): {baseline_f1:.4f}")
        
        # Grid search to find best parameters
        print(f"🔍 Searching hyperparameter space...")
        grid_search = GridSearchCV(
            model_class(), param_grid, cv=cv_folds, 
            scoring='f1', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_pred = grid_search.predict(X_test)
        best_f1 = f1_score(y_test, best_pred)
        
        print(f"🎯 Best F1 Score: {best_f1:.4f}")
        print(f"📈 Improvement: {best_f1 - baseline_f1:.4f}")
        print(f"🔧 Best Parameters: {grid_search.best_params_}")
        
        # Analyze parameter importance
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Plot parameter impact
        self._plot_hyperparameter_impact(model_name, results_df, param_grid)
        
        return {
            'baseline_f1': baseline_f1,
            'best_f1': best_f1,
            'improvement': best_f1 - baseline_f1,
            'best_params': grid_search.best_params_,
            'cv_results': results_df
        }
    
    def _plot_hyperparameter_impact(self, model_name, results_df, param_grid):
        """Plot hyperparameter impact analysis"""
        
        # For each parameter, show its impact on performance
        param_names = list(param_grid.keys())
        n_params = len(param_names)
        
        if n_params > 4:
            n_params = 4  # Limit to 4 most important parameters
            param_names = param_names[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, param_name in enumerate(param_names):
            if i >= 4:
                break
                
            # Group by parameter value and calculate mean score
            param_col = f'param_{param_name}'
            if param_col in results_df.columns:
                grouped = results_df.groupby(param_col)['mean_test_score'].agg(['mean', 'std'])
                
                x_values = list(grouped.index)
                y_values = grouped['mean'].values
                y_errors = grouped['std'].values
                
                axes[i].bar(range(len(x_values)), y_values, yerr=y_errors, capsize=5)
                axes[i].set_title(f'Impact of {param_name}')
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('CV F1 Score')
                axes[i].set_xticks(range(len(x_values)))
                axes[i].set_xticklabels([str(x) for x in x_values], rotation=45)
        
        # Hide unused subplots
        for i in range(n_params, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{model_name} - Hyperparameter Impact Analysis')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{model_name}_hyperparameter_impact.png')
    
    def data_size_ablation_study(self, model, model_name, X_train, X_test, y_train, y_test,
                                size_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]):
        """
        Study the impact of training data size on model performance
        """
        print(f"\n📊 Data Size Ablation Study for {model_name}")
        print("=" * 50)
        
        results = []
        
        for fraction in size_fractions:
            # Sample training data
            n_samples = int(len(X_train) * fraction)
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
            
            # Train and evaluate
            start_time = time.time()
            model.fit(X_train_sample, y_train_sample)
            train_time = time.time() - start_time
            
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'data_fraction': fraction,
                'n_samples': n_samples,
                'accuracy': accuracy_score(y_test, pred),
                'f1': f1_score(y_test, pred),
                'auc': roc_auc_score(y_test, proba) if proba is not None else None,
                'train_time': train_time
            }
            
            results.append(metrics)
            print(f"   {fraction*100:3.0f}% data: F1 = {metrics['f1']:.4f}, Time = {train_time:.2f}s")
        
        # Plot learning curve
        self._plot_learning_curve(model_name, results)
        
        return results
    
    def _plot_learning_curve(self, model_name, results):
        """Plot learning curve showing performance vs data size"""
        
        fractions = [r['data_fraction'] for r in results]
        f1_scores = [r['f1'] for r in results]
        train_times = [r['train_time'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance vs data size
        ax1.plot(fractions, f1_scores, 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Training Data Fraction')
        ax1.set_ylabel('F1 Score')
        ax1.set_title(f'{model_name} - Learning Curve')
        ax1.grid(True, alpha=0.3)
        
        # Training time vs data size
        ax2.plot(fractions, train_times, 'o-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Training Data Fraction')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title(f'{model_name} - Training Time vs Data Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{model_name}_learning_curve.png')
    
    def generate_ablation_report(self, model_name):
        """Generate comprehensive ablation study report"""
        ablation_key = f'{model_name}_ablation'
        if ablation_key not in self.results:
            print(f"No ablation results available for {model_name}")
            return
        
        results = self.results[ablation_key]
        
        print(f"\n📋 Ablation Study Report for {model_name}")
        print("=" * 60)
        
        # Baseline performance
        baseline = results['baseline']
        print(f"\n📊 Baseline Performance:")
        print(f"   Accuracy: {baseline['accuracy']:.4f}")
        print(f"   F1-Score: {baseline['f1']:.4f}")
        if baseline['auc']:
            print(f"   AUC: {baseline['auc']:.4f}")
        
        # Most important individual features
        if results['single_removal']:
            print(f"\n🎯 Most Critical Individual Features:")
            for i, result in enumerate(results['single_removal'][:3]):
                print(f"   {i+1}. {result['removed_feature']}: F1 drop = {result['f1_drop']:.4f}")
        
        # Most important feature groups
        if results['group_removal']:
            print(f"\n🎯 Most Critical Feature Groups:")
            for i, result in enumerate(results['group_removal'][:3]):
                print(f"   {i+1}. {result['group_name']}: F1 drop = {result['f1_drop']:.4f}")
        
        print(f"\n✅ Ablation study analysis complete for {model_name}")

# Usage example
def run_ablation_studies():
    """Example of how to use the ablation study analyzer"""
    
    analyzer = AblationStudyAnalyzer()
    
    # Example usage:
    # analyzer.feature_ablation_study(xgb_model, "XGBoost", X_train, X_test, y_train, y_test, feature_names)
    # analyzer.hyperparameter_ablation_study(xgb.XGBClassifier, "XGBoost", X_train, X_test, y_train, y_test, param_grid)
    # analyzer.data_size_ablation_study(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)
    
    print("Ablation study framework ready!")
    return analyzer
