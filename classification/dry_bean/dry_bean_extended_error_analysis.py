"""
Extended Error Analysis for Dry Bean Dataset

This module provides comprehensive error analysis functions for the dry bean classification models,
adapted from the online shoppers extended error analysis. It includes functions for:

1. Cross-Model Error Comparison
2. Model-Specific Confidence Analysis  
3. Feature-based Error Analysis
4. Error Overlap Analysis
5. Model-Specific Error Insights
6. Comprehensive Summary and Recommendations

Usage:
    from dry_bean_extended_error_analysis import DryBeanErrorAnalyzer
    
    analyzer = DryBeanErrorAnalyzer()
    analyzer.load_data('dry_bean_section2_results.pkl')
    analyzer.run_complete_analysis()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from scipy.stats import ttest_ind
import warnings
import pickle
import os
import torch
import gc

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DryBeanErrorAnalyzer:
    """
    Comprehensive error analysis for dry bean classification models.
    """
    
    def __init__(self):
        """Initialize the error analyzer."""
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        self.misclassified_indices = {}
        self.feature_error_analysis = {}
        self.overlap_analysis = {}
        self.confidence_stats = []
        self.error_df = None
        self.model_names = []
        
        # Data variables
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
        print("📊 Dry Bean Extended Error Analysis Framework Initialized")
        print("Ready to analyze model error patterns")
    
    def load_data(self, data_path='dry_bean_section2_results.pkl'):
        """
        Load trained models and data from Section 2 results.
        
        Args:
            data_path (str): Path to the saved Section 2 results
        """
        try:
            with open(data_path, 'rb') as f:
                section2_data = pickle.load(f)
            
            # Extract variables
            self.models = section2_data['models']
            self.evaluator = section2_data['evaluator']
            self.X_train_scaled = section2_data['X_train_scaled']
            self.X_val_scaled = section2_data.get('X_val_scaled')
            self.X_test_scaled = section2_data['X_test_scaled']
            self.y_train = section2_data['y_train']
            self.y_val = section2_data.get('y_val')
            self.y_test = section2_data['y_test']
            self.feature_names = section2_data['feature_names']
            self.class_mapping = section2_data.get('class_mapping', {})
            self.class_names = section2_data.get('class_names', [])
            self.label_encoder = section2_data.get('label_encoder')
            
            # If class_names not available, create from class_mapping
            if not self.class_names and self.class_mapping:
                self.class_names = list(self.class_mapping.keys())
            elif not self.class_names:
                # Create generic class names
                unique_classes = np.unique(self.y_test)
                self.class_names = [f'Class_{i}' for i in unique_classes]
            
            self.model_names = list(self.models.keys())
            
            print("✅ Section 2 results loaded successfully!")
            print(f"Models available: {self.model_names}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Classes: {len(self.class_names)}")
            print(f"Test samples: {len(self.X_test_scaled):,}")
            
            # Display feature names for reference
            print(f"\n📋 Feature Names:")
            for i, feature in enumerate(self.feature_names[:10]):  # Show first 10
                print(f"   {i+1:2d}. {feature}")
            if len(self.feature_names) > 10:
                print(f"   ... and {len(self.feature_names) - 10} more features")
                
            return True
            
        except FileNotFoundError:
            print("❌ Section 2 results not found!")
            print("Please run Section 2 (Model Training) notebook first.")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_predictions(self):
        """
        Generate predictions and probabilities from all models.
        """
        print("🔮 Generating predictions from all models...")
        
        for model_name, model in self.models.items():
            print(f"   Processing {model_name}...")
            
            try:
                # Handle FT-Transformer specially if it's a PyTorch model
                if 'FT-Transformer' in model_name and hasattr(model, 'eval'):
                    # Create wrapper for FT-Transformer if needed
                    if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'create_pytorch_wrapper'):
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        ft_wrapper = self.evaluator.create_pytorch_wrapper(
                            model=model,
                            device=device,
                            batch_size=256
                        )
                        model = ft_wrapper
                        self.models[model_name] = ft_wrapper
                
                # Get predictions
                y_pred = model.predict(self.X_test_scaled)
                self.predictions[model_name] = y_pred
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test_scaled)
                    self.probabilities[model_name] = y_proba
                else:
                    self.probabilities[model_name] = None
                
                # Find misclassified samples
                misclassified_mask = (y_pred != self.y_test)
                self.misclassified_indices[model_name] = np.where(misclassified_mask)[0]
                
                print(f"     Misclassified: {len(self.misclassified_indices[model_name])} / {len(self.y_test)} "
                      f"({len(self.misclassified_indices[model_name])/len(self.y_test)*100:.2f}%)")
                      
            except Exception as e:
                print(f"     ❌ Error processing {model_name}: {e}")
                continue
        
        print("\n✅ Predictions generated for all models")
    
    def cross_model_error_comparison(self):
        """
        Compare error patterns across all models.
        """
        print("📊 CROSS-MODEL ERROR COMPARISON")
        print("=" * 80)
        
        error_comparison = []
        
        for model_name in self.model_names:
            if model_name not in self.predictions:
                continue
                
            y_pred = self.predictions[model_name]
            
            # Calculate detailed metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics for multi-class
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average=None, zero_division=0
            )
            
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Calculate error rates
            total_errors = len(self.misclassified_indices[model_name])
            error_rate = total_errors / len(self.y_test) * 100
            
            error_comparison.append({
                'Model': model_name,
                'Total_Errors': total_errors,
                'Error_Rate_%': error_rate,
                'Accuracy': (len(self.y_test) - total_errors) / len(self.y_test),
                'Precision_Weighted': precision,
                'Recall_Weighted': recall,
                'F1_Weighted': f1,
                'Precision_Macro': np.mean(precision_per_class),
                'Recall_Macro': np.mean(recall_per_class),
                'F1_Macro': np.mean(f1_per_class),
                'Precision_Std': np.std(precision_per_class),
                'Recall_Std': np.std(recall_per_class),
                'F1_Std': np.std(f1_per_class)
            })
        
        self.error_df = pd.DataFrame(error_comparison)
        
        print(self.error_df.round(4).to_string(index=False))
        
        # Save results
        self.error_df.to_csv('dry_bean_cross_model_error_comparison.csv', index=False)
        print("\n💾 Results saved to 'dry_bean_cross_model_error_comparison.csv'")
        
        return self.error_df
    
    def visualize_error_comparison(self):
        """
        Visualize error comparison across models.
        """
        if self.error_df is None:
            print("❌ No error comparison data available. Run cross_model_error_comparison() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Total Error Rate Comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.model_names)))
        ax1.bar(self.error_df['Model'], self.error_df['Error_Rate_%'], color=colors)
        ax1.set_title('Total Error Rate Comparison - Dry Bean Dataset')
        ax1.set_ylabel('Error Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(self.error_df['Error_Rate_%']):
            ax1.text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom')
        
        # 2. Accuracy Comparison
        ax2.bar(self.error_df['Model'], self.error_df['Accuracy'], color=colors)
        ax2.set_title('Accuracy Comparison')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(self.error_df['Accuracy']):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Weighted vs Macro F1 Comparison
        x = np.arange(len(self.model_names))
        width = 0.35
        
        ax3.bar(x - width/2, self.error_df['F1_Weighted'], width, 
                label='Weighted F1', color='orange', alpha=0.7)
        ax3.bar(x + width/2, self.error_df['F1_Macro'], width, 
                label='Macro F1', color='green', alpha=0.7)
        ax3.set_title('F1-Score Comparison (Weighted vs Macro)')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.error_df['Model'], rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Performance Variability (Standard Deviation)
        ax4.bar(x - width/2, self.error_df['F1_Std'], width, 
                label='F1 Std Dev', color='red', alpha=0.7)
        ax4.bar(x + width/2, self.error_df['Precision_Std'], width, 
                label='Precision Std Dev', color='blue', alpha=0.7)
        ax4.set_title('Performance Variability Across Classes')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.error_df['Model'], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('dry_bean_cross_model_error_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def confidence_analysis(self):
        """
        Analyze prediction confidence for each model.
        """
        print("🎯 CONFIDENCE ANALYSIS")
        print("=" * 60)
        
        models_with_proba = [name for name in self.model_names 
                           if name in self.probabilities and self.probabilities[name] is not None]
        
        if not models_with_proba:
            print("❌ No models with probability predictions available")
            return
        
        fig, axes = plt.subplots(1, len(models_with_proba), figsize=(5*len(models_with_proba), 5))
        if len(models_with_proba) == 1:
            axes = [axes]
        
        self.confidence_stats = []
        
        for i, model_name in enumerate(models_with_proba):
            y_proba = self.probabilities[model_name]
            misclassified_mask = (self.predictions[model_name] != self.y_test)
            
            # Calculate confidence as max probability
            confidence_scores = np.max(y_proba, axis=1)
            misclassified_confidence = confidence_scores[misclassified_mask]
            correct_confidence = confidence_scores[~misclassified_mask]
            
            # Plot confidence distributions
            axes[i].hist(misclassified_confidence, bins=20, alpha=0.7, label='Misclassified', 
                        color='red', density=True)
            axes[i].hist(correct_confidence, bins=20, alpha=0.7, label='Correct', 
                        color='green', density=True)
            axes[i].set_xlabel('Prediction Confidence')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{model_name}\nConfidence Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Calculate confidence statistics
            self.confidence_stats.append({
                'Model': model_name,
                'Avg_Confidence_Correct': np.mean(correct_confidence),
                'Avg_Confidence_Misclassified': np.mean(misclassified_confidence),
                'Std_Confidence_Correct': np.std(correct_confidence),
                'Std_Confidence_Misclassified': np.std(misclassified_confidence),
                'Low_Confidence_Errors_%': np.mean(misclassified_confidence < 0.6) * 100,
                'High_Confidence_Errors_%': np.mean(misclassified_confidence > 0.8) * 100
            })
            
            print(f"\n{model_name} Confidence Analysis:")
            print(f"   Average confidence (correct): {self.confidence_stats[-1]['Avg_Confidence_Correct']:.4f}")
            print(f"   Average confidence (misclassified): {self.confidence_stats[-1]['Avg_Confidence_Misclassified']:.4f}")
            print(f"   Low confidence errors (<60%): {self.confidence_stats[-1]['Low_Confidence_Errors_%']:.1f}%")
            print(f"   High confidence errors (>80%): {self.confidence_stats[-1]['High_Confidence_Errors_%']:.1f}%")
        
        plt.tight_layout()
        plt.savefig('dry_bean_model_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create confidence statistics table
        if self.confidence_stats:
            confidence_df = pd.DataFrame(self.confidence_stats)
            print("\n📊 CONFIDENCE STATISTICS")
            print(confidence_df.round(4).to_string(index=False))
            confidence_df.to_csv('dry_bean_model_confidence_statistics.csv', index=False)
            
        return self.confidence_stats
    
    def feature_based_error_analysis(self):
        """
        Analyze which features are most associated with errors in each model.
        """
        print("🔍 FEATURE-BASED ERROR ANALYSIS")
        print("=" * 60)
        
        self.feature_error_analysis = {}
        
        for model_name in self.model_names:
            if model_name not in self.predictions:
                continue
                
            print(f"\nAnalyzing {model_name}...")
            
            misclassified_mask = (self.predictions[model_name] != self.y_test)
            
            if np.sum(misclassified_mask) > 0:
                misclassified_features = self.X_test_scaled[misclassified_mask]
                correct_features = self.X_test_scaled[~misclassified_mask]
                
                # Calculate feature differences
                feature_diff = np.mean(misclassified_features, axis=0) - np.mean(correct_features, axis=0)
                feature_std_diff = np.std(misclassified_features, axis=0) - np.std(correct_features, axis=0)
                
                # Statistical significance test (t-test)
                p_values = []
                for i in range(len(self.feature_names)):
                    try:
                        _, p_val = ttest_ind(misclassified_features[:, i], correct_features[:, i])
                        p_values.append(p_val)
                    except:
                        p_values.append(1.0)  # No significant difference if test fails
                
                self.feature_error_analysis[model_name] = {
                    'feature_diff': feature_diff,
                    'feature_std_diff': feature_std_diff,
                    'p_values': np.array(p_values),
                    'significant_features': np.array(p_values) < 0.05
                }
                
                # Print top features with significant differences
                significant_indices = np.where(np.array(p_values) < 0.05)[0]
                if len(significant_indices) > 0:
                    sorted_sig_indices = significant_indices[np.argsort(np.abs(feature_diff[significant_indices]))[::-1]]
                    
                    print(f"   Top 5 significant features (p < 0.05):")
                    for i, idx in enumerate(sorted_sig_indices[:5]):
                        print(f"     {i+1}. {self.feature_names[idx]}: diff={feature_diff[idx]:.4f}, p={p_values[idx]:.4f}")
                else:
                    print(f"   No statistically significant feature differences found")
            else:
                print(f"   No misclassified samples for {model_name}")
                self.feature_error_analysis[model_name] = None
        
        return self.feature_error_analysis
    
    def visualize_feature_error_analysis(self):
        """
        Visualize feature differences across models.
        """
        n_models = len([m for m in self.model_names if self.feature_error_analysis.get(m) is not None])
        if n_models == 0:
            print("❌ No feature error analysis data available")
            return
        
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 5*n_models))
        if n_models == 1:
            axes = [axes]
        
        plot_idx = 0
        for model_name in self.model_names:
            if self.feature_error_analysis.get(model_name) is not None:
                feature_diff = self.feature_error_analysis[model_name]['feature_diff']
                p_values = self.feature_error_analysis[model_name]['p_values']
                
                # Sort features by absolute difference
                sorted_indices = np.argsort(np.abs(feature_diff))[::-1][:15]  # Top 15 features
                
                # Create bar plot
                bars = axes[plot_idx].bar(range(15), feature_diff[sorted_indices], 
                                         color=['red' if p_values[i] < 0.05 else 'lightblue' 
                                               for i in sorted_indices])
                
                axes[plot_idx].set_xlabel('Features')
                axes[plot_idx].set_ylabel('Mean Difference (Misclassified - Correct)')
                axes[plot_idx].set_title(f'{model_name}: Feature Differences in Misclassified Samples\n(Red bars: p < 0.05)')
                axes[plot_idx].set_xticks(range(15))
                axes[plot_idx].set_xticklabels([self.feature_names[i] for i in sorted_indices], 
                                              rotation=45, ha='right')
                axes[plot_idx].grid(True, alpha=0.3)
                
                # Add value labels for significant features
                for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
                    if p_values[idx] < 0.05:
                        height = bar.get_height()
                        axes[plot_idx].text(bar.get_x() + bar.get_width()/2., 
                                           height + np.sign(height)*0.001,
                                           f'{height:.3f}', ha='center', 
                                           va='bottom' if height > 0 else 'top')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('dry_bean_feature_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def error_overlap_analysis(self):
        """
        Analyze which samples are commonly misclassified across models.
        """
        print("🔄 ERROR OVERLAP ANALYSIS")
        print("=" * 60)
        
        # Create error matrix (samples x models)
        valid_models = [name for name in self.model_names if name in self.predictions]
        error_matrix = np.zeros((len(self.y_test), len(valid_models)), dtype=bool)
        
        for i, model_name in enumerate(valid_models):
            misclassified_mask = (self.predictions[model_name] != self.y_test)
            error_matrix[:, i] = misclassified_mask
        
        # Analyze overlap patterns
        self.overlap_analysis = {}
        
        # Common errors (misclassified by all models)
        if len(valid_models) > 1:
            common_errors = np.all(error_matrix, axis=1)
            self.overlap_analysis['common_errors'] = np.sum(common_errors)
        else:
            self.overlap_analysis['common_errors'] = 0
        
        # Model-specific errors
        for i, model_name in enumerate(valid_models):
            if len(valid_models) > 1:
                # Errors unique to this model
                other_models_correct = ~np.any(error_matrix[:, [j for j in range(len(valid_models)) if j != i]], axis=1)
                unique_errors = error_matrix[:, i] & other_models_correct
                self.overlap_analysis[f'{model_name}_unique'] = np.sum(unique_errors)
            else:
                self.overlap_analysis[f'{model_name}_unique'] = np.sum(error_matrix[:, i])
        
        # Pairwise overlaps
        pairwise_overlaps = {}
        for i in range(len(valid_models)):
            for j in range(i+1, len(valid_models)):
                overlap = np.sum(error_matrix[:, i] & error_matrix[:, j])
                pairwise_overlaps[f'{valid_models[i]}_vs_{valid_models[j]}'] = overlap
        
        print(f"\nError Overlap Statistics:")
        print(f"   Common errors (all models): {self.overlap_analysis['common_errors']}")
        for model_name in valid_models:
            print(f"   {model_name} unique errors: {self.overlap_analysis[f'{model_name}_unique']}")
        
        if pairwise_overlaps:
            print(f"\nPairwise Error Overlaps:")
            for pair, overlap in pairwise_overlaps.items():
                print(f"   {pair}: {overlap}")
        
        return self.overlap_analysis, pairwise_overlaps
    
    def visualize_error_overlap(self):
        """
        Visualize error overlap patterns.
        """
        if not self.overlap_analysis:
            print("❌ No overlap analysis data available")
            return
        
        valid_models = [name for name in self.model_names if name in self.predictions]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Error overlap heatmap
        if len(valid_models) > 1:
            error_matrix = np.zeros((len(self.y_test), len(valid_models)), dtype=bool)
            for i, model_name in enumerate(valid_models):
                misclassified_mask = (self.predictions[model_name] != self.y_test)
                error_matrix[:, i] = misclassified_mask
            
            overlap_matrix = np.zeros((len(valid_models), len(valid_models)))
            for i in range(len(valid_models)):
                for j in range(len(valid_models)):
                    if i == j:
                        overlap_matrix[i, j] = np.sum(error_matrix[:, i])  # Total errors for diagonal
                    else:
                        overlap_matrix[i, j] = np.sum(error_matrix[:, i] & error_matrix[:, j])  # Overlap
            
            sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='Reds', 
                        xticklabels=valid_models, yticklabels=valid_models, ax=ax1)
            ax1.set_title('Error Overlap Matrix\n(Diagonal: Total Errors, Off-diagonal: Overlaps)')
        else:
            ax1.text(0.5, 0.5, 'Need multiple models\nfor overlap analysis', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Error Overlap Matrix')
        
        # 2. Unique vs shared errors
        unique_errors = [self.overlap_analysis[f'{model}_unique'] for model in valid_models]
        total_errors = [len(self.misclassified_indices[model]) for model in valid_models]
        shared_errors = [total - unique for total, unique in zip(total_errors, unique_errors)]
        
        x = np.arange(len(valid_models))
        width = 0.35
        
        ax2.bar(x, unique_errors, width, label='Unique Errors', color='red', alpha=0.7)
        ax2.bar(x, shared_errors, width, bottom=unique_errors, label='Shared Errors', color='blue', alpha=0.7)
        ax2.set_title('Unique vs Shared Errors by Model')
        ax2.set_ylabel('Number of Errors')
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_models, rotation=45)
        ax2.legend()
        
        # 3. Error rates comparison
        error_rates = [len(self.misclassified_indices[model])/len(self.y_test)*100 for model in valid_models]
        ax3.bar(valid_models, error_rates, color='lightcoral')
        ax3.set_title('Error Rate Comparison')
        ax3.set_ylabel('Error Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(error_rates):
            ax3.text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom')
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""ERROR OVERLAP SUMMARY

Total Test Samples: {len(self.y_test):,}
Common Errors (All Models): {self.overlap_analysis['common_errors']}

Model-Specific Errors:
"""
        
        for model_name in valid_models:
            unique_count = self.overlap_analysis[f'{model_name}_unique']
            total_count = len(self.misclassified_indices[model_name])
            summary_text += f"  {model_name}: {unique_count}/{total_count} unique\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('dry_bean_error_overlap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def model_specific_insights(self):
        """
        Generate detailed insights for each model's error patterns.
        """
        print("🔍 MODEL-SPECIFIC ERROR INSIGHTS")
        print("=" * 80)
        
        insights = {}
        
        for model_name in self.model_names:
            if model_name not in self.predictions:
                continue
                
            print(f"\n📊 {model_name.upper()} ERROR ANALYSIS")
            print("-" * 60)
            
            # Basic error statistics
            total_errors = len(self.misclassified_indices[model_name])
            error_rate = total_errors / len(self.y_test) * 100
            unique_errors = self.overlap_analysis.get(f'{model_name}_unique', 0)
            
            print(f"Total Errors: {total_errors} ({error_rate:.2f}%)")
            if total_errors > 0:
                print(f"Unique Errors: {unique_errors} ({unique_errors/total_errors*100:.1f}% of model errors)")
            
            # Confidence analysis (if available)
            if model_name in [cs['Model'] for cs in self.confidence_stats]:
                conf_stats = next((cs for cs in self.confidence_stats if cs['Model'] == model_name), None)
                if conf_stats:
                    print(f"Average Confidence (Correct): {conf_stats['Avg_Confidence_Correct']:.3f}")
                    print(f"Average Confidence (Errors): {conf_stats['Avg_Confidence_Misclassified']:.3f}")
                    print(f"Low Confidence Errors: {conf_stats['Low_Confidence_Errors_%']:.1f}%")
                    print(f"High Confidence Errors: {conf_stats['High_Confidence_Errors_%']:.1f}%")
            
            # Feature analysis
            if self.feature_error_analysis.get(model_name) is not None:
                feature_data = self.feature_error_analysis[model_name]
                significant_features = np.sum(feature_data['significant_features'])
                print(f"Significant Feature Differences: {significant_features}/{len(self.feature_names)}")
                
                # Top problematic features
                if significant_features > 0:
                    sig_indices = np.where(feature_data['significant_features'])[0]
                    top_sig = sig_indices[np.argsort(np.abs(feature_data['feature_diff'][sig_indices]))[::-1]][:3]
                    print("Top 3 Problematic Features:")
                    for i, idx in enumerate(top_sig):
                        print(f"  {i+1}. {self.feature_names[idx]}: {feature_data['feature_diff'][idx]:.4f}")
            
            # Model-specific insights based on model type
            model_insights = self._get_model_specific_insights(model_name, total_errors, error_rate)
            insights[model_name] = model_insights
            
            for insight in model_insights:
                print(f"  {insight}")
        
        return insights
    
    def _get_model_specific_insights(self, model_name, total_errors, error_rate):
        """
        Get model-specific insights based on model type and performance.
        """
        insights = []
        
        if 'TabPFN' in model_name:
            insights.extend([
                "🧠 TabPFN-Specific Insights:",
                "- Prior-based predictions may struggle with out-of-distribution samples",
                "- Consider context size optimization for better performance",
                "- Errors may indicate dataset shift from pretraining distribution",
                "- TabPFN works best with datasets similar to its training distribution"
            ])
        
        elif 'TabICL' in model_name:
            insights.extend([
                "🎯 TabICL-Specific Insights:",
                "- In-context learning errors may indicate poor example selection",
                "- Consider diverse example selection strategies",
                "- Context window utilization may need optimization",
                "- Performance depends heavily on the quality of in-context examples"
            ])
        
        elif 'XGBoost' in model_name:
            insights.extend([
                "🌳 XGBoost-Specific Insights:",
                "- Tree-based errors often indicate feature interaction issues",
                "- Consider feature engineering or hyperparameter tuning",
                "- May benefit from ensemble methods",
                "- Check for overfitting if validation performance differs significantly"
            ])
        
        elif 'FT-Transformer' in model_name or 'Transformer' in model_name:
            insights.extend([
                "🤖 FT-Transformer-Specific Insights:",
                "- Attention-based errors may indicate complex feature interactions",
                "- Consider adjusting model depth or attention heads",
                "- May need more training data or regularization",
                "- Feature tokenization strategy could be optimized"
            ])
        
        # Performance-based insights
        if error_rate < 5:
            insights.append("✅ Excellent performance - focus on edge cases")
        elif error_rate < 10:
            insights.append("✅ Good performance - minor optimizations needed")
        elif error_rate < 20:
            insights.append("⚠️ Moderate performance - significant improvements possible")
        else:
            insights.append("❌ Poor performance - major changes needed")
        
        return insights
    
    def comprehensive_summary(self):
        """
        Generate comprehensive summary and recommendations.
        """
        print("📋 COMPREHENSIVE ERROR ANALYSIS SUMMARY")
        print("=" * 80)
        
        if not self.predictions:
            print("❌ No predictions available for summary")
            return
        
        # Overall performance ranking
        valid_models = [name for name in self.model_names if name in self.predictions]
        error_rates = [(model, len(self.misclassified_indices[model])/len(self.y_test)*100) 
                      for model in valid_models]
        error_rates.sort(key=lambda x: x[1])
        
        print("\n🏆 PERFORMANCE RANKING (by error rate):")
        for i, (model, rate) in enumerate(error_rates):
            print(f"   {i+1}. {model}: {rate:.2f}% error rate")
        
        # Key findings
        print("\n🔍 KEY FINDINGS:")
        if error_rates:
            print(f"   • Best performing model: {error_rates[0][0]} ({error_rates[0][1]:.2f}% error rate)")
            print(f"   • Worst performing model: {error_rates[-1][0]} ({error_rates[-1][1]:.2f}% error rate)")
        
        if self.overlap_analysis:
            print(f"   • Common errors across all models: {self.overlap_analysis['common_errors']} samples")
            if valid_models:
                most_unique = max(valid_models, key=lambda m: self.overlap_analysis.get(f'{m}_unique', 0))
                print(f"   • Most unique errors: {most_unique}")
        
        if self.confidence_stats:
            most_confident = max(self.confidence_stats, key=lambda x: x['Avg_Confidence_Correct'])
            print(f"   • Most confident model: {most_confident['Model']} ({most_confident['Avg_Confidence_Correct']:.3f} avg confidence)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        print("\n1. Model Selection:")
        if error_rates:
            print(f"   • For production: {error_rates[0][0]} (lowest error rate)")
            if self.confidence_stats:
                most_reliable = min(self.confidence_stats, 
                                  key=lambda x: abs(x['Avg_Confidence_Correct'] - x['Avg_Confidence_Misclassified']))
                print(f"   • For reliability: {most_reliable['Model']} (most consistent confidence)")
        
        print("\n2. Error Reduction Strategies:")
        for model_name in valid_models:
            unique_count = self.overlap_analysis.get(f'{model_name}_unique', 0)
            if unique_count > 0:
                print(f"   • {model_name}: Focus on {unique_count} unique error cases")
                
                if self.feature_error_analysis.get(model_name) is not None:
                    sig_features = np.sum(self.feature_error_analysis[model_name]['significant_features'])
                    if sig_features > 0:
                        print(f"     - Address {sig_features} problematic features")
        
        print("\n3. Ensemble Opportunities:")
        if len(valid_models) > 1 and self.overlap_analysis:
            print(f"   • Models show complementary errors (only {self.overlap_analysis['common_errors']} common errors)")
            print("   • Consider ensemble methods to leverage model diversity")
            print("   • Weighted voting based on confidence scores could improve performance")
        
        print("\n4. Data Quality Improvements:")
        if self.overlap_analysis and self.overlap_analysis['common_errors'] > 0:
            print(f"   • Investigate {self.overlap_analysis['common_errors']} samples misclassified by all models")
            print("   • These may indicate data quality issues or inherently difficult cases")
            print("   • Consider data cleaning or feature engineering for these samples")
        
        print("\n5. Model-Specific Improvements:")
        for model_name in valid_models:
            if 'TabPFN' in model_name:
                print(f"   • {model_name}: Optimize context size and check for distribution shift")
            elif 'TabICL' in model_name:
                print(f"   • {model_name}: Improve example selection and context utilization")
            elif 'XGBoost' in model_name:
                print(f"   • {model_name}: Feature engineering and hyperparameter optimization")
            elif 'FT-Transformer' in model_name:
                print(f"   • {model_name}: Adjust architecture or training strategy")
        
        print("\n✅ Error analysis complete! Use these insights to improve model performance.")
    
    def save_analysis_results(self, output_dir='dry_bean_error_analysis_results'):
        """
        Save all analysis results to files.
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving analysis results to '{output_dir}/'...")
        
        # Save error comparison
        if self.error_df is not None:
            self.error_df.to_csv(f'{output_dir}/cross_model_error_comparison.csv', index=False)
        
        # Save confidence statistics
        if self.confidence_stats:
            confidence_df = pd.DataFrame(self.confidence_stats)
            confidence_df.to_csv(f'{output_dir}/confidence_statistics.csv', index=False)
        
        # Save feature error analysis
        for model_name, analysis in self.feature_error_analysis.items():
            if analysis is not None:
                feature_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Mean_Difference': analysis['feature_diff'],
                    'Std_Difference': analysis['feature_std_diff'],
                    'P_Value': analysis['p_values'],
                    'Significant': analysis['significant_features']
                })
                feature_df.to_csv(f'{output_dir}/{model_name}_feature_error_analysis.csv', index=False)
        
        # Save overlap analysis
        if self.overlap_analysis:
            overlap_df = pd.DataFrame([self.overlap_analysis])
            overlap_df.to_csv(f'{output_dir}/error_overlap_analysis.csv', index=False)
        
        print(f"✅ Analysis results saved to '{output_dir}/'")
    
    def run_complete_analysis(self, save_results=True):
        """
        Run the complete error analysis pipeline.
        """
        print("🚀 STARTING COMPLETE DRY BEAN ERROR ANALYSIS")
        print("=" * 80)
        
        # Step 1: Generate predictions
        self.generate_predictions()
        
        # Step 2: Cross-model error comparison
        self.cross_model_error_comparison()
        self.visualize_error_comparison()
        
        # Step 3: Confidence analysis
        self.confidence_analysis()
        
        # Step 4: Feature-based error analysis
        self.feature_based_error_analysis()
        self.visualize_feature_error_analysis()
        
        # Step 5: Error overlap analysis
        self.error_overlap_analysis()
        self.visualize_error_overlap()
        
        # Step 6: Model-specific insights
        self.model_specific_insights()
        
        # Step 7: Comprehensive summary
        self.comprehensive_summary()
        
        # Step 8: Save results
        if save_results:
            self.save_analysis_results()
        
        print("\n🎉 COMPLETE ERROR ANALYSIS FINISHED!")
        print("Check the generated plots and CSV files for detailed results.")


# Convenience functions for direct usage
def run_dry_bean_error_analysis(data_path='dry_bean_section2_results.pkl'):
    """
    Convenience function to run complete error analysis.
    
    Args:
        data_path (str): Path to the Section 2 results file
    
    Returns:
        DryBeanErrorAnalyzer: The analyzer instance with results
    """
    analyzer = DryBeanErrorAnalyzer()
    
    if analyzer.load_data(data_path):
        analyzer.run_complete_analysis()
        return analyzer
    else:
        print("❌ Failed to load data. Please check the file path.")
        return None


def quick_error_comparison(data_path='dry_bean_section2_results.pkl'):
    """
    Quick function to just run cross-model error comparison.
    
    Args:
        data_path (str): Path to the Section 2 results file
    
    Returns:
        pd.DataFrame: Error comparison results
    """
    analyzer = DryBeanErrorAnalyzer()
    
    if analyzer.load_data(data_path):
        analyzer.generate_predictions()
        return analyzer.cross_model_error_comparison()
    else:
        print("❌ Failed to load data. Please check the file path.")
        return None


if __name__ == "__main__":
    # Example usage
    print("🔬 Dry Bean Extended Error Analysis")
    print("Usage examples:")
    print("1. Complete analysis: run_dry_bean_error_analysis()")
    print("2. Quick comparison: quick_error_comparison()")
    print("3. Custom analysis:")
    print("   analyzer = DryBeanErrorAnalyzer()")
    print("   analyzer.load_data('dry_bean_section2_results.pkl')")
    print("   analyzer.run_complete_analysis()")
