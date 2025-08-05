import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, 
    average_precision_score, confusion_matrix, classification_report,
    matthews_corrcoef, balanced_accuracy_score, log_loss,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import time

# PyTorch imports for deep learning model support
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning model evaluation will be limited.")
    
class PyTorchModelWrapper:
    """Wrapper to make PyTorch models compatible with sklearn-like interface"""
    
    def __init__(self, model=None, device='cpu', batch_size=256):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.is_fitted = True  # Assume model is already trained
        
    def fit(self, X, y):
        """Dummy fit method - assumes model is already trained"""
        return self
        
    def predict(self, X):
        """Make predictions using PyTorch model"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.model is None:
            raise ValueError("Model is None. This wrapper was not properly initialized.")
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                
                # Handle different model types (rtdl FTTransformer expects (numerical, categorical))
                try:
                    output = self.model(batch_X, None)
                except:
                    # Fallback for standard PyTorch models
                    output = self.model(batch_X)
                
                batch_predictions = output.argmax(dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
                
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities using PyTorch model"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.model is None:
            raise ValueError("Model is None. This wrapper was not properly initialized.")
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        probabilities = []
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                
                # Handle different model types
                try:
                    output = self.model(batch_X, None)
                except:
                    output = self.model(batch_X)
                
                # Apply softmax to get probabilities
                batch_proba = F.softmax(output, dim=1).cpu().numpy()
                probabilities.extend(batch_proba)
                
        return np.array(probabilities)
    
    def score(self, X, y):
        """Calculate accuracy score (sklearn-compatible interface)"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn-compatible)"""
        return {
            'model': self.model,
            'device': self.device,
            'batch_size': self.batch_size
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (sklearn-compatible)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def __sklearn_clone__(self):
        """Custom clone method to handle PyTorch models properly"""
        # Return a new instance with the same model reference
        # Note: This shares the model reference, which is appropriate for evaluation
        return PyTorchModelWrapper(
            model=self.model,
            device=self.device,
            batch_size=self.batch_size
        )


class ComprehensiveEvaluator:
    def __init__(self):
        self.results = {}
    
    def create_pytorch_wrapper(self, model, device='cpu', batch_size=256):
        """Create a PyTorch model wrapper for evaluation"""
        return PyTorchModelWrapper(model, device, batch_size)
        
    def evaluate_model(self, model, model_name, X_train, X_test, y_train, y_test, 
                      X_val=None, y_val=None, predict_proba_available=True):
        """Comprehensive model evaluation with timing and multiple metrics"""
        
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Training time
        start_time = time.time()
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities if available
        y_proba = None
        if predict_proba_available and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except:
                predict_proba_available = False
        
        # Basic metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'train_time': train_time,
            'inference_time': inference_time,
            'predictions_per_second': len(y_test) / inference_time if inference_time > 0 else np.inf
        }
        
        # Probability-based metrics
        if predict_proba_available and y_proba is not None:
            metrics.update({
                'auc_roc': roc_auc_score(y_test, y_proba),
                'avg_precision': average_precision_score(y_test, y_proba),
                'log_loss': log_loss(y_test, y_proba),
                'brier_score': brier_score_loss(y_test, y_proba)
            })
        
        # Cross-validation scores (skip for PyTorch models as they can't be properly cloned)
        if hasattr(model, 'fit') and not isinstance(model, PyTorchModelWrapper):
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                metrics.update({
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std()
                })
            except Exception as e:
                print(f"⚠️ Cross-validation failed: {str(e)}")
                print("Skipping cross-validation for this model.")
        elif isinstance(model, PyTorchModelWrapper):
            print("ℹ️ Skipping cross-validation for PyTorch model (already trained)")
        
        self.results[model_name] = metrics
        
        # Print results
        self._print_metrics(metrics, predict_proba_available)
        
        # Generate plots
        self._plot_confusion_matrix(y_test, y_pred, model_name)
        
        if predict_proba_available and y_proba is not None:
            self._plot_roc_curve(y_test, y_proba, model_name)
            self._plot_precision_recall_curve(y_test, y_proba, model_name)
            self._plot_calibration_curve(y_test, y_proba, model_name)
        
        return metrics
    
    def _print_metrics(self, metrics, has_proba):
        """Print formatted metrics"""
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   Matthews Correlation: {metrics['mcc']:.4f}")
        
        if has_proba:
            print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"   Average Precision: {metrics['avg_precision']:.4f}")
            print(f"   Log Loss: {metrics['log_loss']:.4f}")
            print(f"   Brier Score: {metrics['brier_score']:.4f}")
        
        print(f"\n⏱️  Timing:")
        print(f"   Training Time: {metrics['train_time']:.4f}s")
        print(f"   Inference Time: {metrics['inference_time']:.4f}s")
        print(f"   Predictions/sec: {metrics['predictions_per_second']:.0f}")
        
        if 'cv_f1_mean' in metrics:
            print(f"\n🔄 Cross-Validation:")
            print(f"   F1 Score: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Purchase', 'Purchase'],
                   yticklabels=['No Purchase', 'Purchase'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curve(self, y_true, y_proba, model_name):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_precision_recall_curve(self, y_true, y_proba, model_name):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_calibration_curve(self, y_true, y_proba, model_name):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        plt.figure(figsize=(6, 5))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f'{model_name}', linewidth=2)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'{model_name} - Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self):
        """Generate comparison plots and tables"""
        if len(self.results) < 2:
            print("Need at least 2 models for comparison")
            return
        
        df = pd.DataFrame(self.results).T
        
        # Performance comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        metrics_to_show = ['accuracy', 'f1', 'precision', 'recall', 'mcc']
        if 'auc_roc' in df.columns:
            metrics_to_show.append('auc_roc')
        
        comparison_df = df[metrics_to_show].round(4)
        print(comparison_df.to_string())
        
        # Timing comparison
        print(f"\n{'='*50}")
        print("TIMING COMPARISON")
        print(f"{'='*50}")
        timing_df = df[['train_time', 'inference_time', 'predictions_per_second']].round(4)
        print(timing_df.to_string())
        
        # Performance radar chart
        self._plot_performance_radar(df)
        
        # Timing comparison chart
        self._plot_timing_comparison(df)
        
        return df
    
    def _plot_performance_radar(self, df):
        """Create radar chart for performance comparison"""
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'mcc']
        if 'auc_roc' in df.columns:
            metrics.append('auc_roc')
        
        # Normalize MCC to 0-1 scale for visualization
        df_plot = df.copy()
        df_plot['mcc'] = (df_plot['mcc'] + 1) / 2  # MCC ranges from -1 to 1
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in df.index:
            values = df_plot.loc[model_name, metrics].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_timing_comparison(self, df):
        """Create timing comparison charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time
        df['train_time'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Training Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Inference speed
        df['predictions_per_second'].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Inference Speed Comparison')
        ax2.set_ylabel('Predictions per Second')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_timing_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_significance_test(self, model1_name, model2_name, 
                                    X_train, y_train, n_iterations=30):
        """Perform statistical significance test between two models"""
        if model1_name not in self.results or model2_name not in self.results:
            print("Both models must be evaluated first")
            return
        
        print(f"\n🔬 Statistical Significance Test: {model1_name} vs {model2_name}")
        print("-" * 60)
        
        # Bootstrap comparison
        model1_scores = []
        model2_scores = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Split bootstrap sample
            X_train_boot, X_test_boot, y_train_boot, y_test_boot = train_test_split(
                X_boot, y_boot, test_size=0.2, random_state=i, stratify=y_boot
            )
            
            # This is a simplified version - in practice, you'd retrain models
            # For now, we'll simulate with cross-validation scores
            pass
        
        print("Statistical significance testing requires model retraining.")
        print("Consider implementing bootstrap validation for robust comparison.")

# Usage example function
def run_comprehensive_evaluation():
    """Example of how to use the comprehensive evaluator"""
    evaluator = ComprehensiveEvaluator()
    
    # Example usage (you would replace with your actual models):
    # evaluator.evaluate_model(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)
    # evaluator.evaluate_model(tabpfn_model, "TabPFN", X_train, X_test, y_train, y_test)
    # evaluator.evaluate_model(tabicl_model, "TabICL", X_train, X_test, y_train, y_test)
    # evaluator.evaluate_model(ftt_model, "FT-Transformer", X_train, X_test, y_train, y_test)
    
    # comparison_df = evaluator.compare_models()
    # return comparison_df
    
    print("Comprehensive evaluation framework ready!")
    return evaluator
