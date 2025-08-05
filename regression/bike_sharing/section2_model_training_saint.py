# ## 2.7 SAINT Training for Bike Sharing Regression
# 
# This script implements the **SAINT (Self-Attention and Intersample Attention Transformer)** architecture for tabular data regression on the Bike Sharing dataset.
# 
# ## SAINT Overview
# 
# The SAINT architecture is a transformer-based model specifically designed for tabular data that:
# - Uses self-attention mechanisms to capture feature interactions within samples
# - Employs intersample attention to learn patterns across different samples in a batch
# - Applies feature embeddings for numerical features
# - Uses positional encoding to maintain feature order information
# - Provides excellent performance on tabular regression tasks
# 
# ## Implementation Details
# 
# - **Feature Embedding**: Each numerical feature is embedded into a higher-dimensional space
# - **Self-Attention**: Captures interactions between features within each sample
# - **Intersample Attention**: Learns patterns across different samples in the batch
# - **Layer Normalization**: Stabilizes training with residual connections
# - **Regression Head**: Multi-layer perceptron for final prediction
# 
# **Reference**: Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training. arXiv preprint arXiv:2106.01342.

# Import training functions
try:
    from saint_training_functions import *
    print("✅ SAINT training functions imported successfully")
except ImportError:
    print("❌ SAINT training functions not available.")
    print("Please ensure saint_training_functions.py is in the same directory.")
    raise

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("🚴 SAINT Training for Bike Sharing Regression")
print("Dataset: Bike Sharing Dataset")

def main():
    """Main function to run SAINT training"""
    print("\n" + "="*60)
    print("STARTING SAINT TRAINING PIPELINE")
    print("="*60)
    
    # Run the complete training pipeline
    model, history, metrics, predictions, feature_names = run_complete_saint_training(
        data_path='./bike_sharing_preprocessed_data.pkl',
        device=device,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-5,
        d_model=128,
        n_heads=8,
        n_layers=6,
        save_dir='./Section2_Model_Training'
    )
    
    print("\n" + "="*60)
    print("SAINT TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Final Performance:")
    print(f"   R² Score: {metrics['r2_score']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    
    print(f"\n📁 Generated Files:")
    print(f"   - Training History: ./Section2_Model_Training/saint_training_history.csv")
    print(f"   - Evaluation Metrics: ./Section2_Model_Training/saint_evaluation_metrics.csv")
    print(f"   - Predictions: ./Section2_Model_Training/saint_predictions.csv")
    print(f"   - Model Checkpoint: ./Section2_Model_Training/saint_model.pth")
    print(f"   - Model Pickle: ./Section2_Model_Training/saint_model.pkl")
    print(f"   - Training Plots: ./Section2_Model_Training/SAINT_training_history.png")
    print(f"   - Evaluation Plots: ./Section2_Model_Training/SAINT_evaluation_results.png")
    print(f"   - Training Log: ./Section2_Model_Training/saint_training.log")
    
    print(f"\n🚀 Model ready for deployment and comparison with other models!")
    
    return model, history, metrics, predictions, feature_names

if __name__ == "__main__":
    # Run the main function
    model, history, metrics, predictions, feature_names = main()
