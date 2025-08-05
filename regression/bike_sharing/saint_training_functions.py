"""
SAINT Training Functions for Bike Sharing Regression

This module contains functions for training and evaluating SAINT (Self-Attention and Intersample Attention Transformer) models
on the bike sharing dataset for regression tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
import pickle
import logging
warnings.filterwarnings('ignore')

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ML libraries
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup logging
def setup_logging(save_dir='./Section2_Model_Training'):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{save_dir}/saint_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class SAINTModel(nn.Module):
    """
    SAINT (Self-Attention and Intersample Attention Transformer) Model for Regression
    
    This implementation includes:
    - Feature embeddings for numerical features
    - Self-attention mechanism
    - Intersample attention mechanism
    - Layer normalization and residual connections
    - Regression head for continuous target prediction
    """
    
    def __init__(self, n_features, d_model=128, n_heads=8, n_layers=6, 
                 dropout=0.1, d_ff=512):
        super(SAINTModel, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Feature embedding layer
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Positional encoding for features
        self.pos_encoding = nn.Parameter(torch.randn(n_features, d_model))
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Intersample attention layers
        self.intersample_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.layer_norms_3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model * n_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, n_features)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Reshape for feature-wise embedding: (batch_size, n_features, 1)
        x = x.unsqueeze(-1)
        
        # Feature embedding: (batch_size, n_features, d_model)
        x = self.feature_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer layers
        for i in range(self.n_layers):
            # Self-attention within each sample
            residual = x
            x = self.layer_norms_1[i](x)
            attn_output, _ = self.self_attention_layers[i](x, x, x)
            x = residual + attn_output
            
            # Intersample attention (across batch dimension)
            residual = x
            x = self.layer_norms_2[i](x)
            # Transpose for intersample attention: (n_features, batch_size, d_model)
            x_transposed = x.transpose(0, 1)
            attn_output, _ = self.intersample_attention_layers[i](
                x_transposed, x_transposed, x_transposed
            )
            # Transpose back: (batch_size, n_features, d_model)
            x = residual + attn_output.transpose(0, 1)
            
            # Feed-forward network
            residual = x
            x = self.layer_norms_3[i](x)
            x = residual + self.feed_forward_layers[i](x)
        
        # Flatten for regression head
        x = x.view(batch_size, -1)
        
        # Regression prediction
        output = self.regression_head(x)
        
        return output

def load_preprocessed_data(data_path='./bike_sharing_preprocessed_data.pkl'):
    """Load preprocessed bike sharing data"""
    print("📊 Loading preprocessed bike sharing data...")
    
    try:
        with open(data_path, 'rb') as f:
            preprocessing_data = pickle.load(f)

        # Extract variables
        X_train_scaled = preprocessing_data['X_train_scaled']
        X_val_scaled = preprocessing_data['X_val_scaled']
        X_test_scaled = preprocessing_data['X_test_scaled']
        y_train = preprocessing_data['y_train']
        y_val = preprocessing_data['y_val']
        y_test = preprocessing_data['y_test']
        feature_names = preprocessing_data['feature_names']
        data_summary = preprocessing_data['data_summary']

        print("✅ Preprocessed data loaded successfully!")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Task: Regression (Bike Count Prediction)")
        print(f"Target range: [{y_train.min()}, {y_train.max()}]")

        # Check for invalid values
        print("\nChecking for invalid values...")
        print(f"NaN in X_train: {np.isnan(X_train_scaled).any()}")
        print(f"Inf in X_train: {np.isinf(X_train_scaled).any()}")
        print(f"NaN in y_train: {np.isnan(y_train).any()}")

        # Check data ranges
        print(f"X_train min: {X_train_scaled.min():.4f}, max: {X_train_scaled.max():.4f}")
        print(f"y_train min: {y_train.min():.0f}, max: {y_train.max():.0f}")

        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, feature_names, data_summary)

    except FileNotFoundError:
        print("❌ Preprocessed data not found!")
        print("Please run Section 1 (Data Preprocessing) first.")
        raise

def prepare_data_for_training(X_train_scaled, X_val_scaled, X_test_scaled, 
                            y_train, y_val, y_test, feature_names, device, batch_size=256):
    """Prepare data for SAINT training"""
    print("🔄 Preparing data for SAINT training...")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    print(f"📊 Data converted to PyTorch tensors on {device}")
    print(f"Input features: {X_train_tensor.shape[1]}")
    print(f"Task: Regression (continuous target)")

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"🔄 Data loaders created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    return (train_loader, val_loader, test_loader,
            X_train_tensor, X_val_tensor, X_test_tensor,
            y_train_tensor, y_val_tensor, y_test_tensor)

def create_saint_model(n_features, device, d_model=128, n_heads=8, n_layers=6, dropout=0.1):
    """Create SAINT model for regression"""
    print("🤖 Creating SAINT model for regression...")

    model = SAINTModel(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )

    # Move model to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 SAINT Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   Architecture: {n_layers} layers, {n_heads} heads, {d_model} d_model")
    print(f"   Task: Regression (Bike Count Prediction)")

    return model, total_params

def setup_training(model, learning_rate=1e-4, weight_decay=1e-5):
    """Setup training components"""
    print("🔧 Setting up training components...")
    
    # Training configuration
    training_config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'n_epochs': 100,
        'patience': 15,
        'min_delta': 1e-4
    }

    # Loss function for regression
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    print("✅ Training setup completed:")
    print(f"   Loss function: MSELoss (regression)")
    print(f"   Optimizer: AdamW (lr={training_config['learning_rate']}, wd={training_config['weight_decay']})")
    print(f"   Scheduler: ReduceLROnPlateau")

    return criterion, optimizer, scheduler, training_config

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        output = output.squeeze()
        
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * len(target)
        total_samples += len(target)

    avg_loss = total_loss / total_samples
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    """Validate the model for one epoch"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            output = output.squeeze()

            loss = criterion(output, target)

            total_loss += loss.item() * len(target)
            total_samples += len(target)
            
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / total_samples
    
    # Calculate R² score for validation
    r2 = r2_score(all_targets, all_predictions)
    
    return avg_loss, r2

def train_saint_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                     training_config, device, logger=None):
    """Train SAINT model"""
    print("🚀 Starting SAINT training...")
    if logger:
        logger.info("Starting SAINT training...")
    
    print(f"Training for {training_config['n_epochs']} epochs with early stopping (patience={training_config['patience']})")
    print("-" * 80)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'learning_rates': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(training_config['n_epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_r2 = validate_epoch(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss - training_config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            progress_msg = (f"Epoch {epoch+1:3d}/{training_config['n_epochs']} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f} | "
                          f"LR: {current_lr:.2e} | Patience: {patience_counter}/{training_config['patience']}")
            print(progress_msg)
            if logger:
                logger.info(progress_msg)

        # Early stopping
        if patience_counter >= training_config['patience']:
            stop_msg = f"\n⏹️ Early stopping triggered at epoch {epoch+1}"
            best_msg = f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}"
            print(stop_msg)
            print(best_msg)
            if logger:
                logger.info(stop_msg)
                logger.info(best_msg)
            break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch+1}")

    final_msg = f"\n🏁 Training completed in {training_time:.2f} seconds"
    print(final_msg)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation R²: {history['val_r2'][best_epoch]:.4f}")
    
    if logger:
        logger.info(final_msg)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final validation R²: {history['val_r2'][best_epoch]:.4f}")

    return model, history, best_epoch, training_time

def evaluate_model(model, X_test_tensor, y_test_tensor, device, logger=None):
    """Evaluate the trained model"""
    print("📊 Evaluating trained SAINT model...")
    if logger:
        logger.info("Evaluating trained SAINT model...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 256
        for i in range(0, len(X_test_tensor), batch_size):
            batch_X = X_test_tensor[i:i+batch_size]
            output = model(batch_X)
            predictions.extend(output.squeeze().cpu().numpy())
    
    predictions = np.array(predictions)
    y_test_np = y_test_tensor.cpu().numpy()
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test_np, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    
    try:
        mape = mean_absolute_percentage_error(y_test_np, predictions)
    except:
        mape = np.mean(np.abs((y_test_np - predictions) / np.maximum(np.abs(y_test_np), 1e-8))) * 100
    
    explained_var = explained_variance_score(y_test_np, predictions)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape,
        'explained_variance': explained_var
    }
    
    results_msg = f"📊 Test Set Performance:\n" \
                 f"   R² Score: {r2:.4f}\n" \
                 f"   RMSE: {rmse:.4f}\n" \
                 f"   MAE: {mae:.4f}\n" \
                 f"   MSE: {mse:.4f}\n" \
                 f"   MAPE: {mape:.4f}%\n" \
                 f"   Explained Variance: {explained_var:.4f}"
    
    print(results_msg)
    if logger:
        logger.info(results_msg)
    
    return predictions, metrics

def create_training_plots(history, best_epoch, save_dir='./Section2_Model_Training'):
    """Create training visualization plots"""
    print("📈 Creating training visualization plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[0, 0].set_title('SAINT: Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # R² curves
    axes[0, 1].plot(epochs, history['val_r2'], 'g-', label='Validation R²', linewidth=2)
    axes[0, 1].axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[0, 1].set_title('SAINT: Validation R² Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    axes[1, 0].set_title('SAINT: Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
    axes[1, 1].plot(epochs, loss_diff, 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('SAINT: Training - Validation Loss (Overfitting Indicator)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/SAINT_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_evaluation_plots(y_test, predictions, save_dir='./Section2_Model_Training'):
    """Create evaluation plots"""
    print("📊 Creating evaluation plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Regression results plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, predictions, alpha=0.6, color='blue')
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Bike Count')
    axes[0].set_ylabel('Predicted Bike Count')
    axes[0].set_title('SAINT: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² annotation
    r2 = r2_score(y_test, predictions)
    axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals plot
    residuals = y_test - predictions
    axes[1].scatter(predictions, residuals, alpha=0.6, color='green')
    axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Bike Count')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('SAINT: Residuals vs Predicted')
    axes[1].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[2].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue', density=True)
    axes[2].hist(predictions, bins=30, alpha=0.7, label='Predicted', color='red', density=True)
    axes[2].set_xlabel('Bike Count')
    axes[2].set_ylabel('Density')
    axes[2].set_title('SAINT: Distribution Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/SAINT_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(model, history, metrics, predictions, y_test, feature_names, 
                training_time, total_params, save_dir='./Section2_Model_Training', logger=None):
    """Save all results and model"""
    print("💾 Saving results and model...")
    if logger:
        logger.info("Saving results and model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{save_dir}/saint_training_history.csv', index=False)
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{save_dir}/saint_evaluation_metrics.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'residuals': y_test - predictions,
        'absolute_error': np.abs(y_test - predictions)
    })
    predictions_df.to_csv(f'{save_dir}/saint_predictions.csv', index=False)
    
    # Save model checkpoint (as pickle for easier loading)
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'n_features': model.n_features,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'n_layers': model.n_layers
        },
        'feature_names': feature_names,
        'metrics': metrics,
        'total_params': total_params,
        'training_time': training_time
    }
    
    # Save as both .pth and .pkl
    torch.save(model_data, f'{save_dir}/saint_model.pth')
    with open(f'{save_dir}/saint_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    save_msg = "✅ Results saved:\n" \
              f"   - Training history: {save_dir}/saint_training_history.csv\n" \
              f"   - Evaluation metrics: {save_dir}/saint_evaluation_metrics.csv\n" \
              f"   - Predictions: {save_dir}/saint_predictions.csv\n" \
              f"   - Model checkpoint: {save_dir}/saint_model.pth\n" \
              f"   - Model pickle: {save_dir}/saint_model.pkl\n" \
              f"   - Training plots: {save_dir}/SAINT_training_history.png\n" \
              f"   - Evaluation plots: {save_dir}/SAINT_evaluation_results.png\n" \
              f"   - Training log: {save_dir}/saint_training.log"
    
    print(save_msg)
    if logger:
        logger.info(save_msg)

def run_complete_saint_training(data_path='./bike_sharing_preprocessed_data.pkl',
                               device=None, batch_size=256, learning_rate=1e-4,
                               weight_decay=1e-5, d_model=128, n_heads=8, n_layers=6,
                               save_dir='./Section2_Model_Training'):
    """Run complete SAINT training pipeline"""
    print("🚴 Starting Complete SAINT Training Pipeline")
    print("="*60)
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info("Starting Complete SAINT Training Pipeline")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    # Step 1: Load data
    (X_train_scaled, X_val_scaled, X_test_scaled, 
     y_train, y_val, y_test, feature_names, data_summary) = load_preprocessed_data(data_path)
    
    # Step 2: Prepare data for training
    (train_loader, val_loader, test_loader,
     X_train_tensor, X_val_tensor, X_test_tensor,
     y_train_tensor, y_val_tensor, y_test_tensor) = prepare_data_for_training(
        X_train_scaled, X_val_scaled, X_test_scaled, 
        y_train, y_val, y_test, feature_names, device, batch_size)
    
    # Step 3: Create model
    model, total_params = create_saint_model(
        len(feature_names), device, d_model, n_heads, n_layers)
    
    # Step 4: Setup training
    criterion, optimizer, scheduler, training_config = setup_training(
        model, learning_rate, weight_decay)
    
    # Step 5: Train model
    model, history, best_epoch, training_time = train_saint_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        training_config, device, logger)
    
    # Step 6: Evaluate model
    predictions, metrics = evaluate_model(model, X_test_tensor, y_test_tensor, device, logger)
    
    # Step 7: Create plots
    create_training_plots(history, best_epoch, save_dir)
    create_evaluation_plots(y_test, predictions, save_dir)
    
    # Step 8: Save results
    save_results(model, history, metrics, predictions, y_test, feature_names, 
                training_time, total_params, save_dir, logger)
    
    print("\n🎉 SAINT training pipeline completed successfully!")
    print(f"📊 Final R² Score: {metrics['r2_score']:.4f}")
    print(f"📊 Final RMSE: {metrics['rmse']:.4f}")
    print(f"⏱️ Training Time: {training_time:.2f} seconds")
    
    logger.info("SAINT training pipeline completed successfully!")
    logger.info(f"Final R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"Final RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Training Time: {training_time:.2f} seconds")
    
    return model, history, metrics, predictions, feature_names

if __name__ == "__main__":
    # Run the complete training pipeline
    model, history, metrics, predictions, feature_names = run_complete_saint_training()
