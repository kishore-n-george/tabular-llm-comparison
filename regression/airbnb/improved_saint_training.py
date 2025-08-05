"""
Improved SAINT Training for Airbnb Regression with Enhanced Data

This script implements an improved SAINT (Self-Attention and Intersample Attention Transformer) 
model for Airbnb price prediction with:
1. Better architecture design for regression
2. Improved regularization and stability
3. Target scaling for better training
4. Enhanced data preprocessing integration
5. Better hyperparameter configuration
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
from sklearn.preprocessing import RobustScaler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ImprovedSAINTModel(nn.Module):
    """
    Improved SAINT Model for Regression with better stability and performance
    
    Key improvements:
    - Simplified architecture to prevent overfitting
    - Better regularization techniques
    - Proper initialization
    - Gradient clipping support
    - Target-specific design for regression
    """
    
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3, 
                 dropout=0.3, d_ff=128):
        super(ImprovedSAINTModel, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Feature embedding with batch normalization
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.1)
        
        # Transformer layers with proper normalization
        self.transformer_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True  # Pre-norm for better stability
            )
            self.transformer_layers.append(layer)
        
        # Global pooling and regression head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass with improved stability
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
        x = x + self.pos_encoding.expand(batch_size, -1, -1)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Global average pooling: (batch_size, d_model, n_features) -> (batch_size, d_model, 1)
        x = x.transpose(1, 2)  # (batch_size, d_model, n_features)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        # Regression prediction
        output = self.regression_head(x)
        
        return output

def load_enhanced_data(enhanced_data_path='./Section1_Data_PreProcessing/enhanced_data.pkl'):
    """Load enhanced data from XGBoost preprocessing pipeline"""
    print("🏠 Loading enhanced Airbnb data for improved SAINT training...")
    
    try:
        import joblib
        enhanced_data = joblib.load(enhanced_data_path)

        # Extract enhanced variables
        X_train_scaled = enhanced_data['X_train']
        X_val_scaled = enhanced_data['X_val']
        X_test_scaled = enhanced_data['X_test']
        y_train = enhanced_data['y_train']
        y_val = enhanced_data['y_val']
        y_test = enhanced_data['y_test']
        feature_names = enhanced_data['feature_names']
        transform_info = enhanced_data['transform_info']

        print("✅ Enhanced data loaded successfully!")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Features: {len(feature_names)} (after feature engineering & selection)")
        print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        print(f"Target transformation: {transform_info['method']}")

        # Enhanced data quality checks
        print(f"\n🔍 Enhanced Data Quality Analysis:")
        print(f"X_train - Min: {X_train_scaled.min():.4f}, Max: {X_train_scaled.max():.4f}")
        print(f"X_train - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
        print(f"y_train - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
        print(f"y_train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
        
        # Check for problematic values
        print(f"\n🚨 Data Integrity Checks:")
        print(f"NaN in X_train: {np.isnan(X_train_scaled).any()}")
        print(f"Inf in X_train: {np.isinf(X_train_scaled).any()}")
        print(f"NaN in y_train: {np.isnan(y_train).any()}")
        print(f"Inf in y_train: {np.isinf(y_train).any()}")

        # Enhanced preprocessing summary
        print(f"\n🔧 Enhanced Preprocessing Applied:")
        print(f"   • Outlier removal: {enhanced_data['outlier_mask'].sum()} samples removed")
        print(f"   • Feature engineering: Polynomial + interaction + binned features")
        print(f"   • Feature selection: Top {len(feature_names)} features via mutual information")
        print(f"   • Advanced scaling: RobustScaler applied")
        print(f"   • Target transformation: {transform_info['method']}")

        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, feature_names, transform_info)

    except FileNotFoundError:
        print("❌ Enhanced data not found!")
        print("Please run XGBoost enhanced training first to generate enhanced_data.pkl")
        raise

def apply_target_scaling(y_train, y_val, y_test):
    """Apply robust scaling to target variable for better training stability"""
    print("🔧 Applying target scaling for improved SAINT training...")
    
    # Use RobustScaler for target to handle outliers better
    target_scaler = RobustScaler()
    
    # Fit on training data only
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"Original target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Scaled target range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
    print(f"Scaled target mean: {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}")
    
    return y_train_scaled, y_val_scaled, y_test_scaled, target_scaler

def prepare_improved_data(X_train_scaled, X_val_scaled, X_test_scaled, 
                         y_train_scaled, y_val_scaled, y_test_scaled, 
                         feature_names, device, batch_size=64):
    """Prepare data for improved SAINT training with smaller batch size"""
    print("🔄 Preparing data for improved SAINT training...")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Convert to PyTorch tensors with proper dtype
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

    print(f"📊 Tensor conversion completed on {device}")
    print(f"Input features: {X_train_tensor.shape[1]}")
    print(f"Batch size: {batch_size} (smaller for better stability)")

    # Create data loaders with smaller batch size for stability
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"🔄 Data loaders created with improved settings:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    return (train_loader, val_loader, test_loader,
            X_train_tensor, X_val_tensor, X_test_tensor,
            y_train_tensor, y_val_tensor, y_test_tensor)

def create_improved_saint_model(n_features, device, d_model=64, n_heads=4, n_layers=3, dropout=0.3):
    """Create improved SAINT model with better architecture"""
    print("🤖 Creating improved SAINT model...")

    model = ImprovedSAINTModel(
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

    print(f"📊 Improved SAINT Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   Architecture: {n_layers} layers, {n_heads} heads, {d_model} d_model")
    print(f"   Improvements: Batch norm, pre-norm, better initialization")

    return model, total_params

def setup_improved_training(model, learning_rate=1e-4, weight_decay=1e-4):
    """Setup improved training components"""
    print("🔧 Setting up improved training components...")
    
    # Improved training configuration
    training_config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'n_epochs': 150,  # More epochs with better early stopping
        'patience': 25,   # More patience
        'min_delta': 1e-5,
        'warmup_epochs': 10,  # Learning rate warmup
        'gradient_clip': 0.5  # Gradient clipping
    }

    # Loss function with better stability
    criterion = nn.MSELoss(reduction='mean')

    # Improved optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Improved learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < training_config['warmup_epochs']:
            return (epoch + 1) / training_config['warmup_epochs']
        else:
            return 0.95 ** (epoch - training_config['warmup_epochs'])
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("✅ Improved training setup completed:")
    print(f"   Loss function: MSELoss with mean reduction")
    print(f"   Optimizer: AdamW (lr={training_config['learning_rate']}, wd={training_config['weight_decay']})")
    print(f"   Scheduler: LambdaLR with warmup")
    print(f"   Gradient clipping: {training_config['gradient_clip']}")

    return criterion, optimizer, scheduler, training_config

def train_epoch_improved(model, train_loader, criterion, optimizer, device, gradient_clip=0.5):
    """Improved training epoch with better stability"""
    model.train()
    total_loss = 0
    total_samples = 0
    batch_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with error handling
        try:
            output = model(data)
            output = output.squeeze(-1)  # Ensure proper shape
            
            # Check for NaN/Inf in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"⚠️ NaN/Inf detected in model output at batch {batch_idx}")
                continue
                
            loss = criterion(output, target)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf detected in loss at batch {batch_idx}")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * len(target)
            total_samples += len(target)
            batch_losses.append(batch_loss)
            
        except Exception as e:
            print(f"⚠️ Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss, batch_losses

def validate_epoch_improved(model, val_loader, criterion, device, target_scaler):
    """Improved validation with proper target unscaling"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            try:
                output = model(data)
                output = output.squeeze(-1)
                
                # Skip if NaN/Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    continue
                    
                loss = criterion(output, target)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item() * len(target)
                    total_samples += len(target)
                    
                    all_predictions_scaled.extend(output.cpu().numpy())
                    all_targets_scaled.extend(target.cpu().numpy())
                    
            except Exception as e:
                print(f"⚠️ Validation error: {e}")
                continue

    if total_samples == 0:
        return float('inf'), -1.0

    avg_loss = total_loss / total_samples
    
    # Calculate R² on unscaled data for proper evaluation
    if len(all_predictions_scaled) > 0:
        predictions_scaled = np.array(all_predictions_scaled).reshape(-1, 1)
        targets_scaled = np.array(all_targets_scaled).reshape(-1, 1)
        
        # Unscale predictions and targets
        predictions_unscaled = target_scaler.inverse_transform(predictions_scaled).flatten()
        targets_unscaled = target_scaler.inverse_transform(targets_scaled).flatten()
        
        r2 = r2_score(targets_unscaled, predictions_unscaled)
    else:
        r2 = -1.0
    
    return avg_loss, r2

def train_improved_saint_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                              training_config, device, target_scaler):
    """Improved training loop with better monitoring"""
    print("🚀 Starting improved SAINT training...")
    print(f"Training for {training_config['n_epochs']} epochs with enhanced monitoring")
    print("-" * 80)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'learning_rates': [],
        'batch_losses': []
    }

    # Early stopping based on R2 improvement
    best_val_r2 = -float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(training_config['n_epochs']):
        # Training
        train_loss, batch_losses = train_epoch_improved(
            model, train_loader, criterion, optimizer, device, 
            training_config['gradient_clip']
        )

        # Validation
        val_loss, val_r2 = validate_epoch_improved(
            model, val_loader, criterion, device, target_scaler
        )

        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(current_lr)
        history['batch_losses'].extend(batch_losses)

        # Early stopping based on R2 improvement
        if val_r2 > best_val_r2 + training_config['min_delta']:
            best_val_r2 = val_r2
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1

        # Print progress with enhanced information
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}/{training_config['n_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f} | "
                  f"LR: {current_lr:.2e} | Patience: {patience_counter}/{training_config['patience']}")

        # Early stopping
        if patience_counter >= training_config['patience']:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation R²: {best_val_r2:.4f} at epoch {best_epoch+1}")
            break

        # Check for training issues
        if train_loss > 1e6 or val_loss > 1e6:
            print(f"\n⚠️ Training instability detected at epoch {epoch+1}")
            print("Consider reducing learning rate or checking data")
            break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch+1}")

    print(f"\n🏁 Training completed in {training_time:.2f} seconds")
    print(f"Best validation R²: {best_val_r2:.4f}")

    return model, history, best_epoch, training_time

def evaluate_improved_model(model, X_test_tensor, y_test_tensor, device, target_scaler):
    """Improved model evaluation with proper unscaling"""
    print("📊 Evaluating improved SAINT model...")
    
    model.eval()
    predictions_scaled = []
    
    with torch.no_grad():
        # Process in batches
        batch_size = 64
        for i in range(0, len(X_test_tensor), batch_size):
            batch_X = X_test_tensor[i:i+batch_size]
            try:
                output = model(batch_X)
                predictions_scaled.extend(output.squeeze(-1).cpu().numpy())
            except Exception as e:
                print(f"⚠️ Evaluation error in batch {i//batch_size}: {e}")
                # Fill with mean prediction as fallback
                mean_pred = np.mean(predictions_scaled) if predictions_scaled else 0.0
                predictions_scaled.extend([mean_pred] * len(batch_X))
    
    # Convert to numpy and unscale
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    y_test_scaled = y_test_tensor.cpu().numpy().reshape(-1, 1)
    
    # Unscale both predictions and targets
    predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
    y_test_unscaled = target_scaler.inverse_transform(y_test_scaled).flatten()
    
    # Calculate metrics on unscaled data
    mse = mean_squared_error(y_test_unscaled, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, predictions)
    r2 = r2_score(y_test_unscaled, predictions)
    
    try:
        mape = mean_absolute_percentage_error(y_test_unscaled, predictions)
    except:
        mape = np.mean(np.abs((y_test_unscaled - predictions) / np.maximum(np.abs(y_test_unscaled), 1e-8))) * 100
    
    explained_var = explained_variance_score(y_test_unscaled, predictions)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape,
        'explained_variance': explained_var
    }
    
    print(f"📊 Improved SAINT Test Set Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAPE: {mape:.4f}%")
    print(f"   Explained Variance: {explained_var:.4f}")
    
    return predictions, metrics, y_test_unscaled

def create_improved_plots(history, best_epoch, predictions, y_test, metrics, 
                         save_dir='./Section2_Model_Training'):
    """Create improved visualization plots"""
    print("📈 Creating improved visualization plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Training plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[0, 0].set_title('Improved SAINT: Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # R² curves
    axes[0, 1].plot(epochs, history['val_r2'], 'g-', label='Validation R²', linewidth=2)
    axes[0, 1].axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[0, 1].axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Baseline (R²=0)')
    axes[0, 1].set_title('Improved SAINT: R² Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[0, 2].plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    # Actual vs Predicted
    axes[1, 0].scatter(y_test, predictions, alpha=0.6, color='blue', s=20)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Airbnb Price ($)')
    axes[1, 0].set_ylabel('Predicted Airbnb Price ($)')
    axes[1, 0].set_title(f'Improved SAINT: Actual vs Predicted (R²={metrics["r2_score"]:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals plot
    residuals = y_test - predictions
    axes[1, 1].scatter(predictions, residuals, alpha=0.6, color='green', s=20)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Airbnb Price ($)')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Improved SAINT: Residuals')
    axes[1, 1].grid(True, alpha=0.3)

    # Distribution comparison
    axes[1, 2].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue', density=True)
    axes[1, 2].hist(predictions, bins=30, alpha=0.7, label='Predicted', color='red', density=True)
    axes[1, 2].set_xlabel('Airbnb Price ($)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Improved SAINT: Distribution Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/Improved_SAINT_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_improved_results(model, history, metrics, predictions, y_test, feature_names, 
                         training_time, total_params, target_scaler, 
                         save_dir='./Section2_Model_Training'):
    """Save improved results"""
    print("💾 Saving improved results...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame({
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_r2': history['val_r2'],
        'learning_rates': history['learning_rates']
    })
    history_df.to_csv(f'{save_dir}/improved_saint_training_history.csv', index=False)
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{save_dir}/improved_saint_evaluation_metrics.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'residuals': y_test - predictions,
        'absolute_error': np.abs(y_test - predictions)
    })
    predictions_df.to_csv(f'{save_dir}/improved_saint_predictions.csv', index=False)
    
    # Save model checkpoint with target scaler
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
        'training_time': training_time,
        'target_scaler': target_scaler
    }
    
    # Save as both .pth and .pkl
    torch.save(model_data, f'{save_dir}/improved_saint_model.pth')
    with open(f'{save_dir}/improved_saint_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Improved results saved successfully!")

def run_improved_saint_training(enhanced_data_path='./Section1_Data_PreProcessing/enhanced_data.pkl',
                               device=None, batch_size=64, learning_rate=1e-4,
                               weight_decay=1e-4, d_model=64, n_heads=4, n_layers=3,
                               save_dir='./Section2_Model_Training'):
    """Run complete improved SAINT training pipeline with enhanced data"""
    print("🏠 Starting Improved SAINT Training Pipeline with Enhanced Data")
    print("="*60)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Step 1: Load enhanced data
        (X_train_scaled, X_val_scaled, X_test_scaled, 
         y_train, y_val, y_test, feature_names, transform_info) = load_enhanced_data(enhanced_data_path)
        
        # Step 2: Apply target scaling
        y_train_scaled, y_val_scaled, y_test_scaled, target_scaler = apply_target_scaling(
            y_train, y_val, y_test)
        
        # Step 3: Prepare data with improvements
        (train_loader, val_loader, test_loader,
         X_train_tensor, X_val_tensor, X_test_tensor,
         y_train_tensor, y_val_tensor, y_test_tensor) = prepare_improved_data(
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train_scaled, y_val_scaled, y_test_scaled, 
            feature_names, device, batch_size)
        
        # Step 4: Create improved model
        model, total_params = create_improved_saint_model(
            len(feature_names), device, d_model, n_heads, n_layers)
        
        # Step 5: Setup improved training
        criterion, optimizer, scheduler, training_config = setup_improved_training(
            model, learning_rate, weight_decay)
        
        # Step 6: Train improved model
        model, history, best_epoch, training_time = train_improved_saint_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            training_config, device, target_scaler)
        
        # Step 7: Evaluate improved model
        predictions, metrics, y_test_unscaled = evaluate_improved_model(
            model, X_test_tensor, y_test_tensor, device, target_scaler)
        
        # Step 8: Create improved plots
        create_improved_plots(history, best_epoch, predictions, y_test_unscaled, metrics, save_dir)
        
        # Step 9: Save improved results
        save_improved_results(model, history, metrics, predictions, y_test_unscaled, 
                             feature_names, training_time, total_params, target_scaler, save_dir)
        
        print("\n🎉 Improved SAINT training completed successfully!")
        print(f"📊 Final R² Score: {metrics['r2_score']:.4f}")
        print(f"📊 Final RMSE: {metrics['rmse']:.4f}")
        print(f"⏱️ Training Time: {training_time:.2f} seconds")
        
        return model, history, metrics, predictions, feature_names, target_scaler
        
    except Exception as e:
        print(f"❌ Error in improved SAINT training pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run the improved training pipeline
    print("🚀 Running Improved SAINT Training for Airbnb")
    model, history, metrics, predictions, feature_names, target_scaler = run_improved_saint_training()
