"""
FT-Transformer Training Functions for Bike Sharing Regression

This module contains functions for training and evaluating FT-Transformer models
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

# Import rtdl library for FT-Transformer
try:
    import rtdl
    print("✅ rtdl library imported successfully")
    print(f"rtdl version: {rtdl.__version__ if hasattr(rtdl, '__version__') else 'unknown'}")
except ImportError:
    print("❌ rtdl library not available. Install with: pip install rtdl")
    print("This module requires the rtdl library to run.")
    raise ImportError("Please install rtdl: pip install rtdl")

# Import enhanced evaluation framework
try:
    from enhanced_evaluation import ComprehensiveEvaluator
    print("✅ Enhanced evaluation imported successfully")
except ImportError:
    print("⚠️ Enhanced evaluation not available. Using basic evaluation.")
    ComprehensiveEvaluator = None

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
    """Prepare data for FT-Transformer training"""
    print("🔄 Preparing data for FT-Transformer training...")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)  # Float for regression
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    print(f"📊 Data converted to PyTorch tensors on {device}")
    print(f"Input features: {X_train_tensor.shape[1]}")
    print(f"Task: Regression (continuous target)")

    # Feature info for FT-Transformer (all features are numerical after preprocessing)
    feature_info = {
        'n_num_features': len(feature_names),
        'n_cat_features': 0,
        'cat_cardinalities': []
    }

    print(f"📋 Feature Information:")
    print(f"   Numerical features: {feature_info['n_num_features']}")
    print(f"   Categorical features: {feature_info['n_cat_features']}")

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

    return (train_loader, val_loader, test_loader, feature_info,
            X_train_tensor, X_val_tensor, X_test_tensor,
            y_train_tensor, y_val_tensor, y_test_tensor)

def create_ft_transformer_model(feature_info, device):
    """Create FT-Transformer model for regression"""
    print("🤖 Creating FT-Transformer for regression...")

    # Create FT-Transformer using rtdl library for regression (d_out=1)
    fttransformer_model = rtdl.FTTransformer.make_default(
        n_num_features=feature_info['n_num_features'],
        cat_cardinalities=feature_info['cat_cardinalities'],
        d_out=1  # Single output for regression
    )

    # Move model to device
    fttransformer_model = fttransformer_model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in fttransformer_model.parameters())
    trainable_params = sum(p.numel() for p in fttransformer_model.parameters() if p.requires_grad)

    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   Task: Regression (Bike Count Prediction)")

    return fttransformer_model, total_params

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
    criterion = nn.MSELoss()  # Mean Squared Error for regression

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

        # For rtdl FTTransformer, pass None for categorical features
        output = model(data, None)
        
        # Squeeze output to match target shape for regression
        output = output.squeeze()
        
        loss = criterion(output, target)
        loss.backward()
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

            output = model(data, None)
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

def train_ft_transformer(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                        training_config, device):
    """Train FT-Transformer model"""
    print("🚀 Starting FT-Transformer training...")
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
            print(f"Epoch {epoch+1:3d}/{training_config['n_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f} | "
                  f"LR: {current_lr:.2e} | Patience: {patience_counter}/{training_config['patience']}")

        # Early stopping
        if patience_counter >= training_config['patience']:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
            break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch+1}")

    print(f"\n🏁 Training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation R²: {history['val_r2'][best_epoch]:.4f}")

    return model, history, best_epoch, training_time

def evaluate_model(model, X_test_tensor, y_test_tensor, device):
    """Evaluate the trained model"""
    print("📊 Evaluating trained FT-Transformer...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 256
        for i in range(0, len(X_test_tensor), batch_size):
            batch_X = X_test_tensor[i:i+batch_size]
            output = model(batch_X, None)
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
    
    print(f"📊 Test Set Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAPE: {mape:.4f}%")
    print(f"   Explained Variance: {explained_var:.4f}")
    
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
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # R² curves
    axes[0, 1].plot(epochs, history['val_r2'], 'g-', label='Validation R²', linewidth=2)
    axes[0, 1].axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    axes[0, 1].set_title('Validation R² Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
    axes[1, 1].plot(epochs, loss_diff, 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Training - Validation Loss (Overfitting Indicator)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/FT_Transformer_training_history.png', dpi=300, bbox_inches='tight')
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
    axes[0].set_title('FT-Transformer: Actual vs Predicted')
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
    axes[1].set_title('FT-Transformer: Residuals vs Predicted')
    axes[1].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[2].hist(y_test, bins=30, alpha=0.7, label='Actual', color='blue', density=True)
    axes[2].hist(predictions, bins=30, alpha=0.7, label='Predicted', color='red', density=True)
    axes[2].set_xlabel('Bike Count')
    axes[2].set_ylabel('Density')
    axes[2].set_title('FT-Transformer: Distribution Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/FT_Transformer_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(model, history, metrics, predictions, y_test, feature_names, 
                training_time, total_params, save_dir='./Section2_Model_Training'):
    """Save all results and model"""
    print("💾 Saving results and model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{save_dir}/ft_transformer_training_history.csv', index=False)
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{save_dir}/ft_transformer_evaluation_metrics.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'residuals': y_test - predictions,
        'absolute_error': np.abs(y_test - predictions)
    })
    predictions_df.to_csv(f'{save_dir}/ft_transformer_predictions.csv', index=False)
    
    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'metrics': metrics,
        'total_params': total_params,
        'training_time': training_time
    }, f'{save_dir}/ft_transformer_model.pth')
    
    print("✅ Results saved:")
    print(f"   - Training history: {save_dir}/ft_transformer_training_history.csv")
    print(f"   - Evaluation metrics: {save_dir}/ft_transformer_evaluation_metrics.csv")
    print(f"   - Predictions: {save_dir}/ft_transformer_predictions.csv")
    print(f"   - Model checkpoint: {save_dir}/ft_transformer_model.pth")
    print(f"   - Training plots: {save_dir}/FT_Transformer_training_history.png")
    print(f"   - Evaluation plots: {save_dir}/FT_Transformer_evaluation_results.png")

def run_complete_ft_transformer_training(data_path='./bike_sharing_preprocessed_data.pkl',
                                       device=None, batch_size=256, learning_rate=1e-4,
                                       weight_decay=1e-5, save_dir='./Section2_Model_Training'):
    """Run complete FT-Transformer training pipeline"""
    print("🚴 Starting Complete FT-Transformer Training Pipeline")
    print("="*60)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load data
    (X_train_scaled, X_val_scaled, X_test_scaled, 
     y_train, y_val, y_test, feature_names, data_summary) = load_preprocessed_data(data_path)
    
    # Step 2: Prepare data for training
    (train_loader, val_loader, test_loader, feature_info,
     X_train_tensor, X_val_tensor, X_test_tensor,
     y_train_tensor, y_val_tensor, y_test_tensor) = prepare_data_for_training(
        X_train_scaled, X_val_scaled, X_test_scaled, 
        y_train, y_val, y_test, feature_names, device, batch_size)
    
    # Step 3: Create model
    model, total_params = create_ft_transformer_model(feature_info, device)
    
    # Step 4: Setup training
    criterion, optimizer, scheduler, training_config = setup_training(
        model, learning_rate, weight_decay)
    
    # Step 5: Train model
    model, history, best_epoch, training_time = train_ft_transformer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        training_config, device)
    
    # Step 6: Evaluate model
    predictions, metrics = evaluate_model(model, X_test_tensor, y_test_tensor, device)
    
    # Step 7: Create plots
    create_training_plots(history, best_epoch, save_dir)
    create_evaluation_plots(y_test, predictions, save_dir)
    
    # Step 8: Save results
    save_results(model, history, metrics, predictions, y_test, feature_names, 
                training_time, total_params, save_dir)
    
    print("\n🎉 FT-Transformer training pipeline completed successfully!")
    print(f"📊 Final R² Score: {metrics['r2_score']:.4f}")
    print(f"📊 Final RMSE: {metrics['rmse']:.4f}")
    print(f"⏱️ Training Time: {training_time:.2f} seconds")
    
    return model, history, metrics, predictions, feature_names

if __name__ == "__main__":
    # Run the complete training pipeline
    model, history, metrics, predictions, feature_names = run_complete_ft_transformer_training()
