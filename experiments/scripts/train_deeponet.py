"""
Training script for DeepONet Koopman operator learning.

This script demonstrates how to train a DeepONet model on fractal trajectory data
for learning Koopman operators with operator extraction for spectral analysis.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from typing import Dict, Any

from src.data.generators.fractal_generator import FractalGenerator
from src.data.datasets.trajectory_dataset import TrajectoryDataset
from src.training.trainers.deeponet_trainer import DeepONetTrainer
from src.config.config_manager import ConfigManager


def create_deeponet_config() -> Dict[str, Any]:
    """
    Create configuration for DeepONet training.
    
    Returns:
        Configuration dictionary
    """
    config = {
        # Model architecture
        'trajectory_length': 10,
        'state_dim': 2,
        'coordinate_dim': 2,
        'branch_hidden_dims': [64, 128, 64],
        'trunk_hidden_dims': [64, 128, 64],
        'latent_dim': 64,
        'activation': 'relu',
        'dropout_rate': 0.1,
        
        # Training parameters
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'weight_decay': 0.0001,
        
        # Loss function parameters
        'prediction_weight': 1.0,
        'consistency_weight': 0.1,
        
        # Validation and checkpointing
        'validation_freq': 10,
        'checkpoint_freq': 25,
        'early_stopping_patience': 20,
        'min_delta': 1e-6,
        
        # Output
        'output_dir': 'results/deeponet_training'
    }
    
    return config


def main():
    """Main training function."""
    print("Starting DeepONet Koopman Operator Training")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create configuration
    config = create_deeponet_config()
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Generate fractal data
    print("Generating fractal trajectory data...")
    generator = FractalGenerator()
    
    # Generate Sierpinski gasket data
    sierpinski_data = generator.generate_trajectories(
        system_type='sierpinski',
        n_points=10000,
        save_path='data/sierpinski_trajectories.npy'
    )
    
    print(f"Generated {len(sierpinski_data)} Sierpinski trajectory points")
    
    # Create trajectory dataset
    print("Creating trajectory dataset...")
    dataset = TrajectoryDataset(
        data=sierpinski_data,
        sequence_length=1,  # Single step prediction
        normalize=True
    )
    
    # Split dataset
    dataset_split = dataset.create_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(dataset_split.train)} samples")
    print(f"  Validation: {len(dataset_split.validation)} samples")
    print(f"  Test: {len(dataset_split.test)} samples")
    print()
    
    # Initialize trainer
    print("Initializing DeepONet trainer...")
    trainer = DeepONetTrainer(config)
    
    # Train model
    print("Starting training...")
    results = trainer.train(dataset_split)
    
    # Print results
    print("\nTraining completed!")
    print("=" * 50)
    print("Final Results:")
    print(f"  Final training loss: {results['final_train_loss']:.6f}")
    print(f"  Best validation loss: {results['best_val_loss']:.6f}")
    print(f"  Total epochs: {results['total_epochs']}")
    
    print("\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric}: {value:.6f}")
    
    # Save operator matrix
    operator_matrix = results['operator_matrix']
    operator_path = Path(config['output_dir']) / 'operator_matrix.npy'
    np.save(operator_path, operator_matrix)
    print(f"\nSaved operator matrix to {operator_path}")
    print(f"Operator matrix shape: {operator_matrix.shape}")
    
    # Save complete results
    results_path = Path(config['output_dir']) / 'training_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'final_train_loss': results['final_train_loss'],
        'best_val_loss': results['best_val_loss'],
        'total_epochs': results['total_epochs'],
        'test_metrics': results['test_metrics'],
        'training_history': results['training_history'],
        'config': config
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved complete results to {results_path}")
    
    print("\nDeepONet training completed successfully!")


if __name__ == "__main__":
    main()