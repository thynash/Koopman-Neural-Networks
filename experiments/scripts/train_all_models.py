#!/usr/bin/env python3
"""
Comprehensive training script for all neural architectures.

This script trains MLP, DeepONet, and LSTM models on fractal trajectory data
with unified configuration and result tracking.
"""

import sys
import os
import argparse
import yaml
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.datasets.trajectory_dataset import TrajectoryDataset
from training.trainers.mlp_trainer import MLPTrainer
from training.trainers.deeponet_trainer import DeepONetTrainer
from config.config_manager import ConfigManager


def create_default_configs() -> Dict[str, Dict[str, Any]]:
    """Create default configurations for all model types."""
    return {
        'mlp': {
            'model': {
                'input_dim': 2,
                'hidden_dims': [128, 256, 128, 64],
                'output_dim': 2,
                'activation': 'relu',
                'dropout_rate': 0.2,
                'use_batch_norm': True
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 200,
                'weight_decay': 0.0001,
                'scheduler_patience': 15,
                'scheduler_factor': 0.5,
                'early_stopping_patience': 30,
                'min_delta': 1e-6
            },
            'output_dir': 'results/mlp_training'
        },
        'deeponet': {
            'model': {
                'trajectory_length': 10,
                'state_dim': 2,
                'coordinate_dim': 2,
                'branch_hidden_dims': [64, 128, 64],
                'trunk_hidden_dims': [64, 128, 64],
                'latent_dim': 64,
                'activation': 'relu',
                'dropout_rate': 0.1
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 150,
                'weight_decay': 0.0001,
                'prediction_weight': 1.0,
                'consistency_weight': 0.1,
                'validation_freq': 10,
                'checkpoint_freq': 25,
                'early_stopping_patience': 20,
                'min_delta': 1e-6
            },
            'output_dir': 'results/deeponet_training'
        },
        'lstm': {
            'model': {
                'input_dim': 2,
                'hidden_dim': 128,
                'num_layers': 2,
                'output_dim': 2,
                'dropout_rate': 0.2,
                'sequence_length': 20
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 150,
                'weight_decay': 0.0001,
                'gradient_clip_norm': 1.0,
                'scheduler_patience': 10,
                'scheduler_factor': 0.7,
                'early_stopping_patience': 25,
                'min_delta': 1e-6
            },
            'output_dir': 'results/lstm_training'
        }
    }


def load_data(data_path: str, system_type: str = 'sierpinski') -> TrajectoryDataset:
    """Load trajectory data and create dataset."""
    print(f"Loading {system_type} trajectory data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load trajectory data
    trajectory_data = np.load(data_path)
    
    # Split into states and next_states
    n_points = len(trajectory_data) // 2
    states = trajectory_data[:n_points]
    next_states = trajectory_data[n_points:]
    
    print(f"Loaded {len(states)} trajectory points")
    
    # Create dataset
    dataset = TrajectoryDataset(
        states=states,
        next_states=next_states,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize=True,
        seed=42
    )
    
    return dataset


def train_mlp_model(config: Dict[str, Any], dataset: TrajectoryDataset, 
                   output_dir: Path) -> Dict[str, Any]:
    """Train MLP model."""
    print("\nTraining MLP Model")
    print("-" * 30)
    
    # Create trainer
    trainer = MLPTrainer(config['model'])
    
    # Train model
    results = trainer.train(dataset, config['training'])
    
    # Save model and results
    model_path = output_dir / 'mlp_model.pth'
    trainer.save_model(str(model_path))
    
    results_path = output_dir / 'mlp_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"MLP training completed. Results saved to {output_dir}")
    return results


def train_deeponet_model(config: Dict[str, Any], dataset: TrajectoryDataset,
                        output_dir: Path) -> Dict[str, Any]:
    """Train DeepONet model."""
    print("\nTraining DeepONet Model")
    print("-" * 30)
    
    # Create trainer
    trainer = DeepONetTrainer(config)
    
    # Train model
    results = trainer.train(dataset)
    
    # Save operator matrix
    if 'operator_matrix' in results:
        operator_path = output_dir / 'operator_matrix.npy'
        np.save(operator_path, results['operator_matrix'])
    
    # Save results
    results_path = output_dir / 'deeponet_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'final_train_loss': results['final_train_loss'],
        'best_val_loss': results['best_val_loss'],
        'total_epochs': results['total_epochs'],
        'test_metrics': results['test_metrics'],
        'training_history': results['training_history']
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"DeepONet training completed. Results saved to {output_dir}")
    return results


def train_lstm_model(config: Dict[str, Any], dataset: TrajectoryDataset,
                    output_dir: Path) -> Dict[str, Any]:
    """Train LSTM model (placeholder - would need LSTM trainer implementation)."""
    print("\nLSTM Model Training")
    print("-" * 30)
    print("Note: LSTM trainer not yet implemented. Skipping LSTM training.")
    
    # Return placeholder results
    return {
        'status': 'skipped',
        'reason': 'LSTM trainer not implemented',
        'final_train_loss': None,
        'best_val_loss': None,
        'total_epochs': 0,
        'test_metrics': {}
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train all neural architectures')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/sierpinski_trajectories.npy',
        help='Path to trajectory data file'
    )
    parser.add_argument(
        '--system-type',
        type=str,
        default='sierpinski',
        choices=['sierpinski', 'barnsley', 'julia'],
        help='Type of fractal system'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['mlp', 'deeponet'],  # Exclude LSTM until implemented
        choices=['mlp', 'deeponet', 'lstm'],
        help='Models to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/all_models_training',
        help='Base output directory'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoints if available'
    )
    
    args = parser.parse_args()
    
    print("Multi-Model Training Pipeline")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"System type: {args.system_type}")
    print(f"Models to train: {args.models}")
    print(f"Output directory: {args.output_dir}")
    
    # Load or create configurations
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            configs = yaml.safe_load(f)
    else:
        configs = create_default_configs()
    
    # Create base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    config_path = base_output_dir / 'training_configs.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(configs, f, default_flow_style=False)
    
    # Load dataset
    try:
        dataset = load_data(args.data_path, args.system_type)
        print(f"Dataset loaded successfully:")
        print(f"  Training samples: {len(dataset.train_loader.dataset)}")
        print(f"  Validation samples: {len(dataset.val_loader.dataset)}")
        print(f"  Test samples: {len(dataset.test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train each requested model
    all_results = {}
    
    for model_name in args.models:
        if model_name not in configs:
            print(f"Warning: No configuration found for {model_name}, skipping...")
            continue
        
        # Create model-specific output directory
        model_output_dir = base_output_dir / f"{model_name}_results"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update output directory in config
        model_config = configs[model_name].copy()
        model_config['output_dir'] = str(model_output_dir)
        
        try:
            # Train model based on type
            if model_name == 'mlp':
                results = train_mlp_model(model_config, dataset, model_output_dir)
            elif model_name == 'deeponet':
                results = train_deeponet_model(model_config, dataset, model_output_dir)
            elif model_name == 'lstm':
                results = train_lstm_model(model_config, dataset, model_output_dir)
            else:
                print(f"Unknown model type: {model_name}")
                continue
            
            all_results[model_name] = results
            
            # Print summary for this model
            if results.get('status') != 'skipped':
                print(f"\n{model_name.upper()} Results:")
                print(f"  Final training loss: {results.get('final_train_loss', 'N/A'):.6f}")
                print(f"  Best validation loss: {results.get('best_val_loss', 'N/A'):.6f}")
                print(f"  Total epochs: {results.get('total_epochs', 'N/A')}")
                
                if 'test_metrics' in results:
                    print("  Test metrics:")
                    for metric, value in results['test_metrics'].items():
                        print(f"    {metric}: {value:.6f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            all_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            continue
    
    # Save combined results
    combined_results_path = base_output_dir / 'combined_results.json'
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\nTraining Summary")
    print("=" * 30)
    
    successful_models = []
    failed_models = []
    skipped_models = []
    
    for model_name, results in all_results.items():
        status = results.get('status', 'completed')
        if status == 'completed' or ('final_train_loss' in results and results['final_train_loss'] is not None):
            successful_models.append(model_name)
        elif status == 'skipped':
            skipped_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print(f"Successfully trained: {successful_models}")
    if skipped_models:
        print(f"Skipped: {skipped_models}")
    if failed_models:
        print(f"Failed: {failed_models}")
    
    print(f"\nAll results saved to: {base_output_dir}")
    print("Multi-model training completed!")


if __name__ == '__main__':
    main()