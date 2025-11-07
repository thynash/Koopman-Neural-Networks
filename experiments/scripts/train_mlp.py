#!/usr/bin/env python3
"""
Training script for MLP Koopman operator learning.

This script demonstrates how to use the MLP training pipeline to train
a model on fractal trajectory data with proper configuration management.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from training.trainers.mlp_trainer import MLPTrainer
from data.datasets.trajectory_dataset import TrajectoryDataset
from data.generators.fractal_generator import FractalGenerator
from config.config_manager import ConfigManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MLP Koopman model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='experiments/configs/default_mlp.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    print(f"Training MLP model with config: {args.config}")
    print(f"Output directory: {config['output_dir']}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Generate or load fractal data
    print("Generating fractal trajectory data...")
    
    # Initialize fractal generator
    generator = FractalGenerator()
    
    # Generate data based on config
    data_config = config['data']
    system_type = data_config['system_type']
    n_points = data_config['n_points']
    
    # Generate trajectory data
    if system_type == 'sierpinski':
        states, next_states = generator.generate_sierpinski_trajectories(n_points)
    elif system_type == 'barnsley':
        states, next_states = generator.generate_barnsley_trajectories(n_points)
    elif system_type == 'julia':
        states, next_states = generator.generate_julia_trajectories(n_points)
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    print(f"Generated {len(states)} trajectory points for {system_type} system")
    
    # Create dataset
    dataset = TrajectoryDataset(
        states=states,
        next_states=next_states,
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        normalize=data_config.get('normalize', True),
        seed=data_config.get('seed', 42)
    )
    
    # Initialize trainer
    trainer = MLPTrainer(config['model'])
    
    # Train model
    print("Starting training...")
    results = trainer.train(dataset, resume_from=args.resume)
    
    # Print results
    print("\nTraining completed!")
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Total epochs: {results['total_epochs']}")
    
    print("\nTest evaluation results:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"\nResults saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()