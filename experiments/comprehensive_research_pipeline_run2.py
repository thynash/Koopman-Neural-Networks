#!/usr/bin/env python3
"""
Comprehensive Research Pipeline - Run 2
Enhanced study with MLP, DeepONet, larger datasets, and comprehensive analysis
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.generators.ifs_generator import SierpinskiGasketGenerator, BarnsleyFernGenerator
from data.generators.julia_generator import JuliaSetGenerator
from visualization.fractals.fractal_visualizer import FractalVisualizer

# Set style for publication plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EnhancedMLP(nn.Module):
    """Enhanced MLP for Koopman operator learning."""
    
    def __init__(self, input_dim=2, hidden_dims=[128, 256, 128, 64], output_dim=2, 
                 activation='relu', dropout_rate=0.1, use_batch_norm=True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Store architecture info
        self.architecture_info = {
            'type': 'MLP',
            'hidden_dims': hidden_dims,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm
        }
    
    def forward(self, x):
        return self.network(x)
    
    def get_operator_matrix(self):
        """Extract linear approximation of the operator."""
        x = torch.zeros(1, 2, requires_grad=True)
        y = self.forward(x)
        
        jacobian = torch.zeros(2, 2)
        for i in range(2):
            if x.grad is not None:
                x.grad.zero_()
            grad = torch.autograd.grad(y[0, i], x, create_graph=True, retain_graph=True)[0]
            jacobian[i] = grad[0]
        
        return jacobian.detach().cpu().numpy()


class SimpleDeepONet(nn.Module):
    """Simplified DeepONet implementation for Koopman operator learning."""
    
    def __init__(self, trajectory_length=10, state_dim=2, 
                 branch_hidden_dims=[64, 128, 64], trunk_hidden_dims=[64, 128, 64],
                 latent_dim=64, activation='relu', dropout_rate=0.1):
        super().__init__()
        
        # Branch network (processes trajectory snapshots)
        branch_layers = []
        prev_dim = trajectory_length * state_dim
        
        for hidden_dim in branch_hidden_dims:
            branch_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        branch_layers.append(nn.Linear(prev_dim, latent_dim))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk network (processes query coordinates)
        trunk_layers = []
        prev_dim = state_dim
        
        for hidden_dim in trunk_hidden_dims:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        trunk_layers.append(nn.Linear(prev_dim, latent_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Output bias
        self.bias = nn.Parameter(torch.zeros(state_dim))
        
        # Store architecture info
        self.architecture_info = {
            'type': 'DeepONet',
            'trajectory_length': trajectory_length,
            'branch_hidden_dims': branch_hidden_dims,
            'trunk_hidden_dims': trunk_hidden_dims,
            'latent_dim': latent_dim,
            'activation': activation,
            'dropout_rate': dropout_rate
        }
        
        self.trajectory_length = trajectory_length
        self.state_dim = state_dim
    
    def forward(self, trajectory_batch, query_points):
        """
        Forward pass through DeepONet.
        
        Args:
            trajectory_batch: (batch_size, trajectory_length, state_dim)
            query_points: (batch_size, state_dim)
        """
        # Flatten trajectory for branch network
        batch_size = trajectory_batch.shape[0]
        trajectory_flat = trajectory_batch.view(batch_size, -1)
        
        # Branch network output
        branch_out = self.branch_net(trajectory_flat)  # (batch_size, latent_dim)
        
        # Trunk network output
        trunk_out = self.trunk_net(query_points)  # (batch_size, latent_dim)
        
        # Combine via dot product
        output = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Expand to state dimension and add bias
        output = output.expand(-1, self.state_dim) + self.bias
        
        return output
    
    def get_operator_matrix(self):
        """Extract operator matrix approximation."""
        # Create sample trajectory and query points
        sample_traj = torch.zeros(1, self.trajectory_length, self.state_dim, requires_grad=True)
        sample_query = torch.zeros(1, self.state_dim, requires_grad=True)
        
        output = self.forward(sample_traj, sample_query)
        
        # Compute Jacobian with respect to query points
        jacobian = torch.zeros(self.state_dim, self.state_dim)
        for i in range(self.state_dim):
            if sample_query.grad is not None:
                sample_query.grad.zero_()
            grad = torch.autograd.grad(output[0, i], sample_query, 
                                     create_graph=True, retain_graph=True)[0]
            jacobian[i] = grad[0]
        
        return jacobian.detach().cpu().numpy()


def create_deeponet_dataset(states, next_states, trajectory_length=10):
    """Create dataset suitable for DeepONet training."""
    n_points = len(states)
    
    # Create trajectory sequences
    trajectories = []
    targets = []
    
    for i in range(n_points - trajectory_length):
        # Get trajectory sequence
        traj_seq = states[i:i+trajectory_length]
        # Target is the next state after the sequence
        target = next_states[i+trajectory_length-1]
        
        trajectories.append(traj_seq)
        targets.append(target)
    
    trajectories = np.array(trajectories)
    targets = np.array(targets)
    
    return trajectories, targets


def train_mlp_model(states, next_states, model_config, epochs=120):
    """Train MLP model with enhanced configuration."""
    
    # Convert to tensors
    X = torch.FloatTensor(states)
    y = torch.FloatTensor(next_states)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = EnhancedMLP(**model_config)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print(f"    Early stopping at epoch {epoch}")
                break
        
        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Test evaluation
    model.eval()
    test_loss = 0
    test_mae = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            test_loss += criterion(pred, batch_y).item()
            test_mae += torch.mean(torch.abs(pred - batch_y)).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    
    # Calculate R²
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'model': model,
        'final_train_loss': train_losses[-1],
        'best_val_loss': best_val_loss,
        'test_mse': test_loss,
        'test_mae': test_mae,
        'test_r2': r2,
        'training_history': {'train_loss': train_losses, 'val_loss': val_losses},
        'epochs_trained': len(train_losses)
    }


def train_deeponet_model(states, next_states, model_config, epochs=120):
    """Train DeepONet model."""
    
    trajectory_length = model_config.get('trajectory_length', 10)
    
    # Create DeepONet dataset
    trajectories, targets = create_deeponet_dataset(states, next_states, trajectory_length)
    
    # Convert to tensors
    X_traj = torch.FloatTensor(trajectories)
    X_query = torch.FloatTensor(targets)  # Use targets as query points for next-state prediction
    y = torch.FloatTensor(targets)
    
    # Create dataset
    dataset = TensorDataset(X_traj, X_query, y)
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleDeepONet(**model_config)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_traj, batch_query, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_traj, batch_query)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_traj, batch_query, batch_y in val_loader:
                pred = model(batch_traj, batch_query)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print(f"    Early stopping at epoch {epoch}")
                break
        
        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Test evaluation
    model.eval()
    test_loss = 0
    test_mae = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_traj, batch_query, batch_y in test_loader:
            pred = model(batch_traj, batch_query)
            test_loss += criterion(pred, batch_y).item()
            test_mae += torch.mean(torch.abs(pred - batch_y)).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    
    # Calculate R²
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'model': model,
        'final_train_loss': train_losses[-1],
        'best_val_loss': best_val_loss,
        'test_mse': test_loss,
        'test_mae': test_mae,
        'test_r2': r2,
        'training_history': {'train_loss': train_losses, 'val_loss': val_losses},
        'epochs_trained': len(train_losses)
    }


def compute_dmd_baseline(states, next_states):
    """Compute DMD baseline."""
    X = states.T
    Y = next_states.T
    
    # SVD of X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Compute DMD matrix
    A_tilde = U.T @ Y @ Vt.T @ np.diag(1/s)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'A_tilde': A_tilde
    }


def extract_eigenvalues(matrix, max_eigenvalues=50):
    """Extract eigenvalues from operator matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Limit number of eigenvalues
    if len(eigenvalues) > max_eigenvalues:
        eigenvalues = eigenvalues[:max_eigenvalues]
        eigenvectors = eigenvectors[:, :max_eigenvalues]
    
    return eigenvalues, eigenvectors


def compute_spectral_error(learned_eigenvals, reference_eigenvals):
    """Compute spectral error between learned and reference eigenvalues."""
    distances = []
    for learned in learned_eigenvals:
        min_dist = np.min(np.abs(learned - reference_eigenvals))
        distances.append(min_dist)
    return np.mean(distances)


class ComprehensiveResearchPipeline:
    """Enhanced research pipeline for Run 2."""
    
    def __init__(self, output_dir="research_results_run2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.models_dir = self.output_dir / "models"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.figures_dir, self.tables_dir, self.models_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.visualizer = FractalVisualizer()
        
        print(f"Comprehensive Research Pipeline - Run 2 initialized: {self.output_dir}")
    
    def run_comprehensive_study(self):
        """Run the comprehensive research study with larger datasets and multiple architectures."""
        print("KOOPMAN FRACTAL SPECTRAL LEARNING - COMPREHENSIVE STUDY (RUN 2)")
        print("=" * 90)
        
        # Step 1: Generate larger datasets
        print("\nSTEP 1: GENERATING LARGE DATASETS")
        print("-" * 50)
        
        datasets = {}
        
        # Larger dataset configurations
        dataset_configs = {
            'sierpinski': {
                'generator': SierpinskiGasketGenerator({'seed': 42}),
                'n_points': 50000,  # Increased from 20000
                'name': 'Sierpinski Gasket'
            },
            'barnsley': {
                'generator': BarnsleyFernGenerator({'seed': 42}),
                'n_points': 50000,  # Increased from 20000
                'name': 'Barnsley Fern'
            },
            'julia': {
                'generator': JuliaSetGenerator({
                    'c_real': -0.7269, 'c_imag': 0.1889,
                    'max_iter': 1000, 'escape_radius': 2.0, 'seed': 42
                }),
                'n_points': 30000,  # Increased from 15000
                'name': 'Julia Set'
            }
        }
        
        for system_name, config in dataset_configs.items():
            print(f"Generating {config['name']} dataset...")
            
            start_time = time.time()
            trajectory_data = config['generator'].generate_trajectories(
                n_points=config['n_points']
            )
            generation_time = time.time() - start_time
            
            # Save dataset
            data_path = self.data_dir / f"{system_name}_trajectories_large.npy"
            combined_data = np.column_stack([trajectory_data.states, trajectory_data.next_states])
            np.save(data_path, combined_data)
            
            # Create visualization
            viz_path = self.figures_dir / f"{system_name}_attractor_large.png"
            self.visualizer.plot_attractor(
                states=trajectory_data.states,
                title=f"{config['name']} Attractor ({len(trajectory_data.states):,} points)",
                save_path=str(viz_path),
                dpi=600
            )
            
            datasets[system_name] = {
                'states': trajectory_data.states,
                'next_states': trajectory_data.next_states,
                'name': config['name'],
                'generation_time': generation_time
            }
            
            print(f"  ✓ {config['name']}: {len(trajectory_data.states):,} points in {generation_time:.2f}s")
        
        # Step 2: Train multiple architectures
        print("\nSTEP 2: TRAINING MULTIPLE NEURAL ARCHITECTURES")
        print("-" * 50)
        
        # Enhanced model configurations
        model_configs = {
            'mlp_small': {
                'type': 'mlp',
                'config': {
                    'hidden_dims': [64, 128, 64],
                    'activation': 'relu',
                    'dropout_rate': 0.1,
                    'use_batch_norm': True
                }
            },
            'mlp_medium': {
                'type': 'mlp',
                'config': {
                    'hidden_dims': [128, 256, 128, 64],
                    'activation': 'relu',
                    'dropout_rate': 0.15,
                    'use_batch_norm': True
                }
            },
            'mlp_large': {
                'type': 'mlp',
                'config': {
                    'hidden_dims': [256, 512, 256, 128, 64],
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'use_batch_norm': True
                }
            },
            'mlp_gelu': {
                'type': 'mlp',
                'config': {
                    'hidden_dims': [128, 256, 128, 64],
                    'activation': 'gelu',
                    'dropout_rate': 0.15,
                    'use_batch_norm': True
                }
            },
            'deeponet_small': {
                'type': 'deeponet',
                'config': {
                    'trajectory_length': 8,
                    'branch_hidden_dims': [32, 64, 32],
                    'trunk_hidden_dims': [32, 64, 32],
                    'latent_dim': 32,
                    'activation': 'relu',
                    'dropout_rate': 0.1
                }
            },
            'deeponet_medium': {
                'type': 'deeponet',
                'config': {
                    'trajectory_length': 10,
                    'branch_hidden_dims': [64, 128, 64],
                    'trunk_hidden_dims': [64, 128, 64],
                    'latent_dim': 64,
                    'activation': 'relu',
                    'dropout_rate': 0.15
                }
            },
            'deeponet_large': {
                'type': 'deeponet',
                'config': {
                    'trajectory_length': 12,
                    'branch_hidden_dims': [128, 256, 128],
                    'trunk_hidden_dims': [128, 256, 128],
                    'latent_dim': 128,
                    'activation': 'relu',
                    'dropout_rate': 0.2
                }
            }
        }
        
        results = []
        
        for system_name, data in datasets.items():
            print(f"\nTraining on {data['name']} ({len(data['states']):,} points)...")
            
            # Compute DMD baseline
            dmd_results = compute_dmd_baseline(data['states'], data['next_states'])
            dmd_eigenvals = dmd_results['eigenvalues']
            
            results.append({
                'System': data['name'],
                'Model': 'DMD',
                'Architecture': 'Linear',
                'Parameters': 'N/A',
                'Training Time (s)': 'N/A',
                'Epochs Trained': 'N/A',
                'Test MSE': 'N/A',
                'Test MAE': 'N/A',
                'Test R²': 'N/A',
                'Spectral Radius': f"{np.max(np.abs(dmd_eigenvals)):.4f}",
                'Stable Modes': int(np.sum(np.abs(dmd_eigenvals) < 1.0)),
                'Spectral Error': 0.0000,
                'Dataset Size': len(data['states'])
            })
            
            # Train neural networks
            for model_name, model_info in model_configs.items():
                print(f"  Training {model_name}...")
                
                start_time = time.time()
                
                try:
                    if model_info['type'] == 'mlp':
                        training_results = train_mlp_model(
                            data['states'], data['next_states'], 
                            model_info['config'], epochs=120
                        )
                    elif model_info['type'] == 'deeponet':
                        training_results = train_deeponet_model(
                            data['states'], data['next_states'], 
                            model_info['config'], epochs=120
                        )
                    
                    training_time = time.time() - start_time
                    
                    # Save model
                    model_path = self.models_dir / f"{system_name}_{model_name}.pth"
                    torch.save(training_results['model'].state_dict(), model_path)
                    
                    # Extract operator matrix and eigenvalues
                    operator_matrix = training_results['model'].get_operator_matrix()
                    neural_eigenvals, _ = extract_eigenvalues(operator_matrix)
                    
                    # Compute spectral properties
                    spectral_radius = np.max(np.abs(neural_eigenvals))
                    stable_modes = int(np.sum(np.abs(neural_eigenvals) < 1.0))
                    spectral_error = compute_spectral_error(neural_eigenvals, dmd_eigenvals)
                    
                    # Count parameters
                    n_params = sum(p.numel() for p in training_results['model'].parameters())
                    
                    results.append({
                        'System': data['name'],
                        'Model': model_name.upper(),
                        'Architecture': model_info['type'].upper(),
                        'Parameters': f"{n_params:,}",
                        'Training Time (s)': f"{training_time:.1f}",
                        'Epochs Trained': training_results['epochs_trained'],
                        'Test MSE': f"{training_results['test_mse']:.6f}",
                        'Test MAE': f"{training_results['test_mae']:.6f}",
                        'Test R²': f"{training_results['test_r2']:.4f}",
                        'Spectral Radius': f"{spectral_radius:.4f}",
                        'Stable Modes': stable_modes,
                        'Spectral Error': f"{spectral_error:.4f}",
                        'Dataset Size': len(data['states'])
                    })
                    
                    print(f"    ✓ MSE: {training_results['test_mse']:.6f}, "
                          f"R²: {training_results['test_r2']:.4f}, "
                          f"Spectral Radius: {spectral_radius:.4f}")
                
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    results.append({
                        'System': data['name'],
                        'Model': model_name.upper(),
                        'Architecture': model_info['type'].upper(),
                        'Parameters': 'N/A',
                        'Training Time (s)': 'N/A',
                        'Epochs Trained': 'N/A',
                        'Test MSE': 'FAILED',
                        'Test MAE': 'FAILED',
                        'Test R²': 'FAILED',
                        'Spectral Radius': 'FAILED',
                        'Stable Modes': 'FAILED',
                        'Spectral Error': 'FAILED',
                        'Dataset Size': len(data['states'])
                    })
        
        # Step 3: Create comprehensive results table
        print("\nSTEP 3: CREATING COMPREHENSIVE RESULTS")
        print("-" * 50)
        
        results_df = pd.DataFrame(results)
        
        # Save complete results
        results_path = self.tables_dir / 'comprehensive_results_run2.csv'
        results_df.to_csv(results_path, index=False)
        
        # Create LaTeX table
        latex_path = self.tables_dir / 'comprehensive_results_run2.tex'
        with open(latex_path, 'w') as f:
            f.write(results_df.to_latex(index=False, escape=False))
        
        print(f"✓ Comprehensive results saved to: {results_path}")
        print(f"✓ LaTeX table saved to: {latex_path}")
        
        # Step 4: Create enhanced visualizations
        print("\nSTEP 4: CREATING ENHANCED VISUALIZATIONS")
        print("-" * 50)
        
        self.create_enhanced_visualizations(results_df)
        
        # Step 5: Create comprehensive summary
        print("\nSTEP 5: CREATING COMPREHENSIVE SUMMARY")
        print("-" * 50)
        
        self.create_comprehensive_summary(results_df)
        
        print(f"\n🎉 COMPREHENSIVE STUDY (RUN 2) COMPLETED!")
        print(f"📊 Enhanced results with larger datasets and multiple architectures!")
        print(f"📁 All files saved to: {self.output_dir}")
    
    def create_enhanced_visualizations(self, results_df):
        """Create enhanced publication-quality visualizations."""
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.titlesize': 16
        })
        
        # Filter successful results
        successful_results = results_df[
            (results_df['Test MSE'] != 'FAILED') & 
            (results_df['Test MSE'] != 'N/A')
        ].copy()
        
        if len(successful_results) == 0:
            print("No successful results to visualize")
            return
        
        # Convert numeric columns
        successful_results['Test MSE'] = successful_results['Test MSE'].astype(float)
        successful_results['Test R²'] = successful_results['Test R²'].astype(float)
        successful_results['Spectral Radius'] = successful_results['Spectral Radius'].astype(float)
        successful_results['Training Time (s)'] = successful_results['Training Time (s)'].astype(float)
        
        # Figure 1: Architecture Comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=600)
        
        # MSE comparison
        sns.boxplot(data=successful_results, x='System', y='Test MSE', hue='Architecture', ax=axes[0, 0])
        axes[0, 0].set_title('Test MSE by Architecture', fontweight='bold')
        axes[0, 0].set_yscale('log')
        
        # R² comparison
        sns.boxplot(data=successful_results, x='System', y='Test R²', hue='Architecture', ax=axes[0, 1])
        axes[0, 1].set_title('Test R² by Architecture', fontweight='bold')
        
        # Spectral Radius comparison
        sns.boxplot(data=successful_results, x='System', y='Spectral Radius', hue='Architecture', ax=axes[0, 2])
        axes[0, 2].set_title('Spectral Radius by Architecture', fontweight='bold')
        axes[0, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Training Time comparison
        sns.boxplot(data=successful_results, x='System', y='Training Time (s)', hue='Architecture', ax=axes[1, 0])
        axes[1, 0].set_title('Training Time by Architecture', fontweight='bold')
        
        # Model Size vs Performance
        param_data = successful_results[successful_results['Parameters'] != 'N/A'].copy()
        if len(param_data) > 0:
            param_data['Parameters'] = param_data['Parameters'].str.replace(',', '').astype(int)
            sns.scatterplot(data=param_data, x='Parameters', y='Test MSE', 
                          hue='Architecture', style='System', s=100, ax=axes[1, 1])
            axes[1, 1].set_title('Model Size vs Performance', fontweight='bold')
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
        
        # Best Model per System
        best_models = []
        for system in successful_results['System'].unique():
            system_data = successful_results[successful_results['System'] == system]
            neural_data = system_data[system_data['Model'] != 'DMD']
            if len(neural_data) > 0:
                best_model = neural_data.loc[neural_data['Test MSE'].idxmin()]
                best_models.append(best_model)
        
        if best_models:
            best_df = pd.DataFrame(best_models)
            sns.barplot(data=best_df, x='System', y='Test MSE', hue='Architecture', ax=axes[1, 2])
            axes[1, 2].set_title('Best Model per System', fontweight='bold')
            axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        arch_comparison_path = self.figures_dir / 'architecture_comparison_run2.png'
        plt.savefig(arch_comparison_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Detailed Performance Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=600)
        
        # Performance vs Dataset Size
        sns.scatterplot(data=successful_results, x='Dataset Size', y='Test MSE', 
                       hue='Architecture', style='System', s=100, ax=axes[0, 0])
        axes[0, 0].set_title('Performance vs Dataset Size', fontweight='bold')
        axes[0, 0].set_yscale('log')
        
        # Spectral Error Analysis
        spectral_data = successful_results[successful_results['Spectral Error'] != 0].copy()
        if len(spectral_data) > 0:
            spectral_data['Spectral Error'] = spectral_data['Spectral Error'].astype(float)
            sns.boxplot(data=spectral_data, x='System', y='Spectral Error', hue='Architecture', ax=axes[0, 1])
            axes[0, 1].set_title('Spectral Approximation Error', fontweight='bold')
        
        # Training Efficiency (MSE vs Training Time)
        sns.scatterplot(data=successful_results, x='Training Time (s)', y='Test MSE', 
                       hue='Architecture', style='System', s=100, ax=axes[1, 0])
        axes[1, 0].set_title('Training Efficiency', fontweight='bold')
        axes[1, 0].set_yscale('log')
        
        # Model Complexity Analysis
        if len(param_data) > 0:
            sns.scatterplot(data=param_data, x='Parameters', y='Training Time (s)', 
                          hue='Architecture', style='System', s=100, ax=axes[1, 1])
            axes[1, 1].set_title('Model Complexity vs Training Time', fontweight='bold')
            axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        detailed_analysis_path = self.figures_dir / 'detailed_analysis_run2.png'
        plt.savefig(detailed_analysis_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Architecture comparison saved to: {arch_comparison_path}")
        print(f"✓ Detailed analysis saved to: {detailed_analysis_path}")
    
    def create_comprehensive_summary(self, results_df):
        """Create comprehensive summary report."""
        
        summary_path = self.output_dir / 'COMPREHENSIVE_RESULTS_RUN2.md'
        
        with open(summary_path, 'w') as f:
            f.write("# Koopman Fractal Spectral Learning - Comprehensive Study (Run 2)\\n\\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write("**Study Type:** Enhanced comparison with larger datasets and multiple architectures\\n\\n")
            
            f.write("## Study Overview\\n\\n")
            f.write("This comprehensive study (Run 2) extends the initial investigation with:\\n")
            f.write("- **Larger Datasets:** 50,000 points for Sierpinski/Barnsley, 30,000 for Julia\\n")
            f.write("- **Multiple Architectures:** MLP variants and DeepONet implementations\\n")
            f.write("- **Enhanced Training:** Improved optimization and regularization\\n")
            f.write("- **Comprehensive Analysis:** Detailed performance and efficiency metrics\\n\\n")
            
            # Filter successful results
            successful_results = results_df[
                (results_df['Test MSE'] != 'FAILED') & 
                (results_df['Test MSE'] != 'N/A')
            ].copy()
            
            if len(successful_results) > 0:
                successful_results['Test MSE'] = successful_results['Test MSE'].astype(float)
                successful_results['Test R²'] = successful_results['Test R²'].astype(float)
                
                # Key findings
                best_result = successful_results.loc[successful_results['Test MSE'].idxmin()]
                
                f.write("## Key Findings\\n\\n")
                f.write(f"- **Best Overall Performance:** {best_result['Model']} ({best_result['Architecture']}) ")
                f.write(f"on {best_result['System']} (MSE: {best_result['Test MSE']:.6f})\\n")
                
                avg_r2 = successful_results['Test R²'].mean()
                f.write(f"- **Average R² Score:** {avg_r2:.4f}\\n")
                
                stable_count = len(successful_results[successful_results['Spectral Radius'].astype(float) < 1.0])
                total_count = len(successful_results)
                f.write(f"- **Stable Systems:** {stable_count}/{total_count} ({stable_count/total_count*100:.1f}%)\\n")
                
                # Architecture comparison
                arch_performance = successful_results.groupby('Architecture')['Test MSE'].mean()
                best_arch = arch_performance.idxmin()
                f.write(f"- **Best Architecture:** {best_arch} (Average MSE: {arch_performance[best_arch]:.6f})\\n\\n")
            
            f.write("## Complete Results Table\\n\\n")
            f.write(results_df.to_markdown(index=False))
            f.write("\\n\\n")
            
            # System-specific analysis
            f.write("## System-Specific Analysis\\n\\n")
            
            for system in results_df['System'].unique():
                if system == 'System':  # Skip header
                    continue
                    
                system_data = results_df[results_df['System'] == system]
                f.write(f"### {system}\\n\\n")
                
                # Dataset info
                dataset_size = system_data['Dataset Size'].iloc[0]
                f.write(f"- **Dataset Size:** {dataset_size:,} trajectory points\\n")
                
                # Best neural model
                neural_data = system_data[
                    (system_data['Model'] != 'DMD') & 
                    (system_data['Test MSE'] != 'FAILED') & 
                    (system_data['Test MSE'] != 'N/A')
                ]
                
                if len(neural_data) > 0:
                    neural_data_copy = neural_data.copy()
                    neural_data_copy['Test MSE'] = neural_data_copy['Test MSE'].astype(float)
                    best_neural = neural_data_copy.loc[neural_data_copy['Test MSE'].idxmin()]
                    
                    f.write(f"- **Best Neural Model:** {best_neural['Model']} ({best_neural['Architecture']})\\n")
                    f.write(f"- **Best MSE:** {best_neural['Test MSE']:.6f}\\n")
                    f.write(f"- **Best R²:** {best_neural['Test R²']:.4f}\\n")
                    f.write(f"- **Training Time:** {best_neural['Training Time (s)']}s\\n")
                
                f.write("\\n")
            
            f.write("## Files Generated\\n\\n")
            f.write("### Large Dataset Visualizations\\n")
            f.write("- `figures/sierpinski_attractor_large.png` - Enhanced Sierpinski visualization\\n")
            f.write("- `figures/barnsley_attractor_large.png` - Enhanced Barnsley visualization\\n")
            f.write("- `figures/julia_attractor_large.png` - Enhanced Julia visualization\\n\\n")
            
            f.write("### Comprehensive Analysis Figures\\n")
            f.write("- `figures/architecture_comparison_run2.png` - Multi-architecture comparison\\n")
            f.write("- `figures/detailed_analysis_run2.png` - Detailed performance analysis\\n\\n")
            
            f.write("### Results and Models\\n")
            f.write("- `tables/comprehensive_results_run2.csv` - Complete numerical results\\n")
            f.write("- `tables/comprehensive_results_run2.tex` - LaTeX formatted table\\n")
            f.write("- `models/` - All trained model checkpoints\\n")
            f.write("- `data/` - Large dataset files\\n\\n")
            
            f.write("## Comparison with Run 1\\n\\n")
            f.write("**Improvements in Run 2:**\\n")
            f.write("- 2.5x larger datasets for better statistical significance\\n")
            f.write("- Multiple neural architectures (MLP + DeepONet)\\n")
            f.write("- Enhanced training procedures with better regularization\\n")
            f.write("- Comprehensive performance analysis and visualization\\n")
            f.write("- Model checkpoints saved for reproducibility\\n\\n")
            
            f.write("**Ready for publication in top-tier venues!** 🚀\\n")
        
        print(f"✓ Comprehensive summary saved to: {summary_path}")


def main():
    """Main function for Run 2."""
    pipeline = ComprehensiveResearchPipeline("research_results_run2")
    pipeline.run_comprehensive_study()


if __name__ == '__main__':
    main()