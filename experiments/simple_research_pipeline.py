#!/usr/bin/env python3
"""
Simplified Research Pipeline for Koopman Fractal Spectral Learning

This script generates publication-ready results with minimal dependencies.
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


class SimpleMLP(nn.Module):
    """Simple MLP for Koopman operator learning."""
    
    def __init__(self, input_dim=2, hidden_dims=[128, 256, 128, 64], output_dim=2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_operator_matrix(self):
        """Extract linear approximation of the operator."""
        # Compute Jacobian at origin as operator approximation
        x = torch.zeros(1, 2, requires_grad=True)
        y = self.forward(x)
        
        jacobian = torch.zeros(2, 2)
        for i in range(2):
            if x.grad is not None:
                x.grad.zero_()
            grad = torch.autograd.grad(y[0, i], x, create_graph=True, retain_graph=True)[0]
            jacobian[i] = grad[0]
        
        return jacobian.detach().cpu().numpy()


def train_model(states, next_states, model_config, epochs=100):
    """Train a simple MLP model."""
    
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
    model = SimpleMLP(**model_config)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
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
        'training_history': {'train_loss': train_losses, 'val_loss': val_losses}
    }


def compute_dmd_baseline(states, next_states):
    """Compute DMD baseline."""
    # Center the data
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
    # Simple Hausdorff distance approximation
    distances = []
    for learned in learned_eigenvals:
        min_dist = np.min(np.abs(learned - reference_eigenvals))
        distances.append(min_dist)
    return np.mean(distances)


class SimpleResearchPipeline:
    """Simplified research pipeline."""
    
    def __init__(self, output_dir="research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        self.visualizer = FractalVisualizer()
        
        print(f"Research pipeline initialized: {self.output_dir}")
    
    def run_complete_study(self):
        """Run the complete research study."""
        print("KOOPMAN FRACTAL SPECTRAL LEARNING - RESEARCH STUDY")
        print("=" * 80)
        
        # Step 1: Generate datasets
        print("\nSTEP 1: GENERATING DATASETS")
        print("-" * 40)
        
        datasets = {}
        
        # Sierpinski Gasket
        print("Generating Sierpinski Gasket...")
        sierpinski_gen = SierpinskiGasketGenerator({'seed': 42})
        sierpinski_data = sierpinski_gen.generate_trajectories(n_points=20000)
        datasets['sierpinski'] = {
            'states': sierpinski_data.states,
            'next_states': sierpinski_data.next_states,
            'name': 'Sierpinski Gasket'
        }
        
        # Barnsley Fern
        print("Generating Barnsley Fern...")
        barnsley_gen = BarnsleyFernGenerator({'seed': 42})
        barnsley_data = barnsley_gen.generate_trajectories(n_points=20000)
        datasets['barnsley'] = {
            'states': barnsley_data.states,
            'next_states': barnsley_data.next_states,
            'name': 'Barnsley Fern'
        }
        
        # Julia Set
        print("Generating Julia Set...")
        julia_gen = JuliaSetGenerator({
            'c_real': -0.7269, 'c_imag': 0.1889,
            'max_iter': 1000, 'escape_radius': 2.0, 'seed': 42
        })
        julia_data = julia_gen.generate_trajectories(n_points=15000)
        datasets['julia'] = {
            'states': julia_data.states,
            'next_states': julia_data.next_states,
            'name': 'Julia Set'
        }
        
        # Create attractor visualizations
        for system_name, data in datasets.items():
            viz_path = self.figures_dir / f"{system_name}_attractor.png"
            self.visualizer.plot_attractor(
                states=data['states'],
                title=f"{data['name']} Attractor ({len(data['states']):,} points)",
                save_path=str(viz_path),
                dpi=600
            )
            print(f"  ✓ {data['name']}: {len(data['states']):,} points")
        
        # Step 2: Train models
        print("\nSTEP 2: TRAINING NEURAL NETWORKS")
        print("-" * 40)
        
        model_configs = {
            'small': {'hidden_dims': [64, 128, 64]},
            'medium': {'hidden_dims': [128, 256, 128, 64]},
            'large': {'hidden_dims': [256, 512, 256, 128, 64]}
        }
        
        results = []
        
        for system_name, data in datasets.items():
            print(f"\nTraining on {data['name']}...")
            
            # Compute DMD baseline
            dmd_results = compute_dmd_baseline(data['states'], data['next_states'])
            dmd_eigenvals = dmd_results['eigenvalues']
            
            results.append({
                'System': data['name'],
                'Model': 'DMD',
                'Architecture': 'Linear',
                'Parameters': 'N/A',
                'Training Time (s)': 'N/A',
                'Test MSE': 'N/A',
                'Test MAE': 'N/A',
                'Test R²': 'N/A',
                'Spectral Radius': f"{np.max(np.abs(dmd_eigenvals)):.4f}",
                'Stable Modes': int(np.sum(np.abs(dmd_eigenvals) < 1.0)),
                'Spectral Error': 0.0000
            })
            
            # Train neural networks
            for model_name, config in model_configs.items():
                print(f"  Training {model_name} model...")
                
                start_time = time.time()
                training_results = train_model(
                    data['states'], data['next_states'], config, epochs=80
                )
                training_time = time.time() - start_time
                
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
                    'Architecture': 'MLP',
                    'Parameters': f"{n_params:,}",
                    'Training Time (s)': f"{training_time:.1f}",
                    'Test MSE': f"{training_results['test_mse']:.6f}",
                    'Test MAE': f"{training_results['test_mae']:.6f}",
                    'Test R²': f"{training_results['test_r2']:.4f}",
                    'Spectral Radius': f"{spectral_radius:.4f}",
                    'Stable Modes': stable_modes,
                    'Spectral Error': f"{spectral_error:.4f}"
                })
                
                print(f"    ✓ MSE: {training_results['test_mse']:.6f}, "
                      f"Spectral Radius: {spectral_radius:.4f}")
        
        # Step 3: Create results table
        print("\nSTEP 3: CREATING RESULTS TABLE")
        print("-" * 40)
        
        results_df = pd.DataFrame(results)
        
        # Save complete results
        results_path = self.tables_dir / 'complete_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # Create LaTeX table
        latex_path = self.tables_dir / 'results_table.tex'
        with open(latex_path, 'w') as f:
            f.write(results_df.to_latex(index=False, escape=False))
        
        print(f"✓ Results table saved to: {results_path}")
        print(f"✓ LaTeX table saved to: {latex_path}")
        
        # Step 4: Create publication figures
        print("\nSTEP 4: CREATING PUBLICATION FIGURES")
        print("-" * 40)
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.titlesize': 16
        })
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
        
        # Test MSE comparison
        neural_data = results_df[results_df['Model'] != 'DMD'].copy()
        neural_data['Test MSE'] = neural_data['Test MSE'].astype(float)
        
        sns.barplot(data=neural_data, x='System', y='Test MSE', hue='Model', ax=axes[0, 0])
        axes[0, 0].set_title('Test Mean Squared Error', fontweight='bold')
        axes[0, 0].set_ylabel('MSE')
        
        # Spectral Radius comparison
        spectral_data = results_df.copy()
        spectral_data['Spectral Radius'] = spectral_data['Spectral Radius'].astype(float)
        sns.barplot(data=spectral_data, x='System', y='Spectral Radius', hue='Model', ax=axes[0, 1])
        axes[0, 1].set_title('Spectral Radius Comparison', fontweight='bold')
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Training Time comparison
        training_data = neural_data[neural_data['Training Time (s)'] != 'N/A'].copy()
        training_data['Training Time (s)'] = training_data['Training Time (s)'].astype(float)
        sns.barplot(data=training_data, x='System', y='Training Time (s)', hue='Model', ax=axes[1, 0])
        axes[1, 0].set_title('Training Time Comparison', fontweight='bold')
        
        # Model Size comparison
        param_data = neural_data.copy()
        param_data['Parameters'] = param_data['Parameters'].str.replace(',', '').astype(int)
        sns.barplot(data=param_data, x='System', y='Parameters', hue='Model', ax=axes[1, 1])
        axes[1, 1].set_title('Model Size Comparison', fontweight='bold')
        
        plt.tight_layout()
        performance_path = self.figures_dir / 'performance_comparison.png'
        plt.savefig(performance_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Performance comparison saved to: {performance_path}")
        
        # Step 5: Create summary report
        print("\nSTEP 5: CREATING SUMMARY REPORT")
        print("-" * 40)
        
        summary_path = self.output_dir / 'RESEARCH_SUMMARY.md'
        
        with open(summary_path, 'w') as f:
            f.write("# Koopman Fractal Spectral Learning - Research Results\\n\\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Executive Summary\\n\\n")
            f.write("This study compares neural network architectures for learning Koopman operators ")
            f.write("on fractal dynamical systems against Dynamic Mode Decomposition baselines.\\n\\n")
            
            # Key findings
            neural_results = results_df[results_df['Model'] != 'DMD']
            best_mse = neural_results['Test MSE'].astype(float).min()
            best_model = neural_results.loc[neural_results['Test MSE'].astype(float).idxmin()]
            
            f.write("## Key Findings\\n\\n")
            f.write(f"- **Best Performance:** {best_model['Model']} on {best_model['System']} ")
            f.write(f"(MSE: {best_mse:.6f})\\n")
            
            avg_spectral_error = neural_results['Spectral Error'].astype(float).mean()
            f.write(f"- **Average Spectral Error:** {avg_spectral_error:.4f}\\n")
            
            stable_count = len(neural_results[neural_results['Spectral Radius'].astype(float) < 1.0])
            total_count = len(neural_results)
            f.write(f"- **Stable Systems:** {stable_count}/{total_count} ")
            f.write(f"({stable_count/total_count*100:.1f}%)\\n\\n")
            
            f.write("## Complete Results Table\\n\\n")
            f.write(results_df.to_markdown(index=False))
            f.write("\\n\\n")
            
            f.write("## Files Generated\\n\\n")
            f.write("### Figures\\n")
            f.write("- `figures/sierpinski_attractor.png` - Sierpinski gasket visualization\\n")
            f.write("- `figures/barnsley_attractor.png` - Barnsley fern visualization\\n")
            f.write("- `figures/julia_attractor.png` - Julia set visualization\\n")
            f.write("- `figures/performance_comparison.png` - Performance metrics comparison\\n\\n")
            
            f.write("### Tables\\n")
            f.write("- `tables/complete_results.csv` - Complete numerical results\\n")
            f.write("- `tables/results_table.tex` - LaTeX formatted table\\n")
        
        print(f"✓ Research summary saved to: {summary_path}")
        
        # Print final summary
        print(f"\n🎉 RESEARCH STUDY COMPLETED!")
        print(f"📊 Results ready for publication!")
        print(f"📁 All files saved to: {self.output_dir}")
        
        # Display key results
        print(f"\n📈 KEY RESULTS:")
        print(f"   Best neural model: {best_model['Model']} on {best_model['System']}")
        print(f"   Best MSE: {best_mse:.6f}")
        print(f"   Average spectral error: {avg_spectral_error:.4f}")
        print(f"   Stable systems: {stable_count}/{total_count}")


def main():
    """Main function."""
    pipeline = SimpleResearchPipeline("research_results")
    pipeline.run_complete_study()


if __name__ == '__main__':
    main()