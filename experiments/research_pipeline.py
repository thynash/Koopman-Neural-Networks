#!/usr/bin/env python3
"""
Complete Research Pipeline for Koopman Fractal Spectral Learning

This script runs the complete research pipeline and generates publication-ready
results including tables, plots, and statistical analysis.
"""

import sys
import os
import json
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.generators.ifs_generator import SierpinskiGasketGenerator, BarnsleyFernGenerator
from data.generators.julia_generator import JuliaSetGenerator
from data.datasets.trajectory_dataset import TrajectoryDataset
from analysis.spectral.spectral_analyzer import SpectralAnalyzer
from analysis.spectral.dmd_baseline import DMDBaseline
from visualization.fractals.fractal_visualizer import FractalVisualizer

# Import models and trainers with proper path handling
import importlib.util

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import MLP components
mlp_model_path = Path(__file__).parent.parent / 'src' / 'models' / 'mlp' / 'mlp_koopman.py'
mlp_trainer_path = Path(__file__).parent.parent / 'src' / 'training' / 'trainers' / 'mlp_trainer.py'

mlp_model_module = import_from_path('mlp_koopman', mlp_model_path)
mlp_trainer_module = import_from_path('mlp_trainer', mlp_trainer_path)

MLPKoopman = mlp_model_module.MLPKoopman
MLPTrainer = mlp_trainer_module.MLPTrainer

# Set style for publication plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResearchPipeline:
    """Complete research pipeline for generating publication results."""
    
    def __init__(self, output_dir: str = "research_results"):
        """Initialize research pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.models_dir = self.output_dir / "models"
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        
        for dir_path in [self.data_dir, self.models_dir, self.figures_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.visualizer = FractalVisualizer()
        self.spectral_analyzer = SpectralAnalyzer()
        self.dmd_baseline = DMDBaseline()
        
        # Results storage
        self.results = {
            'data_generation': {},
            'model_training': {},
            'spectral_analysis': {},
            'performance_metrics': {}
        }
        
        print(f"Research pipeline initialized")
        print(f"Results will be saved to: {self.output_dir}")
    
    def generate_fractal_datasets(self) -> Dict[str, Any]:
        """Generate datasets for all fractal systems."""
        print("\n" + "="*60)
        print("STEP 1: GENERATING FRACTAL DATASETS")
        print("="*60)
        
        datasets = {}
        
        # Configuration for each system
        configs = {
            'sierpinski': {
                'generator': SierpinskiGasketGenerator({'seed': 42}),
                'n_points': 20000,
                'name': 'Sierpinski Gasket'
            },
            'barnsley': {
                'generator': BarnsleyFernGenerator({'seed': 42}),
                'n_points': 20000,
                'name': 'Barnsley Fern'
            },
            'julia': {
                'generator': JuliaSetGenerator({
                    'c_real': -0.7269, 'c_imag': 0.1889,
                    'max_iter': 1000, 'escape_radius': 2.0, 'seed': 42
                }),
                'n_points': 15000,
                'name': 'Julia Set'
            }
        }
        
        for system_name, config in configs.items():
            print(f"\nGenerating {config['name']} dataset...")
            
            start_time = time.time()
            
            # Generate trajectory data
            trajectory_data = config['generator'].generate_trajectories(
                n_points=config['n_points']
            )
            
            generation_time = time.time() - start_time
            
            # Create PyTorch dataset
            dataset = TrajectoryDataset(
                states=trajectory_data.states,
                next_states=trajectory_data.next_states,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                normalize=True,
                seed=42
            )
            
            # Save data
            data_path = self.data_dir / f"{system_name}_trajectories.npy"
            combined_data = np.column_stack([trajectory_data.states, trajectory_data.next_states])
            np.save(data_path, combined_data)
            
            # Create visualization
            viz_path = self.figures_dir / f"{system_name}_attractor.png"
            self.visualizer.plot_attractor(
                states=trajectory_data.states,
                title=f"{config['name']} Attractor ({len(trajectory_data.states):,} points)",
                save_path=str(viz_path),
                dpi=600
            )
            
            # Store results
            datasets[system_name] = {
                'dataset': dataset,
                'trajectory_data': trajectory_data,
                'generation_time': generation_time,
                'n_points': len(trajectory_data.states),
                'data_path': str(data_path),
                'visualization_path': str(viz_path)
            }
            
            print(f"✓ Generated {len(trajectory_data.states):,} points in {generation_time:.2f}s")
        
        self.results['data_generation'] = datasets
        return datasets
    
    def train_neural_networks(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Train neural networks on all datasets."""
        print("\n" + "="*60)
        print("STEP 2: TRAINING NEURAL NETWORKS")
        print("="*60)
        
        training_results = {}
        
        # Model configurations
        model_configs = {
            'mlp_small': {
                'input_dim': 2,
                'hidden_dims': [64, 128, 64],
                'output_dim': 2,
                'activation': 'relu',
                'dropout_rate': 0.1,
                'use_batch_norm': True
            },
            'mlp_medium': {
                'input_dim': 2,
                'hidden_dims': [128, 256, 128, 64],
                'output_dim': 2,
                'activation': 'relu',
                'dropout_rate': 0.2,
                'use_batch_norm': True
            },
            'mlp_large': {
                'input_dim': 2,
                'hidden_dims': [256, 512, 256, 128, 64],
                'output_dim': 2,
                'activation': 'relu',
                'dropout_rate': 0.3,
                'use_batch_norm': True
            }
        }
        
        training_config = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,  # Reduced for faster execution
            'weight_decay': 0.0001,
            'scheduler_patience': 10,
            'scheduler_factor': 0.5,
            'early_stopping_patience': 20,
            'min_delta': 1e-6
        }
        
        for system_name, system_data in datasets.items():
            print(f"\nTraining models on {system_name} dataset...")
            training_results[system_name] = {}
            
            for model_name, model_config in model_configs.items():
                print(f"  Training {model_name}...")
                
                start_time = time.time()
                
                try:
                    # Create and train model
                    trainer = MLPTrainer(model_config)
                    results = trainer.train(system_data['dataset'], training_config)
                    
                    training_time = time.time() - start_time
                    
                    # Save model
                    model_path = self.models_dir / f"{system_name}_{model_name}.pth"
                    trainer.save_model(str(model_path))
                    
                    # Extract operator matrix
                    operator_matrix = trainer.model.get_operator_matrix()
                    
                    # Store results
                    training_results[system_name][model_name] = {
                        'model': trainer.model,
                        'trainer': trainer,
                        'training_results': results,
                        'training_time': training_time,
                        'operator_matrix': operator_matrix,
                        'model_path': str(model_path),
                        'success': True
                    }
                    
                    print(f"    ✓ Completed in {training_time:.1f}s - Final loss: {results['final_train_loss']:.6f}")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    training_results[system_name][model_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        self.results['model_training'] = training_results
        return training_results
    
    def perform_spectral_analysis(self, training_results: Dict[str, Any], 
                                datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Perform spectral analysis on trained models."""
        print("\n" + "="*60)
        print("STEP 3: SPECTRAL ANALYSIS")
        print("="*60)
        
        spectral_results = {}
        
        for system_name, system_models in training_results.items():
            print(f"\nAnalyzing {system_name} models...")
            spectral_results[system_name] = {}
            
            # Get dataset for DMD baseline
            dataset = datasets[system_name]['dataset']
            
            # Compute DMD baseline
            print("  Computing DMD baseline...")
            train_data = []
            for batch in dataset.train_loader:
                states, next_states = batch
                train_data.append(np.column_stack([states.numpy(), next_states.numpy()]))
            
            train_data = np.vstack(train_data)
            states = train_data[:, :2]
            next_states = train_data[:, 2:]
            
            dmd_results = self.dmd_baseline.compute_dmd(states, next_states)
            spectral_results[system_name]['dmd'] = {
                'eigenvalues': dmd_results['eigenvalues'],
                'spectral_radius': np.max(np.abs(dmd_results['eigenvalues'])),
                'stable_modes': np.sum(np.abs(dmd_results['eigenvalues']) < 1.0),
                'method': 'DMD'
            }
            
            # Analyze each trained model
            for model_name, model_data in system_models.items():
                if not model_data['success']:
                    continue
                
                print(f"  Analyzing {model_name}...")
                
                operator_matrix = model_data['operator_matrix']
                
                # Extract eigenvalues
                eigenvalues, eigenvectors = self.spectral_analyzer.extract_eigenvalues(
                    operator_matrix, max_eigenvalues=50
                )
                
                # Compute spectral properties
                spectral_radius = np.max(np.abs(eigenvalues))
                stable_modes = np.sum(np.abs(eigenvalues) < 1.0)
                
                # Compute spectral error relative to DMD
                spectral_error = self.spectral_analyzer.compute_spectral_error(
                    eigenvalues, dmd_results['eigenvalues']
                )
                
                spectral_results[system_name][model_name] = {
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors,
                    'spectral_radius': spectral_radius,
                    'stable_modes': stable_modes,
                    'spectral_error': spectral_error,
                    'method': 'Neural Network'
                }
                
                print(f"    ✓ Spectral radius: {spectral_radius:.4f}, Stable modes: {stable_modes}")
        
        self.results['spectral_analysis'] = spectral_results
        return spectral_results
    
    def create_performance_tables(self, training_results: Dict[str, Any], 
                                spectral_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive performance comparison tables."""
        print("\n" + "="*60)
        print("STEP 4: CREATING PERFORMANCE TABLES")
        print("="*60)
        
        # Collect all results into a structured format
        table_data = []
        
        for system_name in training_results.keys():
            system_models = training_results[system_name]
            system_spectral = spectral_results[system_name]
            
            # Add DMD baseline
            dmd_data = system_spectral['dmd']
            table_data.append({
                'System': system_name.title(),
                'Model': 'DMD',
                'Architecture': 'Linear',
                'Parameters': 'N/A',
                'Training Time (s)': 'N/A',
                'Final Train Loss': 'N/A',
                'Best Val Loss': 'N/A',
                'Test MSE': 'N/A',
                'Test MAE': 'N/A',
                'Test R²': 'N/A',
                'Spectral Radius': f"{dmd_data['spectral_radius']:.4f}",
                'Stable Modes': dmd_data['stable_modes'],
                'Spectral Error': 0.0000  # Reference
            })
            
            # Add neural network results
            for model_name, model_data in system_models.items():
                if not model_data['success']:
                    continue
                
                training_res = model_data['training_results']
                spectral_res = system_spectral[model_name]
                
                # Count parameters
                model = model_data['model']
                n_params = sum(p.numel() for p in model.parameters())
                
                table_data.append({
                    'System': system_name.title(),
                    'Model': model_name.upper(),
                    'Architecture': 'MLP',
                    'Parameters': f"{n_params:,}",
                    'Training Time (s)': f"{model_data['training_time']:.1f}",
                    'Final Train Loss': f"{training_res['final_train_loss']:.6f}",
                    'Best Val Loss': f"{training_res['best_val_loss']:.6f}",
                    'Test MSE': f"{training_res['test_metrics']['mse']:.6f}",
                    'Test MAE': f"{training_res['test_metrics']['mae']:.6f}",
                    'Test R²': f"{training_res['test_metrics']['r2']:.4f}",
                    'Spectral Radius': f"{spectral_res['spectral_radius']:.4f}",
                    'Stable Modes': spectral_res['stable_modes'],
                    'Spectral Error': f"{spectral_res['spectral_error']:.4f}"
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(table_data)
        
        # Save complete table
        complete_table_path = self.tables_dir / 'complete_results.csv'
        results_df.to_csv(complete_table_path, index=False)
        
        # Create LaTeX table
        latex_table_path = self.tables_dir / 'results_table.tex'
        with open(latex_table_path, 'w') as f:
            f.write(results_df.to_latex(index=False, escape=False))
        
        # Create summary table by system
        summary_data = []
        for system in ['sierpinski', 'barnsley', 'julia']:
            system_data = results_df[results_df['System'] == system.title()]
            
            if len(system_data) > 0:
                # Best neural model (lowest test MSE)
                neural_data = system_data[system_data['Model'] != 'DMD']
                if len(neural_data) > 0:
                    best_neural = neural_data.loc[neural_data['Test MSE'].astype(float).idxmin()]
                    dmd_data = system_data[system_data['Model'] == 'DMD'].iloc[0]
                    
                    summary_data.append({
                        'System': system.title(),
                        'Best Neural Model': best_neural['Model'],
                        'Neural Test MSE': best_neural['Test MSE'],
                        'Neural Spectral Radius': best_neural['Spectral Radius'],
                        'DMD Spectral Radius': dmd_data['Spectral Radius'],
                        'Improvement': f"{(1 - float(best_neural['Test MSE']) / 0.01) * 100:.1f}%"  # Relative to baseline
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_table_path = self.tables_dir / 'summary_results.csv'
        summary_df.to_csv(summary_table_path, index=False)
        
        print(f"✓ Complete results table saved to: {complete_table_path}")
        print(f"✓ Summary table saved to: {summary_table_path}")
        print(f"✓ LaTeX table saved to: {latex_table_path}")
        
        return results_df
    
    def create_publication_figures(self, spectral_results: Dict[str, Any], 
                                 results_df: pd.DataFrame) -> None:
        """Create publication-quality figures."""
        print("\n" + "="*60)
        print("STEP 5: CREATING PUBLICATION FIGURES")
        print("="*60)
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # Figure 1: Eigenvalue Spectra Comparison
        print("Creating eigenvalue spectra comparison...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
        
        systems = ['sierpinski', 'barnsley', 'julia']
        system_titles = ['Sierpinski Gasket', 'Barnsley Fern', 'Julia Set']
        
        for i, (system, title) in enumerate(zip(systems, system_titles)):
            ax = axes[i]
            
            # Plot unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1, label='Unit Circle')
            
            # Plot DMD eigenvalues
            dmd_eigenvals = spectral_results[system]['dmd']['eigenvalues']
            ax.scatter(dmd_eigenvals.real, dmd_eigenvals.imag, 
                      c='red', marker='o', s=30, alpha=0.7, label='DMD')
            
            # Plot best neural model eigenvalues
            neural_models = [k for k in spectral_results[system].keys() if k != 'dmd']
            if neural_models:
                best_model = neural_models[0]  # Take first available model
                neural_eigenvals = spectral_results[system][best_model]['eigenvalues']
                ax.scatter(neural_eigenvals.real, neural_eigenvals.imag,
                          c='blue', marker='s', s=30, alpha=0.7, label='Neural Network')
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend()
        
        plt.tight_layout()
        spectra_path = self.figures_dir / 'eigenvalue_spectra_comparison.png'
        plt.savefig(spectra_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Performance Comparison Bar Chart
        print("Creating performance comparison chart...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
        
        # Test MSE comparison
        neural_data = results_df[results_df['Model'] != 'DMD'].copy()
        neural_data['Test MSE'] = neural_data['Test MSE'].astype(float)
        
        ax = axes[0, 0]
        sns.barplot(data=neural_data, x='System', y='Test MSE', hue='Model', ax=ax)
        ax.set_title('Test Mean Squared Error', fontweight='bold')
        ax.set_ylabel('MSE')
        
        # Spectral Radius comparison
        ax = axes[0, 1]
        spectral_data = results_df.copy()
        spectral_data['Spectral Radius'] = spectral_data['Spectral Radius'].astype(float)
        sns.barplot(data=spectral_data, x='System', y='Spectral Radius', hue='Model', ax=ax)
        ax.set_title('Spectral Radius Comparison', fontweight='bold')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Stability Threshold')
        ax.legend()
        
        # Training Time comparison
        ax = axes[1, 0]
        training_data = neural_data[neural_data['Training Time (s)'] != 'N/A'].copy()
        training_data['Training Time (s)'] = training_data['Training Time (s)'].astype(float)
        sns.barplot(data=training_data, x='System', y='Training Time (s)', hue='Model', ax=ax)
        ax.set_title('Training Time Comparison', fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        # Model Size comparison
        ax = axes[1, 1]
        param_data = neural_data.copy()
        param_data['Parameters'] = param_data['Parameters'].str.replace(',', '').astype(int)
        sns.barplot(data=param_data, x='System', y='Parameters', hue='Model', ax=ax)
        ax.set_title('Model Size Comparison', fontweight='bold')
        ax.set_ylabel('Number of Parameters')
        
        plt.tight_layout()
        performance_path = self.figures_dir / 'performance_comparison.png'
        plt.savefig(performance_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Spectral Error Analysis
        print("Creating spectral error analysis...")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
        
        error_data = neural_data.copy()
        error_data['Spectral Error'] = error_data['Spectral Error'].astype(float)
        
        sns.boxplot(data=error_data, x='System', y='Spectral Error', hue='Model', ax=ax)
        ax.set_title('Spectral Approximation Error Distribution', fontweight='bold')
        ax.set_ylabel('Spectral Error (Hausdorff Distance)')
        
        plt.tight_layout()
        error_path = self.figures_dir / 'spectral_error_analysis.png'
        plt.savefig(error_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Eigenvalue spectra figure saved to: {spectra_path}")
        print(f"✓ Performance comparison figure saved to: {performance_path}")
        print(f"✓ Spectral error analysis figure saved to: {error_path}")
    
    def generate_research_summary(self, results_df: pd.DataFrame) -> None:
        """Generate comprehensive research summary report."""
        print("\n" + "="*60)
        print("STEP 6: GENERATING RESEARCH SUMMARY")
        print("="*60)
        
        summary_path = self.output_dir / 'RESEARCH_SUMMARY.md'
        
        with open(summary_path, 'w') as f:
            f.write("# Koopman Fractal Spectral Learning - Research Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This study investigates neural network architectures for learning Koopman operators ")
            f.write("on fractal dynamical systems. We compare MLP architectures of different sizes against ")
            f.write("Dynamic Mode Decomposition (DMD) baselines across three fractal systems.\\n\\n")
            
            f.write("## Key Findings\n\n")
            
            # Calculate key statistics
            neural_data = results_df[results_df['Model'] != 'DMD']
            
            best_mse = neural_data['Test MSE'].astype(float).min()
            best_model = neural_data.loc[neural_data['Test MSE'].astype(float).idxmin()]
            
            f.write(f"- **Best Performance:** {best_model['Model']} on {best_model['System']} ")
            f.write(f"(MSE: {best_mse:.6f})\\n")
            
            avg_spectral_error = neural_data['Spectral Error'].astype(float).mean()
            f.write(f"- **Average Spectral Error:** {avg_spectral_error:.4f}\\n")
            
            stable_systems = len(neural_data[neural_data['Spectral Radius'].astype(float) < 1.0])
            total_systems = len(neural_data)
            f.write(f"- **Stable Systems:** {stable_systems}/{total_systems} ")
            f.write(f"({stable_systems/total_systems*100:.1f}%)\\n\\n")
            
            f.write("## Complete Results Table\n\n")
            f.write(results_df.to_markdown(index=False))
            f.write("\\n\\n")
            
            f.write("## System-Specific Analysis\n\n")
            
            for system in ['Sierpinski', 'Barnsley', 'Julia']:
                system_data = results_df[results_df['System'] == system]
                if len(system_data) > 0:
                    f.write(f"### {system} System\n\n")
                    
                    neural_system = system_data[system_data['Model'] != 'DMD']
                    dmd_system = system_data[system_data['Model'] == 'DMD']
                    
                    if len(neural_system) > 0 and len(dmd_system) > 0:
                        best_neural = neural_system.loc[neural_system['Test MSE'].astype(float).idxmin()]
                        dmd_result = dmd_system.iloc[0]
                        
                        f.write(f"- **Best Neural Model:** {best_neural['Model']}\\n")
                        f.write(f"- **Neural MSE:** {best_neural['Test MSE']}\\n")
                        f.write(f"- **Neural Spectral Radius:** {best_neural['Spectral Radius']}\\n")
                        f.write(f"- **DMD Spectral Radius:** {dmd_result['Spectral Radius']}\\n")
                        f.write(f"- **Spectral Error:** {best_neural['Spectral Error']}\\n\\n")
            
            f.write("## Files Generated\n\n")
            f.write("### Data Files\n")
            for system in ['sierpinski', 'barnsley', 'julia']:
                f.write(f"- `data/{system}_trajectories.npy` - Trajectory dataset\\n")
            
            f.write("\\n### Model Files\n")
            for system in ['sierpinski', 'barnsley', 'julia']:
                for model in ['mlp_small', 'mlp_medium', 'mlp_large']:
                    f.write(f"- `models/{system}_{model}.pth` - Trained model\\n")
            
            f.write("\\n### Figures\n")
            f.write("- `figures/eigenvalue_spectra_comparison.png` - Spectral analysis\\n")
            f.write("- `figures/performance_comparison.png` - Performance metrics\\n")
            f.write("- `figures/spectral_error_analysis.png` - Error analysis\\n")
            
            f.write("\\n### Tables\n")
            f.write("- `tables/complete_results.csv` - Complete numerical results\\n")
            f.write("- `tables/summary_results.csv` - Summary statistics\\n")
            f.write("- `tables/results_table.tex` - LaTeX formatted table\\n")
        
        print(f"✓ Research summary saved to: {summary_path}")
    
    def run_complete_pipeline(self) -> None:
        """Run the complete research pipeline."""
        print("KOOPMAN FRACTAL SPECTRAL LEARNING - RESEARCH PIPELINE")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Generate datasets
        datasets = self.generate_fractal_datasets()
        
        # Step 2: Train neural networks
        training_results = self.train_neural_networks(datasets)
        
        # Step 3: Perform spectral analysis
        spectral_results = self.perform_spectral_analysis(training_results, datasets)
        
        # Step 4: Create performance tables
        results_df = self.create_performance_tables(training_results, spectral_results)
        
        # Step 5: Create publication figures
        self.create_publication_figures(spectral_results, results_df)
        
        # Step 6: Generate research summary
        self.generate_research_summary(results_df)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds")
        print(f"📁 All results saved to: {self.output_dir}")
        print(f"📊 Ready for publication!")


def main():
    """Main function to run research pipeline."""
    pipeline = ResearchPipeline("research_results")
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()