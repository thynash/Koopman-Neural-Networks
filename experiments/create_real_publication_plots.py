#!/usr/bin/env python3
"""
Real Publication Plots - Using Actual Data and Results

Creates high-quality visualizations based on:
- ACTUAL trained models (MLP and DeepONet only)
- REAL fractal data from generators
- ACTUAL results from Run 2
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.generators.ifs_generator import SierpinskiGasketGenerator, BarnsleyFernGenerator
from data.generators.julia_generator import JuliaSetGenerator

# Publication style
plt.style.use('seaborn-v0_8')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def create_fractal_attractors():
    """Create beautiful fractal attractor visualizations with REAL data."""
    print("Creating fractal attractor visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
    
    # Sierpinski Gasket
    print("  Generating Sierpinski Gasket...")
    sierpinski_gen = SierpinskiGasketGenerator({'seed': 42})
    sierpinski_data = sierpinski_gen.generate_trajectories(n_points=50000)
    states_s = sierpinski_data.states
    
    axes[0].scatter(states_s[:, 0], states_s[:, 1], s=0.3, alpha=0.6, c='#2E86AB', edgecolors='none')
    axes[0].set_title('Sierpinski Gasket\n(50,000 points)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_aspect('equal')
    axes[0].set_facecolor('#F8F9FA')
    
    # Barnsley Fern
    print("  Generating Barnsley Fern...")
    barnsley_gen = BarnsleyFernGenerator({'seed': 42})
    barnsley_data = barnsley_gen.generate_trajectories(n_points=50000)
    states_b = barnsley_data.states
    
    axes[1].scatter(states_b[:, 0], states_b[:, 1], s=0.3, alpha=0.6, c='#06A77D', edgecolors='none')
    axes[1].set_title('Barnsley Fern\n(50,000 points)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_aspect('equal')
    axes[1].set_facecolor('#F8F9FA')
    
    # Julia Set
    print("  Generating Julia Set...")
    julia_gen = JuliaSetGenerator({
        'c_real': -0.7269, 'c_imag': 0.1889,
        'max_iter': 1000, 'escape_radius': 2.0, 'seed': 42
    })
    julia_data = julia_gen.generate_trajectories(n_points=30000)
    states_j = julia_data.states
    
    axes[2].scatter(states_j[:, 0], states_j[:, 1], s=0.5, alpha=0.7, c='#A23B72', edgecolors='none')
    axes[2].set_title('Julia Set\n(c = -0.7269 + 0.1889i)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Real', fontsize=12)
    axes[2].set_ylabel('Imaginary', fontsize=12)
    axes[2].set_aspect('equal')
    axes[2].set_facecolor('#F8F9FA')
    
    plt.suptitle('Fractal Dynamical Systems', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path("research_results_run2/publication_figures/figure1_fractal_attractors.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def create_performance_comparison():
    """Create performance comparison using REAL results."""
    print("Creating performance comparison...")
    
    # Load REAL results
    results_df = pd.read_csv("research_results_run2/tables/comprehensive_results_run2.csv")
    
    # Filter to only successful neural models (no DMD, no failed)
    neural_results = results_df[
        (results_df['Model'] != 'DMD') &
        (results_df['Test MSE'] != 'FAILED') &
        (results_df['Test MSE'] != 'N/A')
    ].copy()
    
    neural_results['Test MSE'] = neural_results['Test MSE'].astype(float)
    neural_results['Test R²'] = neural_results['Test R²'].astype(float)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=600)
    
    # MSE by Architecture and System
    mlp_data = neural_results[neural_results['Architecture'] == 'MLP']
    deeponet_data = neural_results[neural_results['Architecture'] == 'DEEPONET']
    
    systems = ['Sierpinski Gasket', 'Barnsley Fern', 'Julia Set']
    x = np.arange(len(systems))
    width = 0.35
    
    mlp_means = [mlp_data[mlp_data['System'] == sys]['Test MSE'].mean() for sys in systems]
    deeponet_means = [deeponet_data[deeponet_data['System'] == sys]['Test MSE'].mean() for sys in systems]
    
    bars1 = axes[0, 0].bar(x - width/2, mlp_means, width, label='MLP', color='#2E86AB', alpha=0.8)
    bars2 = axes[0, 0].bar(x + width/2, deeponet_means, width, label='DeepONet', color='#06A77D', alpha=0.8)
    
    axes[0, 0].set_ylabel('Average Test MSE', fontsize=11)
    axes[0, 0].set_title('Test MSE by Architecture', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(systems, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # R² by Architecture and System
    mlp_r2 = [mlp_data[mlp_data['System'] == sys]['Test R²'].mean() for sys in systems]
    deeponet_r2 = [deeponet_data[deeponet_data['System'] == sys]['Test R²'].mean() for sys in systems]
    
    bars3 = axes[0, 1].bar(x - width/2, mlp_r2, width, label='MLP', color='#2E86AB', alpha=0.8)
    bars4 = axes[0, 1].bar(x + width/2, deeponet_r2, width, label='DeepONet', color='#06A77D', alpha=0.8)
    
    axes[0, 1].set_ylabel('Average Test R²', fontsize=11)
    axes[0, 1].set_title('Test R² by Architecture', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(systems, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # Box plot of MSE distribution
    sns.boxplot(data=neural_results, x='System', y='Test MSE', hue='Architecture',
               palette={'MLP': '#2E86AB', 'DEEPONET': '#06A77D'}, ax=axes[1, 0])
    axes[1, 0].set_title('MSE Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('Test MSE', fontsize=11)
    axes[1, 0].tick_params(axis='x', rotation=15)
    axes[1, 0].set_yscale('log')
    
    # Training time comparison
    neural_results['Training Time (s)'] = neural_results['Training Time (s)'].astype(float)
    
    mlp_time = [mlp_data[mlp_data['System'] == sys]['Training Time (s)'].mean() for sys in systems]
    deeponet_time = [deeponet_data[deeponet_data['System'] == sys]['Training Time (s)'].mean() for sys in systems]
    
    bars5 = axes[1, 1].bar(x - width/2, mlp_time, width, label='MLP', color='#2E86AB', alpha=0.8)
    bars6 = axes[1, 1].bar(x + width/2, deeponet_time, width, label='DeepONet', color='#06A77D', alpha=0.8)
    
    axes[1, 1].set_ylabel('Average Training Time (s)', fontsize=11)
    axes[1, 1].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(systems, rotation=15, ha='right')
    axes[1, 1].legend()
    
    plt.suptitle('Architecture Performance Comparison (MLP vs DeepONet)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path("research_results_run2/publication_figures/figure2_performance_comparison.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def create_spectral_analysis():
    """Create spectral analysis plots using REAL results."""
    print("Creating spectral analysis...")
    
    results_df = pd.read_csv("research_results_run2/tables/comprehensive_results_run2.csv")
    
    # Filter successful results
    successful = results_df[
        (results_df['Test MSE'] != 'FAILED') &
        (results_df['Test MSE'] != 'N/A')
    ].copy()
    
    successful['Spectral Radius'] = successful['Spectral Radius'].astype(float)
    successful['Spectral Error'] = pd.to_numeric(successful['Spectral Error'], errors='coerce')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=600)
    
    # Spectral Radius by Architecture
    neural_data = successful[successful['Model'] != 'DMD']
    
    sns.boxplot(data=neural_data, x='System', y='Spectral Radius', hue='Architecture',
               palette={'MLP': '#2E86AB', 'DEEPONET': '#06A77D'}, ax=axes[0, 0])
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stability Threshold')
    axes[0, 0].set_title('Spectral Radius Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].legend()
    
    # Stable Modes
    sns.barplot(data=neural_data, x='System', y='Stable Modes', hue='Architecture',
               palette={'MLP': '#2E86AB', 'DEEPONET': '#06A77D'}, ax=axes[0, 1])
    axes[0, 1].set_title('Number of Stable Modes', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=15)
    
    # Spectral Error
    spectral_error_data = neural_data[neural_data['Spectral Error'].notna()]
    
    sns.violinplot(data=spectral_error_data, x='Architecture', y='Spectral Error',
                  palette={'MLP': '#2E86AB', 'DEEPONET': '#06A77D'}, ax=axes[1, 0])
    axes[1, 0].set_title('Spectral Approximation Error', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Architecture', fontsize=11)
    
    # Spectral Radius vs MSE
    neural_data['Test MSE'] = neural_data['Test MSE'].astype(float)
    
    for arch, color in [('MLP', '#2E86AB'), ('DEEPONET', '#06A77D')]:
        arch_data = neural_data[neural_data['Architecture'] == arch]
        axes[1, 1].scatter(arch_data['Spectral Radius'], arch_data['Test MSE'],
                          s=100, alpha=0.7, label=arch, c=color, edgecolors='black', linewidth=0.5)
    
    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Spectral Radius vs Prediction Error', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Spectral Radius', fontsize=11)
    axes[1, 1].set_ylabel('Test MSE', fontsize=11)
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    
    plt.suptitle('Spectral Properties Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path("research_results_run2/publication_figures/figure3_spectral_analysis.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def create_detailed_results_table():
    """Create a detailed visual results table."""
    print("Creating detailed results table...")
    
    results_df = pd.read_csv("research_results_run2/tables/comprehensive_results_run2.csv")
    
    # Filter to best models per system
    neural_results = results_df[
        (results_df['Model'] != 'DMD') &
        (results_df['Test MSE'] != 'FAILED') &
        (results_df['Test MSE'] != 'N/A')
    ].copy()
    
    neural_results['Test MSE'] = neural_results['Test MSE'].astype(float)
    
    # Get best model per system
    best_models = []
    for system in neural_results['System'].unique():
        system_data = neural_results[neural_results['System'] == system]
        best_model = system_data.loc[system_data['Test MSE'].idxmin()]
        best_models.append(best_model)
    
    best_df = pd.DataFrame(best_models)
    
    fig, ax = plt.subplots(figsize=(16, 6), dpi=600)
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns to display
    display_cols = ['System', 'Model', 'Architecture', 'Test MSE', 'Test R²', 
                   'Spectral Radius', 'Training Time (s)', 'Parameters']
    table_data = best_df[display_cols].values
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=display_cols,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12, 0.12, 0.12, 0.10, 0.13, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(best_models) + 1):
        for j in range(len(display_cols)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Best Model Performance by System', fontsize=14, fontweight='bold', pad=20)
    
    save_path = Path("research_results_run2/publication_figures/figure4_best_models_table.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def main():
    """Generate all real publication plots."""
    print("\n" + "="*80)
    print("CREATING REAL PUBLICATION PLOTS")
    print("Using actual data and trained model results")
    print("="*80 + "\n")
    
    create_fractal_attractors()
    create_performance_comparison()
    create_spectral_analysis()
    create_detailed_results_table()
    
    print("\n" + "="*80)
    print("✅ ALL REAL PUBLICATION PLOTS CREATED SUCCESSFULLY!")
    print("📁 Location: research_results_run2/publication_figures/")
    print("="*80)


if __name__ == '__main__':
    main()