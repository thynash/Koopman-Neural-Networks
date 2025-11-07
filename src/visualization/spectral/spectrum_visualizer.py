"""
Spectral analysis visualization module with LaTeX mathematical symbols.

This module provides specialized visualization functions for eigenvalue spectra,
eigenfunctions, and spectral comparisons with publication-quality formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.gridspec as gridspec


class SpectrumVisualizer:
    """
    Specialized visualizer for eigenvalue spectra and eigenfunctions.
    """
    
    def __init__(self, output_dir: str = "figures/spectral", dpi: int = 600):
        """
        Initialize spectrum visualizer.
        
        Args:
            output_dir: Directory to save spectral figures
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = max(dpi, 600)
        
        # Configure matplotlib for LaTeX-style mathematical symbols
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
        
        # Color scheme for different models
        self.colors = {
            'MLP': '#1f77b4',
            'DeepONet': '#ff7f0e', 
            'LSTM': '#2ca02c',
            'DMD': '#d62728',
            'Reference': '#9467bd',
            'Analytical': '#8c564b'
        }
        
        self.markers = {
            'MLP': 'o',
            'DeepONet': 's',
            'LSTM': '^',
            'DMD': 'x',
            'Reference': 'D',
            'Analytical': '*'
        }
    
    def plot_eigenvalue_spectrum(self,
                               eigenvalues_dict: Dict[str, np.ndarray],
                               title: str = "Eigenvalue Spectrum in Complex Plane",
                               save_name: Optional[str] = None,
                               include_unit_circle: bool = True,
                               highlight_dominant: bool = True) -> str:
        """
        Plot eigenvalue spectrum in complex plane with LaTeX formatting.
        
        Args:
            eigenvalues_dict: Dict {model_name: eigenvalue_array}
            title: Plot title
            save_name: Custom filename (optional)
            include_unit_circle: Whether to show unit circle
            highlight_dominant: Whether to highlight dominant eigenvalues
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Plot unit circle for stability reference
        if include_unit_circle:
            circle = patches.Circle((0, 0), 1, fill=False, color='gray',
                                  linestyle='--', alpha=0.6, linewidth=2,
                                  label='Unit Circle')
            ax.add_patch(circle)
        
        # Plot eigenvalues for each model
        for model_name, eigenvals in eigenvalues_dict.items():
            color = self.colors.get(model_name, '#333333')
            marker = self.markers.get(model_name, 'o')
            
            real_parts = np.real(eigenvals)
            imag_parts = np.imag(eigenvals)
            
            # Plot all eigenvalues
            scatter = ax.scatter(real_parts, imag_parts, 
                               color=color, marker=marker, s=80, alpha=0.7,
                               label=model_name, edgecolors='black', linewidth=0.8)
            
            # Highlight dominant eigenvalues (largest magnitude)
            if highlight_dominant and len(eigenvals) > 0:
                magnitudes = np.abs(eigenvals)
                dominant_idx = np.argmax(magnitudes)
                ax.scatter(real_parts[dominant_idx], imag_parts[dominant_idx],
                          color=color, marker=marker, s=200, alpha=1.0,
                          edgecolors='red', linewidth=3)
        
        # Formatting with LaTeX symbols
        ax.set_xlabel(r'$\mathrm{Re}(\lambda)$', fontsize=18, fontweight='bold')
        ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        # Add coordinate axes through origin
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=1)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=1)
        
        # Set equal aspect ratio and grid
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=14)
        
        # Legend
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True,
                 loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "eigenvalue_spectrum"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_spectral_comparison_with_reference(self,
                                              learned_eigenvals: Dict[str, np.ndarray],
                                              reference_eigenvals: np.ndarray,
                                              reference_name: str = "DMD Reference",
                                              title: str = "Learned vs Reference Spectrum",
                                              save_name: Optional[str] = None) -> str:
        """
        Compare learned eigenvalues against reference spectrum.
        
        Args:
            learned_eigenvals: Dict of learned eigenvalues from different models
            reference_eigenvals: Reference eigenvalues (e.g., from DMD)
            reference_name: Name for reference method
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Plot reference eigenvalues with special styling
        ref_real = np.real(reference_eigenvals)
        ref_imag = np.imag(reference_eigenvals)
        ax.scatter(ref_real, ref_imag, 
                  color='black', marker='x', s=150, linewidth=4,
                  label=reference_name, alpha=0.9, zorder=10)
        
        # Plot learned eigenvalues
        for model_name, eigenvals in learned_eigenvals.items():
            color = self.colors.get(model_name, '#333333')
            marker = self.markers.get(model_name, 'o')
            
            real_parts = np.real(eigenvals)
            imag_parts = np.imag(eigenvals)
            
            ax.scatter(real_parts, imag_parts, 
                      color=color, marker=marker, s=80, alpha=0.7,
                      label=f'{model_name} (Learned)', 
                      edgecolors='black', linewidth=0.8)
        
        # Add unit circle
        circle = patches.Circle((0, 0), 1, fill=False, color='gray',
                              linestyle='--', alpha=0.6, linewidth=2)
        ax.add_patch(circle)
        
        # Formatting
        ax.set_xlabel(r'$\mathrm{Re}(\lambda)$', fontsize=18, fontweight='bold')
        ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        # Add coordinate axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=1)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=1)
        
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "spectral_comparison_reference"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_eigenfunction_visualization(self,
                                       eigenfunction_data: np.ndarray,
                                       coordinates: np.ndarray,
                                       eigenvalue: complex,
                                       title: Optional[str] = None,
                                       save_name: Optional[str] = None) -> str:
        """
        Visualize eigenfunction with real and imaginary components.
        
        Args:
            eigenfunction_data: Complex eigenfunction values
            coordinates: Spatial coordinates (x, y)
            eigenvalue: Corresponding eigenvalue
            title: Custom title (optional)
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 8), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 0.05])
        
        # Real part
        ax1 = fig.add_subplot(gs[0, 0])
        real_data = np.real(eigenfunction_data)
        scatter1 = ax1.scatter(coordinates[:, 0], coordinates[:, 1], 
                              c=real_data, cmap='RdBu_r', s=25, alpha=0.8)
        ax1.set_title(r'$\mathrm{Re}(\phi)$', fontsize=16, fontweight='bold')
        ax1.set_xlabel(r'$x$', fontsize=14)
        ax1.set_ylabel(r'$y$', fontsize=14)
        ax1.set_aspect('equal', adjustable='box')
        ax1.tick_params(labelsize=12)
        
        # Imaginary part
        ax2 = fig.add_subplot(gs[0, 1])
        imag_data = np.imag(eigenfunction_data)
        scatter2 = ax2.scatter(coordinates[:, 0], coordinates[:, 1], 
                              c=imag_data, cmap='RdBu_r', s=25, alpha=0.8)
        ax2.set_title(r'$\mathrm{Im}(\phi)$', fontsize=16, fontweight='bold')
        ax2.set_xlabel(r'$x$', fontsize=14)
        ax2.set_ylabel(r'$y$', fontsize=14)
        ax2.set_aspect('equal', adjustable='box')
        ax2.tick_params(labelsize=12)
        
        # Magnitude
        ax3 = fig.add_subplot(gs[0, 2])
        magnitude_data = np.abs(eigenfunction_data)
        scatter3 = ax3.scatter(coordinates[:, 0], coordinates[:, 1], 
                              c=magnitude_data, cmap='viridis', s=25, alpha=0.8)
        ax3.set_title(r'$|\phi|$', fontsize=16, fontweight='bold')
        ax3.set_xlabel(r'$x$', fontsize=14)
        ax3.set_ylabel(r'$y$', fontsize=14)
        ax3.set_aspect('equal', adjustable='box')
        ax3.tick_params(labelsize=12)
        
        # Colorbars
        cbar1 = fig.add_subplot(gs[1, 0])
        plt.colorbar(scatter1, cax=cbar1, orientation='horizontal')
        cbar1.set_xlabel(r'$\mathrm{Re}(\phi)$', fontsize=12)
        
        cbar2 = fig.add_subplot(gs[1, 1])
        plt.colorbar(scatter2, cax=cbar2, orientation='horizontal')
        cbar2.set_xlabel(r'$\mathrm{Im}(\phi)$', fontsize=12)
        
        cbar3 = fig.add_subplot(gs[1, 2])
        plt.colorbar(scatter3, cax=cbar3, orientation='horizontal')
        cbar3.set_xlabel(r'$|\phi|$', fontsize=12)
        
        # Overall title with eigenvalue
        if title is None:
            eigenval_str = f'$\\lambda = {eigenvalue.real:.4f} + {eigenvalue.imag:.4f}i$'
            title = f'Eigenfunction Visualization: {eigenval_str}'
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = f"eigenfunction_lambda_{eigenvalue.real:.4f}_{eigenvalue.imag:.4f}".replace('-', 'neg')
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_spectral_error_analysis(self,
                                   error_data: Dict[str, Dict[str, float]],
                                   title: str = "Spectral Approximation Error Analysis",
                                   save_name: Optional[str] = None) -> str:
        """
        Create comprehensive spectral error analysis visualization.
        
        Args:
            error_data: Nested dict {model_name: {error_type: value}}
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        error_types = list(next(iter(error_data.values())).keys())
        n_errors = len(error_types)
        
        fig, axes = plt.subplots(1, n_errors, figsize=(6*n_errors, 8), dpi=self.dpi)
        
        if n_errors == 1:
            axes = [axes]
        
        model_names = list(error_data.keys())
        x_pos = np.arange(len(model_names))
        
        for i, error_type in enumerate(error_types):
            values = [error_data[model][error_type] for model in model_names]
            colors = [self.colors.get(model, '#333333') for model in model_names]
            
            bars = axes[i].bar(x_pos, values, color=colors, alpha=0.8,
                             edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2e}', ha='center', va='bottom', fontsize=11,
                           fontweight='bold')
            
            axes[i].set_title(error_type.replace('_', ' ').title(), 
                            fontsize=16, fontweight='bold')
            axes[i].set_xlabel('Model', fontsize=14)
            axes[i].set_ylabel('Error', fontsize=14)
            axes[i].set_yscale('log')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].tick_params(labelsize=12)
        
        plt.suptitle(title, fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "spectral_error_analysis"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)