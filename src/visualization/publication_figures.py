"""
Publication-ready figure generation module.

This module provides high-resolution figure generation capabilities for:
- Fractal attractor visualizations at 600+ dpi
- Training curve plots with proper formatting and legends
- Eigenvalue spectrum plots with LaTeX mathematical symbols
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path

# Configure matplotlib for publication-quality figures
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 18
rcParams['text.usetex'] = False  # Set to True if LaTeX is available
rcParams['mathtext.fontset'] = 'cm'


class PublicationFigureGenerator:
    """
    Generates publication-ready figures with consistent formatting and high resolution.
    """
    
    def __init__(self, 
                 output_dir: str = "figures",
                 dpi: int = 600,
                 figsize: Tuple[float, float] = (10, 8)):
        """
        Initialize the figure generator.
        
        Args:
            output_dir: Directory to save figures
            dpi: Resolution for saved figures (minimum 600)
            figsize: Default figure size in inches
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = max(dpi, 600)  # Ensure minimum 600 dpi
        self.figsize = figsize
        
        # Color schemes for consistent visualization
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'accent': '#9467bd',
            'neutral': '#7f7f7f'
        }
        
        self.line_styles = ['-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'v', 'D', 'p']
    
    def create_fractal_attractor_plot(self,
                                    trajectory_data: np.ndarray,
                                    title: str,
                                    system_name: str,
                                    save_name: Optional[str] = None,
                                    colormap: str = 'viridis',
                                    point_size: float = 0.1) -> str:
        """
        Create high-resolution fractal attractor visualization.
        
        Args:
            trajectory_data: Array of shape (n_points, 2) containing trajectory points
            title: Plot title
            system_name: Name of the fractal system
            save_name: Custom filename (optional)
            colormap: Matplotlib colormap name
            point_size: Size of plotted points
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create scatter plot with color gradient based on iteration order
        colors = np.arange(len(trajectory_data))
        scatter = ax.scatter(trajectory_data[:, 0], trajectory_data[:, 1], 
                           c=colors, cmap=colormap, s=point_size, alpha=0.7)
        
        # Formatting
        ax.set_title(f'{title}', fontsize=16, fontweight='bold')
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$y$', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Iteration Number', fontsize=12)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = f"{system_name.lower().replace(' ', '_')}_attractor"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def create_training_curves_plot(self,
                                  training_history: Dict[str, List[float]],
                                  model_names: List[str],
                                  title: str = "Training Curves Comparison",
                                  save_name: Optional[str] = None) -> str:
        """
        Create training curve plots with proper formatting and legends.
        
        Args:
            training_history: Dictionary with model names as keys and loss histories as values
            model_names: List of model names for legend
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Plot training loss
        for i, (model_name, history) in enumerate(training_history.items()):
            if 'train_loss' in history:
                color = list(self.colors.values())[i % len(self.colors)]
                line_style = self.line_styles[i % len(self.line_styles)]
                
                ax1.plot(history['train_loss'], 
                        color=color, linestyle=line_style, linewidth=2,
                        label=f'{model_name} (Train)', alpha=0.8)
                
                if 'val_loss' in history:
                    ax1.plot(history['val_loss'], 
                            color=color, linestyle=':', linewidth=2,
                            label=f'{model_name} (Val)', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_title('Training Loss Curves', fontsize=16, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        
        # Plot accuracy if available
        accuracy_plotted = False
        for i, (model_name, history) in enumerate(training_history.items()):
            if 'train_accuracy' in history:
                color = list(self.colors.values())[i % len(self.colors)]
                line_style = self.line_styles[i % len(self.line_styles)]
                
                ax2.plot(history['train_accuracy'], 
                        color=color, linestyle=line_style, linewidth=2,
                        label=f'{model_name} (Train)', alpha=0.8)
                
                if 'val_accuracy' in history:
                    ax2.plot(history['val_accuracy'], 
                            color=color, linestyle=':', linewidth=2,
                            label=f'{model_name} (Val)', alpha=0.8)
                accuracy_plotted = True
        
        if accuracy_plotted:
            ax2.set_xlabel('Epoch', fontsize=14)
            ax2.set_ylabel('Accuracy', fontsize=14)
            ax2.set_title('Training Accuracy Curves', fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(frameon=True, fancybox=True, shadow=True)
        else:
            # If no accuracy data, plot spectral error instead
            for i, (model_name, history) in enumerate(training_history.items()):
                if 'spectral_error' in history:
                    color = list(self.colors.values())[i % len(self.colors)]
                    line_style = self.line_styles[i % len(self.line_styles)]
                    
                    ax2.plot(history['spectral_error'], 
                            color=color, linestyle=line_style, linewidth=2,
                            label=f'{model_name}', alpha=0.8)
            
            ax2.set_xlabel('Epoch', fontsize=14)
            ax2.set_ylabel('Spectral Error', fontsize=14)
            ax2.set_title('Spectral Approximation Error', fontsize=16, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.legend(frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "training_curves_comparison"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath) 
   
    def create_eigenvalue_spectrum_plot(self,
                                      eigenvalues_dict: Dict[str, np.ndarray],
                                      title: str = "Eigenvalue Spectrum Comparison",
                                      save_name: Optional[str] = None,
                                      include_unit_circle: bool = True) -> str:
        """
        Create eigenvalue spectrum plots with LaTeX mathematical symbols.
        
        Args:
            eigenvalues_dict: Dictionary with model names as keys and eigenvalue arrays as values
            title: Plot title
            save_name: Custom filename (optional)
            include_unit_circle: Whether to draw unit circle for reference
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot unit circle for reference
        if include_unit_circle:
            circle = patches.Circle((0, 0), 1, fill=False, color='black', 
                                  linestyle='--', alpha=0.5, linewidth=1)
            ax.add_patch(circle)
        
        # Plot eigenvalues for each model
        for i, (model_name, eigenvals) in enumerate(eigenvalues_dict.items()):
            color = list(self.colors.values())[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            # Separate real and imaginary parts
            real_parts = np.real(eigenvals)
            imag_parts = np.imag(eigenvals)
            
            ax.scatter(real_parts, imag_parts, 
                      color=color, marker=marker, s=60, alpha=0.7,
                      label=model_name, edgecolors='black', linewidth=0.5)
        
        # Formatting with LaTeX symbols
        ax.set_xlabel(r'$\mathrm{Re}(\lambda)$', fontsize=14)
        ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        
        # Add axis lines through origin
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "eigenvalue_spectrum_comparison"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def create_comparative_spectrum_plot(self,
                                       learned_eigenvals: Dict[str, np.ndarray],
                                       reference_eigenvals: np.ndarray,
                                       reference_name: str = "DMD Reference",
                                       title: str = "Learned vs Reference Spectrum",
                                       save_name: Optional[str] = None) -> str:
        """
        Create comparative plot showing learned eigenvalues vs reference (e.g., DMD).
        
        Args:
            learned_eigenvals: Dictionary of learned eigenvalues from different models
            reference_eigenvals: Reference eigenvalues (e.g., from DMD)
            reference_name: Name for reference method
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot reference eigenvalues first
        ref_real = np.real(reference_eigenvals)
        ref_imag = np.imag(reference_eigenvals)
        ax.scatter(ref_real, ref_imag, 
                  color='black', marker='x', s=100, linewidth=3,
                  label=reference_name, alpha=0.8)
        
        # Plot learned eigenvalues
        for i, (model_name, eigenvals) in enumerate(learned_eigenvals.items()):
            color = list(self.colors.values())[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            real_parts = np.real(eigenvals)
            imag_parts = np.imag(eigenvals)
            
            ax.scatter(real_parts, imag_parts, 
                      color=color, marker=marker, s=60, alpha=0.7,
                      label=f'{model_name} (Learned)', 
                      edgecolors='black', linewidth=0.5)
        
        # Add unit circle
        circle = patches.Circle((0, 0), 1, fill=False, color='gray', 
                              linestyle='--', alpha=0.5, linewidth=1)
        ax.add_patch(circle)
        
        # Formatting
        ax.set_xlabel(r'$\mathrm{Re}(\lambda)$', fontsize=14)
        ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add axis lines
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "comparative_spectrum"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def create_performance_comparison_plot(self,
                                         metrics_data: Dict[str, Dict[str, float]],
                                         metrics_to_plot: List[str],
                                         title: str = "Model Performance Comparison",
                                         save_name: Optional[str] = None) -> str:
        """
        Create bar plots comparing model performance across multiple metrics.
        
        Args:
            metrics_data: Nested dict {model_name: {metric_name: value}}
            metrics_to_plot: List of metric names to include in plot
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(metrics_data.keys())
        x_pos = np.arange(len(model_names))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_data[model][metric] for model in model_names]
            
            bars = axes[i].bar(x_pos, values, 
                             color=[list(self.colors.values())[j % len(self.colors)] 
                                   for j in range(len(model_names))],
                             alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            axes[i].set_xlabel('Model', fontsize=12)
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "performance_comparison"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def create_eigenfunction_visualization(self,
                                         eigenfunction_data: np.ndarray,
                                         coordinates: np.ndarray,
                                         eigenvalue: complex,
                                         title: str = "Eigenfunction Visualization",
                                         save_name: Optional[str] = None) -> str:
        """
        Create eigenfunction visualization with proper mathematical formatting.
        
        Args:
            eigenfunction_data: Eigenfunction values at coordinate points
            coordinates: Spatial coordinates (x, y)
            eigenvalue: Corresponding eigenvalue
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Real part
        scatter1 = ax1.scatter(coordinates[:, 0], coordinates[:, 1], 
                              c=np.real(eigenfunction_data), 
                              cmap='RdBu_r', s=20, alpha=0.8)
        ax1.set_title(r'$\mathrm{Re}(\phi)$', fontsize=14, fontweight='bold')
        ax1.set_xlabel(r'$x$', fontsize=12)
        ax1.set_ylabel(r'$y$', fontsize=12)
        ax1.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter1, ax=ax1)
        
        # Imaginary part
        scatter2 = ax2.scatter(coordinates[:, 0], coordinates[:, 1], 
                              c=np.imag(eigenfunction_data), 
                              cmap='RdBu_r', s=20, alpha=0.8)
        ax2.set_title(r'$\mathrm{Im}(\phi)$', fontsize=14, fontweight='bold')
        ax2.set_xlabel(r'$x$', fontsize=12)
        ax2.set_ylabel(r'$y$', fontsize=12)
        ax2.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter2, ax=ax2)
        
        # Add eigenvalue information
        eigenval_str = f'$\\lambda = {eigenvalue.real:.3f} + {eigenvalue.imag:.3f}i$'
        plt.suptitle(f'{title}\n{eigenval_str}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = f"eigenfunction_lambda_{eigenvalue.real:.3f}_{eigenvalue.imag:.3f}"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)