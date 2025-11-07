"""
Eigenfunction visualization utilities for Koopman operator analysis.

This module provides specialized visualization tools for eigenfunctions
extracted from Koopman operators, including 2D spatial plots, temporal
evolution, and comparative visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Optional, Tuple, Any
import warnings

from src.analysis.spectral.spectral_analyzer import SpectralResults


class EigenfunctionVisualizer:
    """
    Specialized visualizer for Koopman operator eigenfunctions.
    
    This class provides methods for creating publication-quality visualizations
    of eigenfunctions in various formats suitable for fractal dynamical systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the eigenfunction visualizer.
        
        Args:
            config: Configuration dictionary containing:
                - figure_size: Default figure size for plots
                - dpi: Resolution for saved figures
                - colormap: Default colormap for visualizations
                - grid_resolution: Resolution for spatial grids
        """
        self.config = config
        self.figure_size = config.get('figure_size', (12, 10))
        self.dpi = config.get('dpi', 300)
        self.colormap = config.get('colormap', 'RdBu')
        self.grid_resolution = config.get('grid_resolution', 100)
    
    def visualize_spatial_eigenfunctions(self, spectrum: SpectralResults,
                                       spatial_domain: np.ndarray,
                                       n_functions: int = 4,
                                       save_path: Optional[str] = None) -> None:
        """
        Visualize eigenfunctions in spatial domain for 2D fractal systems.
        
        Args:
            spectrum: SpectralResults containing eigenvectors
            spatial_domain: Spatial coordinates (N x 2) for evaluation
            n_functions: Number of eigenfunctions to visualize
            save_path: Optional path to save the figure
        """
        if len(spectrum.eigenvectors) == 0:
            warnings.warn("Cannot visualize eigenfunctions: no eigenvectors available")
            return
        
        try:
            n_plot = min(n_functions, spectrum.eigenvectors.shape[1])
            
            # Create subplot grid
            rows = int(np.ceil(n_plot / 2))
            cols = 2 if n_plot > 1 else 1
            
            fig, axes = plt.subplots(rows, cols, figsize=self.figure_size, dpi=self.dpi)
            if n_plot == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i in range(n_plot):
                eigenfunction = spectrum.eigenvectors[:, i]
                eigenvalue = spectrum.eigenvalues[i]
                
                # Handle complex eigenfunctions by plotting magnitude
                if np.iscomplexobj(eigenfunction):
                    plot_values = np.abs(eigenfunction)
                    title_suffix = " (Magnitude)"
                else:
                    plot_values = eigenfunction
                    title_suffix = ""
                
                # Ensure plot_values matches spatial_domain length
                if len(plot_values) != len(spatial_domain):
                    # Truncate or pad to match spatial domain
                    min_len = min(len(plot_values), len(spatial_domain))
                    plot_values = plot_values[:min_len]
                    spatial_domain_subset = spatial_domain[:min_len]
                else:
                    spatial_domain_subset = spatial_domain
                
                # Create scatter plot
                scatter = axes[i].scatter(
                    spatial_domain_subset[:, 0], 
                    spatial_domain_subset[:, 1],
                    c=plot_values,
                    cmap=self.colormap,
                    s=20,
                    alpha=0.7,
                    edgecolors='none'
                )
                
                # Formatting
                axes[i].set_title(f'Eigenfunction {i+1}{title_suffix}\nλ = {eigenvalue:.4f}', 
                                fontsize=12, fontweight='bold')
                axes[i].set_xlabel('x', fontsize=10)
                axes[i].set_ylabel('y', fontsize=10)
                axes[i].set_aspect('equal')
                axes[i].grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[i], shrink=0.8)
            
            # Hide unused subplots
            for i in range(n_plot, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"Spatial eigenfunction visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating spatial eigenfunction visualization: {str(e)}")
    
    def visualize_eigenfunction_components(self, spectrum: SpectralResults,
                                         eigenfunction_idx: int = 0,
                                         save_path: Optional[str] = None) -> None:
        """
        Visualize real and imaginary components of a complex eigenfunction.
        
        Args:
            spectrum: SpectralResults containing eigenvectors
            eigenfunction_idx: Index of eigenfunction to visualize
            save_path: Optional path to save the figure
        """
        if eigenfunction_idx >= spectrum.eigenvectors.shape[1]:
            warnings.warn(f"Eigenfunction index {eigenfunction_idx} out of range")
            return
        
        try:
            eigenfunction = spectrum.eigenvectors[:, eigenfunction_idx]
            eigenvalue = spectrum.eigenvalues[eigenfunction_idx]
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
            
            # Real part
            axes[0, 0].plot(np.real(eigenfunction), 'b-', linewidth=2)
            axes[0, 0].set_title('Real Part', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Component Index')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Imaginary part
            axes[0, 1].plot(np.imag(eigenfunction), 'r-', linewidth=2)
            axes[0, 1].set_title('Imaginary Part', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Component Index')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Magnitude
            axes[1, 0].plot(np.abs(eigenfunction), 'g-', linewidth=2)
            axes[1, 0].set_title('Magnitude', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Component Index')
            axes[1, 0].set_ylabel('|Value|')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Phase
            axes[1, 1].plot(np.angle(eigenfunction), 'm-', linewidth=2)
            axes[1, 1].set_title('Phase', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Component Index')
            axes[1, 1].set_ylabel('Phase (radians)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Overall title
            fig.suptitle(f'Eigenfunction {eigenfunction_idx+1} Components\nλ = {eigenvalue:.4f}', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"Eigenfunction components visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating eigenfunction components visualization: {str(e)}")
    
    def create_eigenfunction_grid_plot(self, spectrum: SpectralResults,
                                     grid_bounds: Tuple[float, float, float, float],
                                     eigenfunction_idx: int = 0,
                                     save_path: Optional[str] = None) -> None:
        """
        Create a grid-based visualization of an eigenfunction over a spatial domain.
        
        Args:
            spectrum: SpectralResults containing eigenvectors
            grid_bounds: (xmin, xmax, ymin, ymax) for the spatial grid
            eigenfunction_idx: Index of eigenfunction to visualize
            save_path: Optional path to save the figure
        """
        if eigenfunction_idx >= spectrum.eigenvectors.shape[1]:
            warnings.warn(f"Eigenfunction index {eigenfunction_idx} out of range")
            return
        
        try:
            eigenfunction = spectrum.eigenvectors[:, eigenfunction_idx]
            eigenvalue = spectrum.eigenvalues[eigenfunction_idx]
            
            # Create spatial grid
            xmin, xmax, ymin, ymax = grid_bounds
            x = np.linspace(xmin, xmax, self.grid_resolution)
            y = np.linspace(ymin, ymax, self.grid_resolution)
            X, Y = np.meshgrid(x, y)
            
            # Interpolate eigenfunction values onto grid
            # For simplicity, assume eigenfunction corresponds to flattened grid
            if len(eigenfunction) == self.grid_resolution ** 2:
                Z = eigenfunction.reshape(self.grid_resolution, self.grid_resolution)
                
                # Handle complex values
                if np.iscomplexobj(Z):
                    Z_real = np.real(Z)
                    Z_imag = np.imag(Z)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
                    
                    # Real part
                    im1 = axes[0].contourf(X, Y, Z_real, levels=20, cmap=self.colormap)
                    axes[0].set_title(f'Real Part - Eigenfunction {eigenfunction_idx+1}', 
                                    fontsize=12, fontweight='bold')
                    axes[0].set_xlabel('x')
                    axes[0].set_ylabel('y')
                    axes[0].set_aspect('equal')
                    plt.colorbar(im1, ax=axes[0])
                    
                    # Imaginary part
                    im2 = axes[1].contourf(X, Y, Z_imag, levels=20, cmap=self.colormap)
                    axes[1].set_title(f'Imaginary Part - Eigenfunction {eigenfunction_idx+1}', 
                                    fontsize=12, fontweight='bold')
                    axes[1].set_xlabel('x')
                    axes[1].set_ylabel('y')
                    axes[1].set_aspect('equal')
                    plt.colorbar(im2, ax=axes[1])
                    
                else:
                    fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
                    
                    im = ax.contourf(X, Y, Z, levels=20, cmap=self.colormap)
                    ax.set_title(f'Eigenfunction {eigenfunction_idx+1}\nλ = {eigenvalue:.4f}', 
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel('x', fontsize=12)
                    ax.set_ylabel('y', fontsize=12)
                    ax.set_aspect('equal')
                    plt.colorbar(im, ax=ax)
                
                # Overall title for complex case
                if np.iscomplexobj(Z):
                    fig.suptitle(f'Eigenfunction {eigenfunction_idx+1} - λ = {eigenvalue:.4f}', 
                               fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                    print(f"Eigenfunction grid plot saved to: {save_path}")
                
                plt.show()
            
            else:
                warnings.warn(f"Eigenfunction size ({len(eigenfunction)}) doesn't match grid size ({self.grid_resolution**2})")
        
        except Exception as e:
            warnings.warn(f"Error creating eigenfunction grid plot: {str(e)}")
    
    def compare_eigenfunctions(self, spectra: Dict[str, SpectralResults],
                             eigenfunction_idx: int = 0,
                             save_path: Optional[str] = None) -> None:
        """
        Compare eigenfunctions across different models.
        
        Args:
            spectra: Dictionary mapping model names to SpectralResults
            eigenfunction_idx: Index of eigenfunction to compare
            save_path: Optional path to save the figure
        """
        try:
            n_models = len(spectra)
            if n_models == 0:
                warnings.warn("No spectra provided for comparison")
                return
            
            fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4), dpi=self.dpi)
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, spectrum) in enumerate(spectra.items()):
                if eigenfunction_idx >= spectrum.eigenvectors.shape[1]:
                    axes[i].text(0.5, 0.5, f'No eigenfunction\n{eigenfunction_idx+1}\navailable', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
                    continue
                
                eigenfunction = spectrum.eigenvectors[:, eigenfunction_idx]
                eigenvalue = spectrum.eigenvalues[eigenfunction_idx]
                
                # Plot magnitude for complex eigenfunctions
                if np.iscomplexobj(eigenfunction):
                    plot_values = np.abs(eigenfunction)
                    ylabel = '|Eigenfunction|'
                else:
                    plot_values = eigenfunction
                    ylabel = 'Eigenfunction'
                
                axes[i].plot(plot_values, linewidth=2, label=f'λ = {eigenvalue:.4f}')
                axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Component Index')
                axes[i].set_ylabel(ylabel)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            fig.suptitle(f'Eigenfunction {eigenfunction_idx+1} Comparison', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"Eigenfunction comparison saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating eigenfunction comparison: {str(e)}")
    
    def visualize_eigenfunction_evolution(self, spectrum: SpectralResults,
                                        time_steps: np.ndarray,
                                        initial_condition: np.ndarray,
                                        eigenfunction_idx: int = 0,
                                        save_path: Optional[str] = None) -> None:
        """
        Visualize temporal evolution of an eigenfunction.
        
        Args:
            spectrum: SpectralResults containing eigenvalues and eigenvectors
            time_steps: Array of time points for evolution
            initial_condition: Initial condition for evolution
            eigenfunction_idx: Index of eigenfunction to evolve
            save_path: Optional path to save the figure
        """
        if eigenfunction_idx >= spectrum.eigenvectors.shape[1]:
            warnings.warn(f"Eigenfunction index {eigenfunction_idx} out of range")
            return
        
        try:
            eigenvalue = spectrum.eigenvalues[eigenfunction_idx]
            eigenfunction = spectrum.eigenvectors[:, eigenfunction_idx]
            
            # Compute temporal evolution: φ(t) = φ(0) * exp(λt)
            evolution = np.zeros((len(time_steps), len(eigenfunction)), dtype=complex)
            
            for i, t in enumerate(time_steps):
                evolution[i] = eigenfunction * np.exp(eigenvalue * t)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
            
            # Magnitude evolution
            magnitude_evolution = np.abs(evolution)
            im1 = axes[0, 0].imshow(magnitude_evolution.T, aspect='auto', cmap='viridis',
                                  extent=[time_steps[0], time_steps[-1], 0, len(eigenfunction)])
            axes[0, 0].set_title('Magnitude Evolution', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Component')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Phase evolution
            phase_evolution = np.angle(evolution)
            im2 = axes[0, 1].imshow(phase_evolution.T, aspect='auto', cmap='hsv',
                                  extent=[time_steps[0], time_steps[-1], 0, len(eigenfunction)])
            axes[0, 1].set_title('Phase Evolution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Component')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Total magnitude over time
            total_magnitude = np.sum(magnitude_evolution, axis=1)
            axes[1, 0].plot(time_steps, total_magnitude, 'b-', linewidth=2)
            axes[1, 0].set_title('Total Magnitude vs Time', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Total |φ(t)|')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Growth rate analysis
            theoretical_growth = np.sum(np.abs(initial_condition)) * np.exp(np.real(eigenvalue) * time_steps)
            axes[1, 1].semilogy(time_steps, total_magnitude, 'b-', linewidth=2, label='Actual')
            axes[1, 1].semilogy(time_steps, theoretical_growth, 'r--', linewidth=2, label='Theoretical')
            axes[1, 1].set_title('Growth Rate Analysis', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('log |φ(t)|')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            fig.suptitle(f'Eigenfunction {eigenfunction_idx+1} Evolution\nλ = {eigenvalue:.4f}', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"Eigenfunction evolution visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating eigenfunction evolution visualization: {str(e)}")