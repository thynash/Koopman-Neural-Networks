"""
Advanced spectrum visualization tools for Koopman operator analysis.

This module provides publication-quality visualizations for eigenvalue spectra,
including comparative plots and complex plane analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any
import warnings

from src.analysis.spectral.spectral_analyzer import SpectralResults


class SpectrumVisualizer:
    """
    Advanced visualization tools for eigenvalue spectra in the complex plane.
    
    This class provides methods for creating publication-ready plots of eigenvalue
    spectra with support for comparative analysis across multiple methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spectrum visualizer.
        
        Args:
            config: Configuration dictionary containing:
                - figure_size: Default figure size (default: (12, 10))
                - dpi: Resolution for saved figures (default: 300)
                - style: Plot style ('publication', 'presentation', 'notebook')
                - colormap: Default colormap (default: 'viridis')
        """
        self.config = config
        self.figure_size = config.get('figure_size', (12, 10))
        self.dpi = config.get('dpi', 300)
        self.style = config.get('style', 'publication')
        self.colormap = config.get('colormap', 'viridis')
        
        # Set matplotlib style based on configuration
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Setup matplotlib plotting style based on configuration."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'font.family': 'serif',
                'text.usetex': False  # Set to True if LaTeX is available
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'xtick.labelsize': 13,
                'ytick.labelsize': 13,
                'legend.fontsize': 13,
                'figure.titlesize': 20,
                'font.family': 'sans-serif'
            })
    
    def plot_single_spectrum(self, spectrum: SpectralResults, 
                           title: str = "Eigenvalue Spectrum",
                           save_path: Optional[str] = None,
                           show_unit_circle: bool = True,
                           show_info: bool = True) -> None:
        """
        Plot a single eigenvalue spectrum in the complex plane.
        
        Args:
            spectrum: SpectralResults to visualize
            title: Plot title
            save_path: Optional path to save figure
            show_unit_circle: Whether to show unit circle
            show_info: Whether to show information text
        """
        if len(spectrum.eigenvalues) == 0:
            warnings.warn("Cannot plot empty spectrum")
            return
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            eigenvals = spectrum.eigenvalues
            real_parts = np.real(eigenvals)
            imag_parts = np.imag(eigenvals)
            magnitudes = np.abs(eigenvals)
            
            # Create scatter plot
            scatter = ax.scatter(real_parts, imag_parts, 
                               c=magnitudes, 
                               cmap=self.colormap, 
                               s=80, 
                               alpha=0.8,
                               edgecolors='black',
                               linewidth=0.8,
                               zorder=3)
            
            # Add unit circle if requested
            if show_unit_circle:
                circle = Circle((0, 0), 1, fill=False, color='red', 
                              linestyle='--', linewidth=2, alpha=0.7, zorder=2)
                ax.add_patch(circle)
                ax.plot([], [], 'r--', linewidth=2, alpha=0.7, label='Unit Circle')
            
            # Formatting
            ax.set_xlabel('Real Part', fontsize=14)
            ax.set_ylabel('Imaginary Part', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, zorder=1)
            ax.set_aspect('equal')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('|λ| (Eigenvalue Magnitude)', fontsize=12)
            
            # Add information text if requested
            if show_info:
                info_text = self._create_info_text(spectrum)
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                               alpha=0.9, edgecolor='gray'))
            
            # Add legend if unit circle is shown
            if show_unit_circle:
                ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                print(f"Spectrum plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating spectrum plot: {str(e)}")
    
    def plot_comparative_spectra(self, spectra: Dict[str, SpectralResults],
                               title: str = "Comparative Eigenvalue Spectra",
                               save_path: Optional[str] = None,
                               show_unit_circle: bool = True) -> None:
        """
        Plot multiple spectra for comparison.
        
        Args:
            spectra: Dictionary mapping method names to SpectralResults
            title: Plot title
            save_path: Optional path to save figure
            show_unit_circle: Whether to show unit circle
        """
        if not spectra:
            warnings.warn("No spectra provided for comparison")
            return
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Define colors and markers for different methods
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            
            legend_elements = []
            
            for i, (method_name, spectrum) in enumerate(spectra.items()):
                if len(spectrum.eigenvalues) == 0:
                    continue
                
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                real_parts = np.real(spectrum.eigenvalues)
                imag_parts = np.imag(spectrum.eigenvalues)
                
                # Plot eigenvalues
                ax.scatter(real_parts, imag_parts, 
                          c=color, marker=marker, s=80, alpha=0.7,
                          edgecolors='black', linewidth=0.5, 
                          label=method_name, zorder=3)
                
                # Add to legend
                legend_elements.append(
                    plt.Line2D([0], [0], marker=marker, color='w', 
                             markerfacecolor=color, markersize=10, 
                             markeredgecolor='black', markeredgewidth=0.5,
                             label=method_name)
                )
            
            # Add unit circle if requested
            if show_unit_circle:
                circle = Circle((0, 0), 1, fill=False, color='red', 
                              linestyle='--', linewidth=2, alpha=0.7, zorder=2)
                ax.add_patch(circle)
                legend_elements.append(
                    plt.Line2D([0], [0], color='red', linestyle='--', 
                             linewidth=2, alpha=0.7, label='Unit Circle')
                )
            
            # Formatting
            ax.set_xlabel('Real Part', fontsize=14)
            ax.set_ylabel('Imaginary Part', fontsize=14)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, zorder=1)
            ax.set_aspect('equal')
            
            # Add legend
            ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                print(f"Comparative spectrum plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating comparative spectrum plot: {str(e)}")
    
    def plot_spectrum_evolution(self, spectra_sequence: List[SpectralResults],
                              time_points: Optional[List[float]] = None,
                              title: str = "Spectrum Evolution",
                              save_path: Optional[str] = None) -> None:
        """
        Plot evolution of eigenvalue spectrum over time or training iterations.
        
        Args:
            spectra_sequence: List of SpectralResults at different time points
            time_points: Optional list of time points (default: iteration numbers)
            title: Plot title
            save_path: Optional path to save figure
        """
        if not spectra_sequence:
            warnings.warn("No spectra provided for evolution plot")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=self.dpi)
            
            if time_points is None:
                time_points = list(range(len(spectra_sequence)))
            
            # Plot 1: Eigenvalues in complex plane with evolution
            colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_sequence)))
            
            for i, (spectrum, t) in enumerate(zip(spectra_sequence, time_points)):
                if len(spectrum.eigenvalues) == 0:
                    continue
                
                real_parts = np.real(spectrum.eigenvalues)
                imag_parts = np.imag(spectrum.eigenvalues)
                
                ax1.scatter(real_parts, imag_parts, 
                           c=[colors[i]], s=60, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           label=f't={t:.2f}' if len(spectra_sequence) <= 10 else None)
            
            # Add unit circle
            circle = Circle((0, 0), 1, fill=False, color='red', 
                          linestyle='--', linewidth=2, alpha=0.7)
            ax1.add_patch(circle)
            
            ax1.set_xlabel('Real Part', fontsize=14)
            ax1.set_ylabel('Imaginary Part', fontsize=14)
            ax1.set_title('Eigenvalues in Complex Plane', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            if len(spectra_sequence) <= 10:
                ax1.legend()
            
            # Plot 2: Spectral radius evolution
            spectral_radii = []
            for spectrum in spectra_sequence:
                if len(spectrum.eigenvalues) > 0:
                    spectral_radii.append(np.max(np.abs(spectrum.eigenvalues)))
                else:
                    spectral_radii.append(0.0)
            
            ax2.plot(time_points, spectral_radii, 'b-o', linewidth=2, markersize=6)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Unit Circle')
            ax2.set_xlabel('Time/Iteration', fontsize=14)
            ax2.set_ylabel('Spectral Radius', fontsize=14)
            ax2.set_title('Spectral Radius Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                print(f"Spectrum evolution plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating spectrum evolution plot: {str(e)}")
    
    def plot_eigenvalue_distribution(self, spectra: Dict[str, SpectralResults],
                                   save_path: Optional[str] = None) -> None:
        """
        Plot distribution of eigenvalue magnitudes and phases.
        
        Args:
            spectra: Dictionary mapping method names to SpectralResults
            save_path: Optional path to save figure
        """
        if not spectra:
            warnings.warn("No spectra provided for distribution plot")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (method_name, spectrum) in enumerate(spectra.items()):
                if len(spectrum.eigenvalues) == 0:
                    continue
                
                color = colors[i % len(colors)]
                eigenvals = spectrum.eigenvalues
                
                # Magnitude distribution
                magnitudes = np.abs(eigenvals)
                ax1.hist(magnitudes, bins=20, alpha=0.6, color=color, 
                        label=method_name, density=True)
                
                # Phase distribution
                phases = np.angle(eigenvals)
                ax2.hist(phases, bins=20, alpha=0.6, color=color, 
                        label=method_name, density=True)
                
                # Real part distribution
                real_parts = np.real(eigenvals)
                ax3.hist(real_parts, bins=20, alpha=0.6, color=color, 
                        label=method_name, density=True)
                
                # Imaginary part distribution
                imag_parts = np.imag(eigenvals)
                ax4.hist(imag_parts, bins=20, alpha=0.6, color=color, 
                        label=method_name, density=True)
            
            # Formatting
            ax1.set_xlabel('|λ| (Magnitude)', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title('Eigenvalue Magnitude Distribution', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('arg(λ) (Phase)', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Eigenvalue Phase Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3.set_xlabel('Re(λ) (Real Part)', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('Real Part Distribution', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.set_xlabel('Im(λ) (Imaginary Part)', fontsize=12)
            ax4.set_ylabel('Density', fontsize=12)
            ax4.set_title('Imaginary Part Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            fig.suptitle('Eigenvalue Distribution Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                print(f"Eigenvalue distribution plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating eigenvalue distribution plot: {str(e)}")
    
    def _create_info_text(self, spectrum: SpectralResults) -> str:
        """Create information text for spectrum plots."""
        info_lines = [
            f"N eigenvalues: {len(spectrum.eigenvalues)}",
        ]
        
        if len(spectrum.eigenvalues) > 0:
            spectral_radius = np.max(np.abs(spectrum.eigenvalues))
            info_lines.append(f"Spectral radius: {spectral_radius:.4f}")
            
            # Add dominant eigenvalue info
            dominant_idx = np.argmax(np.abs(spectrum.eigenvalues))
            dominant_val = spectrum.eigenvalues[dominant_idx]
            info_lines.append(f"Dominant λ: {dominant_val:.4f}")
        
        # Add extraction method if available
        if 'extraction_method' in spectrum.metadata:
            method = spectrum.metadata['extraction_method']
            info_lines.append(f"Method: {method}")
        
        return '\n'.join(info_lines)