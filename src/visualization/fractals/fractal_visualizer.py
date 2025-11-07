"""
High-resolution fractal attractor visualization module.

This module provides specialized visualization functions for fractal attractors
with publication-quality formatting and high DPI output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import matplotlib.patches as patches


class FractalVisualizer:
    """
    Specialized visualizer for fractal attractors with high-resolution output.
    """
    
    def __init__(self, output_dir: str = "figures/fractals", dpi: int = 600):
        """
        Initialize fractal visualizer.
        
        Args:
            output_dir: Directory to save fractal figures
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = max(dpi, 600)
        
        # Custom colormaps for different fractal types
        self.colormaps = {
            'sierpinski': self._create_sierpinski_colormap(),
            'barnsley': self._create_barnsley_colormap(),
            'julia': self._create_julia_colormap()
        }
    
    def _create_sierpinski_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for Sierpinski gasket."""
        colors = ['#000033', '#000066', '#003366', '#006699', '#0099CC', '#33CCFF']
        return LinearSegmentedColormap.from_list('sierpinski', colors)
    
    def _create_barnsley_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for Barnsley fern."""
        colors = ['#001100', '#003300', '#006600', '#009900', '#00CC00', '#33FF33']
        return LinearSegmentedColormap.from_list('barnsley', colors)
    
    def _create_julia_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for Julia sets."""
        colors = ['#330000', '#660000', '#990033', '#CC0066', '#FF3399', '#FF99CC']
        return LinearSegmentedColormap.from_list('julia', colors)
    
    def visualize_ifs_attractor(self,
                               trajectory_data: np.ndarray,
                               system_name: str,
                               title: Optional[str] = None,
                               point_size: float = 0.1,
                               alpha: float = 0.6,
                               save_name: Optional[str] = None) -> str:
        """
        Visualize IFS attractor with high resolution.
        
        Args:
            trajectory_data: Array of shape (n_points, 2)
            system_name: Name of IFS system ('sierpinski' or 'barnsley')
            title: Custom title (optional)
            point_size: Size of plotted points
            alpha: Transparency of points
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Select appropriate colormap
        cmap = self.colormaps.get(system_name.lower(), 'viridis')
        
        # Create color gradient based on iteration order
        colors = np.arange(len(trajectory_data))
        
        # Create scatter plot
        scatter = ax.scatter(trajectory_data[:, 0], trajectory_data[:, 1],
                           c=colors, cmap=cmap, s=point_size, alpha=alpha,
                           edgecolors='none')
        
        # Formatting
        if title is None:
            title = f'{system_name.title()} Attractor'
        
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$y$', fontsize=16)
        ax.tick_params(labelsize=14)
        
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = f"{system_name.lower()}_attractor_hires"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', 
                   pad_inches=0.1)
        plt.close()
        
        return str(filepath)
    
    def visualize_julia_set(self,
                           trajectory_data: np.ndarray,
                           c_parameter: complex,
                           title: Optional[str] = None,
                           point_size: float = 0.05,
                           alpha: float = 0.7,
                           save_name: Optional[str] = None) -> str:
        """
        Visualize Julia set with high resolution.
        
        Args:
            trajectory_data: Array of shape (n_points, 2) representing complex points
            c_parameter: Complex parameter for Julia set
            title: Custom title (optional)
            point_size: Size of plotted points
            alpha: Transparency of points
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 12), dpi=self.dpi)
        
        # Use Julia colormap
        cmap = self.colormaps['julia']
        
        # Create color based on distance from origin or iteration count
        distances = np.sqrt(trajectory_data[:, 0]**2 + trajectory_data[:, 1]**2)
        
        # Create scatter plot
        scatter = ax.scatter(trajectory_data[:, 0], trajectory_data[:, 1],
                           c=distances, cmap=cmap, s=point_size, alpha=alpha,
                           edgecolors='none')
        
        # Formatting
        if title is None:
            title = f'Julia Set: $c = {c_parameter.real:.3f} + {c_parameter.imag:.3f}i$'
        
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel(r'$\mathrm{Re}(z)$', fontsize=16)
        ax.set_ylabel(r'$\mathrm{Im}(z)$', fontsize=16)
        ax.tick_params(labelsize=14)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid for reference
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            c_str = f"{c_parameter.real:.3f}_{c_parameter.imag:.3f}".replace('-', 'neg')
            save_name = f"julia_set_c_{c_str}_hires"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   pad_inches=0.1)
        plt.close()
        
        return str(filepath)
    
    def create_multi_fractal_comparison(self,
                                      fractal_data: Dict[str, np.ndarray],
                                      titles: Optional[Dict[str, str]] = None,
                                      save_name: str = "fractal_comparison") -> str:
        """
        Create side-by-side comparison of multiple fractals.
        
        Args:
            fractal_data: Dictionary with fractal names as keys and trajectory data as values
            titles: Custom titles for each fractal (optional)
            save_name: Filename for saved figure
            
        Returns:
            Path to saved figure
        """
        n_fractals = len(fractal_data)
        cols = min(3, n_fractals)
        rows = (n_fractals + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), dpi=self.dpi)
        
        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fractal_names = list(fractal_data.keys())
        
        for i, (name, data) in enumerate(fractal_data.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Select appropriate colormap
            cmap = self.colormaps.get(name.lower(), 'viridis')
            colors = np.arange(len(data))
            
            # Create scatter plot
            ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, 
                      s=0.1, alpha=0.6, edgecolors='none')
            
            # Formatting
            title = titles.get(name, name.title()) if titles else name.title()
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Hide unused subplots
        for i in range(n_fractals, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.set_visible(False)
        
        plt.suptitle('Fractal Attractor Comparison', fontsize=24, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_attractor(self, states: np.ndarray, title: str = "Fractal Attractor",
                      save_path: Optional[str] = None, dpi: int = 600) -> None:
        """
        Plot fractal attractor from trajectory data.
        
        Args:
            states: Trajectory states, shape (n_points, 2)
            title: Plot title
            save_path: Path to save figure (optional)
            dpi: Figure resolution for saving
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=max(dpi, 600))
        
        # Create scatter plot
        ax.scatter(states[:, 0], states[:, 1], s=0.5, alpha=0.7, color='blue')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=max(dpi, 600), bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        else:
            plt.show()