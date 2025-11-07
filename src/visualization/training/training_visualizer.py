"""
Training curve visualization module with publication-quality formatting.

This module provides specialized visualization functions for training curves,
loss plots, and performance metrics with consistent formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import seaborn as sns


class TrainingVisualizer:
    """
    Specialized visualizer for training curves and performance metrics.
    """
    
    def __init__(self, output_dir: str = "figures/training", dpi: int = 600):
        """
        Initialize training visualizer.
        
        Args:
            output_dir: Directory to save training figures
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = max(dpi, 600)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            # Fallback if seaborn style not available
            plt.style.use('default')
            plt.rcParams['axes.grid'] = True
        
        # Color palette for different models
        self.colors = {
            'MLP': '#1f77b4',
            'DeepONet': '#ff7f0e',
            'LSTM': '#2ca02c',
            'DMD': '#d62728',
            'Reference': '#9467bd'
        }
        
        self.line_styles = {
            'train': '-',
            'validation': '--',
            'test': ':',
            'reference': '-.'
        }
    
    def plot_loss_curves(self,
                        training_histories: Dict[str, Dict[str, List[float]]],
                        title: str = "Training Loss Comparison",
                        save_name: Optional[str] = None,
                        log_scale: bool = True) -> str:
        """
        Plot training and validation loss curves for multiple models.
        
        Args:
            training_histories: Nested dict {model_name: {metric_name: values}}
            title: Plot title
            save_name: Custom filename (optional)
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        for model_name, history in training_histories.items():
            color = self.colors.get(model_name, '#333333')
            
            # Plot training loss
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax.plot(epochs, history['train_loss'], 
                       color=color, linestyle=self.line_styles['train'],
                       linewidth=2.5, label=f'{model_name} (Train)', alpha=0.8)
            
            # Plot validation loss
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], 
                       color=color, linestyle=self.line_styles['validation'],
                       linewidth=2.5, label=f'{model_name} (Val)', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "loss_curves_comparison"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_spectral_error_evolution(self,
                                    spectral_histories: Dict[str, List[float]],
                                    title: str = "Spectral Error Evolution",
                                    save_name: Optional[str] = None) -> str:
        """
        Plot evolution of spectral approximation error during training.
        
        Args:
            spectral_histories: Dict {model_name: spectral_error_values}
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        for model_name, error_history in spectral_histories.items():
            color = self.colors.get(model_name, '#333333')
            epochs = range(1, len(error_history) + 1)
            
            ax.plot(epochs, error_history, 
                   color=color, linewidth=2.5, marker='o', markersize=4,
                   label=model_name, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel('Spectral Error', fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "spectral_error_evolution"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_comprehensive_training_dashboard(self,
                                            training_data: Dict[str, Dict[str, List[float]]],
                                            title: str = "Training Dashboard",
                                            save_name: Optional[str] = None) -> str:
        """
        Create comprehensive training dashboard with multiple metrics.
        
        Args:
            training_data: Nested dict with training metrics for each model
            title: Overall title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        for model_name, data in training_data.items():
            if 'train_loss' in data:
                color = self.colors.get(model_name, '#333333')
                epochs = range(1, len(data['train_loss']) + 1)
                ax1.plot(epochs, data['train_loss'], color=color, linewidth=2,
                        label=model_name, alpha=0.8)
        
        ax1.set_title('Training Loss', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation Loss
        ax2 = fig.add_subplot(gs[0, 1])
        for model_name, data in training_data.items():
            if 'val_loss' in data:
                color = self.colors.get(model_name, '#333333')
                epochs = range(1, len(data['val_loss']) + 1)
                ax2.plot(epochs, data['val_loss'], color=color, linewidth=2,
                        label=model_name, alpha=0.8)
        
        ax2.set_title('Validation Loss', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Spectral Error
        ax3 = fig.add_subplot(gs[0, 2])
        for model_name, data in training_data.items():
            if 'spectral_error' in data:
                color = self.colors.get(model_name, '#333333')
                epochs = range(1, len(data['spectral_error']) + 1)
                ax3.plot(epochs, data['spectral_error'], color=color, linewidth=2,
                        label=model_name, alpha=0.8)
        
        ax3.set_title('Spectral Error', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Error', fontsize=12)
        ax3.set_yscale('log')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Rate Schedule (if available)
        ax4 = fig.add_subplot(gs[1, 0])
        for model_name, data in training_data.items():
            if 'learning_rate' in data:
                color = self.colors.get(model_name, '#333333')
                epochs = range(1, len(data['learning_rate']) + 1)
                ax4.plot(epochs, data['learning_rate'], color=color, linewidth=2,
                        label=model_name, alpha=0.8)
        
        ax4.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_yscale('log')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Training Time per Epoch (if available)
        ax5 = fig.add_subplot(gs[1, 1])
        for model_name, data in training_data.items():
            if 'epoch_time' in data:
                color = self.colors.get(model_name, '#333333')
                epochs = range(1, len(data['epoch_time']) + 1)
                ax5.plot(epochs, data['epoch_time'], color=color, linewidth=2,
                        label=model_name, alpha=0.8)
        
        ax5.set_title('Training Time per Epoch', fontsize=16, fontweight='bold')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Time (seconds)', fontsize=12)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Memory Usage (if available)
        ax6 = fig.add_subplot(gs[1, 2])
        for model_name, data in training_data.items():
            if 'memory_usage' in data:
                color = self.colors.get(model_name, '#333333')
                epochs = range(1, len(data['memory_usage']) + 1)
                ax6.plot(epochs, data['memory_usage'], color=color, linewidth=2,
                        label=model_name, alpha=0.8)
        
        ax6.set_title('Memory Usage', fontsize=16, fontweight='bold')
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('Memory (MB)', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=24, fontweight='bold')
        
        # Save figure
        if save_name is None:
            save_name = "training_dashboard"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_convergence_analysis(self,
                                convergence_data: Dict[str, Dict[str, float]],
                                metrics: List[str] = ['final_loss', 'convergence_epoch', 'best_val_loss'],
                                title: str = "Model Convergence Analysis",
                                save_name: Optional[str] = None) -> str:
        """
        Create bar plots comparing convergence metrics across models.
        
        Args:
            convergence_data: Dict {model_name: {metric_name: value}}
            metrics: List of metrics to plot
            title: Plot title
            save_name: Custom filename (optional)
            
        Returns:
            Path to saved figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(convergence_data.keys())
        x_pos = np.arange(len(model_names))
        
        for i, metric in enumerate(metrics):
            values = [convergence_data[model].get(metric, 0) for model in model_names]
            colors = [self.colors.get(model, '#333333') for model in model_names]
            
            bars = axes[i].bar(x_pos, values, color=colors, alpha=0.8,
                             edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=12,
                           fontweight='bold')
            
            axes[i].set_title(metric.replace('_', ' ').title(), 
                            fontsize=16, fontweight='bold')
            axes[i].set_xlabel('Model', fontsize=14)
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].tick_params(labelsize=12)
        
        plt.suptitle(title, fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_name is None:
            save_name = "convergence_analysis"
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)