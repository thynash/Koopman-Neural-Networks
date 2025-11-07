"""
Visualization generator for comparative analysis of Koopman operator learning models.

This module creates publication-ready visualizations for comparing multiple
neural architectures including loss curves, spectral plots, and performance tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import asdict

from .model_comparator import ComparisonResults, ModelMetrics
from .benchmark_runner import BenchmarkResults
from src.analysis.spectral.spectral_analyzer import SpectralResults


class ComparisonVisualizer:
    """
    Comprehensive visualization generator for model comparison results.
    
    Creates side-by-side loss curves, comparative spectral plots, performance
    tables, and efficiency analysis charts for publication-ready output.
    """
    
    def __init__(self, output_dir: str = 'results/comparison_plots'):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-ready style
        self._setup_plot_style()
        
    def _setup_plot_style(self) -> None:
        """Setup matplotlib style for publication-ready plots."""
        plt.style.use('seaborn-v0_8')
        
        # Custom parameters for publication quality
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'grid.alpha': 0.3,
            'axes.grid': True
        })
        
        # Color palette for different models
        self.model_colors = {
            'mlp': '#1f77b4',      # Blue
            'deeponet': '#ff7f0e', # Orange
            'lstm': '#2ca02c',     # Green
            'dmd': '#d62728'       # Red (for baseline)
        }
    
    def plot_training_curves_comparison(self, training_histories: Dict[str, Dict],
                                      save_path: Optional[str] = None) -> None:
        """
        Create side-by-side training and validation loss curves for all models.
        
        Args:
            training_histories: Dictionary mapping model names to training history
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training loss comparison
        for model_name, history in training_histories.items():
            if 'train_losses' in history:
                epochs = range(len(history['train_losses']))
                color = self.model_colors.get(model_name, None)
                ax1.plot(epochs, history['train_losses'], 
                        label=f'{model_name.upper()}', color=color, linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss comparison
        for model_name, history in training_histories.items():
            if 'val_losses' in history and history['val_losses']:
                # Create epoch indices for validation losses (assuming validation every N epochs)
                val_epochs = np.linspace(0, len(history.get('train_losses', [])), 
                                       len(history['val_losses']))
                color = self.model_colors.get(model_name, None)
                ax2.plot(val_epochs, history['val_losses'], 
                        label=f'{model_name.upper()}', color=color, linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Comparison')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'training_curves_comparison.png', 
                       dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def plot_spectral_comparison(self, spectral_results: Dict[str, SpectralResults],
                               save_path: Optional[str] = None) -> None:
        """
        Create comparative spectral plots with eigenvalue overlays.
        
        Args:
            spectral_results: Dictionary mapping model names to spectral results
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Complex plane eigenvalue plot
        for model_name, results in spectral_results.items():
            eigenvals = results.eigenvalues
            color = self.model_colors.get(model_name, None)
            
            ax1.scatter(eigenvals.real, eigenvals.imag, 
                       label=f'{model_name.upper()}', 
                       color=color, alpha=0.7, s=50)
        
        # Add unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title('Eigenvalue Spectrum in Complex Plane')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Eigenvalue magnitude comparison
        for model_name, results in spectral_results.items():
            eigenvals = results.eigenvalues
            magnitudes = np.abs(eigenvals)
            sorted_mags = np.sort(magnitudes)[::-1]  # Descending order
            
            color = self.model_colors.get(model_name, None)
            ax2.plot(range(len(sorted_mags)), sorted_mags, 
                    'o-', label=f'{model_name.upper()}', 
                    color=color, markersize=4)
        
        ax2.set_xlabel('Eigenvalue Index')
        ax2.set_ylabel('Eigenvalue Magnitude')
        ax2.set_title('Eigenvalue Magnitudes (Sorted)')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'spectral_comparison.png', 
                       dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def plot_performance_metrics_comparison(self, comparison_results: ComparisonResults,
                                          save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance metrics comparison chart.
        
        Args:
            comparison_results: ComparisonResults object
            save_path: Optional path to save the plot
        """
        metrics_data = []
        
        for model_name, metrics in comparison_results.model_metrics.items():
            metrics_data.append({
                'Model': model_name.upper(),
                'Test MSE': metrics.test_mse,
                'Test MAE': metrics.test_mae,
                'Relative Error': metrics.test_relative_error,
                'R² Score': metrics.test_r2_score,
                'Spectral Error': metrics.spectral_error
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots for different metric categories
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('Test MSE', 'Test MSE', True),
            ('Test MAE', 'Test MAE', True),
            ('Relative Error', 'Relative Error (%)', True),
            ('R² Score', 'R² Score', False),
            ('Spectral Error', 'Spectral Error', True)
        ]
        
        for i, (metric, ylabel, log_scale) in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(df['Model'], df[metric], 
                         color=[self.model_colors.get(model.lower(), '#gray') 
                               for model in df['Model']])
            
            ax.set_ylabel(ylabel)
            ax.set_title(f'{metric} Comparison')
            
            if log_scale:
                ax.set_yscale('log')
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                if not np.isnan(value) and not np.isinf(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.2e}' if log_scale else f'{value:.3f}',
                           ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Remove unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'performance_metrics_comparison.png', 
                       dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def plot_efficiency_comparison(self, benchmark_results: List[BenchmarkResults],
                                 save_path: Optional[str] = None) -> None:
        """
        Create computational efficiency comparison charts.
        
        Args:
            benchmark_results: List of BenchmarkResults for different models
            save_path: Optional path to save the plot
        """
        # Prepare data
        efficiency_data = []
        for result in benchmark_results:
            efficiency_data.append({
                'Model': result.model_name.upper(),
                'Training Time (s)': result.training_time,
                'Inference Time (ms)': result.inference_time_mean * 1000,  # Convert to ms
                'Memory Usage (MB)': result.inference_memory,
                'Model Size (MB)': result.model_size_mb,
                'Parameters (M)': result.total_parameters / 1e6  # Convert to millions
            })
        
        df = pd.DataFrame(efficiency_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        efficiency_metrics = [
            ('Training Time (s)', 'Training Time (seconds)', True),
            ('Inference Time (ms)', 'Inference Time (ms)', True),
            ('Memory Usage (MB)', 'Memory Usage (MB)', True),
            ('Model Size (MB)', 'Model Size (MB)', True),
            ('Parameters (M)', 'Parameters (Millions)', True)
        ]
        
        for i, (metric, ylabel, log_scale) in enumerate(efficiency_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(df['Model'], df[metric], 
                         color=[self.model_colors.get(model.lower(), '#gray') 
                               for model in df['Model']])
            
            ax.set_ylabel(ylabel)
            ax.set_title(f'{metric.split(" (")[0]} Comparison')
            
            if log_scale and df[metric].min() > 0:
                ax.set_yscale('log')
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                if not np.isnan(value) and not np.isinf(value) and value > 0:
                    if log_scale:
                        label = f'{value:.2e}' if value < 0.01 else f'{value:.2f}'
                    else:
                        label = f'{value:.2f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           label, ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Remove unused subplots
        for i in range(len(efficiency_metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'efficiency_comparison.png', 
                       dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def plot_batch_scaling_analysis(self, benchmark_results: List[BenchmarkResults],
                                  save_path: Optional[str] = None) -> None:
        """
        Create batch size scaling analysis plots.
        
        Args:
            benchmark_results: List of BenchmarkResults with scaling data
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Processing time vs batch size
        for result in benchmark_results:
            if result.batch_processing_times:
                batch_sizes = list(result.batch_processing_times.keys())
                times = list(result.batch_processing_times.values())
                
                color = self.model_colors.get(result.model_name.lower(), None)
                ax1.plot(batch_sizes, times, 'o-', 
                        label=result.model_name.upper(), 
                        color=color, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Processing Time per Sample (s)')
        ax1.set_title('Inference Time vs Batch Size')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory usage vs batch size
        for result in benchmark_results:
            if result.memory_scaling:
                batch_sizes = list(result.memory_scaling.keys())
                memory = list(result.memory_scaling.values())
                
                color = self.model_colors.get(result.model_name.lower(), None)
                ax2.plot(batch_sizes, memory, 's-', 
                        label=result.model_name.upper(), 
                        color=color, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.set_xscale('log', base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'batch_scaling_analysis.png', 
                       dpi=600, bbox_inches='tight')
        
        plt.close()
    
    def generate_performance_table(self, comparison_results: ComparisonResults,
                                 save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive performance comparison table.
        
        Args:
            comparison_results: ComparisonResults object
            save_path: Optional path to save the table
            
        Returns:
            DataFrame containing the performance table
        """
        table_data = []
        
        for model_name, metrics in comparison_results.model_metrics.items():
            table_data.append({
                'Model': model_name.upper(),
                'Test MSE': f'{metrics.test_mse:.2e}',
                'Test MAE': f'{metrics.test_mae:.2e}',
                'Relative Error': f'{metrics.test_relative_error:.3f}',
                'R² Score': f'{metrics.test_r2_score:.3f}',
                'Spectral Error': f'{metrics.spectral_error:.2e}',
                'Training Time (s)': f'{metrics.training_time:.1f}',
                'Inference Time (ms)': f'{metrics.inference_time * 1000:.2f}',
                'Parameters (M)': f'{metrics.model_parameters / 1e6:.2f}',
                'Memory (MB)': f'{metrics.peak_memory_usage:.1f}',
                'Converged': '✓' if metrics.converged else '✗'
            })
        
        df = pd.DataFrame(table_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(self.output_dir / 'performance_comparison_table.csv', index=False)
        
        # Also save as LaTeX table for publications
        latex_path = save_path.replace('.csv', '.tex') if save_path else self.output_dir / 'performance_table.tex'
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        return df
    
    def create_summary_dashboard(self, comparison_results: ComparisonResults,
                               benchmark_results: List[BenchmarkResults],
                               spectral_results: Dict[str, SpectralResults],
                               training_histories: Dict[str, Dict],
                               save_path: Optional[str] = None) -> None:
        """
        Create comprehensive summary dashboard with all comparison plots.
        
        Args:
            comparison_results: ComparisonResults object
            benchmark_results: List of BenchmarkResults
            spectral_results: Dictionary of spectral results
            training_histories: Dictionary of training histories
            save_path: Optional path to save the dashboard
        """
        # Create individual plots
        self.plot_training_curves_comparison(training_histories)
        self.plot_spectral_comparison(spectral_results)
        self.plot_performance_metrics_comparison(comparison_results)
        self.plot_efficiency_comparison(benchmark_results)
        
        if benchmark_results and any(r.batch_processing_times for r in benchmark_results):
            self.plot_batch_scaling_analysis(benchmark_results)
        
        # Generate performance table
        self.generate_performance_table(comparison_results)
        
        # Create summary report
        self._generate_summary_report(comparison_results, benchmark_results)
        
        print(f"Comprehensive comparison dashboard generated in {self.output_dir}")
    
    def _generate_summary_report(self, comparison_results: ComparisonResults,
                               benchmark_results: List[BenchmarkResults]) -> None:
        """
        Generate text summary report of comparison results.
        
        Args:
            comparison_results: ComparisonResults object
            benchmark_results: List of BenchmarkResults
        """
        report_lines = []
        report_lines.append("# Model Comparison Summary Report")
        report_lines.append(f"Generated on: {comparison_results.timestamp}")
        report_lines.append("")
        
        # Dataset information
        report_lines.append("## Dataset Information")
        for key, value in comparison_results.dataset_info.items():
            report_lines.append(f"- {key}: {value}")
        report_lines.append("")
        
        # Performance rankings
        if 'performance_rankings' in comparison_results.comparison_summary:
            report_lines.append("## Performance Rankings")
            rankings = comparison_results.comparison_summary['performance_rankings']
            
            for metric, ranking in rankings.items():
                report_lines.append(f"### {metric.replace('_', ' ').title()}")
                for i, (model, value) in enumerate(ranking, 1):
                    report_lines.append(f"{i}. {model.upper()}: {value:.2e}")
                report_lines.append("")
        
        # Best overall model
        if 'best_overall_model' in comparison_results.comparison_summary:
            best = comparison_results.comparison_summary['best_overall_model']
            report_lines.append("## Best Overall Model")
            report_lines.append(f"**{best['name'].upper()}** (Score: {best['score']:.3f})")
            report_lines.append("")
        
        # Efficiency summary
        if benchmark_results:
            report_lines.append("## Computational Efficiency Summary")
            fastest_training = min(benchmark_results, key=lambda r: r.training_time)
            fastest_inference = min(benchmark_results, key=lambda r: r.inference_time_mean)
            most_efficient_memory = min(benchmark_results, key=lambda r: r.inference_memory)
            
            report_lines.append(f"- Fastest Training: {fastest_training.model_name.upper()} ({fastest_training.training_time:.1f}s)")
            report_lines.append(f"- Fastest Inference: {fastest_inference.model_name.upper()} ({fastest_inference.inference_time_mean*1000:.2f}ms)")
            report_lines.append(f"- Most Memory Efficient: {most_efficient_memory.model_name.upper()} ({most_efficient_memory.inference_memory:.1f}MB)")
            report_lines.append("")
        
        # Save report
        report_path = self.output_dir / 'comparison_summary_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to {report_path}")