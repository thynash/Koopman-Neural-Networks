"""
Comprehensive result documentation and automated figure saving module.

This module provides automated documentation generation, figure saving with
descriptive filenames, and performance metrics summarization.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import yaml


@dataclass
class ExperimentConfig:
    """Configuration data for experiments."""
    experiment_name: str
    timestamp: str
    fractal_system: str
    model_architectures: List[str]
    hyperparameters: Dict[str, Any]
    data_parameters: Dict[str, Any]
    training_parameters: Dict[str, Any]


@dataclass
class ModelResults:
    """Results data for a single model."""
    model_name: str
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    convergence_epoch: int
    training_time: float
    memory_usage: float
    spectral_error: float
    eigenvalue_count: int
    dominant_eigenvalue: complex
    

@dataclass
class ExperimentResults:
    """Complete experiment results."""
    config: ExperimentConfig
    model_results: Dict[str, ModelResults]
    comparative_metrics: Dict[str, float]
    figure_paths: Dict[str, str]
    summary_statistics: Dict[str, Any]


class ResultDocumentationGenerator:
    """
    Generates comprehensive documentation for experimental results.
    """
    
    def __init__(self, 
                 output_dir: str = "results",
                 figures_dir: str = "figures"):
        """
        Initialize result documentation generator.
        
        Args:
            output_dir: Directory for result documentation
            figures_dir: Directory containing generated figures
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        (self.output_dir / "detailed").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
    
    def generate_descriptive_filename(self,
                                    figure_type: str,
                                    experiment_name: str,
                                    model_names: Optional[List[str]] = None,
                                    fractal_system: Optional[str] = None,
                                    timestamp: Optional[str] = None) -> str:
        """
        Generate descriptive filename for figures.
        
        Args:
            figure_type: Type of figure (e.g., 'spectrum', 'training_curves', 'attractor')
            experiment_name: Name of the experiment
            model_names: List of model names included (optional)
            fractal_system: Fractal system name (optional)
            timestamp: Timestamp string (optional)
            
        Returns:
            Descriptive filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename_parts = [
            experiment_name.lower().replace(' ', '_'),
            figure_type.lower().replace(' ', '_')
        ]
        
        if fractal_system:
            filename_parts.append(fractal_system.lower().replace(' ', '_'))
        
        if model_names:
            models_str = "_vs_".join([m.lower().replace(' ', '_') for m in model_names])
            filename_parts.append(models_str)
        
        filename_parts.append(timestamp)
        
        return "_".join(filename_parts) + ".png"
    
    def save_experiment_config(self,
                             config: ExperimentConfig,
                             filename: Optional[str] = None) -> str:
        """
        Save experiment configuration to file.
        
        Args:
            config: Experiment configuration
            filename: Custom filename (optional)
            
        Returns:
            Path to saved config file
        """
        if filename is None:
            filename = f"config_{config.experiment_name}_{config.timestamp}.yaml"
        
        config_path = self.output_dir / "configs" / filename
        
        # Convert to dictionary and handle numpy types
        config_dict = asdict(config)
        config_dict = self._convert_numpy_types(config_dict)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        return str(config_path)
    
    def save_model_results(self,
                          results: Dict[str, ModelResults],
                          experiment_name: str,
                          timestamp: str,
                          format: str = 'both') -> Dict[str, str]:
        """
        Save model results in multiple formats.
        
        Args:
            results: Dictionary of model results
            experiment_name: Name of experiment
            timestamp: Timestamp string
            format: Output format ('json', 'csv', or 'both')
            
        Returns:
            Dictionary of saved file paths
        """
        base_filename = f"results_{experiment_name}_{timestamp}"
        saved_paths = {}
        
        # Convert to DataFrame for easier handling
        results_data = []
        for model_name, result in results.items():
            result_dict = asdict(result)
            result_dict = self._convert_numpy_types(result_dict)
            results_data.append(result_dict)
        
        df = pd.DataFrame(results_data)
        
        if format in ['json', 'both']:
            json_path = self.output_dir / "detailed" / f"{base_filename}.json"
            results_dict = {name: asdict(result) for name, result in results.items()}
            results_dict = self._convert_numpy_types(results_dict)
            
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            saved_paths['json'] = str(json_path)
        
        if format in ['csv', 'both']:
            csv_path = self.output_dir / "detailed" / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            saved_paths['csv'] = str(csv_path)
        
        return saved_paths
    
    def generate_performance_summary(self,
                                   results: Dict[str, ModelResults],
                                   experiment_name: str,
                                   timestamp: str) -> str:
        """
        Generate human-readable performance summary.
        
        Args:
            results: Dictionary of model results
            experiment_name: Name of experiment
            timestamp: Timestamp string
            
        Returns:
            Path to saved summary file
        """
        summary_path = self.output_dir / "summaries" / f"summary_{experiment_name}_{timestamp}.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Experiment Results Summary\n\n")
            f.write(f"**Experiment:** {experiment_name}\n")
            f.write(f"**Timestamp:** {timestamp}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Performance Comparison\n\n")
            
            # Create performance table
            f.write("| Model | Final Train Loss | Final Val Loss | Best Val Loss | Convergence Epoch | Training Time (s) | Spectral Error |\n")
            f.write("|-------|------------------|----------------|---------------|-------------------|-------------------|----------------|\n")
            
            for model_name, result in results.items():
                f.write(f"| {model_name} | {result.final_train_loss:.2e} | {result.final_val_loss:.2e} | "
                       f"{result.best_val_loss:.2e} | {result.convergence_epoch} | {result.training_time:.1f} | "
                       f"{result.spectral_error:.2e} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Find best performing models
            best_train_loss = min(results.values(), key=lambda x: x.final_train_loss)
            best_val_loss = min(results.values(), key=lambda x: x.best_val_loss)
            best_spectral = min(results.values(), key=lambda x: x.spectral_error)
            fastest_convergence = min(results.values(), key=lambda x: x.convergence_epoch)
            
            f.write(f"- **Best Training Loss:** {best_train_loss.model_name} ({best_train_loss.final_train_loss:.2e})\n")
            f.write(f"- **Best Validation Loss:** {best_val_loss.model_name} ({best_val_loss.best_val_loss:.2e})\n")
            f.write(f"- **Best Spectral Approximation:** {best_spectral.model_name} ({best_spectral.spectral_error:.2e})\n")
            f.write(f"- **Fastest Convergence:** {fastest_convergence.model_name} ({fastest_convergence.convergence_epoch} epochs)\n\n")
            
            f.write("## Eigenvalue Analysis\n\n")
            
            for model_name, result in results.items():
                dominant_real = result.dominant_eigenvalue.real
                dominant_imag = result.dominant_eigenvalue.imag
                f.write(f"- **{model_name}:** {result.eigenvalue_count} eigenvalues, "
                       f"dominant lambda = {dominant_real:.4f} + {dominant_imag:.4f}i\n")
            
            f.write("\n## Computational Efficiency\n\n")
            
            total_time = sum(result.training_time for result in results.values())
            avg_memory = np.mean([result.memory_usage for result in results.values()])
            
            f.write(f"- **Total Training Time:** {total_time:.1f} seconds\n")
            f.write(f"- **Average Memory Usage:** {avg_memory:.1f} MB\n")
            
            # Training time comparison
            f.write("\n### Training Time Breakdown\n\n")
            for model_name, result in results.items():
                percentage = (result.training_time / total_time) * 100
                f.write(f"- **{model_name}:** {result.training_time:.1f}s ({percentage:.1f}%)\n")
        
        return str(summary_path)
    
    def create_metrics_dashboard_data(self,
                                    results: Dict[str, ModelResults],
                                    experiment_name: str,
                                    timestamp: str) -> str:
        """
        Create structured metrics data for dashboard visualization.
        
        Args:
            results: Dictionary of model results
            experiment_name: Name of experiment
            timestamp: Timestamp string
            
        Returns:
            Path to saved metrics file
        """
        metrics_path = self.output_dir / "metrics" / f"metrics_{experiment_name}_{timestamp}.json"
        
        # Organize metrics for visualization
        metrics_data = {
            "experiment_info": {
                "name": experiment_name,
                "timestamp": timestamp,
                "date": datetime.now().isoformat(),
                "n_models": len(results)
            },
            "performance_metrics": {},
            "convergence_metrics": {},
            "computational_metrics": {},
            "spectral_metrics": {}
        }
        
        for model_name, result in results.items():
            # Performance metrics
            metrics_data["performance_metrics"][model_name] = {
                "final_train_loss": float(result.final_train_loss),
                "final_val_loss": float(result.final_val_loss),
                "best_val_loss": float(result.best_val_loss),
                "loss_improvement": float(result.final_train_loss - result.best_val_loss)
            }
            
            # Convergence metrics
            metrics_data["convergence_metrics"][model_name] = {
                "convergence_epoch": int(result.convergence_epoch),
                "convergence_rate": float(1.0 / result.convergence_epoch) if result.convergence_epoch > 0 else 0.0
            }
            
            # Computational metrics
            metrics_data["computational_metrics"][model_name] = {
                "training_time": float(result.training_time),
                "memory_usage": float(result.memory_usage),
                "time_per_epoch": float(result.training_time / max(result.convergence_epoch, 1))
            }
            
            # Spectral metrics
            metrics_data["spectral_metrics"][model_name] = {
                "spectral_error": float(result.spectral_error),
                "eigenvalue_count": int(result.eigenvalue_count),
                "dominant_eigenvalue_real": float(result.dominant_eigenvalue.real),
                "dominant_eigenvalue_imag": float(result.dominant_eigenvalue.imag),
                "dominant_eigenvalue_magnitude": float(abs(result.dominant_eigenvalue))
            }
        
        # Add summary statistics
        metrics_data["summary_statistics"] = self._calculate_summary_statistics(results)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return str(metrics_path)
    
    def generate_complete_experiment_report(self,
                                          experiment_results: ExperimentResults) -> Dict[str, str]:
        """
        Generate complete experiment report with all documentation.
        
        Args:
            experiment_results: Complete experiment results
            
        Returns:
            Dictionary of generated file paths
        """
        config = experiment_results.config
        timestamp = config.timestamp
        experiment_name = config.experiment_name
        
        generated_files = {}
        
        # Save configuration
        config_path = self.save_experiment_config(config)
        generated_files['config'] = config_path
        
        # Save detailed results
        results_paths = self.save_model_results(
            experiment_results.model_results, 
            experiment_name, 
            timestamp
        )
        generated_files.update(results_paths)
        
        # Generate summary
        summary_path = self.generate_performance_summary(
            experiment_results.model_results,
            experiment_name,
            timestamp
        )
        generated_files['summary'] = summary_path
        
        # Create metrics dashboard data
        metrics_path = self.create_metrics_dashboard_data(
            experiment_results.model_results,
            experiment_name,
            timestamp
        )
        generated_files['metrics'] = metrics_path
        
        # Save figure paths registry
        figures_registry_path = self.output_dir / "summaries" / f"figures_{experiment_name}_{timestamp}.json"
        with open(figures_registry_path, 'w') as f:
            json.dump(experiment_results.figure_paths, f, indent=2)
        generated_files['figures_registry'] = str(figures_registry_path)
        
        return generated_files
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.complexfloating, np.complex128, np.complex64)):
            return complex(obj)
        else:
            return obj
    
    def _calculate_summary_statistics(self, results: Dict[str, ModelResults]) -> Dict[str, Any]:
        """Calculate summary statistics across all models."""
        train_losses = [r.final_train_loss for r in results.values()]
        val_losses = [r.best_val_loss for r in results.values()]
        spectral_errors = [r.spectral_error for r in results.values()]
        training_times = [r.training_time for r in results.values()]
        
        return {
            "train_loss_stats": {
                "mean": float(np.mean(train_losses)),
                "std": float(np.std(train_losses)),
                "min": float(np.min(train_losses)),
                "max": float(np.max(train_losses))
            },
            "val_loss_stats": {
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses))
            },
            "spectral_error_stats": {
                "mean": float(np.mean(spectral_errors)),
                "std": float(np.std(spectral_errors)),
                "min": float(np.min(spectral_errors)),
                "max": float(np.max(spectral_errors))
            },
            "training_time_stats": {
                "total": float(np.sum(training_times)),
                "mean": float(np.mean(training_times)),
                "std": float(np.std(training_times))
            }
        }