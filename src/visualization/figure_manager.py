"""
Automated figure management and organization system.

This module provides centralized figure management with automated saving,
descriptive naming, and organization for publication-ready outputs.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np

from .publication_figures import PublicationFigureGenerator
from .fractals.fractal_visualizer import FractalVisualizer
from .training.training_visualizer import TrainingVisualizer
from .spectral.spectrum_visualizer import SpectrumVisualizer
from .result_documentation import ResultDocumentationGenerator, ExperimentConfig, ModelResults


class FigureManager:
    """
    Centralized figure management system for automated figure generation and organization.
    """
    
    def __init__(self, 
                 base_output_dir: str = "figures",
                 experiment_name: str = "koopman_fractal_experiment",
                 dpi: int = 600):
        """
        Initialize figure manager.
        
        Args:
            base_output_dir: Base directory for all figures
            experiment_name: Name of current experiment
            dpi: Resolution for all figures
        """
        self.base_output_dir = Path(base_output_dir)
        self.experiment_name = experiment_name
        self.dpi = dpi
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment-specific directory
        self.experiment_dir = self.base_output_dir / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'fractals': self.experiment_dir / "fractals",
            'training': self.experiment_dir / "training", 
            'spectral': self.experiment_dir / "spectral",
            'comparative': self.experiment_dir / "comparative",
            'publication': self.experiment_dir / "publication_ready"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # Initialize visualizers
        self.pub_generator = PublicationFigureGenerator(
            output_dir=str(self.subdirs['publication']), dpi=dpi
        )
        self.fractal_viz = FractalVisualizer(
            output_dir=str(self.subdirs['fractals']), dpi=dpi
        )
        self.training_viz = TrainingVisualizer(
            output_dir=str(self.subdirs['training']), dpi=dpi
        )
        self.spectrum_viz = SpectrumVisualizer(
            output_dir=str(self.subdirs['spectral']), dpi=dpi
        )
        
        # Initialize documentation generator
        self.doc_generator = ResultDocumentationGenerator(
            output_dir=str(self.experiment_dir / "documentation"),
            figures_dir=str(self.experiment_dir)
        )
        
        # Track generated figures
        self.figure_registry = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'dpi': dpi,
            'created_date': datetime.now().isoformat()
        }
    
    def generate_fractal_figures(self,
                                fractal_data: Dict[str, np.ndarray],
                                fractal_configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate all fractal-related figures.
        
        Args:
            fractal_data: Dictionary {fractal_name: trajectory_data}
            fractal_configs: Dictionary {fractal_name: config_dict}
            
        Returns:
            Dictionary of generated figure paths
        """
        fractal_figures = {}
        
        # Individual fractal attractors
        for fractal_name, data in fractal_data.items():
            config = fractal_configs.get(fractal_name, {})
            
            if 'julia' in fractal_name.lower():
                c_param = config.get('c_parameter', -0.7 + 0.27015j)
                filename = self.doc_generator.generate_descriptive_filename(
                    'julia_attractor', self.experiment_name, 
                    fractal_system=fractal_name, timestamp=self.timestamp
                )
                path = self.fractal_viz.visualize_julia_set(
                    data, c_param, save_name=filename.replace('.png', '')
                )
            else:
                filename = self.doc_generator.generate_descriptive_filename(
                    'ifs_attractor', self.experiment_name,
                    fractal_system=fractal_name, timestamp=self.timestamp
                )
                path = self.fractal_viz.visualize_ifs_attractor(
                    data, fractal_name, save_name=filename.replace('.png', '')
                )
            
            fractal_figures[f'{fractal_name}_attractor'] = path
        
        # Multi-fractal comparison
        if len(fractal_data) > 1:
            comparison_filename = self.doc_generator.generate_descriptive_filename(
                'fractal_comparison', self.experiment_name,
                model_names=list(fractal_data.keys()), timestamp=self.timestamp
            )
            comparison_path = self.fractal_viz.create_multi_fractal_comparison(
                fractal_data, save_name=comparison_filename.replace('.png', '')
            )
            fractal_figures['fractal_comparison'] = comparison_path
        
        self.figure_registry.update(fractal_figures)
        return fractal_figures
    
    def generate_training_figures(self,
                                training_histories: Dict[str, Dict[str, List[float]]],
                                model_names: List[str]) -> Dict[str, str]:
        """
        Generate all training-related figures.
        
        Args:
            training_histories: Training history data for each model
            model_names: List of model names
            
        Returns:
            Dictionary of generated figure paths
        """
        training_figures = {}
        
        # Loss curves comparison
        loss_filename = self.doc_generator.generate_descriptive_filename(
            'training_curves', self.experiment_name,
            model_names=model_names, timestamp=self.timestamp
        )
        loss_path = self.training_viz.plot_loss_curves(
            training_histories, save_name=loss_filename.replace('.png', '')
        )
        training_figures['loss_curves'] = loss_path
        
        # Spectral error evolution
        spectral_histories = {}
        for model_name, history in training_histories.items():
            if 'spectral_error' in history:
                spectral_histories[model_name] = history['spectral_error']
        
        if spectral_histories:
            spectral_filename = self.doc_generator.generate_descriptive_filename(
                'spectral_error_evolution', self.experiment_name,
                model_names=list(spectral_histories.keys()), timestamp=self.timestamp
            )
            spectral_path = self.training_viz.plot_spectral_error_evolution(
                spectral_histories, save_name=spectral_filename.replace('.png', '')
            )
            training_figures['spectral_error_evolution'] = spectral_path
        
        # Comprehensive training dashboard
        dashboard_filename = self.doc_generator.generate_descriptive_filename(
            'training_dashboard', self.experiment_name,
            model_names=model_names, timestamp=self.timestamp
        )
        dashboard_path = self.training_viz.plot_comprehensive_training_dashboard(
            training_histories, save_name=dashboard_filename.replace('.png', '')
        )
        training_figures['training_dashboard'] = dashboard_path
        
        self.figure_registry.update(training_figures)
        return training_figures
    
    def generate_spectral_figures(self,
                                eigenvalues_dict: Dict[str, np.ndarray],
                                reference_eigenvals: Optional[np.ndarray] = None,
                                eigenfunction_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, complex]]] = None) -> Dict[str, str]:
        """
        Generate all spectral analysis figures.
        
        Args:
            eigenvalues_dict: Dictionary {model_name: eigenvalues}
            reference_eigenvals: Reference eigenvalues (e.g., from DMD)
            eigenfunction_data: Dictionary {model_name: (eigenfunction, coordinates, eigenvalue)}
            
        Returns:
            Dictionary of generated figure paths
        """
        spectral_figures = {}
        
        # Eigenvalue spectrum comparison
        spectrum_filename = self.doc_generator.generate_descriptive_filename(
            'eigenvalue_spectrum', self.experiment_name,
            model_names=list(eigenvalues_dict.keys()), timestamp=self.timestamp
        )
        spectrum_path = self.spectrum_viz.plot_eigenvalue_spectrum(
            eigenvalues_dict, save_name=spectrum_filename.replace('.png', '')
        )
        spectral_figures['eigenvalue_spectrum'] = spectrum_path
        
        # Comparison with reference if available
        if reference_eigenvals is not None:
            comparison_filename = self.doc_generator.generate_descriptive_filename(
                'spectrum_vs_reference', self.experiment_name,
                model_names=list(eigenvalues_dict.keys()), timestamp=self.timestamp
            )
            comparison_path = self.spectrum_viz.plot_spectral_comparison_with_reference(
                eigenvalues_dict, reference_eigenvals,
                save_name=comparison_filename.replace('.png', '')
            )
            spectral_figures['spectrum_vs_reference'] = comparison_path
        
        # Eigenfunction visualizations
        if eigenfunction_data:
            for model_name, (eigenfunction, coordinates, eigenvalue) in eigenfunction_data.items():
                eigenfunction_filename = self.doc_generator.generate_descriptive_filename(
                    f'eigenfunction_{model_name}', self.experiment_name,
                    timestamp=self.timestamp
                )
                eigenfunction_path = self.spectrum_viz.plot_eigenfunction_visualization(
                    eigenfunction, coordinates, eigenvalue,
                    save_name=eigenfunction_filename.replace('.png', '')
                )
                spectral_figures[f'eigenfunction_{model_name}'] = eigenfunction_path
        
        self.figure_registry.update(spectral_figures)
        return spectral_figures
    
    def generate_comparative_figures(self,
                                   model_results: Dict[str, ModelResults],
                                   performance_metrics: List[str] = None) -> Dict[str, str]:
        """
        Generate comparative analysis figures.
        
        Args:
            model_results: Dictionary of model results
            performance_metrics: List of metrics to compare
            
        Returns:
            Dictionary of generated figure paths
        """
        if performance_metrics is None:
            performance_metrics = ['final_train_loss', 'best_val_loss', 'spectral_error', 'training_time']
        
        comparative_figures = {}
        
        # Performance comparison
        metrics_data = {}
        for model_name, result in model_results.items():
            metrics_data[model_name] = {
                'final_train_loss': result.final_train_loss,
                'best_val_loss': result.best_val_loss,
                'spectral_error': result.spectral_error,
                'training_time': result.training_time,
                'memory_usage': result.memory_usage,
                'convergence_epoch': result.convergence_epoch
            }
        
        performance_filename = self.doc_generator.generate_descriptive_filename(
            'performance_comparison', self.experiment_name,
            model_names=list(model_results.keys()), timestamp=self.timestamp
        )
        performance_path = self.pub_generator.create_performance_comparison_plot(
            metrics_data, performance_metrics,
            save_name=performance_filename.replace('.png', '')
        )
        comparative_figures['performance_comparison'] = performance_path
        
        # Convergence analysis
        convergence_data = {}
        for model_name, result in model_results.items():
            convergence_data[model_name] = {
                'final_loss': result.final_train_loss,
                'convergence_epoch': result.convergence_epoch,
                'best_val_loss': result.best_val_loss
            }
        
        convergence_filename = self.doc_generator.generate_descriptive_filename(
            'convergence_analysis', self.experiment_name,
            model_names=list(model_results.keys()), timestamp=self.timestamp
        )
        convergence_path = self.training_viz.plot_convergence_analysis(
            convergence_data, save_name=convergence_filename.replace('.png', '')
        )
        comparative_figures['convergence_analysis'] = convergence_path
        
        self.figure_registry.update(comparative_figures)
        return comparative_figures
    
    def create_publication_ready_collection(self) -> Dict[str, str]:
        """
        Create a collection of publication-ready figures with consistent formatting.
        
        Returns:
            Dictionary of publication-ready figure paths
        """
        pub_collection = {}
        
        # Copy key figures to publication directory with standardized names
        key_figures = {
            'fractal_comparison': 'Figure_1_Fractal_Attractors.png',
            'training_dashboard': 'Figure_2_Training_Analysis.png', 
            'eigenvalue_spectrum': 'Figure_3_Eigenvalue_Spectra.png',
            'performance_comparison': 'Figure_4_Performance_Comparison.png'
        }
        
        for fig_key, pub_name in key_figures.items():
            if fig_key in self.figure_registry:
                source_path = Path(self.figure_registry[fig_key])
                dest_path = self.subdirs['publication'] / pub_name
                
                if source_path.exists():
                    shutil.copy2(source_path, dest_path)
                    pub_collection[fig_key] = str(dest_path)
        
        return pub_collection
    
    def save_figure_registry(self) -> str:
        """
        Save complete figure registry with metadata.
        
        Returns:
            Path to saved registry file
        """
        registry_data = {
            'metadata': self.metadata,
            'figures': self.figure_registry,
            'subdirectories': {name: str(path) for name, path in self.subdirs.items()},
            'experiment_directory': str(self.experiment_dir)
        }
        
        registry_path = self.experiment_dir / "figure_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        return str(registry_path)
    
    def generate_complete_figure_set(self,
                                   fractal_data: Dict[str, np.ndarray],
                                   fractal_configs: Dict[str, Dict[str, Any]],
                                   training_histories: Dict[str, Dict[str, List[float]]],
                                   eigenvalues_dict: Dict[str, np.ndarray],
                                   model_results: Dict[str, ModelResults],
                                   reference_eigenvals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate complete set of figures for the experiment.
        
        Args:
            fractal_data: Fractal trajectory data
            fractal_configs: Fractal system configurations
            training_histories: Training history data
            eigenvalues_dict: Eigenvalue data for each model
            model_results: Complete model results
            reference_eigenvals: Reference eigenvalues (optional)
            
        Returns:
            Dictionary containing all generated figure information
        """
        print(f"Generating complete figure set for experiment: {self.experiment_name}")
        
        # Generate all figure categories
        fractal_figs = self.generate_fractal_figures(fractal_data, fractal_configs)
        print(f"Generated {len(fractal_figs)} fractal figures")
        
        training_figs = self.generate_training_figures(
            training_histories, list(model_results.keys())
        )
        print(f"Generated {len(training_figs)} training figures")
        
        spectral_figs = self.generate_spectral_figures(
            eigenvalues_dict, reference_eigenvals
        )
        print(f"Generated {len(spectral_figs)} spectral figures")
        
        comparative_figs = self.generate_comparative_figures(model_results)
        print(f"Generated {len(comparative_figs)} comparative figures")
        
        # Create publication-ready collection
        pub_figs = self.create_publication_ready_collection()
        print(f"Created {len(pub_figs)} publication-ready figures")
        
        # Save registry
        registry_path = self.save_figure_registry()
        
        return {
            'fractal_figures': fractal_figs,
            'training_figures': training_figs,
            'spectral_figures': spectral_figs,
            'comparative_figures': comparative_figs,
            'publication_figures': pub_figs,
            'registry_path': registry_path,
            'experiment_directory': str(self.experiment_dir),
            'total_figures': len(self.figure_registry)
        }