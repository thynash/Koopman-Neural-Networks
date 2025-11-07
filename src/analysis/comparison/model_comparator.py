"""
Unified model comparison framework for Koopman operator learning.

This module implements a comprehensive comparison system that evaluates multiple
neural architectures on identical datasets with standardized metrics and protocols.
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict

from src.models.base.koopman_model import KoopmanModel
from src.models.mlp.mlp_koopman import MLPKoopman
from src.models.deeponet.deeponet_koopman import DeepONetKoopman
from src.training.trainers.mlp_trainer import MLPTrainer
from src.training.trainers.deeponet_trainer import DeepONetTrainer
from src.data.datasets.trajectory_dataset import DatasetSplit
from src.analysis.spectral.koopman_spectral_extractor import KoopmanSpectralExtractor
from src.analysis.spectral.dmd_baseline import DMDBaseline


@dataclass
class ModelMetrics:
    """Data structure for storing comprehensive model evaluation metrics."""
    
    # Training metrics
    final_train_loss: float
    best_val_loss: float
    total_epochs: int
    training_time: float
    
    # Test performance metrics
    test_mse: float
    test_mae: float
    test_relative_error: float
    test_r2_score: float
    
    # Spectral analysis metrics
    spectral_error: float
    n_eigenvalues: int
    dominant_eigenvalue: complex
    spectral_radius: float
    
    # Computational efficiency metrics
    peak_memory_usage: float  # MB
    inference_time: float     # seconds per sample
    model_parameters: int
    
    # Additional metadata
    converged: bool
    early_stopped: bool
    metadata: Dict[str, Any]


@dataclass
class ComparisonResults:
    """Data structure for storing comparison results across multiple models."""
    
    model_metrics: Dict[str, ModelMetrics]
    dataset_info: Dict[str, Any]
    comparison_summary: Dict[str, Any]
    timestamp: str
    config: Dict[str, Any]


class ModelComparator:
    """
    Unified framework for comparing multiple Koopman operator learning models.
    
    This class implements fair comparison protocols with identical datasets,
    preprocessing, and evaluation metrics across different neural architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model comparator with configuration.
        
        Args:
            config: Configuration dictionary containing comparison parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Comparison parameters
        self.models_to_compare = config.get('models_to_compare', ['mlp', 'deeponet'])
        self.random_seed = config.get('random_seed', 42)
        self.n_runs = config.get('n_runs', 1)  # Number of runs for statistical significance
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'results/model_comparison'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Spectral analysis configuration
        self.spectral_config = config.get('spectral_analysis', {})
        
        # Initialize components
        self.spectral_extractor = KoopmanSpectralExtractor(self.spectral_config)
        self.dmd_baseline = DMDBaseline(self.spectral_config)
        
        # Results storage
        self.comparison_results = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / 'comparison.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible comparisons.
        
        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def create_model_and_trainer(self, model_type: str, model_config: Dict[str, Any]) -> Tuple[KoopmanModel, Any]:
        """
        Create model and trainer instances based on model type.
        
        Args:
            model_type: Type of model ('mlp', 'deeponet', 'lstm')
            model_config: Configuration for the model
            
        Returns:
            Tuple of (model, trainer) instances
        """
        if model_type == 'mlp':
            model = MLPKoopman(model_config)
            trainer = MLPTrainer(model_config)
            return model, trainer
            
        elif model_type == 'deeponet':
            model = DeepONetKoopman(model_config)
            trainer = DeepONetTrainer(model_config)
            return model, trainer
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def measure_computational_efficiency(self, model: KoopmanModel, 
                                       test_data: torch.Tensor) -> Dict[str, float]:
        """
        Measure computational efficiency metrics for a model.
        
        Args:
            model: Trained model to evaluate
            test_data: Test dataset for inference timing
            
        Returns:
            Dictionary containing efficiency metrics
        """
        model.eval()
        
        # Measure inference time
        n_samples = min(1000, len(test_data))
        sample_data = test_data[:n_samples]
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_data[:10])
        
        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, n_samples, 32):  # Process in batches
                batch = sample_data[i:i+32]
                _ = model(batch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        inference_time = (end_time - start_time) / n_samples
        
        # Count model parameters
        model_parameters = sum(p.numel() for p in model.parameters())
        
        return {
            'inference_time': inference_time,
            'model_parameters': model_parameters
        }
    
    def evaluate_single_model(self, model_type: str, dataset_split: DatasetSplit, 
                            run_id: int = 0) -> ModelMetrics:
        """
        Evaluate a single model on the given dataset.
        
        Args:
            model_type: Type of model to evaluate
            dataset_split: Dataset split for training/validation/testing
            run_id: Run identifier for multiple runs
            
        Returns:
            ModelMetrics object containing all evaluation results
        """
        self.logger.info(f"Evaluating {model_type} model (run {run_id})")
        
        # Set random seed for this run
        self.set_random_seed(self.random_seed + run_id)
        
        # Get model configuration
        model_config = self.config.get(f'{model_type}_config', {})
        model_config['output_dir'] = str(self.output_dir / f'{model_type}_run_{run_id}')
        
        # Create model and trainer
        model, trainer = self.create_model_and_trainer(model_type, model_config)
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        # Training phase
        start_time = time.time()
        
        try:
            training_results = trainer.train(dataset_split)
            training_time = time.time() - start_time
            
            # Update peak memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            # Extract training metrics
            final_train_loss = training_results['final_train_loss']
            best_val_loss = training_results['best_val_loss']
            total_epochs = training_results['total_epochs']
            test_metrics = training_results['test_metrics']
            
            # Check convergence
            converged = total_epochs < model_config.get('epochs', 100)
            early_stopped = converged
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_type}: {str(e)}")
            # Return default metrics for failed training
            return ModelMetrics(
                final_train_loss=float('inf'),
                best_val_loss=float('inf'),
                total_epochs=0,
                training_time=0.0,
                test_mse=float('inf'),
                test_mae=float('inf'),
                test_relative_error=float('inf'),
                test_r2_score=0.0,
                spectral_error=float('inf'),
                n_eigenvalues=0,
                dominant_eigenvalue=0+0j,
                spectral_radius=0.0,
                peak_memory_usage=0.0,
                inference_time=0.0,
                model_parameters=0,
                converged=False,
                early_stopped=False,
                metadata={'error': str(e)}
            )
        
        # Computational efficiency evaluation
        sample_states, _ = dataset_split.test[0]
        test_tensor = torch.stack([dataset_split.test[i][0] for i in range(min(100, len(dataset_split.test)))])
        efficiency_metrics = self.measure_computational_efficiency(trainer.model, test_tensor)
        
        # Spectral analysis
        try:
            operator_matrix = trainer.model.get_operator_matrix()
            spectral_results = self.spectral_extractor.extract_spectrum(operator_matrix)
            
            # Compute spectral error against DMD baseline
            # Create trajectory data for DMD
            all_states = []
            all_next_states = []
            for i in range(min(1000, len(dataset_split.test))):
                state, next_state = dataset_split.test[i]
                all_states.append(state.numpy())
                all_next_states.append(next_state.numpy())
            
            trajectory_data = np.column_stack([np.array(all_states), np.array(all_next_states)])
            dmd_results = self.dmd_baseline.compute_dmd_spectrum(trajectory_data)
            
            spectral_error = self.spectral_extractor.compute_spectral_error(spectral_results, dmd_results)
            
            # Extract spectral properties
            n_eigenvalues = len(spectral_results.eigenvalues)
            dominant_eigenvalue = spectral_results.eigenvalues[np.argmax(np.abs(spectral_results.eigenvalues))]
            spectral_radius = np.max(np.abs(spectral_results.eigenvalues))
            
        except Exception as e:
            self.logger.warning(f"Spectral analysis failed for {model_type}: {str(e)}")
            spectral_error = float('inf')
            n_eigenvalues = 0
            dominant_eigenvalue = 0+0j
            spectral_radius = 0.0
        
        # Create comprehensive metrics
        metrics = ModelMetrics(
            final_train_loss=final_train_loss,
            best_val_loss=best_val_loss,
            total_epochs=total_epochs,
            training_time=training_time,
            test_mse=test_metrics['mse_loss'],
            test_mae=test_metrics['mae_loss'],
            test_relative_error=test_metrics['relative_error'],
            test_r2_score=test_metrics['r2_score'],
            spectral_error=spectral_error,
            n_eigenvalues=n_eigenvalues,
            dominant_eigenvalue=dominant_eigenvalue,
            spectral_radius=spectral_radius,
            peak_memory_usage=peak_memory - initial_memory,
            inference_time=efficiency_metrics['inference_time'],
            model_parameters=efficiency_metrics['model_parameters'],
            converged=converged,
            early_stopped=early_stopped,
            metadata={
                'model_type': model_type,
                'run_id': run_id,
                'device': str(self.device)
            }
        )
        
        self.logger.info(f"Completed evaluation of {model_type} (run {run_id})")
        return metrics
    
    def run_comparison(self, dataset_split: DatasetSplit) -> ComparisonResults:
        """
        Run comprehensive comparison across all specified models.
        
        Args:
            dataset_split: Dataset split for training/validation/testing
            
        Returns:
            ComparisonResults object containing all comparison data
        """
        self.logger.info("Starting comprehensive model comparison")
        self.logger.info(f"Models to compare: {self.models_to_compare}")
        self.logger.info(f"Number of runs per model: {self.n_runs}")
        
        all_results = {}
        
        # Run evaluation for each model type
        for model_type in self.models_to_compare:
            model_results = []
            
            for run_id in range(self.n_runs):
                metrics = self.evaluate_single_model(model_type, dataset_split, run_id)
                model_results.append(metrics)
            
            # Store results (average across runs if multiple)
            if self.n_runs == 1:
                all_results[model_type] = model_results[0]
            else:
                # Compute average metrics across runs
                all_results[model_type] = self._average_metrics(model_results)
        
        # Create dataset info
        dataset_info = {
            'train_size': len(dataset_split.train),
            'val_size': len(dataset_split.validation),
            'test_size': len(dataset_split.test),
            'state_dim': dataset_split.train[0][0].shape[0]
        }
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(all_results)
        
        # Create final results object
        results = ComparisonResults(
            model_metrics=all_results,
            dataset_info=dataset_info,
            comparison_summary=comparison_summary,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            config=self.config
        )
        
        # Save results
        self._save_comparison_results(results)
        
        self.logger.info("Model comparison completed successfully")
        return results
    
    def _average_metrics(self, metrics_list: List[ModelMetrics]) -> ModelMetrics:
        """
        Average metrics across multiple runs.
        
        Args:
            metrics_list: List of ModelMetrics from different runs
            
        Returns:
            Averaged ModelMetrics
        """
        if not metrics_list:
            raise ValueError("Empty metrics list")
        
        if len(metrics_list) == 1:
            return metrics_list[0]
        
        # Average numerical fields
        avg_metrics = ModelMetrics(
            final_train_loss=np.mean([m.final_train_loss for m in metrics_list]),
            best_val_loss=np.mean([m.best_val_loss for m in metrics_list]),
            total_epochs=int(np.mean([m.total_epochs for m in metrics_list])),
            training_time=np.mean([m.training_time for m in metrics_list]),
            test_mse=np.mean([m.test_mse for m in metrics_list]),
            test_mae=np.mean([m.test_mae for m in metrics_list]),
            test_relative_error=np.mean([m.test_relative_error for m in metrics_list]),
            test_r2_score=np.mean([m.test_r2_score for m in metrics_list]),
            spectral_error=np.mean([m.spectral_error for m in metrics_list]),
            n_eigenvalues=int(np.mean([m.n_eigenvalues for m in metrics_list])),
            dominant_eigenvalue=np.mean([m.dominant_eigenvalue for m in metrics_list]),
            spectral_radius=np.mean([m.spectral_radius for m in metrics_list]),
            peak_memory_usage=np.mean([m.peak_memory_usage for m in metrics_list]),
            inference_time=np.mean([m.inference_time for m in metrics_list]),
            model_parameters=metrics_list[0].model_parameters,  # Should be same across runs
            converged=all(m.converged for m in metrics_list),
            early_stopped=any(m.early_stopped for m in metrics_list),
            metadata={
                'n_runs': len(metrics_list),
                'std_test_mse': np.std([m.test_mse for m in metrics_list]),
                'std_spectral_error': np.std([m.spectral_error for m in metrics_list])
            }
        )
        
        return avg_metrics
    
    def _generate_comparison_summary(self, results: Dict[str, ModelMetrics]) -> Dict[str, Any]:
        """
        Generate summary statistics and rankings for model comparison.
        
        Args:
            results: Dictionary mapping model names to their metrics
            
        Returns:
            Dictionary containing comparison summary
        """
        summary = {}
        
        # Performance rankings (lower is better for errors, higher for R²)
        test_mse_ranking = sorted(results.items(), key=lambda x: x[1].test_mse)
        spectral_error_ranking = sorted(results.items(), key=lambda x: x[1].spectral_error)
        r2_ranking = sorted(results.items(), key=lambda x: x[1].test_r2_score, reverse=True)
        
        summary['performance_rankings'] = {
            'test_mse': [(name, metrics.test_mse) for name, metrics in test_mse_ranking],
            'spectral_error': [(name, metrics.spectral_error) for name, metrics in spectral_error_ranking],
            'r2_score': [(name, metrics.test_r2_score) for name, metrics in r2_ranking]
        }
        
        # Efficiency rankings (lower is better)
        training_time_ranking = sorted(results.items(), key=lambda x: x[1].training_time)
        memory_ranking = sorted(results.items(), key=lambda x: x[1].peak_memory_usage)
        inference_time_ranking = sorted(results.items(), key=lambda x: x[1].inference_time)
        
        summary['efficiency_rankings'] = {
            'training_time': [(name, metrics.training_time) for name, metrics in training_time_ranking],
            'memory_usage': [(name, metrics.peak_memory_usage) for name, metrics in memory_ranking],
            'inference_time': [(name, metrics.inference_time) for name, metrics in inference_time_ranking]
        }
        
        # Best performing model overall (weighted score)
        weights = {
            'test_mse': 0.3,
            'spectral_error': 0.3,
            'r2_score': 0.2,
            'training_time': 0.1,
            'inference_time': 0.1
        }
        
        model_scores = {}
        for name, metrics in results.items():
            # Normalize metrics to [0, 1] range for scoring
            score = 0
            
            # Lower is better metrics (invert)
            mse_values = [m.test_mse for m in results.values()]
            normalized_mse = 1 - (metrics.test_mse - min(mse_values)) / (max(mse_values) - min(mse_values) + 1e-8)
            score += weights['test_mse'] * normalized_mse
            
            spectral_values = [m.spectral_error for m in results.values()]
            normalized_spectral = 1 - (metrics.spectral_error - min(spectral_values)) / (max(spectral_values) - min(spectral_values) + 1e-8)
            score += weights['spectral_error'] * normalized_spectral
            
            # Higher is better metrics
            r2_values = [m.test_r2_score for m in results.values()]
            normalized_r2 = (metrics.test_r2_score - min(r2_values)) / (max(r2_values) - min(r2_values) + 1e-8)
            score += weights['r2_score'] * normalized_r2
            
            # Efficiency metrics (lower is better, invert)
            time_values = [m.training_time for m in results.values()]
            normalized_time = 1 - (metrics.training_time - min(time_values)) / (max(time_values) - min(time_values) + 1e-8)
            score += weights['training_time'] * normalized_time
            
            inference_values = [m.inference_time for m in results.values()]
            normalized_inference = 1 - (metrics.inference_time - min(inference_values)) / (max(inference_values) - min(inference_values) + 1e-8)
            score += weights['inference_time'] * normalized_inference
            
            model_scores[name] = score
        
        best_model = max(model_scores.items(), key=lambda x: x[1])
        summary['best_overall_model'] = {
            'name': best_model[0],
            'score': best_model[1],
            'scoring_weights': weights
        }
        
        return summary
    
    def _save_comparison_results(self, results: ComparisonResults) -> None:
        """
        Save comparison results to files.
        
        Args:
            results: ComparisonResults to save
        """
        # Save as JSON
        results_dict = asdict(results)
        
        # Convert complex numbers to string representation for JSON serialization
        for model_name, metrics in results_dict['model_metrics'].items():
            if isinstance(metrics['dominant_eigenvalue'], complex):
                metrics['dominant_eigenvalue'] = str(metrics['dominant_eigenvalue'])
        
        json_path = self.output_dir / 'comparison_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        metrics_df = pd.DataFrame([
            {
                'model': name,
                **{k: v for k, v in asdict(metrics).items() if k != 'metadata'}
            }
            for name, metrics in results.model_metrics.items()
        ])
        
        csv_path = self.output_dir / 'comparison_metrics.csv'
        metrics_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Saved comparison results to {json_path} and {csv_path}")