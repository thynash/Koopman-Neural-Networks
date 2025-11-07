"""
Performance metrics tracking for Koopman operator learning models.

This module provides comprehensive tracking of training, validation, and test
metrics for model comparison and analysis.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TrainingMetrics:
    """Data structure for tracking training metrics over time."""
    
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # Additional metrics
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    batch_time: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Data structure for comprehensive model evaluation metrics."""
    
    # Prediction accuracy metrics
    mse_loss: float
    mae_loss: float
    rmse_loss: float
    relative_error: float
    r2_score: float
    
    # Spectral analysis metrics
    spectral_error: Optional[float] = None
    eigenvalue_accuracy: Optional[float] = None
    spectral_radius: Optional[float] = None
    
    # Computational metrics
    inference_time: Optional[float] = None
    memory_footprint: Optional[float] = None
    
    # Statistical metrics
    prediction_variance: Optional[float] = None
    bias: Optional[float] = None
    
    # Metadata
    n_samples: int = 0
    timestamp: float = field(default_factory=time.time)


class PerformanceTracker:
    """
    Comprehensive performance tracking system for model comparison.
    
    Tracks training progress, evaluation metrics, computational efficiency,
    and provides utilities for analysis and visualization.
    """
    
    def __init__(self, model_name: str, output_dir: Optional[str] = None):
        """
        Initialize performance tracker.
        
        Args:
            model_name: Name of the model being tracked
            output_dir: Directory to save tracking results
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path(f'results/{model_name}_tracking')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.training_history: List[TrainingMetrics] = []
        self.evaluation_history: List[EvaluationMetrics] = []
        self.custom_metrics: Dict[str, List[Any]] = defaultdict(list)
        
        # Timing information
        self.start_time = None
        self.end_time = None
        
        # Configuration
        self.save_frequency = 10  # Save every N epochs
        
    def start_tracking(self) -> None:
        """Start tracking session."""
        self.start_time = time.time()
        
    def end_tracking(self) -> None:
        """End tracking session."""
        self.end_time = time.time()
        self.save_all_metrics()
        
    def log_training_metrics(self, epoch: int, train_loss: float, 
                           val_loss: Optional[float] = None,
                           learning_rate: float = 0.0,
                           **kwargs) -> None:
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            learning_rate: Current learning rate
            **kwargs: Additional metrics to track
        """
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            gradient_norm=kwargs.get('gradient_norm'),
            memory_usage=kwargs.get('memory_usage'),
            batch_time=kwargs.get('batch_time')
        )
        
        self.training_history.append(metrics)
        
        # Save periodically
        if epoch % self.save_frequency == 0:
            self.save_training_history()
    
    def log_evaluation_metrics(self, mse_loss: float, mae_loss: float,
                             relative_error: float, r2_score: float,
                             n_samples: int, **kwargs) -> None:
        """
        Log comprehensive evaluation metrics.
        
        Args:
            mse_loss: Mean squared error
            mae_loss: Mean absolute error
            relative_error: Relative prediction error
            r2_score: R-squared coefficient
            n_samples: Number of samples evaluated
            **kwargs: Additional metrics to track
        """
        rmse_loss = np.sqrt(mse_loss)
        
        metrics = EvaluationMetrics(
            mse_loss=mse_loss,
            mae_loss=mae_loss,
            rmse_loss=rmse_loss,
            relative_error=relative_error,
            r2_score=r2_score,
            n_samples=n_samples,
            spectral_error=kwargs.get('spectral_error'),
            eigenvalue_accuracy=kwargs.get('eigenvalue_accuracy'),
            spectral_radius=kwargs.get('spectral_radius'),
            inference_time=kwargs.get('inference_time'),
            memory_footprint=kwargs.get('memory_footprint'),
            prediction_variance=kwargs.get('prediction_variance'),
            bias=kwargs.get('bias')
        )
        
        self.evaluation_history.append(metrics)
        
    def log_custom_metric(self, metric_name: str, value: Any, 
                         timestamp: Optional[float] = None) -> None:
        """
        Log custom metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.custom_metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of training progress.
        
        Returns:
            Dictionary containing training summary
        """
        if not self.training_history:
            return {}
        
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history if m.val_loss is not None]
        
        summary = {
            'total_epochs': len(self.training_history),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'best_train_loss': min(train_losses) if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'training_time': self.get_total_training_time(),
            'convergence_epoch': self._find_convergence_epoch(),
            'loss_improvement': self._calculate_loss_improvement()
        }
        
        return summary
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of evaluation metrics.
        
        Returns:
            Dictionary containing evaluation summary
        """
        if not self.evaluation_history:
            return {}
        
        latest_eval = self.evaluation_history[-1]
        
        summary = {
            'final_mse': latest_eval.mse_loss,
            'final_mae': latest_eval.mae_loss,
            'final_rmse': latest_eval.rmse_loss,
            'final_relative_error': latest_eval.relative_error,
            'final_r2_score': latest_eval.r2_score,
            'spectral_error': latest_eval.spectral_error,
            'spectral_radius': latest_eval.spectral_radius,
            'inference_time': latest_eval.inference_time,
            'memory_footprint': latest_eval.memory_footprint
        }
        
        return summary
    
    def get_total_training_time(self) -> Optional[float]:
        """
        Get total training time in seconds.
        
        Returns:
            Training time in seconds, or None if tracking not completed
        """
        if self.start_time is None:
            return None
        
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def _find_convergence_epoch(self) -> Optional[int]:
        """
        Find the epoch where training converged (loss stopped improving significantly).
        
        Returns:
            Convergence epoch, or None if not converged
        """
        if len(self.training_history) < 10:
            return None
        
        val_losses = [m.val_loss for m in self.training_history if m.val_loss is not None]
        if len(val_losses) < 10:
            return None
        
        # Look for plateau in validation loss
        window_size = 5
        improvement_threshold = 0.001
        
        for i in range(window_size, len(val_losses)):
            recent_losses = val_losses[i-window_size:i]
            current_loss = val_losses[i]
            
            # Check if improvement is below threshold
            min_recent = min(recent_losses)
            if (min_recent - current_loss) / min_recent < improvement_threshold:
                return self.training_history[i].epoch
        
        return None
    
    def _calculate_loss_improvement(self) -> Optional[float]:
        """
        Calculate relative improvement in loss from start to end.
        
        Returns:
            Relative improvement ratio, or None if insufficient data
        """
        if len(self.training_history) < 2:
            return None
        
        initial_loss = self.training_history[0].train_loss
        final_loss = self.training_history[-1].train_loss
        
        if initial_loss == 0:
            return None
        
        return (initial_loss - final_loss) / initial_loss
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history:
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history if m.val_loss is not None]
        val_epochs = [m.epoch for m in self.training_history if m.val_loss is not None]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', color='blue')
        
        if val_losses:
            plt.plot(val_epochs, val_losses, label='Validation Loss', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_learning_rate_schedule(self, save_path: Optional[str] = None) -> None:
        """
        Plot learning rate schedule over training.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history:
            return
        
        epochs = [m.epoch for m in self.training_history]
        learning_rates = [m.learning_rate for m in self.training_history]
        
        if all(lr == 0 for lr in learning_rates):
            return  # No learning rate data
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, learning_rates, color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule - {self.model_name}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'learning_rate_schedule.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save_training_history(self) -> None:
        """Save training history to CSV file."""
        if not self.training_history:
            return
        
        df = pd.DataFrame([
            {
                'epoch': m.epoch,
                'train_loss': m.train_loss,
                'val_loss': m.val_loss,
                'learning_rate': m.learning_rate,
                'gradient_norm': m.gradient_norm,
                'memory_usage': m.memory_usage,
                'batch_time': m.batch_time,
                'timestamp': m.timestamp
            }
            for m in self.training_history
        ])
        
        csv_path = self.output_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
    
    def save_evaluation_history(self) -> None:
        """Save evaluation history to CSV file."""
        if not self.evaluation_history:
            return
        
        df = pd.DataFrame([
            {
                'mse_loss': m.mse_loss,
                'mae_loss': m.mae_loss,
                'rmse_loss': m.rmse_loss,
                'relative_error': m.relative_error,
                'r2_score': m.r2_score,
                'spectral_error': m.spectral_error,
                'eigenvalue_accuracy': m.eigenvalue_accuracy,
                'spectral_radius': m.spectral_radius,
                'inference_time': m.inference_time,
                'memory_footprint': m.memory_footprint,
                'prediction_variance': m.prediction_variance,
                'bias': m.bias,
                'n_samples': m.n_samples,
                'timestamp': m.timestamp
            }
            for m in self.evaluation_history
        ])
        
        csv_path = self.output_dir / 'evaluation_history.csv'
        df.to_csv(csv_path, index=False)
    
    def save_custom_metrics(self) -> None:
        """Save custom metrics to JSON file."""
        if not self.custom_metrics:
            return
        
        json_path = self.output_dir / 'custom_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(dict(self.custom_metrics), f, indent=2, default=str)
    
    def save_all_metrics(self) -> None:
        """Save all tracked metrics to files."""
        self.save_training_history()
        self.save_evaluation_history()
        self.save_custom_metrics()
        
        # Save summary
        summary = {
            'model_name': self.model_name,
            'training_summary': self.get_training_summary(),
            'evaluation_summary': self.get_evaluation_summary(),
            'tracking_start': self.start_time,
            'tracking_end': self.end_time
        }
        
        summary_path = self.output_dir / 'tracking_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def load_metrics(self, base_dir: Optional[str] = None) -> None:
        """
        Load previously saved metrics.
        
        Args:
            base_dir: Base directory to load from (defaults to self.output_dir)
        """
        load_dir = Path(base_dir) if base_dir else self.output_dir
        
        # Load training history
        training_csv = load_dir / 'training_history.csv'
        if training_csv.exists():
            df = pd.read_csv(training_csv)
            self.training_history = [
                TrainingMetrics(
                    epoch=row['epoch'],
                    train_loss=row['train_loss'],
                    val_loss=row['val_loss'] if pd.notna(row['val_loss']) else None,
                    learning_rate=row['learning_rate'],
                    gradient_norm=row['gradient_norm'] if pd.notna(row['gradient_norm']) else None,
                    memory_usage=row['memory_usage'] if pd.notna(row['memory_usage']) else None,
                    batch_time=row['batch_time'] if pd.notna(row['batch_time']) else None,
                    timestamp=row['timestamp']
                )
                for _, row in df.iterrows()
            ]
        
        # Load evaluation history
        eval_csv = load_dir / 'evaluation_history.csv'
        if eval_csv.exists():
            df = pd.read_csv(eval_csv)
            self.evaluation_history = [
                EvaluationMetrics(
                    mse_loss=row['mse_loss'],
                    mae_loss=row['mae_loss'],
                    rmse_loss=row['rmse_loss'],
                    relative_error=row['relative_error'],
                    r2_score=row['r2_score'],
                    spectral_error=row['spectral_error'] if pd.notna(row['spectral_error']) else None,
                    eigenvalue_accuracy=row['eigenvalue_accuracy'] if pd.notna(row['eigenvalue_accuracy']) else None,
                    spectral_radius=row['spectral_radius'] if pd.notna(row['spectral_radius']) else None,
                    inference_time=row['inference_time'] if pd.notna(row['inference_time']) else None,
                    memory_footprint=row['memory_footprint'] if pd.notna(row['memory_footprint']) else None,
                    prediction_variance=row['prediction_variance'] if pd.notna(row['prediction_variance']) else None,
                    bias=row['bias'] if pd.notna(row['bias']) else None,
                    n_samples=int(row['n_samples']),
                    timestamp=row['timestamp']
                )
                for _, row in df.iterrows()
            ]
        
        # Load custom metrics
        custom_json = load_dir / 'custom_metrics.json'
        if custom_json.exists():
            with open(custom_json, 'r') as f:
                self.custom_metrics = defaultdict(list, json.load(f))