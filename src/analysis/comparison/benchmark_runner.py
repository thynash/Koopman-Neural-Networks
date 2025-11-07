"""
Benchmark runner for computational efficiency measurement.

This module provides utilities for measuring and comparing computational
efficiency metrics across different Koopman operator learning models.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from contextlib import contextmanager

from src.models.base.koopman_model import KoopmanModel
from src.data.datasets.trajectory_dataset import TrajectoryDataset


@dataclass
class BenchmarkResults:
    """Data structure for storing benchmark results."""
    
    model_name: str
    
    # Training efficiency
    training_time: float  # seconds
    training_memory_peak: float  # MB
    training_memory_average: float  # MB
    
    # Inference efficiency
    inference_time_mean: float  # seconds per sample
    inference_time_std: float   # standard deviation
    inference_memory: float     # MB
    
    # Model complexity
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    
    # Hardware utilization
    cpu_utilization: float  # percentage
    gpu_utilization: Optional[float] = None  # percentage
    gpu_memory_used: Optional[float] = None  # MB
    
    # Scalability metrics
    batch_processing_times: Dict[int, float] = None  # batch_size -> time
    memory_scaling: Dict[int, float] = None  # batch_size -> memory
    
    # Additional metadata
    device: str = 'cpu'
    timestamp: float = 0.0
    config: Dict[str, Any] = None


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize memory monitor.
        
        Args:
            device: Device to monitor ('cpu' or 'cuda')
        """
        self.device = device
        self.process = psutil.Process()
        self.initial_memory = 0
        self.peak_memory = 0
        self.memory_samples = []
        
    def __enter__(self):
        """Start memory monitoring."""
        if self.device == 'cpu':
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        elif self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        self.peak_memory = self.initial_memory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop memory monitoring."""
        pass
    
    def sample_memory(self) -> float:
        """
        Sample current memory usage.
        
        Returns:
            Current memory usage in MB
        """
        if self.device == 'cpu':
            current_memory = self.process.memory_info().rss / 1024 / 1024
        elif self.device == 'cuda' and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            current_memory = 0
        
        self.memory_samples.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return self.peak_memory
    
    def get_average_memory(self) -> float:
        """Get average memory usage in MB."""
        if not self.memory_samples:
            return self.initial_memory
        return np.mean(self.memory_samples)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for model efficiency evaluation.
    
    Measures training time, inference speed, memory usage, and scalability
    across different batch sizes and input configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark runner.
        
        Args:
            config: Configuration dictionary for benchmarking
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Benchmark parameters
        self.warmup_iterations = config.get('warmup_iterations', 10)
        self.benchmark_iterations = config.get('benchmark_iterations', 100)
        self.batch_sizes = config.get('batch_sizes', [1, 8, 16, 32, 64])
        self.memory_sampling_interval = config.get('memory_sampling_interval', 0.1)  # seconds
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'results/benchmarks'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / 'benchmark.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def measure_model_complexity(self, model: KoopmanModel) -> Dict[str, Any]:
        """
        Measure model complexity metrics.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary containing complexity metrics
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_bytes = param_size + buffer_size
        model_size_mb = model_size_bytes / 1024 / 1024
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def benchmark_inference_speed(self, model: KoopmanModel, 
                                test_data: torch.Tensor,
                                batch_size: int = 32) -> Dict[str, float]:
        """
        Benchmark inference speed for a model.
        
        Args:
            model: Model to benchmark
            test_data: Test data for inference
            batch_size: Batch size for inference
            
        Returns:
            Dictionary containing inference timing metrics
        """
        model.eval()
        model.to(self.device)
        
        # Prepare data
        n_samples = min(len(test_data), self.benchmark_iterations * batch_size)
        test_subset = test_data[:n_samples].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for i in range(self.warmup_iterations):
                start_idx = (i * batch_size) % (len(test_subset) - batch_size)
                batch = test_subset[start_idx:start_idx + batch_size]
                _ = model(batch)
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark inference
        inference_times = []
        
        with torch.no_grad():
            for i in range(self.benchmark_iterations):
                start_idx = (i * batch_size) % (len(test_subset) - batch_size)
                batch = test_subset[start_idx:start_idx + batch_size]
                
                start_time = time.perf_counter()
                _ = model(batch)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) / batch_size)  # Per sample
        
        return {
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'inference_time_median': np.median(inference_times),
            'inference_time_min': np.min(inference_times),
            'inference_time_max': np.max(inference_times)
        }
    
    def benchmark_batch_scaling(self, model: KoopmanModel,
                              test_data: torch.Tensor) -> Dict[str, Dict[int, float]]:
        """
        Benchmark how performance scales with batch size.
        
        Args:
            model: Model to benchmark
            test_data: Test data for benchmarking
            
        Returns:
            Dictionary containing scaling metrics for different batch sizes
        """
        model.eval()
        model.to(self.device)
        
        batch_times = {}
        memory_usage = {}
        
        for batch_size in self.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            self.logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Memory monitoring
            with MemoryMonitor(str(self.device)) as monitor:
                # Benchmark timing
                timing_results = self.benchmark_inference_speed(model, test_data, batch_size)
                batch_times[batch_size] = timing_results['inference_time_mean']
                
                # Sample memory during inference
                monitor.sample_memory()
                memory_usage[batch_size] = monitor.get_peak_memory()
        
        return {
            'batch_processing_times': batch_times,
            'memory_scaling': memory_usage
        }
    
    def measure_hardware_utilization(self, model: KoopmanModel,
                                   test_data: torch.Tensor,
                                   duration: float = 10.0) -> Dict[str, float]:
        """
        Measure hardware utilization during model inference.
        
        Args:
            model: Model to benchmark
            test_data: Test data for inference
            duration: Duration to measure utilization (seconds)
            
        Returns:
            Dictionary containing utilization metrics
        """
        model.eval()
        model.to(self.device)
        
        # Prepare data
        batch_size = 32
        test_subset = test_data[:batch_size * 10].to(self.device)
        
        # Monitor CPU utilization
        cpu_samples = []
        gpu_samples = []
        gpu_memory_samples = []
        
        start_time = time.time()
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                # Run inference
                for i in range(10):  # Multiple batches per sample
                    start_idx = (i * batch_size) % (len(test_subset) - batch_size)
                    batch = test_subset[start_idx:start_idx + batch_size]
                    _ = model(batch)
                
                # Sample utilization
                cpu_samples.append(psutil.cpu_percent(interval=None))
                
                if torch.cuda.is_available():
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_samples.append(gpu_util.gpu)
                        
                        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_samples.append(gpu_mem.used / 1024 / 1024)  # MB
                    except ImportError:
                        # pynvml not available, use torch memory info
                        gpu_memory_samples.append(torch.cuda.memory_allocated() / 1024 / 1024)
                
                time.sleep(0.1)  # Sample every 100ms
        
        results = {
            'cpu_utilization': np.mean(cpu_samples) if cpu_samples else 0.0
        }
        
        if gpu_samples:
            results['gpu_utilization'] = np.mean(gpu_samples)
        
        if gpu_memory_samples:
            results['gpu_memory_used'] = np.mean(gpu_memory_samples)
        
        return results
    
    def run_comprehensive_benchmark(self, model: KoopmanModel,
                                  test_data: torch.Tensor,
                                  model_name: str,
                                  training_time: Optional[float] = None,
                                  training_memory: Optional[float] = None) -> BenchmarkResults:
        """
        Run comprehensive benchmark suite for a model.
        
        Args:
            model: Model to benchmark
            test_data: Test data for benchmarking
            model_name: Name of the model
            training_time: Optional training time (if available)
            training_memory: Optional training memory usage (if available)
            
        Returns:
            BenchmarkResults object containing all metrics
        """
        self.logger.info(f"Running comprehensive benchmark for {model_name}")
        
        # Model complexity
        complexity_metrics = self.measure_model_complexity(model)
        
        # Inference speed (default batch size)
        inference_metrics = self.benchmark_inference_speed(model, test_data, batch_size=32)
        
        # Batch scaling
        scaling_metrics = self.benchmark_batch_scaling(model, test_data)
        
        # Hardware utilization
        utilization_metrics = self.measure_hardware_utilization(model, test_data)
        
        # Inference memory usage
        with MemoryMonitor(str(self.device)) as monitor:
            _ = self.benchmark_inference_speed(model, test_data, batch_size=32)
            inference_memory = monitor.get_peak_memory()
        
        # Create results object
        results = BenchmarkResults(
            model_name=model_name,
            training_time=training_time or 0.0,
            training_memory_peak=training_memory or 0.0,
            training_memory_average=training_memory or 0.0,
            inference_time_mean=inference_metrics['inference_time_mean'],
            inference_time_std=inference_metrics['inference_time_std'],
            inference_memory=inference_memory,
            total_parameters=complexity_metrics['total_parameters'],
            trainable_parameters=complexity_metrics['trainable_parameters'],
            model_size_mb=complexity_metrics['model_size_mb'],
            cpu_utilization=utilization_metrics['cpu_utilization'],
            gpu_utilization=utilization_metrics.get('gpu_utilization'),
            gpu_memory_used=utilization_metrics.get('gpu_memory_used'),
            batch_processing_times=scaling_metrics['batch_processing_times'],
            memory_scaling=scaling_metrics['memory_scaling'],
            device=str(self.device),
            timestamp=time.time(),
            config=self.config
        )
        
        # Save results
        self._save_benchmark_results(results)
        
        self.logger.info(f"Benchmark completed for {model_name}")
        return results
    
    def compare_models(self, benchmark_results: List[BenchmarkResults]) -> Dict[str, Any]:
        """
        Compare benchmark results across multiple models.
        
        Args:
            benchmark_results: List of BenchmarkResults to compare
            
        Returns:
            Dictionary containing comparison analysis
        """
        if not benchmark_results:
            return {}
        
        comparison = {
            'models': [r.model_name for r in benchmark_results],
            'rankings': {},
            'relative_performance': {},
            'summary': {}
        }
        
        # Performance rankings (lower is better for time/memory, higher for utilization)
        metrics_to_rank = [
            ('training_time', False),  # Lower is better
            ('inference_time_mean', False),
            ('inference_memory', False),
            ('model_size_mb', False),
            ('total_parameters', False),
            ('cpu_utilization', True),  # Higher is better (more efficient use)
        ]
        
        for metric, higher_better in metrics_to_rank:
            values = [(r.model_name, getattr(r, metric, 0)) for r in benchmark_results]
            sorted_values = sorted(values, key=lambda x: x[1], reverse=higher_better)
            comparison['rankings'][metric] = sorted_values
        
        # Relative performance (normalized to best performer)
        for metric, higher_better in metrics_to_rank:
            values = [getattr(r, metric, 0) for r in benchmark_results]
            if not values or all(v == 0 for v in values):
                continue
            
            if higher_better:
                best_value = max(values)
                relative_perf = [v / best_value if best_value > 0 else 0 for v in values]
            else:
                best_value = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1
                relative_perf = [best_value / v if v > 0 else 0 for v in values]
            
            comparison['relative_performance'][metric] = {
                r.model_name: rel_perf 
                for r, rel_perf in zip(benchmark_results, relative_perf)
            }
        
        # Summary statistics
        comparison['summary'] = {
            'fastest_training': min(benchmark_results, key=lambda r: r.training_time).model_name,
            'fastest_inference': min(benchmark_results, key=lambda r: r.inference_time_mean).model_name,
            'most_memory_efficient': min(benchmark_results, key=lambda r: r.inference_memory).model_name,
            'smallest_model': min(benchmark_results, key=lambda r: r.total_parameters).model_name,
        }
        
        return comparison
    
    def _save_benchmark_results(self, results: BenchmarkResults) -> None:
        """
        Save benchmark results to file.
        
        Args:
            results: BenchmarkResults to save
        """
        # Convert to dictionary for JSON serialization
        results_dict = {
            'model_name': results.model_name,
            'training_time': results.training_time,
            'training_memory_peak': results.training_memory_peak,
            'training_memory_average': results.training_memory_average,
            'inference_time_mean': results.inference_time_mean,
            'inference_time_std': results.inference_time_std,
            'inference_memory': results.inference_memory,
            'total_parameters': results.total_parameters,
            'trainable_parameters': results.trainable_parameters,
            'model_size_mb': results.model_size_mb,
            'cpu_utilization': results.cpu_utilization,
            'gpu_utilization': results.gpu_utilization,
            'gpu_memory_used': results.gpu_memory_used,
            'batch_processing_times': results.batch_processing_times,
            'memory_scaling': results.memory_scaling,
            'device': results.device,
            'timestamp': results.timestamp,
            'config': results.config
        }
        
        # Save to JSON
        json_path = self.output_dir / f'{results.model_name}_benchmark.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Saved benchmark results to {json_path}")