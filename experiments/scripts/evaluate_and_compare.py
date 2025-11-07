#!/usr/bin/env python3
"""
Evaluation and comparison script for all trained models.

This script loads trained models, performs spectral analysis, and generates
comprehensive comparative visualizations and metrics.
"""

import sys
import os
import argparse
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.datasets.trajectory_dataset import TrajectoryDataset
from analysis.spectral.spectral_analyzer import SpectralAnalyzer
from analysis.spectral.dmd_baseline import DMDBaseline
from analysis.comparison.model_comparator import ModelComparator
from analysis.comparison.benchmark_runner import BenchmarkRunner
from analysis.comparison.results_generator import ResultsGenerator
from visualization.spectral.spectrum_visualizer import SpectrumVisualizer
from visualization.training.training_visualizer import TrainingVisualizer
from visualization.publication_figures import PublicationFigures


def load_model_results(results_dir: Path) -> Dict[str, Any]:
    """Load training results for a model."""
    # Try different result file names
    result_files = [
        'mlp_results.json',
        'deeponet_results.json',
        'lstm_results.json',
        'training_results.json',
        'results.json'
    ]
    
    for result_file in result_files:
        result_path = results_dir / result_file
        if result_path.exists():
            with open(result_path, 'r') as f:
                return json.load(f)
    
    return {}


def load_operator_matrix(results_dir: Path) -> Optional[np.ndarray]:
    """Load operator matrix if available."""
    operator_path = results_dir / 'operator_matrix.npy'
    if operator_path.exists():
        return np.load(operator_path)
    return None


def create_evaluation_config() -> Dict[str, Any]:
    """Create default evaluation configuration."""
    return {
        'spectral_analysis': {
            'compute_eigenvalues': True,
            'compute_eigenfunctions': True,
            'max_eigenvalues': 50,
            'tolerance': 1e-10
        },
        'dmd_baseline': {
            'compute_dmd': True,
            'rank_truncation': None,
            'exact_dmd': True
        },
        'comparison_metrics': [
            'prediction_error',
            'spectral_error',
            'training_efficiency',
            'memory_usage'
        ],
        'visualization': {
            'create_spectrum_plots': True,
            'create_training_curves': True,
            'create_comparison_plots': True,
            'dpi': 600,
            'format': 'png'
        },
        'output_dir': 'results/evaluation_comparison'
    }


def perform_spectral_analysis(model_results: Dict[str, Any], 
                            operator_matrix: Optional[np.ndarray],
                            dataset: TrajectoryDataset,
                            config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform spectral analysis on a trained model."""
    print("  Performing spectral analysis...")
    
    analyzer = SpectralAnalyzer()
    spectral_results = {}
    
    if operator_matrix is not None:
        # Extract eigenvalues and eigenvectors
        eigenvalues, eigenvectors = analyzer.extract_eigenvalues(
            operator_matrix,
            max_eigenvalues=config['spectral_analysis']['max_eigenvalues']
        )
        
        spectral_results['eigenvalues'] = eigenvalues
        spectral_results['eigenvectors'] = eigenvectors
        spectral_results['num_eigenvalues'] = len(eigenvalues)
        
        # Compute spectral radius
        spectral_results['spectral_radius'] = np.max(np.abs(eigenvalues))
        
        print(f"    Extracted {len(eigenvalues)} eigenvalues")
        print(f"    Spectral radius: {spectral_results['spectral_radius']:.6f}")
    else:
        print("    No operator matrix available for spectral analysis")
        spectral_results['eigenvalues'] = None
        spectral_results['eigenvectors'] = None
    
    return spectral_results


def compute_dmd_baseline(dataset: TrajectoryDataset, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute DMD baseline for comparison."""
    print("Computing DMD baseline...")
    
    dmd = DMDBaseline()
    
    # Get training data
    train_data = []
    for batch in dataset.train_loader:
        states, next_states = batch
        train_data.append(np.column_stack([states.numpy(), next_states.numpy()]))
    
    train_data = np.vstack(train_data)
    states = train_data[:, :2]
    next_states = train_data[:, 2:]
    
    # Compute DMD
    dmd_results = dmd.compute_dmd(
        states=states,
        next_states=next_states,
        rank=config['dmd_baseline'].get('rank_truncation'),
        exact=config['dmd_baseline']['exact_dmd']
    )
    
    print(f"  DMD eigenvalues: {len(dmd_results['eigenvalues'])}")
    print(f"  DMD spectral radius: {np.max(np.abs(dmd_results['eigenvalues'])):.6f}")
    
    return dmd_results


def generate_comparison_metrics(model_results: Dict[str, Dict[str, Any]],
                              dmd_results: Dict[str, Any],
                              config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive comparison metrics."""
    print("Generating comparison metrics...")
    
    comparator = ModelComparator()
    
    # Prepare model data for comparison
    model_data = {}
    for model_name, results in model_results.items():
        if results.get('spectral_results', {}).get('eigenvalues') is not None:
            model_data[model_name] = {
                'eigenvalues': results['spectral_results']['eigenvalues'],
                'training_loss': results.get('final_train_loss'),
                'validation_loss': results.get('best_val_loss'),
                'test_metrics': results.get('test_metrics', {}),
                'training_time': results.get('training_time'),
                'total_epochs': results.get('total_epochs')
            }
    
    # Add DMD baseline
    model_data['DMD'] = {
        'eigenvalues': dmd_results['eigenvalues'],
        'training_loss': None,
        'validation_loss': None,
        'test_metrics': {},
        'training_time': dmd_results.get('computation_time'),
        'total_epochs': 0
    }
    
    # Compute comparison metrics
    comparison_results = comparator.compare_models(
        model_data=model_data,
        reference_eigenvalues=dmd_results['eigenvalues']
    )
    
    return comparison_results


def create_visualizations(model_results: Dict[str, Dict[str, Any]],
                         dmd_results: Dict[str, Any],
                         comparison_results: Dict[str, Any],
                         config: Dict[str, Any],
                         output_dir: Path) -> None:
    """Create comprehensive visualizations."""
    print("Creating visualizations...")
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizers
    spectrum_viz = SpectrumVisualizer()
    training_viz = TrainingVisualizer()
    pub_figs = PublicationFigures()
    
    # Create spectrum comparison plot
    if config['visualization']['create_spectrum_plots']:
        print("  Creating spectrum comparison plots...")
        
        eigenvalue_data = {}
        for model_name, results in model_results.items():
            eigenvals = results.get('spectral_results', {}).get('eigenvalues')
            if eigenvals is not None:
                eigenvalue_data[model_name] = eigenvals
        
        # Add DMD baseline
        eigenvalue_data['DMD'] = dmd_results['eigenvalues']
        
        spectrum_path = figures_dir / 'spectrum_comparison.png'
        spectrum_viz.plot_comparative_spectrum(
            eigenvalue_data=eigenvalue_data,
            title='Eigenvalue Spectrum Comparison',
            save_path=str(spectrum_path),
            dpi=config['visualization']['dpi']
        )
    
    # Create training curves
    if config['visualization']['create_training_curves']:
        print("  Creating training curve plots...")
        
        training_data = {}
        for model_name, results in model_results.items():
            if 'training_history' in results:
                training_data[model_name] = results['training_history']
        
        if training_data:
            training_path = figures_dir / 'training_curves.png'
            training_viz.plot_comparative_training_curves(
                training_histories=training_data,
                save_path=str(training_path),
                dpi=config['visualization']['dpi']
            )
    
    # Create comparison summary plots
    if config['visualization']['create_comparison_plots']:
        print("  Creating comparison summary plots...")
        
        # Performance comparison bar chart
        performance_path = figures_dir / 'performance_comparison.png'
        pub_figs.create_performance_comparison(
            comparison_results=comparison_results,
            save_path=str(performance_path),
            dpi=config['visualization']['dpi']
        )
    
    print(f"  All visualizations saved to {figures_dir}")


def main():
    """Main evaluation and comparison function."""
    parser = argparse.ArgumentParser(description='Evaluate and compare trained models')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/all_models_training',
        help='Directory containing trained model results'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/sierpinski_trajectories.npy',
        help='Path to trajectory data file'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['mlp', 'deeponet'],
        help='Models to evaluate'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to evaluation configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation_comparison',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--skip-dmd',
        action='store_true',
        help='Skip DMD baseline computation'
    )
    
    args = parser.parse_args()
    
    print("Model Evaluation and Comparison")
    print("=" * 50)
    print(f"Results directory: {args.results_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Models to evaluate: {args.models}")
    print(f"Output directory: {args.output_dir}")
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_evaluation_config()
    
    # Update output directory
    config['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'evaluation_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        trajectory_data = np.load(args.data_path)
        n_points = len(trajectory_data) // 2
        states = trajectory_data[:n_points]
        next_states = trajectory_data[n_points:]
        
        dataset = TrajectoryDataset(
            states=states,
            next_states=next_states,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            normalize=True,
            seed=42
        )
        print(f"Dataset loaded: {len(states)} trajectory points")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load model results and perform analysis
    results_dir = Path(args.results_dir)
    model_results = {}
    
    for model_name in args.models:
        model_dir = results_dir / f"{model_name}_results"
        if not model_dir.exists():
            print(f"Warning: Results directory not found for {model_name}: {model_dir}")
            continue
        
        print(f"\nEvaluating {model_name} model...")
        
        # Load training results
        training_results = load_model_results(model_dir)
        if not training_results:
            print(f"  Warning: No training results found for {model_name}")
            continue
        
        # Load operator matrix
        operator_matrix = load_operator_matrix(model_dir)
        
        # Perform spectral analysis
        spectral_results = perform_spectral_analysis(
            training_results, operator_matrix, dataset, config
        )
        
        # Combine results
        model_results[model_name] = {
            **training_results,
            'spectral_results': spectral_results
        }
        
        print(f"  {model_name} evaluation completed")
    
    if not model_results:
        print("No model results found. Exiting.")
        return
    
    # Compute DMD baseline
    dmd_results = {}
    if not args.skip_dmd:
        print("\n" + "="*30)
        try:
            dmd_results = compute_dmd_baseline(dataset, config)
        except Exception as e:
            print(f"Error computing DMD baseline: {e}")
            dmd_results = {'eigenvalues': np.array([])}
    
    # Generate comparison metrics
    print("\n" + "="*30)
    comparison_results = generate_comparison_metrics(
        model_results, dmd_results, config
    )
    
    # Create visualizations
    print("\n" + "="*30)
    create_visualizations(
        model_results, dmd_results, comparison_results, config, output_dir
    )
    
    # Save all results
    print("\nSaving results...")
    
    # Save individual model results
    for model_name, results in model_results.items():
        model_results_path = output_dir / f"{model_name}_evaluation.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_results[key][subkey] = subvalue.tolist()
                    else:
                        json_results[key][subkey] = subvalue
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        with open(model_results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    # Save DMD results
    if dmd_results:
        dmd_path = output_dir / 'dmd_baseline.json'
        dmd_json = {key: (value.tolist() if isinstance(value, np.ndarray) else value) 
                   for key, value in dmd_results.items()}
        with open(dmd_path, 'w') as f:
            json.dump(dmd_json, f, indent=2, default=str)
    
    # Save comparison results
    comparison_path = output_dir / 'comparison_results.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\nEvaluation Summary")
    print("=" * 30)
    print(f"Models evaluated: {list(model_results.keys())}")
    print(f"DMD baseline computed: {bool(dmd_results)}")
    print(f"Comparison metrics generated: {bool(comparison_results)}")
    print(f"Visualizations created: {config['visualization']['create_spectrum_plots']}")
    print(f"\nAll results saved to: {output_dir}")
    print("Evaluation and comparison completed!")


if __name__ == '__main__':
    main()