#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Koopman Fractal Spectral Learning

This script runs systematic experiments with different parameters and documents
all results for comprehensive analysis and comparison.
"""

import sys
import os
import json
import yaml
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our modules
from data.generators.ifs_generator import SierpinskiGasketGenerator, BarnsleyFernGenerator
from data.generators.julia_generator import JuliaSetGenerator
from visualization.fractals.fractal_visualizer import FractalVisualizer


class ExperimentRunner:
    """
    Comprehensive experiment runner for systematic parameter studies.
    """
    
    def __init__(self, base_output_dir: str = "experiments/results"):
        """Initialize experiment runner."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this experiment session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results tracking
        self.experiment_results = []
        self.visualizer = FractalVisualizer()
        
        print(f"Experiment session started: {self.session_id}")
        print(f"Results will be saved to: {self.session_dir}")
    
    def create_experiment_config(self, system_type: str, system_params: Dict[str, Any],
                               data_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create experiment configuration."""
        return {
            'experiment_id': f"{system_type}_{len(self.experiment_results):03d}",
            'system_type': system_type,
            'system_params': system_params,
            'data_params': data_params,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
    
    def run_data_generation_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single data generation experiment."""
        print(f"\nRunning experiment: {config['experiment_id']}")
        print(f"System: {config['system_type']}")
        print(f"Parameters: {config['system_params']}")
        
        start_time = time.time()
        
        try:
            # Create generator based on system type
            if config['system_type'] == 'sierpinski':
                generator = SierpinskiGasketGenerator(config['system_params'])
            elif config['system_type'] == 'barnsley':
                generator = BarnsleyFernGenerator(config['system_params'])
            elif config['system_type'] == 'julia':
                generator = JuliaSetGenerator(config['system_params'])
            else:
                raise ValueError(f"Unknown system type: {config['system_type']}")
            
            # Generate trajectory data
            trajectory_data = generator.generate_trajectories(**config['data_params'])
            
            # Calculate statistics
            states = trajectory_data.states
            next_states = trajectory_data.next_states
            
            stats = {
                'n_points': len(states),
                'state_bounds': {
                    'x_min': float(states[:, 0].min()),
                    'x_max': float(states[:, 0].max()),
                    'y_min': float(states[:, 1].min()),
                    'y_max': float(states[:, 1].max())
                },
                'state_statistics': {
                    'x_mean': float(states[:, 0].mean()),
                    'x_std': float(states[:, 0].std()),
                    'y_mean': float(states[:, 1].mean()),
                    'y_std': float(states[:, 1].std())
                },
                'step_statistics': {
                    'mean_step_size': float(np.linalg.norm(next_states - states, axis=1).mean()),
                    'max_step_size': float(np.linalg.norm(next_states - states, axis=1).max()),
                    'step_size_std': float(np.linalg.norm(next_states - states, axis=1).std())
                }
            }
            
            # Save data
            experiment_dir = self.session_dir / config['experiment_id']
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            data_path = experiment_dir / 'trajectory_data.npy'
            combined_data = np.column_stack([states, next_states])
            np.save(data_path, combined_data)
            
            # Create visualization
            viz_path = experiment_dir / 'attractor_visualization.png'
            self.visualizer.plot_attractor(
                states=states,
                title=f"{config['system_type'].title()} Attractor - {config['experiment_id']}",
                save_path=str(viz_path),
                dpi=600
            )
            
            # Save configuration and metadata
            config_path = experiment_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            metadata_path = experiment_dir / 'metadata.json'
            metadata = {
                **trajectory_data.metadata,
                'statistics': stats,
                'generation_time': time.time() - start_time,
                'data_path': str(data_path),
                'visualization_path': str(viz_path)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            result = {
                'config': config,
                'statistics': stats,
                'metadata': metadata,
                'success': True,
                'error': None,
                'experiment_dir': str(experiment_dir)
            }
            
            print(f"✓ Experiment completed successfully")
            print(f"  Generated {stats['n_points']} points")
            print(f"  Time: {metadata['generation_time']:.2f}s")
            
        except Exception as e:
            print(f"✗ Experiment failed: {e}")
            result = {
                'config': config,
                'statistics': None,
                'metadata': None,
                'success': False,
                'error': str(e),
                'experiment_dir': None
            }
        
        self.experiment_results.append(result)
        return result
    
    def run_sierpinski_parameter_study(self) -> List[Dict[str, Any]]:
        """Run parameter study for Sierpinski gasket with different data sizes."""
        print("\n" + "="*60)
        print("SIERPINSKI GASKET PARAMETER STUDY")
        print("="*60)
        
        results = []
        
        # Test different data sizes
        data_sizes = [5000, 10000, 20000, 50000]
        
        for n_points in data_sizes:
            config = self.create_experiment_config(
                system_type='sierpinski',
                system_params={'seed': 42},
                data_params={'n_points': n_points}
            )
            
            result = self.run_data_generation_experiment(config)
            results.append(result)
        
        return results
    
    def run_barnsley_parameter_study(self) -> List[Dict[str, Any]]:
        """Run parameter study for Barnsley fern with different data sizes."""
        print("\n" + "="*60)
        print("BARNSLEY FERN PARAMETER STUDY")
        print("="*60)
        
        results = []
        
        # Test different data sizes
        data_sizes = [5000, 10000, 20000, 50000]
        
        for n_points in data_sizes:
            config = self.create_experiment_config(
                system_type='barnsley',
                system_params={'seed': 42},
                data_params={'n_points': n_points}
            )
            
            result = self.run_data_generation_experiment(config)
            results.append(result)
        
        return results
    
    def run_julia_parameter_study(self) -> List[Dict[str, Any]]:
        """Run parameter study for Julia sets with different parameters."""
        print("\n" + "="*60)
        print("JULIA SET PARAMETER STUDY")
        print("="*60)
        
        results = []
        
        # Test different Julia set parameters
        julia_params = [
            {'c_real': -0.7269, 'c_imag': 0.1889, 'name': 'Dragon'},
            {'c_real': -0.123, 'c_imag': 0.745, 'name': 'Rabbit'},
            {'c_real': -0.7, 'c_imag': 0.27015, 'name': 'Airplane'},
            {'c_real': -0.75, 'c_imag': 0.11, 'name': 'Spiral'}
        ]
        
        # Test different data sizes for each parameter set
        data_sizes = [5000, 15000]
        
        for params in julia_params:
            for n_points in data_sizes:
                config = self.create_experiment_config(
                    system_type='julia',
                    system_params={
                        'c_real': params['c_real'],
                        'c_imag': params['c_imag'],
                        'max_iter': 1000,
                        'escape_radius': 2.0,
                        'seed': 42
                    },
                    data_params={
                        'n_points': n_points,
                        'trajectory_length': 50,
                        'filter_divergent': True
                    }
                )
                
                # Add parameter set name to experiment ID
                config['experiment_id'] += f"_{params['name']}"
                
                result = self.run_data_generation_experiment(config)
                results.append(result)
        
        return results
    
    def create_summary_report(self) -> None:
        """Create comprehensive summary report of all experiments."""
        print("\n" + "="*60)
        print("CREATING SUMMARY REPORT")
        print("="*60)
        
        # Create summary statistics
        summary_data = []
        
        for result in self.experiment_results:
            if result['success']:
                config = result['config']
                stats = result['statistics']
                
                summary_data.append({
                    'experiment_id': config['experiment_id'],
                    'system_type': config['system_type'],
                    'n_points': stats['n_points'],
                    'x_range': stats['state_bounds']['x_max'] - stats['state_bounds']['x_min'],
                    'y_range': stats['state_bounds']['y_max'] - stats['state_bounds']['y_min'],
                    'mean_step_size': stats['step_statistics']['mean_step_size'],
                    'max_step_size': stats['step_statistics']['max_step_size'],
                    'generation_time': result['metadata']['generation_time'],
                    'success': True
                })
            else:
                summary_data.append({
                    'experiment_id': result['config']['experiment_id'],
                    'system_type': result['config']['system_type'],
                    'success': False,
                    'error': result['error']
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary report
        summary_path = self.session_dir / 'experiment_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Create detailed report
        report_path = self.session_dir / 'experiment_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Koopman Fractal Spectral Learning - Experiment Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Total Experiments:** {len(self.experiment_results)}\\n")
            f.write(f"**Successful Experiments:** {len([r for r in self.experiment_results if r['success']])}\\n\\n")
            
            f.write("## Experiment Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")
            
            # System-specific summaries
            for system_type in ['sierpinski', 'barnsley', 'julia']:
                system_results = [r for r in self.experiment_results 
                                if r['success'] and r['config']['system_type'] == system_type]
                
                if system_results:
                    f.write(f"## {system_type.title()} System Results\n\n")
                    
                    for result in system_results:
                        config = result['config']
                        stats = result['statistics']
                        
                        f.write(f"### {config['experiment_id']}\n\n")
                        f.write(f"- **Points Generated:** {stats['n_points']:,}\\n")
                        f.write(f"- **State Space Bounds:** x ∈ [{stats['state_bounds']['x_min']:.3f}, {stats['state_bounds']['x_max']:.3f}], y ∈ [{stats['state_bounds']['y_min']:.3f}, {stats['state_bounds']['y_max']:.3f}]\\n")
                        f.write(f"- **Mean Step Size:** {stats['step_statistics']['mean_step_size']:.6f}\\n")
                        f.write(f"- **Generation Time:** {result['metadata']['generation_time']:.2f}s\\n")
                        
                        if system_type == 'julia':
                            sys_params = config['system_params']
                            f.write(f"- **Julia Parameter:** c = {sys_params['c_real']:.4f} + {sys_params['c_imag']:.4f}i\\n")
                        
                        f.write(f"- **Data Path:** `{result['experiment_dir']}/trajectory_data.npy`\\n")
                        f.write(f"- **Visualization:** `{result['experiment_dir']}/attractor_visualization.png`\\n\\n")
        
        # Save complete results as JSON
        results_path = self.session_dir / 'complete_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2, default=str)
        
        print(f"✓ Summary report saved to: {report_path}")
        print(f"✓ Summary CSV saved to: {summary_path}")
        print(f"✓ Complete results saved to: {results_path}")
        
        # Print summary statistics
        successful_experiments = [r for r in self.experiment_results if r['success']]
        
        print(f"\nEXPERIMENT SESSION SUMMARY")
        print(f"Total experiments: {len(self.experiment_results)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(self.experiment_results) - len(successful_experiments)}")
        
        if successful_experiments:
            total_points = sum(r['statistics']['n_points'] for r in successful_experiments)
            total_time = sum(r['metadata']['generation_time'] for r in successful_experiments)
            
            print(f"Total data points generated: {total_points:,}")
            print(f"Total generation time: {total_time:.2f}s")
            print(f"Average generation rate: {total_points/total_time:.0f} points/second")


def main():
    """Main experiment runner function."""
    parser = argparse.ArgumentParser(description='Run comprehensive fractal experiments')
    parser.add_argument(
        '--systems',
        nargs='+',
        default=['sierpinski', 'barnsley', 'julia'],
        choices=['sierpinski', 'barnsley', 'julia'],
        help='Fractal systems to test'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results',
        help='Base output directory for results'
    )
    
    args = parser.parse_args()
    
    print("KOOPMAN FRACTAL SPECTRAL LEARNING - COMPREHENSIVE EXPERIMENTS")
    print("=" * 80)
    print(f"Systems to test: {args.systems}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.output_dir)
    
    # Run experiments for each requested system
    if 'sierpinski' in args.systems:
        runner.run_sierpinski_parameter_study()
    
    if 'barnsley' in args.systems:
        runner.run_barnsley_parameter_study()
    
    if 'julia' in args.systems:
        runner.run_julia_parameter_study()
    
    # Create comprehensive summary report
    runner.create_summary_report()
    
    print(f"\n🎉 All experiments completed successfully!")
    print(f"📊 Results available in: {runner.session_dir}")


if __name__ == '__main__':
    main()