#!/usr/bin/env python3
"""
Data generation script for all fractal systems.

This script generates trajectory datasets for all supported fractal systems
(Sierpinski gasket, Barnsley fern, Julia sets) with configurable parameters.
"""

import sys
import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.generators.ifs_generator import IFSGenerator
from data.generators.julia_generator import JuliaSetGenerator
from data.datasets.trajectory_dataset import TrajectoryDataset
from visualization.fractals.fractal_visualizer import FractalVisualizer


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for data generation."""
    return {
        'systems': {
            'sierpinski': {
                'n_points': 20000,
                'save_path': 'data/sierpinski_trajectories.npy',
                'visualize': True
            },
            'barnsley': {
                'n_points': 20000,
                'save_path': 'data/barnsley_trajectories.npy',
                'visualize': True
            },
            'julia': {
                'n_points': 15000,
                'c_real': -0.7269,
                'c_imag': 0.1889,
                'save_path': 'data/julia_trajectories.npy',
                'visualize': True
            }
        },
        'output_dir': 'data',
        'figures_dir': 'figures/fractal_attractors',
        'dataset_config': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'normalize': True,
            'seed': 42
        }
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return create_default_config()


def generate_system_data(system_name: str, 
                        system_config: Dict[str, Any], output_dir: Path,
                        figures_dir: Path, visualizer: FractalVisualizer) -> np.ndarray:
    """Generate data for a specific fractal system."""
    print(f"\nGenerating {system_name} trajectory data...")
    
    # Generate trajectories based on system type
    if system_name == 'sierpinski':
        from data.generators.ifs_generator import SierpinskiGasketGenerator
        generator = SierpinskiGasketGenerator({'seed': 42})
        trajectory_data = generator.generate_trajectories(
            n_points=system_config['n_points']
        )
        states = trajectory_data.states
        next_states = trajectory_data.next_states
    elif system_name == 'barnsley':
        from data.generators.ifs_generator import BarnsleyFernGenerator
        generator = BarnsleyFernGenerator({'seed': 42})
        trajectory_data = generator.generate_trajectories(
            n_points=system_config['n_points']
        )
        states = trajectory_data.states
        next_states = trajectory_data.next_states
    elif system_name == 'julia':
        generator = JuliaSetGenerator({
            'c_real': system_config.get('c_real', -0.7269),
            'c_imag': system_config.get('c_imag', 0.1889),
            'seed': 42
        })
        trajectory_data = generator.generate_trajectories(
            n_points=system_config['n_points']
        )
        states = trajectory_data.states
        next_states = trajectory_data.next_states
    else:
        raise ValueError(f"Unknown system type: {system_name}")
    
    print(f"Generated {len(states)} trajectory points")
    
    # Save trajectory data
    save_path = output_dir / system_config['save_path']
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    trajectory_data = np.column_stack([states, next_states])
    np.save(save_path, trajectory_data)
    print(f"Saved trajectory data to {save_path}")
    
    # Generate visualization if requested
    if system_config.get('visualize', False):
        print(f"Creating {system_name} attractor visualization...")
        figure_path = figures_dir / f"{system_name}_attractor.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualizer.plot_attractor(
            states=states,
            title=f"{system_name.title()} Attractor",
            save_path=str(figure_path),
            dpi=600
        )
        print(f"Saved attractor plot to {figure_path}")
    
    return trajectory_data


def main():
    """Main data generation function."""
    parser = argparse.ArgumentParser(description='Generate fractal trajectory datasets')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--systems',
        nargs='+',
        default=['sierpinski', 'barnsley', 'julia'],
        help='Fractal systems to generate (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override settings from command line
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.no_visualize:
        for system_config in config['systems'].values():
            system_config['visualize'] = False
    
    print("Fractal Data Generation")
    print("=" * 50)
    print(f"Systems to generate: {args.systems}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Figures directory: {config['figures_dir']}")
    
    # Create output directories
    output_dir = Path(config['output_dir'])
    figures_dir = Path(config['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'data_generation_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved configuration to {config_path}")
    
    # Initialize visualizer
    visualizer = FractalVisualizer()
    
    # Generate data for each requested system
    generated_data = {}
    
    for system_name in args.systems:
        if system_name not in config['systems']:
            print(f"Warning: No configuration found for system '{system_name}', skipping...")
            continue
        
        try:
            trajectory_data = generate_system_data(
                system_name=system_name,
                system_config=config['systems'][system_name],
                output_dir=output_dir,
                figures_dir=figures_dir,
                visualizer=visualizer
            )
            generated_data[system_name] = trajectory_data
            
        except Exception as e:
            print(f"Error generating data for {system_name}: {e}")
            continue
    
    # Create combined dataset summary
    print(f"\nData Generation Summary")
    print("=" * 30)
    
    total_points = 0
    for system_name, data in generated_data.items():
        n_points = len(data) // 2  # Each row has state and next_state
        total_points += n_points
        print(f"{system_name}: {n_points:,} trajectory points")
    
    print(f"Total: {total_points:,} trajectory points across {len(generated_data)} systems")
    
    # Save summary
    summary = {
        'systems_generated': list(generated_data.keys()),
        'total_points': total_points,
        'individual_counts': {
            name: len(data) // 2 for name, data in generated_data.items()
        },
        'config_used': config
    }
    
    summary_path = output_dir / 'generation_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"\nGeneration summary saved to {summary_path}")
    print("Data generation completed successfully!")


if __name__ == '__main__':
    main()