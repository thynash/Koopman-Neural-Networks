"""
Demonstration of publication-ready visualization pipeline.

This script shows how to use the comprehensive visualization system to generate
high-resolution figures with automated documentation and organization.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from visualization import (
    FigureManager, 
    ExperimentConfig, 
    ModelResults,
    ExperimentResults
)


def create_sample_data():
    """Create sample data for demonstration."""
    
    # Sample fractal data
    fractal_data = {
        'sierpinski': np.random.rand(10000, 2) * 2 - 1,  # Placeholder data
        'barnsley': np.random.rand(8000, 2) * 3 - 1.5,
        'julia': np.random.rand(12000, 2) * 4 - 2
    }
    
    fractal_configs = {
        'sierpinski': {'system_type': 'ifs', 'n_transforms': 3},
        'barnsley': {'system_type': 'ifs', 'n_transforms': 4},
        'julia': {'system_type': 'julia', 'c_parameter': -0.7 + 0.27015j}
    }
    
    # Sample training histories
    epochs = 100
    training_histories = {
        'MLP': {
            'train_loss': np.exp(-np.linspace(0, 4, epochs)) + 0.01 * np.random.rand(epochs),
            'val_loss': np.exp(-np.linspace(0, 3.5, epochs)) + 0.02 * np.random.rand(epochs),
            'spectral_error': np.exp(-np.linspace(0, 3, epochs)) + 0.005 * np.random.rand(epochs),
            'learning_rate': 0.001 * np.exp(-np.linspace(0, 2, epochs)),
            'epoch_time': 2.5 + 0.5 * np.random.rand(epochs),
            'memory_usage': 150 + 20 * np.random.rand(epochs)
        },
        'DeepONet': {
            'train_loss': np.exp(-np.linspace(0, 3.8, epochs)) + 0.015 * np.random.rand(epochs),
            'val_loss': np.exp(-np.linspace(0, 3.2, epochs)) + 0.025 * np.random.rand(epochs),
            'spectral_error': np.exp(-np.linspace(0, 2.8, epochs)) + 0.008 * np.random.rand(epochs),
            'learning_rate': 0.0005 * np.exp(-np.linspace(0, 1.5, epochs)),
            'epoch_time': 4.2 + 0.8 * np.random.rand(epochs),
            'memory_usage': 280 + 40 * np.random.rand(epochs)
        },
        'LSTM': {
            'train_loss': np.exp(-np.linspace(0, 3.5, epochs)) + 0.02 * np.random.rand(epochs),
            'val_loss': np.exp(-np.linspace(0, 3.0, epochs)) + 0.03 * np.random.rand(epochs),
            'spectral_error': np.exp(-np.linspace(0, 2.5, epochs)) + 0.01 * np.random.rand(epochs),
            'learning_rate': 0.002 * np.exp(-np.linspace(0, 1.8, epochs)),
            'epoch_time': 3.8 + 0.6 * np.random.rand(epochs),
            'memory_usage': 220 + 30 * np.random.rand(epochs)
        }
    }
    
    # Sample eigenvalues
    eigenvalues_dict = {
        'MLP': np.array([0.95 + 0.1j, 0.85 - 0.15j, 0.75 + 0.2j, 0.65, 0.55 - 0.1j]),
        'DeepONet': np.array([0.92 + 0.08j, 0.88 - 0.12j, 0.78 + 0.18j, 0.68, 0.58 - 0.08j]),
        'LSTM': np.array([0.90 + 0.12j, 0.82 - 0.18j, 0.72 + 0.22j, 0.62, 0.52 - 0.12j])
    }
    
    # Reference eigenvalues (DMD)
    reference_eigenvals = np.array([0.93 + 0.09j, 0.86 - 0.14j, 0.76 + 0.19j, 0.66, 0.56 - 0.09j])
    
    # Sample model results
    model_results = {
        'MLP': ModelResults(
            model_name='MLP',
            final_train_loss=0.0123,
            final_val_loss=0.0156,
            best_val_loss=0.0145,
            convergence_epoch=78,
            training_time=195.5,
            memory_usage=165.2,
            spectral_error=0.0089,
            eigenvalue_count=5,
            dominant_eigenvalue=0.95 + 0.1j
        ),
        'DeepONet': ModelResults(
            model_name='DeepONet',
            final_train_loss=0.0098,
            final_val_loss=0.0134,
            best_val_loss=0.0128,
            convergence_epoch=85,
            training_time=357.2,
            memory_usage=295.8,
            spectral_error=0.0067,
            eigenvalue_count=5,
            dominant_eigenvalue=0.92 + 0.08j
        ),
        'LSTM': ModelResults(
            model_name='LSTM',
            final_train_loss=0.0145,
            final_val_loss=0.0178,
            best_val_loss=0.0165,
            convergence_epoch=92,
            training_time=285.7,
            memory_usage=235.4,
            spectral_error=0.0112,
            eigenvalue_count=5,
            dominant_eigenvalue=0.90 + 0.12j
        )
    }
    
    return (fractal_data, fractal_configs, training_histories, 
            eigenvalues_dict, reference_eigenvals, model_results)


def demonstrate_publication_pipeline():
    """Demonstrate the complete publication visualization pipeline."""
    
    print("=== Publication Visualization Pipeline Demo ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    (fractal_data, fractal_configs, training_histories, 
     eigenvalues_dict, reference_eigenvals, model_results) = create_sample_data()
    print("   ✓ Sample data created")
    
    # Initialize figure manager
    print("\n2. Initializing figure manager...")
    figure_manager = FigureManager(
        base_output_dir="figures",
        experiment_name="koopman_fractal_demo",
        dpi=600
    )
    print(f"   ✓ Figure manager initialized")
    print(f"   ✓ Experiment directory: {figure_manager.experiment_dir}")
    
    # Generate complete figure set
    print("\n3. Generating complete figure set...")
    figure_results = figure_manager.generate_complete_figure_set(
        fractal_data=fractal_data,
        fractal_configs=fractal_configs,
        training_histories=training_histories,
        eigenvalues_dict=eigenvalues_dict,
        model_results=model_results,
        reference_eigenvals=reference_eigenvals
    )
    
    print(f"   ✓ Generated {figure_results['total_figures']} figures total")
    print(f"   ✓ Fractal figures: {len(figure_results['fractal_figures'])}")
    print(f"   ✓ Training figures: {len(figure_results['training_figures'])}")
    print(f"   ✓ Spectral figures: {len(figure_results['spectral_figures'])}")
    print(f"   ✓ Comparative figures: {len(figure_results['comparative_figures'])}")
    print(f"   ✓ Publication-ready figures: {len(figure_results['publication_figures'])}")
    
    # Generate documentation
    print("\n4. Generating comprehensive documentation...")
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        experiment_name="koopman_fractal_demo",
        timestamp=figure_manager.timestamp,
        fractal_system="multi_system",
        model_architectures=list(model_results.keys()),
        hyperparameters={
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'optimizer': 'Adam'
        },
        data_parameters={
            'n_trajectories': 10000,
            'trajectory_length': 1000,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        },
        training_parameters={
            'early_stopping': True,
            'patience': 10,
            'lr_scheduler': 'ExponentialLR'
        }
    )
    
    # Create complete experiment results
    experiment_results = ExperimentResults(
        config=experiment_config,
        model_results=model_results,
        comparative_metrics={
            'best_model_train_loss': 'DeepONet',
            'best_model_spectral_error': 'DeepONet',
            'fastest_convergence': 'MLP'
        },
        figure_paths=figure_manager.figure_registry,
        summary_statistics={
            'total_training_time': sum(r.training_time for r in model_results.values()),
            'average_spectral_error': np.mean([r.spectral_error for r in model_results.values()])
        }
    )
    
    # Generate complete documentation
    doc_files = figure_manager.doc_generator.generate_complete_experiment_report(
        experiment_results
    )
    
    print("   ✓ Documentation generated:")
    for doc_type, path in doc_files.items():
        print(f"     - {doc_type}: {Path(path).name}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Experiment directory: {figure_manager.experiment_dir}")
    print(f"Total figures generated: {figure_results['total_figures']}")
    print(f"Documentation files: {len(doc_files)}")
    print(f"Figure registry: {figure_results['registry_path']}")
    
    print("\n=== Publication-Ready Outputs ===")
    pub_dir = figure_manager.subdirs['publication']
    print(f"Publication directory: {pub_dir}")
    
    if pub_dir.exists():
        pub_files = list(pub_dir.glob("*.png"))
        for pub_file in pub_files:
            print(f"  - {pub_file.name}")
    
    print("\n✓ Publication visualization pipeline demonstration complete!")
    
    return figure_manager.experiment_dir


if __name__ == "__main__":
    experiment_dir = demonstrate_publication_pipeline()
    print(f"\nAll outputs saved to: {experiment_dir}")
    print("You can now use these figures in publications, presentations, or reports.")