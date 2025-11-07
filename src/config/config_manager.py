"""
Configuration management system for experiments.

This module provides utilities for loading, validating, and managing
experiment configurations across different components of the system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    architecture: str           # 'mlp', 'deeponet', 'lstm'
    hidden_dims: list          # Layer dimensions
    learning_rate: float       # Training learning rate
    batch_size: int           # Training batch size
    epochs: int               # Training epochs
    activation: str           # Activation function type
    dropout_rate: float = 0.0 # Dropout rate
    weight_decay: float = 0.0 # L2 regularization


@dataclass
class DataConfig:
    """Configuration for data generation and preprocessing."""
    system_type: str          # 'sierpinski', 'barnsley', 'julia'
    n_points: int            # Number of trajectory points
    train_ratio: float = 0.7 # Training data fraction
    val_ratio: float = 0.15  # Validation data fraction
    test_ratio: float = 0.15 # Test data fraction
    seed: int = 42           # Random seed
    normalize: bool = True   # Whether to normalize data


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis."""
    reference_method: str = 'dmd'  # Reference method for comparison
    n_eigenvalues: int = 10        # Number of eigenvalues to extract
    tolerance: float = 1e-6        # Numerical tolerance
    max_iterations: int = 1000     # Maximum iterations for eigensolvers


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str                    # Experiment name
    description: str            # Experiment description
    model: ModelConfig          # Model configuration
    data: DataConfig           # Data configuration
    spectral: SpectralConfig   # Spectral analysis configuration
    output_dir: str           # Output directory
    save_checkpoints: bool = True  # Whether to save model checkpoints
    log_level: str = 'INFO'   # Logging level


class ConfigManager:
    """
    Manages experiment configurations with validation and serialization.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Load experiment configuration from file.
        
        Args:
            config_path: Path to configuration file (.json or .yaml)
            
        Returns:
            ExperimentConfig object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Convert to ExperimentConfig
        return self._dict_to_config(config_dict)
    
    def save_config(self, config: ExperimentConfig, config_path: Union[str, Path]) -> None:
        """
        Save experiment configuration to file.
        
        Args:
            config: ExperimentConfig to save
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_dict = asdict(config)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """
        Convert dictionary to ExperimentConfig object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig object
        """
        # Extract nested configurations
        model_config = ModelConfig(**config_dict['model'])
        data_config = DataConfig(**config_dict['data'])
        spectral_config = SpectralConfig(**config_dict.get('spectral', {}))
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            model=model_config,
            data=data_config,
            spectral=spectral_config,
            output_dir=config_dict['output_dir'],
            save_checkpoints=config_dict.get('save_checkpoints', True),
            log_level=config_dict.get('log_level', 'INFO')
        )
        
        return experiment_config
    
    def validate_config(self, config: ExperimentConfig) -> bool:
        """
        Validate experiment configuration.
        
        Args:
            config: ExperimentConfig to validate
            
        Returns:
            True if configuration is valid
        """
        try:
            # Validate model configuration
            assert config.model.architecture in ['mlp', 'deeponet', 'lstm'], \
                f"Invalid architecture: {config.model.architecture}"
            assert config.model.learning_rate > 0, "Learning rate must be positive"
            assert config.model.batch_size > 0, "Batch size must be positive"
            assert config.model.epochs > 0, "Epochs must be positive"
            
            # Validate data configuration
            assert config.data.system_type in ['sierpinski', 'barnsley', 'julia'], \
                f"Invalid system type: {config.data.system_type}"
            assert config.data.n_points > 0, "Number of points must be positive"
            
            # Validate split ratios
            total_ratio = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
            assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total_ratio}"
            
            # Validate spectral configuration
            assert config.spectral.n_eigenvalues > 0, "Number of eigenvalues must be positive"
            assert config.spectral.tolerance > 0, "Tolerance must be positive"
            
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_default_config(self, name: str, architecture: str = 'mlp') -> ExperimentConfig:
        """
        Create a default experiment configuration.
        
        Args:
            name: Experiment name
            architecture: Neural network architecture
            
        Returns:
            Default ExperimentConfig
        """
        model_config = ModelConfig(
            architecture=architecture,
            hidden_dims=[64, 128, 64],
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            activation='relu'
        )
        
        data_config = DataConfig(
            system_type='sierpinski',
            n_points=10000
        )
        
        spectral_config = SpectralConfig()
        
        return ExperimentConfig(
            name=name,
            description=f"Default {architecture} experiment on fractal dynamics",
            model=model_config,
            data=data_config,
            spectral=spectral_config,
            output_dir=f"results/{name}"
        )