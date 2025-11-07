"""
Hyperparameter management utilities for MLP training.

This module provides utilities for hyperparameter configuration, validation,
and optimization for MLP Koopman operator learning.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class HyperparameterConfig:
    """
    Hyperparameter configuration and validation for MLP training.
    
    Provides default values, validation, and optimization utilities
    for MLP model hyperparameters.
    """
    
    # Default hyperparameter values
    DEFAULT_CONFIG = {
        'hidden_dims': [64, 128, 128, 64],
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'activation': 'relu',
        'dropout_rate': 0.1,
        'weight_decay': 0.0001,
        'validation_freq': 10,
        'checkpoint_freq': 25,
        'early_stopping_patience': 20,
        'min_delta': 1e-6
    }
    
    # Valid activation functions
    VALID_ACTIVATIONS = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
    
    # Parameter ranges for validation
    PARAM_RANGES = {
        'learning_rate': (1e-5, 1e-1),
        'batch_size': (1, 512),
        'epochs': (1, 1000),
        'dropout_rate': (0.0, 0.9),
        'weight_decay': (0.0, 1e-2),
        'validation_freq': (1, 100),
        'checkpoint_freq': (1, 100),
        'early_stopping_patience': (1, 100)
    }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default hyperparameter configuration.
        
        Returns:
            Dictionary with default hyperparameter values
        """
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and complete hyperparameter configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated and completed configuration
            
        Raises:
            ValueError: If configuration contains invalid values
        """
        # Start with default config
        validated_config = cls.get_default_config()
        
        # Update with provided config
        validated_config.update(config)
        
        # Validate individual parameters
        cls._validate_hidden_dims(validated_config['hidden_dims'])
        cls._validate_activation(validated_config['activation'])
        cls._validate_numeric_ranges(validated_config)
        
        return validated_config
    
    @classmethod
    def _validate_hidden_dims(cls, hidden_dims: List[int]) -> None:
        """Validate hidden layer dimensions."""
        if not isinstance(hidden_dims, list):
            raise ValueError("hidden_dims must be a list")
        
        if len(hidden_dims) < 1:
            raise ValueError("At least one hidden layer is required")
        
        if len(hidden_dims) > 10:
            raise ValueError("Too many hidden layers (max 10)")
        
        for dim in hidden_dims:
            if not isinstance(dim, int) or dim < 1:
                raise ValueError("Hidden layer dimensions must be positive integers")
            
            if dim > 2048:
                raise ValueError("Hidden layer dimension too large (max 2048)")
    
    @classmethod
    def _validate_activation(cls, activation: str) -> None:
        """Validate activation function."""
        if activation not in cls.VALID_ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}. "
                           f"Valid options: {cls.VALID_ACTIVATIONS}")
    
    @classmethod
    def _validate_numeric_ranges(cls, config: Dict[str, Any]) -> None:
        """Validate numeric parameter ranges."""
        for param, (min_val, max_val) in cls.PARAM_RANGES.items():
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{param} must be numeric")
                
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{param} must be between {min_val} and {max_val}")
    
    @classmethod
    def suggest_config_for_system(cls, system_type: str, n_points: int) -> Dict[str, Any]:
        """
        Suggest hyperparameter configuration based on system type and data size.
        
        Args:
            system_type: Type of fractal system ('sierpinski', 'barnsley', 'julia')
            n_points: Number of training points
            
        Returns:
            Suggested hyperparameter configuration
        """
        config = cls.get_default_config()
        
        # Adjust based on data size
        if n_points < 5000:
            # Small dataset - simpler model
            config['hidden_dims'] = [32, 64, 32]
            config['batch_size'] = 16
            config['epochs'] = 150
            config['learning_rate'] = 0.002
        elif n_points > 30000:
            # Large dataset - more complex model
            config['hidden_dims'] = [128, 256, 256, 128]
            config['batch_size'] = 64
            config['epochs'] = 80
            config['learning_rate'] = 0.0005
        
        # Adjust based on system type
        if system_type == 'julia':
            # Julia sets may need more capacity for complex dynamics
            config['hidden_dims'] = [dim * 2 for dim in config['hidden_dims']]
            config['dropout_rate'] = 0.15
        elif system_type == 'sierpinski':
            # Sierpinski gasket is relatively simple
            config['dropout_rate'] = 0.05
        
        return config
    
    @classmethod
    def create_grid_search_configs(cls, base_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create configurations for grid search hyperparameter optimization.
        
        Args:
            base_config: Base configuration to modify (uses default if None)
            
        Returns:
            List of configurations for grid search
        """
        if base_config is None:
            base_config = cls.get_default_config()
        
        # Define parameter grids
        learning_rates = [0.0001, 0.001, 0.01]
        hidden_architectures = [
            [32, 64, 32],
            [64, 128, 64],
            [64, 128, 128, 64],
            [128, 256, 128]
        ]
        dropout_rates = [0.0, 0.1, 0.2]
        
        configs = []
        
        for lr in learning_rates:
            for hidden_dims in hidden_architectures:
                for dropout in dropout_rates:
                    config = base_config.copy()
                    config.update({
                        'learning_rate': lr,
                        'hidden_dims': hidden_dims,
                        'dropout_rate': dropout
                    })
                    configs.append(config)
        
        return configs
    
    @classmethod
    def create_random_search_configs(cls, n_configs: int, base_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create random configurations for hyperparameter optimization.
        
        Args:
            n_configs: Number of random configurations to generate
            base_config: Base configuration to modify (uses default if None)
            
        Returns:
            List of random configurations
        """
        if base_config is None:
            base_config = cls.get_default_config()
        
        configs = []
        
        for _ in range(n_configs):
            config = base_config.copy()
            
            # Random learning rate (log scale)
            config['learning_rate'] = 10 ** np.random.uniform(-4, -2)
            
            # Random architecture
            n_layers = np.random.randint(2, 6)
            layer_sizes = np.random.choice([32, 64, 128, 256, 512], size=n_layers)
            config['hidden_dims'] = layer_sizes.tolist()
            
            # Random dropout rate
            config['dropout_rate'] = np.random.uniform(0.0, 0.3)
            
            # Random weight decay
            config['weight_decay'] = 10 ** np.random.uniform(-5, -3)
            
            # Random batch size
            config['batch_size'] = np.random.choice([16, 32, 64, 128])
            
            configs.append(config)
        
        return configs