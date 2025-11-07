"""
Abstract base class for fractal dynamical system generators.

This module defines the interface for generating trajectory data from
various fractal systems including IFS and Julia sets.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryData:
    """
    Data structure for storing fractal trajectory information.
    """
    states: np.ndarray          # Shape: (n_samples, state_dim)
    next_states: np.ndarray     # Shape: (n_samples, state_dim)
    system_params: Dict         # System-specific parameters
    metadata: Dict              # Generation metadata (timestamps, seeds, etc.)


class FractalGenerator(ABC):
    """
    Abstract base class for fractal dynamical system generators.
    
    All fractal system implementations (IFS, Julia sets, etc.) must inherit
    from this class and implement the required methods for trajectory generation.
    """
    
    def __init__(self, system_params: Dict[str, Any]):
        """
        Initialize the fractal generator with system parameters.
        
        Args:
            system_params: Dictionary containing system-specific parameters
        """
        self.system_params = system_params
        self.rng = np.random.RandomState(system_params.get('seed', 42))
        
    @abstractmethod
    def generate_trajectories(self, n_points: int, **kwargs) -> TrajectoryData:
        """
        Generate trajectory data from the fractal dynamical system.
        
        Args:
            n_points: Number of trajectory points to generate
            **kwargs: Additional generation parameters
            
        Returns:
            TrajectoryData object containing states and metadata
        """
        pass
    
    @abstractmethod
    def get_system_name(self) -> str:
        """
        Get the name of the fractal system.
        
        Returns:
            String identifier for the fractal system
        """
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate that system parameters are mathematically valid.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
    
    def save_dataset(self, data: TrajectoryData, filepath: str, format: str = 'npy') -> None:
        """
        Save trajectory dataset to file.
        
        Args:
            data: TrajectoryData object to save
            filepath: Path to save the dataset
            format: File format ('npy' or 'csv')
        """
        if format == 'npy':
            np.save(f"{filepath}_states.npy", data.states)
            np.save(f"{filepath}_next_states.npy", data.next_states)
            np.save(f"{filepath}_metadata.npy", data.metadata)
        elif format == 'csv':
            np.savetxt(f"{filepath}_states.csv", data.states, delimiter=',')
            np.savetxt(f"{filepath}_next_states.csv", data.next_states, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_dataset(self, filepath: str, format: str = 'npy') -> TrajectoryData:
        """
        Load trajectory dataset from file.
        
        Args:
            filepath: Path to load the dataset from
            format: File format ('npy' or 'csv')
            
        Returns:
            TrajectoryData object loaded from file
        """
        if format == 'npy':
            states = np.load(f"{filepath}_states.npy")
            next_states = np.load(f"{filepath}_next_states.npy")
            metadata = np.load(f"{filepath}_metadata.npy", allow_pickle=True).item()
        elif format == 'csv':
            states = np.loadtxt(f"{filepath}_states.csv", delimiter=',')
            next_states = np.loadtxt(f"{filepath}_next_states.csv", delimiter=',')
            metadata = {}
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return TrajectoryData(
            states=states,
            next_states=next_states,
            system_params=self.system_params,
            metadata=metadata
        )
    
    def create_train_test_split(self, data: TrajectoryData, 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15) -> Tuple[TrajectoryData, TrajectoryData, TrajectoryData]:
        """
        Split trajectory data into train/validation/test sets.
        
        Args:
            data: TrajectoryData to split
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        n_samples = len(data.states)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle indices
        indices = self.rng.permutation(n_samples)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_data = TrajectoryData(
            states=data.states[train_idx],
            next_states=data.next_states[train_idx],
            system_params=data.system_params,
            metadata={**data.metadata, 'split': 'train', 'indices': train_idx}
        )
        
        val_data = TrajectoryData(
            states=data.states[val_idx],
            next_states=data.next_states[val_idx],
            system_params=data.system_params,
            metadata={**data.metadata, 'split': 'validation', 'indices': val_idx}
        )
        
        test_data = TrajectoryData(
            states=data.states[test_idx],
            next_states=data.next_states[test_idx],
            system_params=data.system_params,
            metadata={**data.metadata, 'split': 'test', 'indices': test_idx}
        )
        
        return train_data, val_data, test_data