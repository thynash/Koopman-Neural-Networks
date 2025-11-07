"""
Abstract base class for Koopman operator learning models.

This module defines the interface that all neural network architectures
must implement for learning Koopman operators on fractal dynamical systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch
import numpy as np


class KoopmanModel(ABC, torch.nn.Module):
    """
    Abstract base class for neural networks that learn Koopman operators.
    
    All neural architectures (MLP, DeepONet, LSTM) must inherit from this class
    and implement the required methods for training, evaluation, and spectral analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Koopman model with configuration parameters.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor containing state vectors or trajectory data
            
        Returns:
            Output tensor with predicted next states or Koopman-lifted observables
        """
        pass
    
    @abstractmethod
    def get_operator_matrix(self) -> np.ndarray:
        """
        Extract the learned Koopman operator matrix from the trained model.
        
        This method should linearize the learned dynamics around the attractor
        to obtain a matrix representation suitable for spectral analysis.
        
        Returns:
            Numpy array representing the learned Koopman operator matrix
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform a single training step on a batch of data.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Training loss for this batch
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model on test data and return performance metrics.
        
        Args:
            test_data: Test dataset for evaluation
            
        Returns:
            Dictionary containing evaluation metrics (loss, accuracy, etc.)
        """
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to the appropriate device (CPU/GPU).
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor moved to the model's device
        """
        return tensor.to(self.device)
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint including state dict and configuration.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'KoopmanModel':
        """
        Load model from checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model