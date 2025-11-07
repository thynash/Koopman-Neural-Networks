"""
Multi-Layer Perceptron implementation for Koopman operator learning.

This module implements a configurable MLP architecture that learns to approximate
Koopman operators on fractal dynamical systems through next-state prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from ..base.koopman_model import KoopmanModel


class MLPKoopman(KoopmanModel):
    """
    Multi-Layer Perceptron for learning Koopman operators on fractal systems.
    
    This model uses a feedforward neural network to learn the mapping from
    current state to next state, approximating the Koopman operator dynamics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLP model with configurable architecture.
        
        Args:
            config: Configuration dictionary containing:
                - hidden_dims: List of hidden layer dimensions
                - activation: Activation function ('relu', 'tanh', 'sigmoid')
                - dropout_rate: Dropout probability for regularization
                - input_dim: Input state dimension (default: 2 for 2D fractals)
                - output_dim: Output state dimension (default: 2 for 2D fractals)
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.hidden_dims = config.get('hidden_dims', [64, 128, 128, 64])
        self.activation_name = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.input_dim = config.get('input_dim', 2)
        self.output_dim = config.get('output_dim', 2)
        
        # Build the network layers
        self.layers = self._build_network()
        
        # Move model to appropriate device
        self.to(self.device)
        
    def _build_network(self) -> nn.ModuleList:
        """
        Build the MLP network architecture.
        
        Returns:
            ModuleList containing all network layers
        """
        layers = nn.ModuleList()
        
        # Get activation function
        activation_fn = self._get_activation_function()
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return layers
    
    def _get_activation_function(self) -> nn.Module:
        """
        Get activation function based on configuration.
        
        Returns:
            PyTorch activation function module
        """
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        
        if self.activation_name not in activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        
        return activation_map[self.activation_name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim) representing next states
        """
        # Ensure input is on correct device
        x = self.to_device(x)
        
        # Forward pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_operator_matrix(self) -> np.ndarray:
        """
        Extract linear approximation of the Koopman operator.
        
        This method computes the Jacobian of the network at the origin
        or uses the final linear layer as an approximation of the operator.
        
        Returns:
            Numpy array representing the learned Koopman operator matrix
        """
        # For MLP, we can use the final linear layer as operator approximation
        # or compute Jacobian at a reference point
        
        # Method 1: Use final linear layer weights
        final_layer = None
        for layer in reversed(self.layers):
            if isinstance(layer, nn.Linear):
                final_layer = layer
                break
        
        if final_layer is not None:
            return final_layer.weight.detach().cpu().numpy()
        
        # Method 2: Compute Jacobian at origin (fallback)
        return self._compute_jacobian_at_point(torch.zeros(1, self.input_dim))
    
    def _compute_jacobian_at_point(self, point: torch.Tensor) -> np.ndarray:
        """
        Compute Jacobian matrix of the network at a given point.
        
        Args:
            point: Point at which to compute Jacobian (shape: 1 x input_dim)
            
        Returns:
            Jacobian matrix as numpy array
        """
        point = self.to_device(point)
        point.requires_grad_(True)
        
        # Forward pass
        output = self.forward(point)
        
        # Compute Jacobian
        jacobian = torch.zeros(self.output_dim, self.input_dim)
        
        for i in range(self.output_dim):
            # Compute gradient of i-th output w.r.t. input
            grad_outputs = torch.zeros_like(output)
            grad_outputs[0, i] = 1.0
            
            grad_input = torch.autograd.grad(
                outputs=output,
                inputs=point,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]
            
            jacobian[i] = grad_input.squeeze()
        
        return jacobian.detach().cpu().numpy()
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform a single training step on a batch of data.
        
        Args:
            batch: Batch tensor containing (current_states, next_states)
            
        Returns:
            Training loss for this batch
        """
        # Extract current and next states from batch
        current_states = batch[:, :self.input_dim]
        next_states = batch[:, self.input_dim:]
        
        # Forward pass
        predicted_next = self.forward(current_states)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(predicted_next, next_states)
        
        return loss.item()
    
    def evaluate(self, test_data: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset tensor
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Extract states
            current_states = test_data[:, :self.input_dim]
            next_states = test_data[:, self.input_dim:]
            
            # Predictions
            predicted_next = self.forward(current_states)
            
            # Compute metrics
            mse_loss = nn.functional.mse_loss(predicted_next, next_states)
            mae_loss = nn.functional.l1_loss(predicted_next, next_states)
            
            # Compute relative error
            relative_error = torch.mean(
                torch.norm(predicted_next - next_states, dim=1) / 
                torch.norm(next_states, dim=1)
            )
            
        return {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'relative_error': relative_error.item()
        }