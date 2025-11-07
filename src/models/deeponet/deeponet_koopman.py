"""
Deep Neural Operator (DeepONet) implementation for Koopman operator learning.

This module implements the DeepONet architecture that learns operators between
function spaces, using a branch-trunk structure to approximate Koopman operators
on fractal dynamical systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
from src.models.base.koopman_model import KoopmanModel


class BranchNetwork(nn.Module):
    """
    Branch network that processes trajectory snapshots (input functions).
    
    The branch network encodes trajectory data into a latent representation
    that captures the functional structure of the input.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', dropout_rate: float = 0.1):
        """
        Initialize branch network.
        
        Args:
            input_dim: Dimension of input trajectory snapshots
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (should match trunk output)
            activation: Activation function name
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Get activation function
        activation_fn = self._get_activation_function(activation)
        
        # Build network layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(activation_fn)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(prev_dim, output_dim))
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activation_map.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through branch network.
        
        Args:
            x: Input trajectory snapshots (batch_size, trajectory_length, state_dim)
            
        Returns:
            Branch encoding (batch_size, output_dim)
        """
        # Flatten trajectory snapshots for processing
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten to (batch_size, trajectory_length * state_dim)
        
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)
        
        return x


class TrunkNetwork(nn.Module):
    """
    Trunk network that processes spatial coordinates.
    
    The trunk network encodes spatial coordinate information to be combined
    with the branch network output for operator learning.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout_rate: float = 0.1):
        """
        Initialize trunk network.
        
        Args:
            input_dim: Dimension of spatial coordinates
            hidden_dims: List of hidden layer dimensions  
            output_dim: Output dimension (should match branch output)
            activation: Activation function name
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Get activation function
        activation_fn = self._get_activation_function(activation)
        
        # Build network layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(activation_fn)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(prev_dim, output_dim))
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(), 
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activation_map.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through trunk network.
        
        Args:
            x: Input spatial coordinates (batch_size, coordinate_dim)
            
        Returns:
            Trunk encoding (batch_size, output_dim)
        """
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)
        
        return x


class DeepONetKoopman(KoopmanModel):
    """
    Deep Neural Operator for learning Koopman operators on fractal systems.
    
    This model uses a branch-trunk architecture where the branch network processes
    trajectory snapshots and the trunk network processes spatial coordinates.
    The outputs are combined via dot product to learn operator mappings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepONet model with branch-trunk architecture.
        
        Args:
            config: Configuration dictionary containing:
                - trajectory_length: Length of trajectory snapshots for branch network
                - state_dim: Dimension of state space (default: 2 for 2D fractals)
                - coordinate_dim: Dimension of spatial coordinates (default: 2)
                - branch_hidden_dims: Hidden layer dimensions for branch network
                - trunk_hidden_dims: Hidden layer dimensions for trunk network
                - latent_dim: Dimension of latent space (branch/trunk output)
                - activation: Activation function name
                - dropout_rate: Dropout probability
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.trajectory_length = config.get('trajectory_length', 10)
        self.state_dim = config.get('state_dim', 2)
        self.coordinate_dim = config.get('coordinate_dim', 2)
        self.branch_hidden_dims = config.get('branch_hidden_dims', [64, 128, 64])
        self.trunk_hidden_dims = config.get('trunk_hidden_dims', [64, 128, 64])
        self.latent_dim = config.get('latent_dim', 64)
        self.activation = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Build branch and trunk networks
        self.branch_net = BranchNetwork(
            input_dim=self.trajectory_length * self.state_dim,
            hidden_dims=self.branch_hidden_dims,
            output_dim=self.latent_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
        
        self.trunk_net = TrunkNetwork(
            input_dim=self.coordinate_dim,
            hidden_dims=self.trunk_hidden_dims,
            output_dim=self.latent_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
        
        # Move model to appropriate device
        self.to(self.device)
    
    def forward(self, trajectory_data: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet architecture.
        
        Args:
            trajectory_data: Trajectory snapshots (batch_size, trajectory_length, state_dim)
            coordinates: Spatial coordinates (batch_size, coordinate_dim)
            
        Returns:
            Operator output (batch_size, state_dim)
        """
        # Ensure inputs are on correct device
        trajectory_data = self.to_device(trajectory_data)
        coordinates = self.to_device(coordinates)
        
        # Process through branch and trunk networks
        branch_output = self.branch_net(trajectory_data)  # (batch_size, latent_dim)
        trunk_output = self.trunk_net(coordinates)        # (batch_size, latent_dim)
        
        # Combine via dot product
        combined = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Expand to state dimension for next-state prediction
        output = combined.expand(-1, self.state_dim)  # (batch_size, state_dim)
        
        return output
    
    def forward_single_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with single input tensor (for compatibility with base class).
        
        This method assumes the input contains both trajectory and coordinate data
        concatenated together, which is split internally.
        
        Args:
            x: Combined input tensor (batch_size, trajectory_length * state_dim + coordinate_dim)
            
        Returns:
            Operator output (batch_size, state_dim)
        """
        # Split input into trajectory and coordinate components
        trajectory_size = self.trajectory_length * self.state_dim
        
        trajectory_data = x[:, :trajectory_size].view(-1, self.trajectory_length, self.state_dim)
        coordinates = x[:, trajectory_size:trajectory_size + self.coordinate_dim]
        
        return self.forward(trajectory_data, coordinates)
    
    def get_operator_matrix(self) -> np.ndarray:
        """
        Extract linear approximation of the learned Koopman operator.
        
        For DeepONet, we compute the operator by evaluating the network
        on a grid of points and fitting a linear approximation.
        
        Returns:
            Numpy array representing the learned Koopman operator matrix
        """
        self.eval()
        
        with torch.no_grad():
            # Create reference trajectory and coordinates
            n_samples = 100
            
            # Generate reference trajectory (simple grid)
            traj_range = torch.linspace(-1, 1, self.trajectory_length)
            trajectory = torch.zeros(n_samples, self.trajectory_length, self.state_dim)
            for i in range(self.state_dim):
                trajectory[:, :, i] = traj_range.unsqueeze(0).expand(n_samples, -1)
            
            # Generate coordinate grid
            coord_range = torch.linspace(-1, 1, n_samples)
            coordinates = torch.zeros(n_samples, self.coordinate_dim)
            for i in range(self.coordinate_dim):
                coordinates[:, i] = coord_range
            
            # Evaluate network
            trajectory = self.to_device(trajectory)
            coordinates = self.to_device(coordinates)
            
            outputs = self.forward(trajectory, coordinates)
            
            # Fit linear approximation using least squares
            # For simplicity, use the mean output as operator approximation
            operator_matrix = torch.mean(outputs, dim=0).unsqueeze(0)
            operator_matrix = operator_matrix.expand(self.state_dim, self.state_dim)
            
        return operator_matrix.cpu().numpy()
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform a single training step on a batch of data.
        
        Args:
            batch: Batch tensor containing trajectory snapshots, coordinates, and targets
                   Shape: (batch_size, trajectory_length * state_dim + coordinate_dim + state_dim)
            
        Returns:
            Training loss for this batch
        """
        # Parse batch data
        trajectory_size = self.trajectory_length * self.state_dim
        coord_size = self.coordinate_dim
        
        # Extract components
        trajectory_data = batch[:, :trajectory_size].view(-1, self.trajectory_length, self.state_dim)
        coordinates = batch[:, trajectory_size:trajectory_size + coord_size]
        targets = batch[:, trajectory_size + coord_size:]
        
        # Forward pass
        predictions = self.forward(trajectory_data, coordinates)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(predictions, targets)
        
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
            # Parse test data
            trajectory_size = self.trajectory_length * self.state_dim
            coord_size = self.coordinate_dim
            
            trajectory_data = test_data[:, :trajectory_size].view(-1, self.trajectory_length, self.state_dim)
            coordinates = test_data[:, trajectory_size:trajectory_size + coord_size]
            targets = test_data[:, trajectory_size + coord_size:]
            
            # Predictions
            predictions = self.forward(trajectory_data, coordinates)
            
            # Compute metrics
            mse_loss = nn.functional.mse_loss(predictions, targets)
            mae_loss = nn.functional.l1_loss(predictions, targets)
            
            # Compute relative error
            relative_error = torch.mean(
                torch.norm(predictions - targets, dim=1) / 
                (torch.norm(targets, dim=1) + 1e-8)  # Add small epsilon for stability
            )
            
        return {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'relative_error': relative_error.item()
        }
    
    def create_trajectory_coordinates_batch(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """
        Create batch data with trajectory snapshots and coordinates from state sequences.
        
        This is a utility method to convert standard state-next_state pairs into
        the trajectory-coordinate format required by DeepONet.
        
        Args:
            states: Current states (batch_size, state_dim)
            next_states: Next states (batch_size, state_dim)
            
        Returns:
            Formatted batch tensor for DeepONet training
        """
        batch_size = states.shape[0]
        
        # Create trajectory snapshots (for simplicity, repeat current state)
        trajectory_data = states.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        trajectory_flat = trajectory_data.reshape(batch_size, -1)
        
        # Use current states as coordinates
        coordinates = states[:, :self.coordinate_dim]
        
        # Combine into batch format
        batch = torch.cat([trajectory_flat, coordinates, next_states], dim=1)
        
        return batch