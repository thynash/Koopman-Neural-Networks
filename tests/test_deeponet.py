"""
Unit tests for DeepONet Koopman operator learning implementation.

This module tests the branch network, trunk network, and complete DeepONet
architecture for operator learning on fractal dynamical systems.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import torch
import numpy as np
from typing import Dict, Any

from src.models.deeponet.deeponet_koopman import DeepONetKoopman, BranchNetwork, TrunkNetwork
from src.training.trainers.deeponet_trainer import DeepONetTrainer, OperatorLoss


class TestBranchNetwork(unittest.TestCase):
    """Test cases for the Branch Network component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 20  # trajectory_length * state_dim (10 * 2)
        self.hidden_dims = [32, 64, 32]
        self.output_dim = 16
        self.batch_size = 8
        
        self.branch_net = BranchNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation='relu',
            dropout_rate=0.1
        )
    
    def test_branch_network_initialization(self):
        """Test branch network initialization."""
        self.assertIsInstance(self.branch_net, BranchNetwork)
        self.assertEqual(len(self.branch_net.layers), 10)  # 3 linear + 3 activation + 3 dropout + 1 output
    
    def test_branch_network_forward_pass(self):
        """Test branch network forward pass with trajectory data."""
        # Create test trajectory data
        trajectory_data = torch.randn(self.batch_size, 10, 2)  # (batch, trajectory_length, state_dim)
        
        # Forward pass
        output = self.branch_net(trajectory_data)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_branch_network_different_activations(self):
        """Test branch network with different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid']
        
        for activation in activations:
            branch_net = BranchNetwork(
                input_dim=self.input_dim,
                hidden_dims=[32, 32],
                output_dim=self.output_dim,
                activation=activation
            )
            
            trajectory_data = torch.randn(4, 10, 2)
            output = branch_net(trajectory_data)
            
            self.assertEqual(output.shape, (4, self.output_dim))
            self.assertFalse(torch.isnan(output).any())


class TestTrunkNetwork(unittest.TestCase):
    """Test cases for the Trunk Network component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 2  # coordinate dimension
        self.hidden_dims = [32, 64, 32]
        self.output_dim = 16
        self.batch_size = 8
        
        self.trunk_net = TrunkNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation='relu',
            dropout_rate=0.1
        )
    
    def test_trunk_network_initialization(self):
        """Test trunk network initialization."""
        self.assertIsInstance(self.trunk_net, TrunkNetwork)
        self.assertEqual(len(self.trunk_net.layers), 10)  # 3 linear + 3 activation + 3 dropout + 1 output
    
    def test_trunk_network_forward_pass(self):
        """Test trunk network forward pass with coordinate data."""
        # Create test coordinate data
        coordinates = torch.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        output = self.trunk_net(coordinates)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_trunk_network_gradient_flow(self):
        """Test gradient flow through trunk network."""
        coordinates = torch.randn(4, self.input_dim, requires_grad=True)
        output = self.trunk_net(coordinates)
        
        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(coordinates.grad)
        self.assertFalse(torch.isnan(coordinates.grad).any())


class TestDeepONetKoopman(unittest.TestCase):
    """Test cases for the complete DeepONet Koopman model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'trajectory_length': 10,
            'state_dim': 2,
            'coordinate_dim': 2,
            'branch_hidden_dims': [32, 64, 32],
            'trunk_hidden_dims': [32, 64, 32],
            'latent_dim': 16,
            'activation': 'relu',
            'dropout_rate': 0.1
        }
        
        self.model = DeepONetKoopman(self.config)
        self.batch_size = 8
    
    def test_model_initialization(self):
        """Test DeepONet model initialization."""
        self.assertIsInstance(self.model, DeepONetKoopman)
        self.assertIsInstance(self.model.branch_net, BranchNetwork)
        self.assertIsInstance(self.model.trunk_net, TrunkNetwork)
        
        # Check configuration parameters
        self.assertEqual(self.model.trajectory_length, 10)
        self.assertEqual(self.model.state_dim, 2)
        self.assertEqual(self.model.coordinate_dim, 2)
        self.assertEqual(self.model.latent_dim, 16)
    
    def test_forward_pass(self):
        """Test DeepONet forward pass with trajectory and coordinate data."""
        # Create test data
        trajectory_data = torch.randn(self.batch_size, 10, 2)
        coordinates = torch.randn(self.batch_size, 2)
        
        # Forward pass
        output = self.model.forward(trajectory_data, coordinates)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 2))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_single_input_forward(self):
        """Test forward pass with single concatenated input."""
        # Create combined input (trajectory + coordinates)
        trajectory_size = 10 * 2  # trajectory_length * state_dim
        coord_size = 2
        combined_input = torch.randn(self.batch_size, trajectory_size + coord_size)
        
        # Forward pass
        output = self.model.forward_single_input(combined_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 2))
        self.assertFalse(torch.isnan(output).any())
    
    def test_operator_matrix_extraction(self):
        """Test extraction of operator matrix from trained model."""
        # Extract operator matrix
        operator_matrix = self.model.get_operator_matrix()
        
        # Check matrix properties
        self.assertIsInstance(operator_matrix, np.ndarray)
        self.assertEqual(operator_matrix.shape, (2, 2))  # state_dim x state_dim
        self.assertFalse(np.isnan(operator_matrix).any())
        self.assertFalse(np.isinf(operator_matrix).any())
    
    def test_batch_creation_utility(self):
        """Test utility method for creating DeepONet batch format."""
        # Create test state data
        states = torch.randn(self.batch_size, 2)
        next_states = torch.randn(self.batch_size, 2)
        
        # Create batch
        batch = self.model.create_trajectory_coordinates_batch(states, next_states)
        
        # Check batch format
        expected_size = 10 * 2 + 2 + 2  # trajectory + coordinates + targets
        self.assertEqual(batch.shape, (self.batch_size, expected_size))
    
    def test_train_step(self):
        """Test single training step."""
        # Create test batch
        states = torch.randn(4, 2)
        next_states = torch.randn(4, 2)
        batch = self.model.create_trajectory_coordinates_batch(states, next_states)
        
        # Perform training step
        loss = self.model.train_step(batch)
        
        # Check loss value
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertFalse(np.isnan(loss))
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Create test data
        states = torch.randn(10, 2)
        next_states = torch.randn(10, 2)
        test_data = self.model.create_trajectory_coordinates_batch(states, next_states)
        
        # Evaluate model
        metrics = self.model.evaluate(test_data)
        
        # Check metrics
        self.assertIn('mse_loss', metrics)
        self.assertIn('mae_loss', metrics)
        self.assertIn('relative_error', metrics)
        
        for metric_value in metrics.values():
            self.assertIsInstance(metric_value, float)
            self.assertGreater(metric_value, 0)
            self.assertFalse(np.isnan(metric_value))


class TestOperatorLoss(unittest.TestCase):
    """Test cases for the specialized operator learning loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = OperatorLoss(prediction_weight=1.0, consistency_weight=0.1)
        self.batch_size = 4
    
    def test_loss_initialization(self):
        """Test operator loss initialization."""
        self.assertEqual(self.loss_fn.prediction_weight, 1.0)
        self.assertEqual(self.loss_fn.consistency_weight, 0.1)
    
    def test_loss_computation(self):
        """Test operator loss computation."""
        # Create test data
        predictions = torch.randn(self.batch_size, 2)
        targets = torch.randn(self.batch_size, 2)
        trajectory_data = torch.randn(self.batch_size, 10, 2)
        coordinates = torch.randn(self.batch_size, 2)
        
        # Compute loss
        loss = self.loss_fn(predictions, targets, trajectory_data, coordinates)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
    
    def test_loss_gradients(self):
        """Test gradient flow through operator loss."""
        # Create test data with gradients
        predictions = torch.randn(self.batch_size, 2, requires_grad=True)
        targets = torch.randn(self.batch_size, 2)
        trajectory_data = torch.randn(self.batch_size, 10, 2)
        coordinates = torch.randn(self.batch_size, 2)
        
        # Compute loss and backpropagate
        loss = self.loss_fn(predictions, targets, trajectory_data, coordinates)
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(predictions.grad)
        self.assertFalse(torch.isnan(predictions.grad).any())


class TestDeepONetTrainer(unittest.TestCase):
    """Test cases for the DeepONet training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'trajectory_length': 5,
            'state_dim': 2,
            'coordinate_dim': 2,
            'branch_hidden_dims': [16, 32, 16],
            'trunk_hidden_dims': [16, 32, 16],
            'latent_dim': 8,
            'activation': 'relu',
            'dropout_rate': 0.0,
            'learning_rate': 0.01,
            'batch_size': 4,
            'epochs': 2,
            'weight_decay': 0.0001,
            'prediction_weight': 1.0,
            'consistency_weight': 0.1,
            'validation_freq': 1,
            'checkpoint_freq': 1,
            'early_stopping_patience': 5,
            'output_dir': 'test_output'
        }
        
        self.trainer = DeepONetTrainer(self.config)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, DeepONetTrainer)
        self.assertEqual(self.trainer.learning_rate, 0.01)
        self.assertEqual(self.trainer.batch_size, 4)
        self.assertEqual(self.trainer.epochs, 2)
    
    def test_model_initialization(self):
        """Test model initialization within trainer."""
        self.trainer.initialize_model()
        
        self.assertIsNotNone(self.trainer.model)
        self.assertIsInstance(self.trainer.model, DeepONetKoopman)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
    
    def test_operator_extraction(self):
        """Test operator matrix extraction from trainer."""
        self.trainer.initialize_model()
        
        # Extract operator matrix
        operator_matrix = self.trainer.extract_operator_matrix()
        
        # Check matrix properties
        self.assertIsInstance(operator_matrix, np.ndarray)
        self.assertEqual(operator_matrix.shape, (2, 2))
        self.assertFalse(np.isnan(operator_matrix).any())


class TestDeepONetIntegration(unittest.TestCase):
    """Integration tests for DeepONet with known operator mappings."""
    
    def setUp(self):
        """Set up test fixtures with simple known mappings."""
        self.config = {
            'trajectory_length': 3,
            'state_dim': 2,
            'coordinate_dim': 2,
            'branch_hidden_dims': [8, 16, 8],
            'trunk_hidden_dims': [8, 16, 8],
            'latent_dim': 4,
            'activation': 'tanh',
            'dropout_rate': 0.0
        }
        
        self.model = DeepONetKoopman(self.config)
    
    def test_identity_operator_learning(self):
        """Test learning of identity operator mapping."""
        # Create identity mapping data
        n_samples = 100
        states = torch.randn(n_samples, 2)
        next_states = states.clone()  # Identity mapping
        
        # Create training batch
        batch = self.model.create_trajectory_coordinates_batch(states, next_states)
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Parse batch
            trajectory_size = self.config['trajectory_length'] * self.config['state_dim']
            coord_size = self.config['coordinate_dim']
            
            trajectory_data = batch[:, :trajectory_size].view(-1, self.config['trajectory_length'], self.config['state_dim'])
            coordinates = batch[:, trajectory_size:trajectory_size + coord_size]
            targets = batch[:, trajectory_size + coord_size:]
            
            # Forward pass
            predictions = self.model.forward(trajectory_data, coordinates)
            loss = criterion(predictions, targets)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
        
        # Check that loss decreased (learning occurred)
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 1.0)  # Should achieve reasonable loss reduction
    
    def test_spectral_extraction_consistency(self):
        """Test that spectral extraction produces consistent results."""
        # Extract operator matrix multiple times
        matrices = []
        for _ in range(3):
            matrix = self.model.get_operator_matrix()
            matrices.append(matrix)
        
        # Check consistency (should be identical for same model state)
        for i in range(1, len(matrices)):
            np.testing.assert_array_almost_equal(matrices[0], matrices[i], decimal=6)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)