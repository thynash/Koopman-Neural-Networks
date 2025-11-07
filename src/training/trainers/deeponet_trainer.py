"""
Training pipeline for DeepONet Koopman operator learning.

This module implements the complete training pipeline for DeepONet models including
specialized loss functions for operator learning, training loops, validation monitoring,
and operator extraction for spectral analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from src.models.deeponet.deeponet_koopman import DeepONetKoopman
from src.data.datasets.trajectory_dataset import TrajectoryDataset, DatasetSplit


class OperatorLoss(nn.Module):
    """
    Specialized loss function for operator learning in DeepONet.
    
    Combines standard prediction loss with operator consistency terms
    to encourage learning of meaningful operator mappings.
    """
    
    def __init__(self, prediction_weight: float = 1.0, consistency_weight: float = 0.1):
        """
        Initialize operator loss function.
        
        Args:
            prediction_weight: Weight for standard prediction loss
            consistency_weight: Weight for operator consistency loss
        """
        super().__init__()
        self.prediction_weight = prediction_weight
        self.consistency_weight = consistency_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                trajectory_data: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Compute operator learning loss.
        
        Args:
            predictions: Model predictions (batch_size, state_dim)
            targets: Target values (batch_size, state_dim)
            trajectory_data: Input trajectory snapshots (batch_size, trajectory_length, state_dim)
            coordinates: Input coordinates (batch_size, coordinate_dim)
            
        Returns:
            Combined loss value
        """
        # Standard prediction loss
        prediction_loss = self.mse_loss(predictions, targets)
        
        # Operator consistency loss (encourage smooth operator behavior)
        consistency_loss = self._compute_consistency_loss(trajectory_data, coordinates)
        
        # Combined loss
        total_loss = (self.prediction_weight * prediction_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss
    
    def _compute_consistency_loss(self, trajectory_data: torch.Tensor, 
                                coordinates: torch.Tensor) -> torch.Tensor:
        """
        Compute operator consistency loss to encourage smooth operator behavior.
        
        Args:
            trajectory_data: Input trajectory snapshots
            coordinates: Input coordinates
            
        Returns:
            Consistency loss value
        """
        # For now, use a simple regularization term
        # In practice, this could include more sophisticated operator consistency checks
        batch_size = trajectory_data.shape[0]
        
        # Compute variance in trajectory data as a simple consistency measure
        traj_variance = torch.var(trajectory_data.view(batch_size, -1), dim=1)
        coord_variance = torch.var(coordinates, dim=1)
        
        # Encourage bounded variance (operator stability)
        consistency_loss = torch.mean(traj_variance + coord_variance)
        
        return consistency_loss


class DeepONetTrainer:
    """
    Training pipeline for DeepONet Koopman models.
    
    Handles training loop with specialized operator learning loss, validation monitoring,
    checkpointing, and operator extraction for spectral analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepONet trainer with configuration.
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.weight_decay = config.get('weight_decay', 0.0001)
        
        # Loss function parameters
        self.prediction_weight = config.get('prediction_weight', 1.0)
        self.consistency_weight = config.get('consistency_weight', 0.1)
        
        # Validation and checkpointing
        self.validation_freq = config.get('validation_freq', 10)
        self.checkpoint_freq = config.get('checkpoint_freq', 25)
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.min_delta = config.get('min_delta', 1e-6)
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'results/deeponet_training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model, optimizer, and scheduler
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = OperatorLoss(self.prediction_weight, self.consistency_weight)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_model(self) -> None:
        """Initialize the DeepONet model."""
        # Create model
        self.model = DeepONetKoopman(self.config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        self.logger.info(f"Initialized DeepONet model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def prepare_deeponet_data(self, dataset_split: DatasetSplit) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for DeepONet training with trajectory-coordinate format.
        
        Args:
            dataset_split: DatasetSplit containing train/val/test datasets
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        def convert_to_deeponet_format(dataset: TrajectoryDataset) -> TensorDataset:
            """Convert standard trajectory dataset to DeepONet format."""
            all_batches = []
            
            for i in range(len(dataset)):
                states, next_states = dataset[i]
                
                # Create DeepONet batch format
                batch = self.model.create_trajectory_coordinates_batch(
                    states.unsqueeze(0), next_states.unsqueeze(0)
                )
                all_batches.append(batch.squeeze(0))
            
            # Stack all batches
            data_tensor = torch.stack(all_batches)
            return TensorDataset(data_tensor)
        
        # Convert datasets to DeepONet format
        train_deeponet = convert_to_deeponet_format(dataset_split.train)
        val_deeponet = convert_to_deeponet_format(dataset_split.validation)
        test_deeponet = convert_to_deeponet_format(dataset_split.test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_deeponet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_deeponet,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_deeponet,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_data,) in enumerate(train_loader):
            # Move data to device
            batch_data = batch_data.to(self.device)
            
            # Parse batch data
            trajectory_size = self.model.trajectory_length * self.model.state_dim
            coord_size = self.model.coordinate_dim
            
            trajectory_data = batch_data[:, :trajectory_size].view(-1, self.model.trajectory_length, self.model.state_dim)
            coordinates = batch_data[:, trajectory_size:trajectory_size + coord_size]
            targets = batch_data[:, trajectory_size + coord_size:]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model.forward(trajectory_data, coordinates)
            
            # Compute loss
            loss = self.criterion(predictions, targets, trajectory_data, coordinates)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, in val_loader:
                # Move data to device
                batch_data = batch_data.to(self.device)
                
                # Parse batch data
                trajectory_size = self.model.trajectory_length * self.model.state_dim
                coord_size = self.model.coordinate_dim
                
                trajectory_data = batch_data[:, :trajectory_size].view(-1, self.model.trajectory_length, self.model.state_dim)
                coordinates = batch_data[:, trajectory_size:trajectory_size + coord_size]
                targets = batch_data[:, trajectory_size + coord_size:]
                
                # Forward pass
                predictions = self.model.forward(trajectory_data, coordinates)
                
                # Compute loss
                loss = self.criterion(predictions, targets, trajectory_data, coordinates)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def extract_operator_matrix(self) -> np.ndarray:
        """
        Extract the learned Koopman operator matrix for spectral analysis.
        
        Returns:
            Numpy array representing the learned operator matrix
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        return self.model.get_operator_matrix()
    
    def train(self, dataset_split: DatasetSplit, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            dataset_split: DatasetSplit for training
            resume_from: Optional path to checkpoint to resume from
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = self.prepare_deeponet_data(dataset_split)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        self.logger.info(f"Starting DeepONet training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Training loop
        for epoch in range(start_epoch, self.epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate if needed
            if epoch % self.validation_freq == 0:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Check for improvement
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += self.validation_freq
                
                self.logger.info(
                    f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}, Best Val = {self.best_val_loss:.6f}"
                )
                
                # Save checkpoint
                if epoch % self.checkpoint_freq == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                # Early stopping check
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.logger.info(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate_model(test_loader)
        
        # Extract operator matrix for spectral analysis
        operator_matrix = self.extract_operator_matrix()
        
        # Save training history
        self._save_training_history()
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False)
        
        return {
            'final_train_loss': self.train_losses[-1],
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'operator_matrix': operator_matrix,
            'total_epochs': epoch + 1,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
        }
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data, in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Parse batch data
                trajectory_size = self.model.trajectory_length * self.model.state_dim
                coord_size = self.model.coordinate_dim
                
                trajectory_data = batch_data[:, :trajectory_size].view(-1, self.model.trajectory_length, self.model.state_dim)
                coordinates = batch_data[:, trajectory_size:trajectory_size + coord_size]
                targets = batch_data[:, trajectory_size + coord_size:]
                
                predictions = self.model.forward(trajectory_data, coordinates)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        mse_loss = nn.functional.mse_loss(all_predictions, all_targets)
        mae_loss = nn.functional.l1_loss(all_predictions, all_targets)
        
        # Relative error
        relative_error = torch.mean(
            torch.norm(all_predictions - all_targets, dim=1) / 
            (torch.norm(all_targets, dim=1) + 1e-8)
        )
        
        # R-squared coefficient
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets, dim=0)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'relative_error': relative_error.item(),
            'r2_score': r2_score.item()
        }
        
        self.logger.info("Test Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def _save_training_history(self) -> None:
        """Save training history to JSON file."""
        history = {
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved training history to {history_path}")