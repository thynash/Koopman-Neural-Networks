"""
Training pipeline for MLP Koopman operator learning.

This module implements the complete training pipeline for MLP models including
training loops, validation monitoring, checkpointing, and hyperparameter management.
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

from ...models.mlp.mlp_koopman import MLPKoopman
from ...data.datasets.trajectory_dataset import TrajectoryDataset, DatasetSplit


class MLPTrainer:
    """
    Training pipeline for MLP Koopman models.
    
    Handles training loop, validation monitoring, checkpointing, and logging
    for MLP-based Koopman operator learning on fractal dynamical systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLP trainer with configuration.
        
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
        
        # Validation and checkpointing
        self.validation_freq = config.get('validation_freq', 10)
        self.checkpoint_freq = config.get('checkpoint_freq', 25)
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.min_delta = config.get('min_delta', 1e-6)
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'results/mlp_training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model, optimizer, and scheduler
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
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
        
    def initialize_model(self, input_dim: int = 2, output_dim: int = 2) -> None:
        """
        Initialize the MLP model with given dimensions.
        
        Args:
            input_dim: Input state dimension
            output_dim: Output state dimension
        """
        # Update config with dimensions
        model_config = self.config.copy()
        model_config['input_dim'] = input_dim
        model_config['output_dim'] = output_dim
        
        # Create model
        self.model = MLPKoopman(model_config)
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
        
        self.logger.info(f"Initialized MLP model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def prepare_data_loaders(self, dataset_split: DatasetSplit) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training, validation, and testing.
        
        Args:
            dataset_split: DatasetSplit containing train/val/test datasets
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Use the datasets directly from the split
        train_dataset = dataset_split.train
        val_dataset = dataset_split.validation
        test_dataset = dataset_split.test
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
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
        
        for batch_idx, (states, next_states) in enumerate(train_loader):
            # Move data to device
            states = states.to(self.device)
            next_states = next_states.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_next = self.model(states)
            
            # Compute loss
            loss = self.criterion(predicted_next, next_states)
            
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
            for states, next_states in val_loader:
                # Move data to device
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                
                # Forward pass
                predicted_next = self.model(states)
                
                # Compute loss
                loss = self.criterion(predicted_next, next_states)
                
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
            # Get dimensions from training dataset
            sample_state, sample_next = dataset_split.train[0]
            input_dim = sample_state.shape[0]
            output_dim = sample_next.shape[0]
            self.initialize_model(input_dim, output_dim)
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = self.prepare_data_loaders(dataset_split)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        self.logger.info(f"Starting training for {self.epochs} epochs")
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
        
        # Save training history
        self._save_training_history()
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False)
        
        return {
            'final_train_loss': self.train_losses[-1],
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
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
        
        all_states = []
        all_next_states = []
        all_predictions = []
        
        with torch.no_grad():
            for states, next_states in test_loader:
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                
                predictions = self.model(states)
                
                all_states.append(states.cpu())
                all_next_states.append(next_states.cpu())
                all_predictions.append(predictions.cpu())
        
        # Concatenate all batches
        all_states = torch.cat(all_states, dim=0)
        all_next_states = torch.cat(all_next_states, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Compute metrics
        mse_loss = nn.functional.mse_loss(all_predictions, all_next_states)
        mae_loss = nn.functional.l1_loss(all_predictions, all_next_states)
        
        # Relative error
        relative_error = torch.mean(
            torch.norm(all_predictions - all_next_states, dim=1) / 
            (torch.norm(all_next_states, dim=1) + 1e-8)
        )
        
        # R-squared coefficient
        ss_res = torch.sum((all_next_states - all_predictions) ** 2)
        ss_tot = torch.sum((all_next_states - torch.mean(all_next_states, dim=0)) ** 2)
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