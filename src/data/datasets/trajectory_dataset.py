"""
Dataset management and preprocessing for fractal trajectory data.

This module provides comprehensive dataset management including train/validation/test
splits, data normalization, preprocessing utilities, and efficient data loading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

from ..generators.fractal_generator import TrajectoryData


@dataclass
class DatasetSplit:
    """
    Container for dataset split information.
    """
    train: TrajectoryData
    validation: TrajectoryData
    test: TrajectoryData
    split_info: Dict[str, Any]


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset class for fractal trajectory data with preprocessing capabilities.
    
    Supports train/validation/test splits, data normalization, and efficient batching
    for neural network training.
    """
    
    def __init__(self, 
                 data: TrajectoryData,
                 normalize: bool = True,
                 normalization_params: Optional[Dict[str, np.ndarray]] = None,
                 device: str = 'cpu'):
        """
        Initialize trajectory dataset.
        
        Args:
            data: TrajectoryData object containing trajectory information
            normalize: Whether to normalize the data
            normalization_params: Pre-computed normalization parameters
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.data = data
        self.device = device
        self.normalize = normalize
        
        # Convert to tensors
        self.states = torch.tensor(data.states, dtype=torch.float32, device=device)
        self.next_states = torch.tensor(data.next_states, dtype=torch.float32, device=device)
        
        # Apply normalization
        if normalize:
            if normalization_params is not None:
                self.normalization_params = normalization_params
            else:
                self.normalization_params = self._compute_normalization_params()
            
            self._apply_normalization()
        else:
            self.normalization_params = None
    
    def _compute_normalization_params(self) -> Dict[str, np.ndarray]:
        """
        Compute normalization parameters (mean and std) for the dataset.
        
        Returns:
            Dictionary containing normalization parameters
        """
        states_np = self.states.cpu().numpy()
        next_states_np = self.next_states.cpu().numpy()
        
        # Compute statistics over all data
        all_data = np.vstack([states_np, next_states_np])
        
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        return {
            'mean': mean,
            'std': std,
            'min': np.min(all_data, axis=0),
            'max': np.max(all_data, axis=0)
        }
    
    def _apply_normalization(self) -> None:
        """Apply normalization to states and next_states."""
        mean = torch.tensor(self.normalization_params['mean'], 
                           dtype=torch.float32, device=self.device)
        std = torch.tensor(self.normalization_params['std'], 
                          dtype=torch.float32, device=self.device)
        
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std
    
    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized data back to original scale.
        
        Args:
            normalized_data: Normalized tensor data
            
        Returns:
            Denormalized tensor data
        """
        if not self.normalize or self.normalization_params is None:
            return normalized_data
        
        mean = torch.tensor(self.normalization_params['mean'], 
                           dtype=torch.float32, device=normalized_data.device)
        std = torch.tensor(self.normalization_params['std'], 
                          dtype=torch.float32, device=normalized_data.device)
        
        return normalized_data * std + mean
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (state, next_state) tensors
        """
        return self.states[idx], self.next_states[idx]
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (states_batch, next_states_batch) tensors
        """
        return self.states[indices], self.next_states[indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        states_np = self.states.cpu().numpy()
        next_states_np = self.next_states.cpu().numpy()
        
        stats = {
            'n_samples': len(self.states),
            'state_dim': self.states.shape[1],
            'states_mean': np.mean(states_np, axis=0).tolist(),
            'states_std': np.std(states_np, axis=0).tolist(),
            'states_min': np.min(states_np, axis=0).tolist(),
            'states_max': np.max(states_np, axis=0).tolist(),
            'next_states_mean': np.mean(next_states_np, axis=0).tolist(),
            'next_states_std': np.std(next_states_np, axis=0).tolist(),
            'system_type': self.data.metadata.get('system_type', 'Unknown'),
            'system_name': self.data.metadata.get('system_name', 'Unknown')
        }
        
        return stats


class DatasetManager:
    """
    Manager class for creating, saving, and loading trajectory datasets with splits.
    
    Handles the complete dataset lifecycle including generation, preprocessing,
    splitting, and persistence.
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory for storing datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def create_dataset_split(self, 
                           data: TrajectoryData,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           normalize: bool = True,
                           seed: Optional[int] = None) -> DatasetSplit:
        """
        Create train/validation/test split from trajectory data.
        
        Args:
            data: TrajectoryData to split
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            normalize: Whether to normalize the data
            seed: Random seed for reproducible splits
            
        Returns:
            DatasetSplit containing train/val/test datasets
        """
        # Validate split ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create indices for splitting
        n_samples = len(data.states)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Create split datasets
        train_data = TrajectoryData(
            states=data.states[train_idx],
            next_states=data.next_states[train_idx],
            system_params=data.system_params,
            metadata={**data.metadata, 'split': 'train', 'indices': train_idx.tolist()}
        )
        
        val_data = TrajectoryData(
            states=data.states[val_idx],
            next_states=data.next_states[val_idx],
            system_params=data.system_params,
            metadata={**data.metadata, 'split': 'validation', 'indices': val_idx.tolist()}
        )
        
        test_data = TrajectoryData(
            states=data.states[test_idx],
            next_states=data.next_states[test_idx],
            system_params=data.system_params,
            metadata={**data.metadata, 'split': 'test', 'indices': test_idx.tolist()}
        )
        
        # Create PyTorch datasets with normalization
        if normalize:
            # Compute normalization parameters from training data only
            train_dataset = TrajectoryDataset(train_data, normalize=True)
            normalization_params = train_dataset.normalization_params
            
            # Apply same normalization to validation and test sets
            val_dataset = TrajectoryDataset(val_data, normalize=True, 
                                          normalization_params=normalization_params)
            test_dataset = TrajectoryDataset(test_data, normalize=True,
                                           normalization_params=normalization_params)
        else:
            train_dataset = TrajectoryDataset(train_data, normalize=False)
            val_dataset = TrajectoryDataset(val_data, normalize=False)
            test_dataset = TrajectoryDataset(test_data, normalize=False)
            normalization_params = None
        
        split_info = {
            'train_size': len(train_data.states),
            'val_size': len(val_data.states),
            'test_size': len(test_data.states),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'normalized': normalize,
            'normalization_params': normalization_params,
            'seed': seed,
            'split_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return DatasetSplit(
            train=train_dataset,
            validation=val_dataset,
            test=test_dataset,
            split_info=split_info
        )
    
    def save_dataset_split(self, 
                          dataset_split: DatasetSplit,
                          name: str,
                          formats: List[str] = ['npy', 'csv']) -> Dict[str, str]:
        """
        Save dataset split to disk in multiple formats.
        
        Args:
            dataset_split: DatasetSplit to save
            name: Name for the dataset files
            formats: List of formats to save ('npy', 'csv', 'pickle')
            
        Returns:
            Dictionary mapping format to file paths
        """
        dataset_dir = self.data_dir / name
        dataset_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for split_name, dataset in [('train', dataset_split.train),
                                   ('val', dataset_split.validation),
                                   ('test', dataset_split.test)]:
            
            for fmt in formats:
                if fmt == 'npy':
                    # Save as NumPy arrays
                    states_path = dataset_dir / f"{split_name}_states.npy"
                    next_states_path = dataset_dir / f"{split_name}_next_states.npy"
                    
                    np.save(states_path, dataset.states.cpu().numpy())
                    np.save(next_states_path, dataset.next_states.cpu().numpy())
                    
                    saved_files[f'{split_name}_states_npy'] = str(states_path)
                    saved_files[f'{split_name}_next_states_npy'] = str(next_states_path)
                
                elif fmt == 'csv':
                    # Save as CSV files
                    states_path = dataset_dir / f"{split_name}_states.csv"
                    next_states_path = dataset_dir / f"{split_name}_next_states.csv"
                    
                    np.savetxt(states_path, dataset.states.cpu().numpy(), delimiter=',')
                    np.savetxt(next_states_path, dataset.next_states.cpu().numpy(), delimiter=',')
                    
                    saved_files[f'{split_name}_states_csv'] = str(states_path)
                    saved_files[f'{split_name}_next_states_csv'] = str(next_states_path)
                
                elif fmt == 'pickle':
                    # Save entire dataset as pickle
                    pickle_path = dataset_dir / f"{split_name}_dataset.pkl"
                    
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(dataset, f)
                    
                    saved_files[f'{split_name}_pickle'] = str(pickle_path)
        
        # Save split information and metadata
        split_info_path = dataset_dir / "split_info.json"
        with open(split_info_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            split_info_serializable = {}
            for key, value in dataset_split.split_info.items():
                if isinstance(value, dict) and 'mean' in value:
                    # Handle normalization parameters
                    split_info_serializable[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    split_info_serializable[key] = value
            
            json.dump(split_info_serializable, f, indent=2)
        
        saved_files['split_info'] = str(split_info_path)
        
        return saved_files
    
    def load_dataset_split(self, name: str, format: str = 'npy') -> DatasetSplit:
        """
        Load dataset split from disk.
        
        Args:
            name: Name of the dataset to load
            format: Format to load ('npy', 'csv', 'pickle')
            
        Returns:
            Loaded DatasetSplit
        """
        dataset_dir = self.data_dir / name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Load split information
        split_info_path = dataset_dir / "split_info.json"
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        
        datasets = {}
        
        for split_name in ['train', 'val', 'test']:
            if format == 'pickle':
                pickle_path = dataset_dir / f"{split_name}_dataset.pkl"
                with open(pickle_path, 'rb') as f:
                    datasets[split_name] = pickle.load(f)
            
            else:
                # Load states and next_states
                if format == 'npy':
                    states_path = dataset_dir / f"{split_name}_states.npy"
                    next_states_path = dataset_dir / f"{split_name}_next_states.npy"
                    
                    states = np.load(states_path)
                    next_states = np.load(next_states_path)
                
                elif format == 'csv':
                    states_path = dataset_dir / f"{split_name}_states.csv"
                    next_states_path = dataset_dir / f"{split_name}_next_states.csv"
                    
                    states = np.loadtxt(states_path, delimiter=',')
                    next_states = np.loadtxt(next_states_path, delimiter=',')
                
                # Reconstruct TrajectoryData
                trajectory_data = TrajectoryData(
                    states=states,
                    next_states=next_states,
                    system_params={},
                    metadata={'split': split_name}
                )
                
                # Create TrajectoryDataset
                normalize = split_info.get('normalized', False)
                normalization_params = split_info.get('normalization_params')
                
                datasets[split_name] = TrajectoryDataset(
                    trajectory_data,
                    normalize=normalize,
                    normalization_params=normalization_params
                )
        
        return DatasetSplit(
            train=datasets['train'],
            validation=datasets['val'],
            test=datasets['test'],
            split_info=split_info
        )
    
    def create_data_loaders(self, 
                           dataset_split: DatasetSplit,
                           batch_size: int = 32,
                           shuffle_train: bool = True,
                           num_workers: int = 0) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test splits.
        
        Args:
            dataset_split: DatasetSplit containing the datasets
            batch_size: Batch size for data loading
            shuffle_train: Whether to shuffle training data
            num_workers: Number of worker processes for data loading
            
        Returns:
            Dictionary containing DataLoaders for each split
        """
        data_loaders = {
            'train': DataLoader(
                dataset_split.train,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers
            ),
            'val': DataLoader(
                dataset_split.validation,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            ),
            'test': DataLoader(
                dataset_split.test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        }
        
        return data_loaders
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a saved dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        dataset_dir = self.data_dir / name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        split_info_path = dataset_dir / "split_info.json"
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        
        return split_info
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        if not self.data_dir.exists():
            return []
        
        datasets = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and (item / "split_info.json").exists():
                datasets.append(item.name)
        
        return datasets