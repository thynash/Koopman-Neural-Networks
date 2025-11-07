"""
Data augmentation utilities for fractal trajectory data.
"""

import numpy as np
from typing import Tuple, Optional


def add_noise(data: np.ndarray, 
              noise_level: float = 0.01,
              noise_type: str = 'gaussian',
              seed: Optional[int] = None) -> np.ndarray:
    """
    Add noise to trajectory data for augmentation.
    
    Args:
        data: Input trajectory data
        noise_level: Standard deviation of noise relative to data scale
        noise_type: Type of noise ('gaussian', 'uniform')
        seed: Random seed for reproducibility
        
    Returns:
        Augmented data with added noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_scale = np.std(data, axis=0)
    noise_std = noise_level * data_scale
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_std, data.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_std, noise_std, data.shape)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return data + noise


def rotate_trajectories(data: np.ndarray, 
                       angle: float,
                       center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rotate 2D trajectory data by a given angle.
    
    Args:
        data: Input trajectory data of shape (n_points, 2)
        angle: Rotation angle in radians
        center: Center of rotation (default: data centroid)
        
    Returns:
        Rotated trajectory data
    """
    if data.shape[1] != 2:
        raise ValueError("Rotation only supported for 2D data")
    
    if center is None:
        center = np.mean(data, axis=0)
    
    # Translate to origin
    centered_data = data - center
    
    # Rotation matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a],
                               [sin_a, cos_a]])
    
    # Apply rotation
    rotated_data = centered_data @ rotation_matrix.T
    
    # Translate back
    return rotated_data + center


def augment_trajectories(data: np.ndarray,
                        augmentation_factor: int = 2,
                        noise_level: float = 0.01,
                        rotation_range: float = np.pi/4,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Apply multiple augmentation techniques to trajectory data.
    
    Args:
        data: Input trajectory data
        augmentation_factor: Number of augmented copies to create
        noise_level: Standard deviation of added noise
        rotation_range: Maximum rotation angle in radians
        seed: Random seed for reproducibility
        
    Returns:
        Augmented dataset with original + augmented data
    """
    if seed is not None:
        np.random.seed(seed)
    
    augmented_data = [data]  # Start with original data
    
    for i in range(augmentation_factor - 1):
        # Create augmented copy
        aug_data = data.copy()
        
        # Add noise
        aug_data = add_noise(aug_data, noise_level)
        
        # Apply rotation (only for 2D data)
        if data.shape[1] == 2:
            angle = np.random.uniform(-rotation_range, rotation_range)
            aug_data = rotate_trajectories(aug_data, angle)
        
        augmented_data.append(aug_data)
    
    return np.vstack(augmented_data)