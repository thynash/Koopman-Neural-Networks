"""
Data filtering utilities for fractal trajectory data.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal


def filter_divergent_trajectories(states: np.ndarray,
                                 next_states: np.ndarray,
                                 max_magnitude: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out trajectory points that have diverged beyond a threshold.
    
    Args:
        states: Current state data
        next_states: Next state data
        max_magnitude: Maximum allowed magnitude for trajectory points
        
    Returns:
        Tuple of filtered (states, next_states)
    """
    # Compute magnitudes
    state_magnitudes = np.linalg.norm(states, axis=1)
    next_state_magnitudes = np.linalg.norm(next_states, axis=1)
    
    # Create mask for non-divergent points
    valid_mask = (state_magnitudes <= max_magnitude) & (next_state_magnitudes <= max_magnitude)
    
    return states[valid_mask], next_states[valid_mask]


def remove_outliers(data: np.ndarray,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outlier points from trajectory data.
    
    Args:
        data: Input trajectory data
        method: Outlier detection method ('iqr', 'zscore', 'isolation')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (filtered_data, outlier_mask)
    """
    if method == 'iqr':
        # Interquartile range method
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Check if any dimension is an outlier
        outlier_mask = np.any((data < lower_bound) | (data > upper_bound), axis=1)
        
    elif method == 'zscore':
        # Z-score method
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / std)
        
        # Check if any dimension has high z-score
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return data[~outlier_mask], outlier_mask


def smooth_trajectories(data: np.ndarray,
                       method: str = 'savgol',
                       window_length: int = 5,
                       polyorder: int = 2) -> np.ndarray:
    """
    Apply smoothing to trajectory data to reduce noise.
    
    Args:
        data: Input trajectory data
        method: Smoothing method ('savgol', 'moving_average', 'gaussian')
        window_length: Length of smoothing window
        polyorder: Polynomial order for Savitzky-Golay filter
        
    Returns:
        Smoothed trajectory data
    """
    if len(data) < window_length:
        return data  # Cannot smooth if data is shorter than window
    
    smoothed_data = np.zeros_like(data)
    
    for dim in range(data.shape[1]):
        if method == 'savgol':
            # Ensure window_length is odd
            if window_length % 2 == 0:
                window_length += 1
            
            smoothed_data[:, dim] = signal.savgol_filter(
                data[:, dim], window_length, polyorder
            )
            
        elif method == 'moving_average':
            # Simple moving average
            smoothed_data[:, dim] = np.convolve(
                data[:, dim], 
                np.ones(window_length) / window_length, 
                mode='same'
            )
            
        elif method == 'gaussian':
            # Gaussian smoothing
            sigma = window_length / 6.0  # Standard deviation
            smoothed_data[:, dim] = signal.gaussian_filter1d(
                data[:, dim], sigma
            )
            
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed_data