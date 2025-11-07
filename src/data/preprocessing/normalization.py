"""
Data normalization utilities for fractal trajectory data.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_normalization_params(data: np.ndarray, 
                                method: str = 'standardize') -> Dict[str, np.ndarray]:
    """
    Compute normalization parameters for trajectory data.
    
    Args:
        data: Input data array of shape (n_samples, n_features)
        method: Normalization method ('standardize', 'minmax', 'robust')
        
    Returns:
        Dictionary containing normalization parameters
    """
    if method == 'standardize':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        return {
            'method': 'standardize',
            'mean': mean,
            'std': std
        }
    
    elif method == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)
        
        return {
            'method': 'minmax',
            'min': min_vals,
            'max': max_vals,
            'range': range_vals
        }
    
    elif method == 'robust':
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        # Avoid division by zero
        mad = np.where(mad < 1e-8, 1.0, mad)
        
        return {
            'method': 'robust',
            'median': median,
            'mad': mad
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_data(data: np.ndarray, 
                  params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalize data using pre-computed parameters.
    
    Args:
        data: Input data to normalize
        params: Normalization parameters from compute_normalization_params
        
    Returns:
        Normalized data array
    """
    method = params['method']
    
    if method == 'standardize':
        return (data - params['mean']) / params['std']
    
    elif method == 'minmax':
        return (data - params['min']) / params['range']
    
    elif method == 'robust':
        return (data - params['median']) / params['mad']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_data(normalized_data: np.ndarray,
                    params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert normalized data back to original scale.
    
    Args:
        normalized_data: Normalized data array
        params: Normalization parameters used for normalization
        
    Returns:
        Denormalized data array
    """
    method = params['method']
    
    if method == 'standardize':
        return normalized_data * params['std'] + params['mean']
    
    elif method == 'minmax':
        return normalized_data * params['range'] + params['min']
    
    elif method == 'robust':
        return normalized_data * params['mad'] + params['median']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")