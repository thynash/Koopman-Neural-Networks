"""
Abstract base class for spectral analysis of Koopman operators.

This module defines the interface for extracting and analyzing spectral
properties from trained neural network models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SpectralResults:
    """
    Data structure for storing spectral analysis results.
    """
    eigenvalues: np.ndarray     # Complex eigenvalues
    eigenvectors: np.ndarray    # Corresponding eigenvectors
    spectral_error: float       # Error vs reference spectrum
    convergence_history: List   # Eigenvalue evolution during training
    metadata: Dict              # Analysis metadata


class SpectralAnalyzer(ABC):
    """
    Abstract base class for spectral analysis of learned Koopman operators.
    
    All spectral analysis implementations must inherit from this class
    and implement the required methods for eigenvalue extraction and comparison.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the spectral analyzer with configuration parameters.
        
        Args:
            config: Dictionary containing analysis configuration parameters
        """
        self.config = config
        
    @abstractmethod
    def extract_spectrum(self, operator_matrix: np.ndarray) -> SpectralResults:
        """
        Extract eigenvalues and eigenvectors from a Koopman operator matrix.
        
        Args:
            operator_matrix: Learned Koopman operator matrix
            
        Returns:
            SpectralResults object containing eigenvalues and eigenvectors
        """
        pass
    
    @abstractmethod
    def compute_spectral_error(self, learned_spectrum: SpectralResults,
                              reference_spectrum: SpectralResults) -> float:
        """
        Compute error between learned and reference spectra.
        
        Args:
            learned_spectrum: Spectrum extracted from neural network
            reference_spectrum: Reference spectrum (e.g., from DMD)
            
        Returns:
            Spectral approximation error
        """
        pass
    
    @abstractmethod
    def visualize_spectrum(self, spectrum: SpectralResults, 
                          save_path: Optional[str] = None) -> None:
        """
        Create visualization of eigenvalue spectrum in complex plane.
        
        Args:
            spectrum: SpectralResults to visualize
            save_path: Optional path to save the figure
        """
        pass
    
    def compare_spectra(self, spectra: Dict[str, SpectralResults]) -> Dict[str, Any]:
        """
        Compare multiple spectra and compute relative errors.
        
        Args:
            spectra: Dictionary mapping model names to their spectral results
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison_results = {}
        
        # Get reference spectrum (typically DMD or analytical)
        reference_key = self.config.get('reference_method', 'dmd')
        if reference_key not in spectra:
            reference_key = list(spectra.keys())[0]
        
        reference_spectrum = spectra[reference_key]
        
        for model_name, spectrum in spectra.items():
            if model_name == reference_key:
                continue
                
            error = self.compute_spectral_error(spectrum, reference_spectrum)
            comparison_results[model_name] = {
                'spectral_error': error,
                'n_eigenvalues': len(spectrum.eigenvalues),
                'dominant_eigenvalue': self._get_dominant_eigenvalue(spectrum),
                'spectral_radius': np.max(np.abs(spectrum.eigenvalues))
            }
        
        return comparison_results
    
    def _get_dominant_eigenvalue(self, spectrum: SpectralResults) -> complex:
        """
        Get the dominant (largest magnitude) eigenvalue.
        
        Args:
            spectrum: SpectralResults object
            
        Returns:
            Dominant eigenvalue
        """
        magnitudes = np.abs(spectrum.eigenvalues)
        dominant_idx = np.argmax(magnitudes)
        return spectrum.eigenvalues[dominant_idx]
    
    def _sort_eigenvalues(self, eigenvalues: np.ndarray, 
                         eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort eigenvalues and eigenvectors by magnitude (descending).
        
        Args:
            eigenvalues: Array of eigenvalues
            eigenvectors: Array of eigenvectors
            
        Returns:
            Tuple of sorted (eigenvalues, eigenvectors)
        """
        magnitudes = np.abs(eigenvalues)
        sort_indices = np.argsort(magnitudes)[::-1]
        
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices]
        
        return sorted_eigenvalues, sorted_eigenvectors
    
    def validate_spectrum(self, spectrum: SpectralResults) -> bool:
        """
        Validate that extracted spectrum is mathematically reasonable.
        
        Args:
            spectrum: SpectralResults to validate
            
        Returns:
            True if spectrum is valid, False otherwise
        """
        # Check for NaN or infinite values
        if np.any(np.isnan(spectrum.eigenvalues)) or np.any(np.isinf(spectrum.eigenvalues)):
            return False
        
        # Check spectral radius (should be reasonable for stable systems)
        spectral_radius = np.max(np.abs(spectrum.eigenvalues))
        if spectral_radius > 100:  # Arbitrary threshold
            return False
        
        # Check that eigenvectors are not all zero
        if np.allclose(spectrum.eigenvectors, 0):
            return False
        
        return True
    
    def save_results(self, spectrum: SpectralResults, filepath: str) -> None:
        """
        Save spectral analysis results to file.
        
        Args:
            spectrum: SpectralResults to save
            filepath: Path to save the results
        """
        np.savez(filepath,
                eigenvalues=spectrum.eigenvalues,
                eigenvectors=spectrum.eigenvectors,
                spectral_error=spectrum.spectral_error,
                metadata=spectrum.metadata)
    
    def load_results(self, filepath: str) -> SpectralResults:
        """
        Load spectral analysis results from file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            SpectralResults object loaded from file
        """
        data = np.load(filepath, allow_pickle=True)
        
        return SpectralResults(
            eigenvalues=data['eigenvalues'],
            eigenvectors=data['eigenvectors'],
            spectral_error=float(data['spectral_error']),
            convergence_history=[],
            metadata=data['metadata'].item() if 'metadata' in data else {}
        )