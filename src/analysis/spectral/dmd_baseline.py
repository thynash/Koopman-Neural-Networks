"""
Dynamic Mode Decomposition (DMD) baseline implementation.

This module provides a standard DMD implementation for comparison against
neural network-based Koopman operator learning methods.
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .spectral_analyzer import SpectralAnalyzer, SpectralResults


@dataclass
class DMDResults:
    """Results from Dynamic Mode Decomposition analysis."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    reconstruction_error: float
    metadata: Dict[str, Any]


class DMDBaseline(SpectralAnalyzer):
    """
    Dynamic Mode Decomposition implementation for Koopman operator analysis.
    
    This class provides a standard DMD algorithm that can be used as a baseline
    for comparison against neural network-based methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DMD baseline analyzer.
        
        Args:
            config: Configuration dictionary containing:
                - rank_truncation: Number of modes to keep (default: None for full rank)
                - svd_tolerance: Tolerance for SVD truncation (default: 1e-10)
                - time_step: Time step between snapshots (default: 1.0)
        """
        super().__init__(config)
        
        self.rank_truncation = config.get('rank_truncation', None)
        self.svd_tolerance = config.get('svd_tolerance', 1e-10)
        self.time_step = config.get('time_step', 1.0)
    
    def compute_dmd(self, data: np.ndarray) -> DMDResults:
        """
        Compute Dynamic Mode Decomposition of trajectory data.
        
        Args:
            data: Trajectory data of shape (n_features, n_snapshots)
            
        Returns:
            DMDResults containing eigenvalues, modes, and other DMD quantities
        """
        try:
            if data.ndim != 2:
                raise ValueError("Data must be 2D array with shape (n_features, n_snapshots)")
            
            n_features, n_snapshots = data.shape
            
            if n_snapshots < 2:
                raise ValueError("Need at least 2 snapshots for DMD")
            
            # Split data into X and Y matrices
            X = data[:, :-1]  # First n-1 snapshots
            Y = data[:, 1:]   # Last n-1 snapshots
            
            # Perform SVD on X
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            
            # Determine rank for truncation
            if self.rank_truncation is not None:
                r = min(self.rank_truncation, len(s))
            else:
                # Automatic rank determination based on tolerance
                r = np.sum(s > self.svd_tolerance)
            
            # Truncate SVD components
            U_r = U[:, :r]
            s_r = s[:r]
            Vt_r = Vt[:r, :]
            
            # Compute DMD operator in reduced space
            A_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / s_r)
            
            # Eigendecomposition of reduced operator
            eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
            
            # Compute DMD modes
            modes = Y @ Vt_r.T @ np.diag(1.0 / s_r) @ eigenvectors
            
            # Compute initial amplitudes
            amplitudes = np.linalg.pinv(modes) @ data[:, 0]
            
            # Compute reconstruction error
            reconstruction = self._reconstruct_data(modes, eigenvalues, amplitudes, n_snapshots)
            reconstruction_error = np.linalg.norm(data - reconstruction, 'fro') / np.linalg.norm(data, 'fro')
            
            # Convert continuous-time eigenvalues to discrete-time if needed
            if self.time_step != 1.0:
                discrete_eigenvalues = np.exp(eigenvalues * self.time_step)
            else:
                discrete_eigenvalues = eigenvalues
            
            return DMDResults(
                eigenvalues=discrete_eigenvalues,
                eigenvectors=eigenvectors,
                modes=modes,
                amplitudes=amplitudes,
                reconstruction_error=reconstruction_error,
                metadata={
                    'rank': r,
                    'original_rank': len(s),
                    'condition_number': np.max(s_r) / np.min(s_r) if np.min(s_r) > 0 else float('inf'),
                    'time_step': self.time_step,
                    'n_snapshots': n_snapshots,
                    'n_features': n_features
                }
            )
            
        except Exception as e:
            warnings.warn(f"DMD computation failed: {str(e)}")
            return DMDResults(
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                modes=np.array([]),
                amplitudes=np.array([]),
                reconstruction_error=float('inf'),
                metadata={'error': str(e)}
            )
    
    def extract_spectrum(self, operator_matrix: np.ndarray) -> SpectralResults:
        """
        Extract spectrum from operator matrix (for compatibility with base class).
        
        Args:
            operator_matrix: Koopman operator matrix
            
        Returns:
            SpectralResults containing eigenvalues and eigenvectors
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(operator_matrix)
            
            # Sort by magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            return SpectralResults(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                spectral_error=0.0,
                convergence_history=[],
                metadata={
                    'extraction_method': 'dmd_operator_matrix',
                    'spectral_radius': np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0.0,
                    'n_eigenvalues': len(eigenvalues)
                }
            )
            
        except Exception as e:
            warnings.warn(f"Spectrum extraction failed: {str(e)}")
            return SpectralResults(
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                spectral_error=float('inf'),
                convergence_history=[],
                metadata={'error': str(e)}
            )
    
    def compute_spectral_error(self, learned_spectrum: SpectralResults,
                              reference_spectrum: SpectralResults) -> float:
        """
        Compute spectral error between two spectra.
        
        Args:
            learned_spectrum: Spectrum from neural network
            reference_spectrum: Reference spectrum (e.g., from DMD)
            
        Returns:
            Relative spectral error
        """
        if len(learned_spectrum.eigenvalues) == 0 or len(reference_spectrum.eigenvalues) == 0:
            return float('inf')
        
        try:
            # Compare dominant eigenvalues
            n_compare = min(len(learned_spectrum.eigenvalues), len(reference_spectrum.eigenvalues), 10)
            
            learned_vals = learned_spectrum.eigenvalues[:n_compare]
            ref_vals = reference_spectrum.eigenvalues[:n_compare]
            
            # Compute relative error in magnitudes
            learned_mags = np.abs(learned_vals)
            ref_mags = np.abs(ref_vals)
            
            relative_errors = np.abs(learned_mags - ref_mags) / (ref_mags + 1e-12)
            
            return np.mean(relative_errors)
            
        except Exception as e:
            warnings.warn(f"Error computing spectral error: {str(e)}")
            return float('inf')
    
    def visualize_spectrum(self, spectrum: SpectralResults, 
                          save_path: Optional[str] = None) -> None:
        """
        Visualize eigenvalue spectrum in complex plane.
        
        Args:
            spectrum: SpectralResults to visualize
            save_path: Optional path to save figure
        """
        if len(spectrum.eigenvalues) == 0:
            warnings.warn("Cannot visualize empty spectrum")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            eigenvals = spectrum.eigenvalues
            real_parts = np.real(eigenvals)
            imag_parts = np.imag(eigenvals)
            
            # Plot eigenvalues
            scatter = ax.scatter(real_parts, imag_parts, 
                               c=np.abs(eigenvals), 
                               cmap='plasma', 
                               s=60, 
                               alpha=0.8,
                               edgecolors='black',
                               linewidth=0.5)
            
            # Add unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1, label='Unit Circle')
            
            ax.set_xlabel('Real Part', fontsize=12)
            ax.set_ylabel('Imaginary Part', fontsize=12)
            ax.set_title('DMD Eigenvalue Spectrum', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend()
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('|λ| (Eigenvalue Magnitude)', fontsize=10)
            
            # Add info text
            info_text = f"N eigenvalues: {len(eigenvals)}\n"
            info_text += f"Spectral radius: {np.max(np.abs(eigenvals)):.4f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"DMD spectrum visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating spectrum visualization: {str(e)}")
    
    def _reconstruct_data(self, modes: np.ndarray, eigenvalues: np.ndarray, 
                         amplitudes: np.ndarray, n_snapshots: int) -> np.ndarray:
        """
        Reconstruct data using DMD modes and eigenvalues.
        
        Args:
            modes: DMD modes
            eigenvalues: DMD eigenvalues
            amplitudes: Initial amplitudes
            n_snapshots: Number of time snapshots
            
        Returns:
            Reconstructed data matrix
        """
        time_dynamics = np.zeros((len(eigenvalues), n_snapshots), dtype=complex)
        
        for i, eigenval in enumerate(eigenvalues):
            time_dynamics[i, :] = amplitudes[i] * (eigenval ** np.arange(n_snapshots))
        
        reconstruction = modes @ time_dynamics
        
        return np.real(reconstruction)
    
    def compare_with_neural_methods(self, dmd_results: DMDResults, 
                                   neural_spectra: Dict[str, SpectralResults]) -> Dict[str, Any]:
        """
        Compare DMD results with neural network methods.
        
        Args:
            dmd_results: Results from DMD analysis
            neural_spectra: Dictionary of neural network spectral results
            
        Returns:
            Comparison metrics and analysis
        """
        comparison = {
            'dmd_reconstruction_error': dmd_results.reconstruction_error,
            'dmd_n_modes': len(dmd_results.eigenvalues),
            'neural_comparisons': {}
        }
        
        # Convert DMD results to SpectralResults format for comparison
        dmd_spectrum = SpectralResults(
            eigenvalues=dmd_results.eigenvalues,
            eigenvectors=dmd_results.eigenvectors,
            spectral_error=dmd_results.reconstruction_error,
            convergence_history=[],
            metadata=dmd_results.metadata
        )
        
        for method_name, neural_spectrum in neural_spectra.items():
            spectral_error = self.compute_spectral_error(neural_spectrum, dmd_spectrum)
            
            comparison['neural_comparisons'][method_name] = {
                'spectral_error_vs_dmd': spectral_error,
                'n_eigenvalues': len(neural_spectrum.eigenvalues),
                'spectral_radius_diff': (
                    np.max(np.abs(neural_spectrum.eigenvalues)) - 
                    np.max(np.abs(dmd_results.eigenvalues))
                ) if len(neural_spectrum.eigenvalues) > 0 and len(dmd_results.eigenvalues) > 0 else float('inf')
            }
        
        return comparison