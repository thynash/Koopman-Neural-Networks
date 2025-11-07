"""
Concrete implementation of spectral analysis for Koopman operators.
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.analysis.spectral.spectral_analyzer import SpectralAnalyzer, SpectralResults
from src.models.base.koopman_model import KoopmanModel


class KoopmanSpectralExtractor(SpectralAnalyzer):
    """
    Concrete implementation for extracting spectral properties from Koopman models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Koopman spectral extractor."""
        super().__init__(config)
        
        self.tolerance = config.get('tolerance', 1e-10)
        self.max_eigenvalues = config.get('max_eigenvalues', 50)
        self.reference_point = config.get('reference_point', None)
        self.visualization_dpi = config.get('visualization_dpi', 300)
        self.figure_size = config.get('figure_size', (10, 8))
    
    def extract_spectrum_from_model(self, model: KoopmanModel) -> SpectralResults:
        """Extract eigenvalues and eigenvectors from a trained Koopman model."""
        try:
            operator_matrix = model.get_operator_matrix()
            
            if not self._validate_operator_matrix(operator_matrix):
                raise ValueError("Invalid operator matrix extracted from model")
            
            spectrum = self.extract_spectrum(operator_matrix)
            
            spectrum.metadata.update({
                'model_type': model.__class__.__name__,
                'model_config': model.config,
                'operator_shape': operator_matrix.shape,
                'extraction_method': 'model_operator_matrix'
            })
            
            return spectrum
            
        except Exception as e:
            warnings.warn(f"Failed to extract spectrum from model: {str(e)}")
            return SpectralResults(
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
                spectral_error=float('inf'),
                convergence_history=[],
                metadata={'error': str(e), 'model_type': model.__class__.__name__}
            )
    
    def extract_spectrum(self, operator_matrix: np.ndarray) -> SpectralResults:
        """Extract eigenvalues and eigenvectors from a Koopman operator matrix."""
        try:
            eigenvalues, eigenvectors = scipy.linalg.eig(operator_matrix)
            
            eigenvalues, eigenvectors = self._sort_eigenvalues(eigenvalues, eigenvectors)
            
            if self.max_eigenvalues > 0:
                n_keep = min(self.max_eigenvalues, len(eigenvalues))
                eigenvalues = eigenvalues[:n_keep]
                eigenvectors = eigenvectors[:, :n_keep]
            
            spectral_radius = np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0.0
            
            spectrum = SpectralResults(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                spectral_error=0.0,
                convergence_history=[],
                metadata={
                    'spectral_radius': spectral_radius,
                    'n_eigenvalues': len(eigenvalues),
                    'operator_condition_number': np.linalg.cond(operator_matrix),
                    'operator_determinant': np.linalg.det(operator_matrix),
                    'extraction_timestamp': np.datetime64('now').astype(str)
                }
            )
            
            if not self.validate_spectrum(spectrum):
                warnings.warn("Extracted spectrum failed validation checks")
            
            return spectrum
            
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Linear algebra error during eigenvalue computation: {str(e)}")
            return self._create_empty_spectrum(error_msg=str(e))
        
        except Exception as e:
            warnings.warn(f"Unexpected error during spectrum extraction: {str(e)}")
            return self._create_empty_spectrum(error_msg=str(e))
    
    def compute_spectral_error(self, learned_spectrum: SpectralResults,
                              reference_spectrum: SpectralResults) -> float:
        """Compute error between learned and reference spectra."""
        if len(learned_spectrum.eigenvalues) == 0 or len(reference_spectrum.eigenvalues) == 0:
            return float('inf')
        
        try:
            learned_vals = learned_spectrum.eigenvalues
            reference_vals = reference_spectrum.eigenvalues
            
            n_learned = len(learned_vals)
            n_ref = len(reference_vals)
            n_compare = min(n_learned, n_ref)
            
            if n_compare == 0:
                return float('inf')
            
            learned_sorted = learned_vals[np.argsort(np.abs(learned_vals))[::-1]][:n_compare]
            ref_sorted = reference_vals[np.argsort(np.abs(reference_vals))[::-1]][:n_compare]
            
            learned_mags = np.abs(learned_sorted)
            ref_mags = np.abs(ref_sorted)
            
            ref_mags_safe = np.maximum(ref_mags, 1e-12)
            relative_errors = np.abs(learned_mags - ref_mags) / ref_mags_safe
            
            return np.mean(relative_errors)
            
        except Exception as e:
            warnings.warn(f"Error computing spectral error: {str(e)}")
            return float('inf')
    
    def visualize_spectrum(self, spectrum: SpectralResults, 
                          save_path: Optional[str] = None) -> None:
        """Create visualization of eigenvalue spectrum in complex plane."""
        if len(spectrum.eigenvalues) == 0:
            warnings.warn("Cannot visualize empty spectrum")
            return
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.visualization_dpi)
            
            real_parts = np.real(spectrum.eigenvalues)
            imag_parts = np.imag(spectrum.eigenvalues)
            
            scatter = ax.scatter(real_parts, imag_parts, 
                               c=np.abs(spectrum.eigenvalues), 
                               cmap='viridis', 
                               s=50, 
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
            
            theta = np.linspace(0, 2*np.pi, 100)
            unit_circle_x = np.cos(theta)
            unit_circle_y = np.sin(theta)
            ax.plot(unit_circle_x, unit_circle_y, 'k--', alpha=0.3, linewidth=1, label='Unit Circle')
            
            ax.set_xlabel('Real Part', fontsize=12)
            ax.set_ylabel('Imaginary Part', fontsize=12)
            ax.set_title('Koopman Operator Eigenvalue Spectrum', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('|λ| (Eigenvalue Magnitude)', fontsize=10)
            
            ax.legend()
            
            info_text = f"N eigenvalues: {len(spectrum.eigenvalues)}\n"
            info_text += f"Spectral radius: {spectrum.metadata.get('spectral_radius', 'N/A'):.4f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.visualization_dpi, bbox_inches='tight')
                print(f"Spectrum visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Error creating spectrum visualization: {str(e)}")
    
    def _validate_operator_matrix(self, matrix: np.ndarray) -> bool:
        """Validate that the operator matrix is suitable for spectral analysis."""
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return False
        
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False
        
        try:
            cond_num = np.linalg.cond(matrix)
            if cond_num > 1e12:
                warnings.warn(f"Matrix is ill-conditioned (cond={cond_num:.2e})")
        except:
            pass
        
        return True
    
    def extract_spectrum_with_jacobian(self, model: KoopmanModel, 
                                       reference_point: Optional[np.ndarray] = None) -> SpectralResults:
        """
        Extract spectrum using Jacobian-based linearization of the model.
        
        This method computes the Jacobian of the neural network at a reference point
        and uses it as the linear operator approximation for spectral analysis.
        
        Args:
            model: Trained Koopman model
            reference_point: Point at which to compute Jacobian (default: origin)
            
        Returns:
            SpectralResults containing eigenvalues and eigenvectors
        """
        try:
            # Check if torch is available and model supports PyTorch operations
            if not TORCH_AVAILABLE:
                # Fallback to operator matrix extraction if torch is not available
                warnings.warn("PyTorch not available, falling back to operator matrix extraction")
                return self.extract_spectrum_from_model(model)
            
            if hasattr(model, 'eval'):
                model.eval()
            
            # Set reference point (default to origin)
            if reference_point is None:
                if hasattr(model, 'input_dim'):
                    input_dim = model.input_dim
                elif hasattr(model, 'state_dim'):
                    input_dim = model.state_dim
                elif hasattr(model, 'config') and 'input_dim' in model.config:
                    input_dim = model.config['input_dim']
                else:
                    input_dim = 2  # Default for 2D fractals
                reference_point = np.zeros(input_dim)
            
            # Convert to tensor and enable gradients
            ref_tensor = torch.tensor(reference_point, dtype=torch.float32, requires_grad=True)
            ref_tensor = ref_tensor.unsqueeze(0)  # Add batch dimension
            if hasattr(model, 'to_device'):
                ref_tensor = model.to_device(ref_tensor)
            
            # Forward pass
            if hasattr(model, 'forward_single_input'):
                # For models like DeepONet that need special handling
                output = model.forward_single_input(ref_tensor)
            else:
                output = model.forward(ref_tensor)
            
            # Compute Jacobian matrix
            output_dim = output.shape[1]
            input_dim = ref_tensor.shape[1]
            jacobian = torch.zeros(output_dim, input_dim)
            
            for i in range(output_dim):
                # Compute gradient of i-th output w.r.t. input
                grad_outputs = torch.zeros_like(output)
                grad_outputs[0, i] = 1.0
                
                grad_input = torch.autograd.grad(
                    outputs=output,
                    inputs=ref_tensor,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                
                if grad_input is not None:
                    jacobian[i] = grad_input.squeeze()
            
            # Extract spectrum from Jacobian
            jacobian_np = jacobian.detach().cpu().numpy()
            spectrum = self.extract_spectrum(jacobian_np)
            
            # Update metadata (preserve existing metadata and add new fields)
            spectrum.metadata.update({
                'model_type': model.__class__.__name__,
                'extraction_method': 'jacobian_based',
                'reference_point': reference_point.tolist(),
                'jacobian_condition_number': np.linalg.cond(jacobian_np) if jacobian_np.size > 0 else float('inf')
            })
            
            return spectrum
            
        except Exception as e:
            warnings.warn(f"Failed to extract spectrum using Jacobian method: {str(e)}")
            return self._create_empty_spectrum(error_msg=f"Jacobian extraction failed: {str(e)}")
    
    def _create_empty_spectrum(self, error_msg: str = "") -> SpectralResults:
        """Create an empty SpectralResults object for error cases."""
        return SpectralResults(
            eigenvalues=np.array([]),
            eigenvectors=np.array([]),
            spectral_error=float('inf'),
            convergence_history=[],
            metadata={'error': error_msg}
        )