#!/usr/bin/env python3
"""
Demonstration of eigenvalue extraction from Koopman models.

This script shows how to use the KoopmanSpectralExtractor to extract
eigenvalues and eigenvectors from trained neural network models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.spectral.koopman_spectral_extractor import KoopmanSpectralExtractor
from src.models.base.koopman_model import KoopmanModel


class SimpleKoopmanModel(KoopmanModel):
    """Simple test model for demonstration."""
    
    def __init__(self, config):
        super().__init__(config)
        # Create a simple 2x2 operator matrix with known eigenvalues
        self.operator_matrix = np.array([[0.8, 0.3], [-0.2, 0.9]])
        
    def forward(self, x):
        """Forward pass through the model."""
        return torch.matmul(x, torch.tensor(self.operator_matrix.T, dtype=torch.float32))
    
    def get_operator_matrix(self):
        """Return the operator matrix."""
        return self.operator_matrix
    
    def train_step(self, batch):
        """Dummy training step."""
        return 0.1
    
    def evaluate(self, test_data):
        """Dummy evaluation."""
        return {'mse_loss': 0.05}


def main():
    """Main demonstration function."""
    print("=== Koopman Spectral Extraction Demo ===\n")
    
    # 1. Create a simple test model
    print("1. Creating test Koopman model...")
    model_config = {'input_dim': 2, 'output_dim': 2}
    model = SimpleKoopmanModel(model_config)
    
    print(f"   Operator matrix:\n{model.get_operator_matrix()}")
    
    # Compute analytical eigenvalues for comparison
    analytical_eigenvals = np.linalg.eigvals(model.get_operator_matrix())
    print(f"   Analytical eigenvalues: {analytical_eigenvals}")
    print()
    
    # 2. Create spectral extractor
    print("2. Creating spectral extractor...")
    extractor_config = {
        'tolerance': 1e-10,
        'max_eigenvalues': 10,
        'visualization_dpi': 150,
        'figure_size': (8, 6)
    }
    extractor = KoopmanSpectralExtractor(extractor_config)
    print("   ✓ Extractor created")
    print()
    
    # 3. Extract spectrum from model
    print("3. Extracting spectrum from model...")
    spectrum = extractor.extract_spectrum_from_model(model)
    
    print(f"   Extracted eigenvalues: {spectrum.eigenvalues}")
    print(f"   Number of eigenvalues: {len(spectrum.eigenvalues)}")
    print(f"   Spectral radius: {spectrum.metadata.get('spectral_radius', 'N/A'):.6f}")
    print(f"   Model type: {spectrum.metadata.get('model_type', 'N/A')}")
    print()
    
    # 4. Validate extraction accuracy
    print("4. Validating extraction accuracy...")
    extracted_eigenvals = spectrum.eigenvalues
    
    # Sort both arrays by magnitude for comparison
    analytical_sorted = analytical_eigenvals[np.argsort(np.abs(analytical_eigenvals))[::-1]]
    extracted_sorted = extracted_eigenvals[np.argsort(np.abs(extracted_eigenvals))[::-1]]
    
    error = np.mean(np.abs(analytical_sorted - extracted_sorted))
    print(f"   Mean absolute error: {error:.2e}")
    
    if error < 1e-10:
        print("   ✓ Extraction is highly accurate!")
    elif error < 1e-6:
        print("   ✓ Extraction is reasonably accurate")
    else:
        print("   ⚠ Extraction may have issues")
    print()
    
    # 5. Test direct matrix extraction
    print("5. Testing direct matrix extraction...")
    test_matrix = np.array([[1.2, 0.4], [-0.3, 0.7]])
    direct_spectrum = extractor.extract_spectrum(test_matrix)
    
    print(f"   Test matrix:\n{test_matrix}")
    print(f"   Extracted eigenvalues: {direct_spectrum.eigenvalues}")
    print(f"   Analytical eigenvalues: {np.linalg.eigvals(test_matrix)}")
    print()
    
    # 6. Test spectral error computation
    print("6. Testing spectral error computation...")
    
    # Create a slightly perturbed spectrum for comparison
    perturbed_eigenvals = spectrum.eigenvalues * (1 + 0.01 * np.random.randn(len(spectrum.eigenvalues)))
    perturbed_spectrum = spectrum
    perturbed_spectrum.eigenvalues = perturbed_eigenvals
    
    spectral_error = extractor.compute_spectral_error(perturbed_spectrum, spectrum)
    print(f"   Spectral error between original and perturbed: {spectral_error:.6f}")
    print()
    
    # 7. Visualization (optional - comment out if running headless)
    print("7. Creating spectrum visualization...")
    try:
        # Create a more interesting spectrum for visualization
        complex_matrix = np.array([[0.8, 0.5], [-0.4, 0.6]])
        complex_spectrum = extractor.extract_spectrum(complex_matrix)
        
        print(f"   Complex eigenvalues: {complex_spectrum.eigenvalues}")
        
        # This will show the plot - comment out if running in headless environment
        # extractor.visualize_spectrum(complex_spectrum, save_path='spectrum_demo.png')
        print("   ✓ Visualization created (commented out for headless operation)")
        
    except Exception as e:
        print(f"   ⚠ Visualization failed: {e}")
    
    print()
    print("=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()