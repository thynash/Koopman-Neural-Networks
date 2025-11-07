"""
Unit tests for spectral analysis and eigenvalue extraction functionality.

This module tests the core spectral analysis capabilities including
eigenvalue extraction from models, spectrum validation, and visualization.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, patch

from src.analysis.spectral.koopman_spectral_extractor import KoopmanSpectralExtractor
from src.analysis.spectral.spectral_analyzer import SpectralResults
from src.analysis.spectral.eigenfunction_visualizer import EigenfunctionVisualizer
from src.models.base.koopman_model import KoopmanModel


class MockKoopmanModel(KoopmanModel):
    """Mock Koopman model for testing purposes."""
    
    def __init__(self, config):
        super().__init__(config)
        self.test_matrix = np.array([[0.8, 0.2], [-0.1, 0.9]])
    
    def forward(self, x):
        return torch.matmul(x, torch.tensor(self.test_matrix.T, dtype=torch.float32))
    
    def get_operator_matrix(self):
        return self.test_matrix
    
    def train_step(self, batch):
        return 0.1
    
    def evaluate(self, test_data):
        return {'mse_loss': 0.05, 'mae_loss': 0.03}


class TestKoopmanSpectralExtractor(unittest.TestCase):
    """Test cases for KoopmanSpectralExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'tolerance': 1e-10,
            'max_eigenvalues': 10,
            'visualization_dpi': 100,
            'figure_size': (8, 6)
        }
        self.extractor = KoopmanSpectralExtractor(self.config)
        
        # Create mock model
        model_config = {'input_dim': 2, 'output_dim': 2}
        self.mock_model = MockKoopmanModel(model_config)
    
    def test_extract_spectrum_from_matrix(self):
        """Test eigenvalue extraction from operator matrix."""
        # Create test matrix with known eigenvalues
        test_matrix = np.array([[2.0, 1.0], [0.0, 1.0]])
        
        spectrum = self.extractor.extract_spectrum(test_matrix)
        
        # Check that eigenvalues are extracted
        self.assertGreater(len(spectrum.eigenvalues), 0)
        self.assertEqual(len(spectrum.eigenvalues), 2)
        
        # Check eigenvalues are approximately correct (2.0 and 1.0)
        eigenvals_sorted = np.sort(np.abs(spectrum.eigenvalues))[::-1]
        np.testing.assert_allclose(eigenvals_sorted, [2.0, 1.0], rtol=1e-10)
        
        # Check metadata
        self.assertIn('spectral_radius', spectrum.metadata)
        self.assertIn('n_eigenvalues', spectrum.metadata)
    
    def test_extract_spectrum_from_model(self):
        """Test eigenvalue extraction from trained model."""
        spectrum = self.extractor.extract_spectrum_from_model(self.mock_model)
        
        # Check that spectrum is extracted successfully
        self.assertGreater(len(spectrum.eigenvalues), 0)
        self.assertIn('model_type', spectrum.metadata)
        self.assertEqual(spectrum.metadata['model_type'], 'MockKoopmanModel')
    
    def test_spectrum_validation(self):
        """Test spectrum validation functionality."""
        # Valid spectrum
        valid_spectrum = SpectralResults(
            eigenvalues=np.array([1.0, 0.5]),
            eigenvectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            spectral_error=0.1,
            convergence_history=[],
            metadata={}
        )
        self.assertTrue(self.extractor.validate_spectrum(valid_spectrum))
        
        # Invalid spectrum with NaN eigenvalues
        invalid_spectrum = SpectralResults(
            eigenvalues=np.array([np.nan, 0.5]),
            eigenvectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            spectral_error=0.1,
            convergence_history=[],
            metadata={}
        )
        self.assertFalse(self.extractor.validate_spectrum(invalid_spectrum))
        
        # Invalid spectrum with zero eigenvectors
        zero_eigenvectors_spectrum = SpectralResults(
            eigenvalues=np.array([1.0, 0.5]),
            eigenvectors=np.zeros((2, 2)),
            spectral_error=0.1,
            convergence_history=[],
            metadata={}
        )
        self.assertFalse(self.extractor.validate_spectrum(zero_eigenvectors_spectrum))
    
    def test_spectral_error_computation(self):
        """Test spectral error computation between spectra."""
        # Create reference spectrum
        reference_spectrum = SpectralResults(
            eigenvalues=np.array([1.0, 0.5]),
            eigenvectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            spectral_error=0.0,
            convergence_history=[],
            metadata={}
        )
        
        # Create learned spectrum with small error
        learned_spectrum = SpectralResults(
            eigenvalues=np.array([1.05, 0.48]),
            eigenvectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            spectral_error=0.0,
            convergence_history=[],
            metadata={}
        )
        
        error = self.extractor.compute_spectral_error(learned_spectrum, reference_spectrum)
        
        # Error should be small but non-zero
        self.assertGreater(error, 0.0)
        self.assertLess(error, 0.2)  # Should be reasonable
    
    def test_jacobian_based_extraction(self):
        """Test Jacobian-based spectrum extraction."""
        # Test with mock model
        spectrum = self.extractor.extract_spectrum_with_jacobian(self.mock_model)
        
        # Should extract some eigenvalues
        self.assertGreaterEqual(len(spectrum.eigenvalues), 0)
        self.assertIn('extraction_method', spectrum.metadata)
        self.assertEqual(spectrum.metadata['extraction_method'], 'jacobian_based')
    
    def test_operator_matrix_validation(self):
        """Test operator matrix validation."""
        # Valid matrix
        valid_matrix = np.array([[1.0, 0.5], [0.2, 0.8]])
        self.assertTrue(self.extractor._validate_operator_matrix(valid_matrix))
        
        # Invalid matrix with NaN
        invalid_matrix = np.array([[np.nan, 0.5], [0.2, 0.8]])
        self.assertFalse(self.extractor._validate_operator_matrix(invalid_matrix))
        
        # Non-square matrix
        non_square_matrix = np.array([[1.0, 0.5, 0.3], [0.2, 0.8, 0.1]])
        self.assertFalse(self.extractor._validate_operator_matrix(non_square_matrix))
    
    def test_save_and_load_results(self):
        """Test saving and loading spectral results."""
        # Create test spectrum
        spectrum = SpectralResults(
            eigenvalues=np.array([1.0, 0.5]),
            eigenvectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            spectral_error=0.1,
            convergence_history=[],
            metadata={'test': 'data'}
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.extractor.save_results(spectrum, tmp_path)
            
            # Load and verify
            loaded_spectrum = self.extractor.load_results(tmp_path)
            
            np.testing.assert_array_equal(spectrum.eigenvalues, loaded_spectrum.eigenvalues)
            np.testing.assert_array_equal(spectrum.eigenvectors, loaded_spectrum.eigenvectors)
            self.assertEqual(spectrum.spectral_error, loaded_spectrum.spectral_error)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('matplotlib.pyplot.show')
    def test_spectrum_visualization(self, mock_show):
        """Test spectrum visualization (without actually showing plots)."""
        # Create test spectrum
        spectrum = SpectralResults(
            eigenvalues=np.array([1.0+0.5j, 0.8-0.3j, 0.6]),
            eigenvectors=np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.3], [0.2, 0.1, 0.8]]),
            spectral_error=0.1,
            convergence_history=[],
            metadata={'spectral_radius': 1.18}
        )
        
        # Test visualization (should not raise exception)
        try:
            self.extractor.visualize_spectrum(spectrum)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"Spectrum visualization raised exception: {e}")


class TestEigenfunctionVisualizer(unittest.TestCase):
    """Test cases for EigenfunctionVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'figure_size': (8, 6),
            'dpi': 100,
            'colormap': 'RdBu',
            'grid_resolution': 50
        }
        self.visualizer = EigenfunctionVisualizer(self.config)
        
        # Create test spectrum
        self.test_spectrum = SpectralResults(
            eigenvalues=np.array([1.0+0.2j, 0.8, 0.6-0.1j]),
            eigenvectors=np.random.rand(100, 3) + 1j * np.random.rand(100, 3),
            spectral_error=0.05,
            convergence_history=[],
            metadata={}
        )
    
    @patch('matplotlib.pyplot.show')
    def test_spatial_eigenfunction_visualization(self, mock_show):
        """Test spatial eigenfunction visualization."""
        # Create spatial domain
        spatial_domain = np.random.rand(100, 2)
        
        try:
            self.visualizer.visualize_spatial_eigenfunctions(
                self.test_spectrum, spatial_domain, n_functions=2
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"Spatial eigenfunction visualization raised exception: {e}")
    
    @patch('matplotlib.pyplot.show')
    def test_eigenfunction_components_visualization(self, mock_show):
        """Test eigenfunction components visualization."""
        try:
            self.visualizer.visualize_eigenfunction_components(
                self.test_spectrum, eigenfunction_idx=0
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"Eigenfunction components visualization raised exception: {e}")
    
    @patch('matplotlib.pyplot.show')
    def test_eigenfunction_comparison(self, mock_show):
        """Test eigenfunction comparison across models."""
        # Create multiple spectra for comparison
        spectra = {
            'Model1': self.test_spectrum,
            'Model2': SpectralResults(
                eigenvalues=np.array([0.9+0.1j, 0.7, 0.5]),
                eigenvectors=np.random.rand(100, 3),
                spectral_error=0.08,
                convergence_history=[],
                metadata={}
            )
        }
        
        try:
            self.visualizer.compare_eigenfunctions(spectra, eigenfunction_idx=0)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"Eigenfunction comparison raised exception: {e}")
    
    @patch('matplotlib.pyplot.show')
    def test_eigenfunction_evolution_visualization(self, mock_show):
        """Test eigenfunction temporal evolution visualization."""
        time_steps = np.linspace(0, 5, 50)
        initial_condition = np.ones(100)
        
        try:
            self.visualizer.visualize_eigenfunction_evolution(
                self.test_spectrum, time_steps, initial_condition, eigenfunction_idx=0
            )
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"Eigenfunction evolution visualization raised exception: {e}")


class TestSpectralAnalysisIntegration(unittest.TestCase):
    """Integration tests for spectral analysis components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.extractor_config = {
            'tolerance': 1e-10,
            'max_eigenvalues': 5,
            'visualization_dpi': 100
        }
        self.extractor = KoopmanSpectralExtractor(self.extractor_config)
        
        self.visualizer_config = {
            'figure_size': (8, 6),
            'dpi': 100,
            'grid_resolution': 20
        }
        self.visualizer = EigenfunctionVisualizer(self.visualizer_config)
    
    def test_end_to_end_spectral_analysis(self):
        """Test complete spectral analysis workflow."""
        # Create mock model
        model_config = {'input_dim': 2, 'output_dim': 2}
        mock_model = MockKoopmanModel(model_config)
        
        # Extract spectrum
        spectrum = self.extractor.extract_spectrum_from_model(mock_model)
        
        # Validate spectrum
        self.assertTrue(self.extractor.validate_spectrum(spectrum))
        
        # Check that we can create visualizations without errors
        with patch('matplotlib.pyplot.show'):
            try:
                self.extractor.visualize_spectrum(spectrum)
                
                # Create spatial domain for eigenfunction visualization
                spatial_domain = np.random.rand(50, 2)
                self.visualizer.visualize_spatial_eigenfunctions(
                    spectrum, spatial_domain, n_functions=1
                )
                
            except Exception as e:
                self.fail(f"End-to-end spectral analysis failed: {e}")
    
    def test_multiple_model_comparison(self):
        """Test spectral comparison across multiple models."""
        # Create multiple mock models with different matrices
        model1_config = {'input_dim': 2, 'output_dim': 2}
        model1 = MockKoopmanModel(model1_config)
        
        model2_config = {'input_dim': 2, 'output_dim': 2}
        model2 = MockKoopmanModel(model2_config)
        model2.test_matrix = np.array([[0.7, 0.3], [-0.2, 0.8]])
        
        # Extract spectra
        spectrum1 = self.extractor.extract_spectrum_from_model(model1)
        spectrum2 = self.extractor.extract_spectrum_from_model(model2)
        
        # Compare spectra
        spectra = {'Model1': spectrum1, 'Model2': spectrum2}
        comparison_results = self.extractor.compare_spectra(spectra)
        
        # Check comparison results
        self.assertIn('Model2', comparison_results)  # Model1 is reference
        self.assertIn('spectral_error', comparison_results['Model2'])
        self.assertIn('spectral_radius', comparison_results['Model2'])


if __name__ == '__main__':
    unittest.main()