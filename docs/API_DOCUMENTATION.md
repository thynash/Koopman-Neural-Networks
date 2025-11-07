# API Documentation

This document provides comprehensive API documentation for all classes and methods in the Koopman Fractal Spectral Learning framework.

## Table of Contents

1. [Data Generation](#data-generation)
2. [Neural Models](#neural-models)
3. [Training](#training)
4. [Analysis](#analysis)
5. [Visualization](#visualization)
6. [Configuration](#configuration)

## Data Generation

### FractalGenerator

Main interface for generating fractal trajectory data.

```python
class FractalGenerator:
    """
    Main generator for fractal dynamical systems.
    
    Provides unified interface for generating trajectory data from
    different fractal systems including IFS and Julia sets.
    """
    
    def __init__(self):
        """Initialize the fractal generator."""
        
    def generate_trajectories(self, system_type: str, n_points: int, 
                            save_path: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Generate trajectory data for specified fractal system.
        
        Args:
            system_type: Type of fractal system ('sierpinski', 'barnsley', 'julia')
            n_points: Number of trajectory points to generate
            save_path: Optional path to save generated data
            **kwargs: System-specific parameters
            
        Returns:
            Generated trajectory data as numpy array
            
        Raises:
            ValueError: If system_type is not supported
        """
        
    def generate_sierpinski_trajectories(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Sierpinski gasket trajectory data.
        
        Args:
            n_points: Number of trajectory points
            
        Returns:
            Tuple of (states, next_states) arrays
        """
        
    def generate_barnsley_trajectories(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Barnsley fern trajectory data.
        
        Args:
            n_points: Number of trajectory points
            
        Returns:
            Tuple of (states, next_states) arrays
        """
        
    def generate_julia_trajectories(self, n_points: int, c_real: float = -0.7269,
                                  c_imag: float = 0.1889) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Julia set trajectory data.
        
        Args:
            n_points: Number of trajectory points
            c_real: Real part of complex parameter c
            c_imag: Imaginary part of complex parameter c
            
        Returns:
            Tuple of (states, next_states) arrays
        """
```

### IFSGenerator

Specialized generator for Iterated Function Systems.

```python
class IFSGenerator:
    """
    Generator for Iterated Function System (IFS) fractals.
    
    Implements deterministic and probabilistic IFS including
    Sierpinski gasket and Barnsley fern.
    """
    
    def __init__(self):
        """Initialize IFS generator with transformation matrices."""
        
    def generate_sierpinski(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Sierpinski gasket using IFS.
        
        Mathematical formulation:
        f₁(x,y) = (0.5x, 0.5y)
        f₂(x,y) = (0.5x + 0.5, 0.5y)  
        f₃(x,y) = (0.5x + 0.25, 0.5y + 0.433)
        
        Args:
            n_points: Number of points to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 2) containing trajectory points
        """
        
    def generate_barnsley_fern(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Barnsley fern using probabilistic IFS.
        
        Transformation probabilities:
        - f₁: 1% (stem)
        - f₂: 85% (main leaflet)
        - f₃: 7% (left leaflet)
        - f₄: 7% (right leaflet)
        
        Args:
            n_points: Number of points to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 2) containing trajectory points
        """
```

### JuliaGenerator

Generator for Julia set dynamics.

```python
class JuliaGenerator:
    """
    Generator for Julia set fractal dynamics.
    
    Implements complex iteration z_{n+1} = z_n^2 + c
    for various complex parameters c.
    """
    
    def __init__(self):
        """Initialize Julia set generator."""
        
    def generate_julia_set(self, n_points: int, c: complex, 
                          max_iter: int = 100, escape_radius: float = 2.0,
                          seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Julia set trajectory data.
        
        Mathematical formulation:
        z_{n+1} = z_n^2 + c
        
        Args:
            n_points: Number of trajectory points
            c: Complex parameter for Julia set
            max_iter: Maximum iterations before escape
            escape_radius: Radius for escape condition
            seed: Random seed for initial conditions
            
        Returns:
            Array of shape (n_points, 2) with real/imaginary parts
        """
        
    def is_in_julia_set(self, z0: complex, c: complex, max_iter: int = 100,
                       escape_radius: float = 2.0) -> bool:
        """
        Check if initial point z0 is in Julia set.
        
        Args:
            z0: Initial complex point
            c: Julia set parameter
            max_iter: Maximum iterations
            escape_radius: Escape radius threshold
            
        Returns:
            True if point is in Julia set, False otherwise
        """
```

### TrajectoryDataset

PyTorch dataset wrapper for trajectory data.

```python
class TrajectoryDataset(Dataset):
    """
    PyTorch dataset for fractal trajectory data.
    
    Handles data loading, normalization, and train/validation/test splits
    with proper batching for neural network training.
    """
    
    def __init__(self, states: np.ndarray, next_states: np.ndarray,
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, normalize: bool = True,
                 seed: int = 42):
        """
        Initialize trajectory dataset.
        
        Args:
            states: Current state data, shape (n_samples, state_dim)
            next_states: Next state data, shape (n_samples, state_dim)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            normalize: Whether to normalize data to [-1, 1]
            seed: Random seed for data splitting
        """
        
    def __len__(self) -> int:
        """Return dataset size."""
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (state, next_state) tensors
        """
        
    def get_data_loaders(self, batch_size: int = 32, 
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for train/val/test splits.
        
        Args:
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
```

## Neural Models

### KoopmanModel (Base Class)

Abstract base class for all Koopman operator learning models.

```python
class KoopmanModel(nn.Module):
    """
    Abstract base class for Koopman operator learning models.
    
    Defines the interface that all neural architectures must implement
    for consistent training and evaluation.
    """
    
    def __init__(self):
        """Initialize base Koopman model."""
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input state tensor, shape (batch_size, state_dim)
            
        Returns:
            Predicted next state tensor, shape (batch_size, state_dim)
        """
        
    @abstractmethod
    def get_operator_matrix(self) -> np.ndarray:
        """
        Extract linear operator matrix from trained model.
        
        Returns:
            Operator matrix as numpy array for spectral analysis
        """
        
    def predict_trajectory(self, initial_state: torch.Tensor, 
                          n_steps: int) -> torch.Tensor:
        """
        Predict trajectory evolution from initial state.
        
        Args:
            initial_state: Starting state, shape (1, state_dim)
            n_steps: Number of prediction steps
            
        Returns:
            Predicted trajectory, shape (n_steps, state_dim)
        """
```

### MLPKoopman

Multi-layer perceptron implementation for Koopman learning.

```python
class MLPKoopman(KoopmanModel):
    """
    Multi-Layer Perceptron for Koopman operator learning.
    
    Implements a feedforward neural network that learns to predict
    next states from current states, approximating the Koopman operator.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout_rate: float = 0.0,
                 use_batch_norm: bool = False):
        """
        Initialize MLP Koopman model.
        
        Args:
            input_dim: Input state dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output state dimension
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input states, shape (batch_size, input_dim)
            
        Returns:
            Predicted next states, shape (batch_size, output_dim)
        """
        
    def get_operator_matrix(self) -> np.ndarray:
        """
        Extract linear approximation of learned operator.
        
        Computes Jacobian matrix at origin or uses final linear layer
        as operator approximation.
        
        Returns:
            Operator matrix, shape (output_dim, input_dim)
        """
```

### DeepONetKoopman

Deep Operator Network implementation for Koopman learning.

```python
class DeepONetKoopman(KoopmanModel):
    """
    Deep Operator Network (DeepONet) for Koopman operator learning.
    
    Implements branch-trunk architecture for learning operators
    between function spaces, specialized for dynamical systems.
    """
    
    def __init__(self, trajectory_length: int, state_dim: int, coordinate_dim: int,
                 branch_hidden_dims: List[int], trunk_hidden_dims: List[int],
                 latent_dim: int, activation: str = 'relu', dropout_rate: float = 0.0):
        """
        Initialize DeepONet Koopman model.
        
        Args:
            trajectory_length: Length of input trajectory sequences
            state_dim: Dimension of state space
            coordinate_dim: Dimension of coordinate space
            branch_hidden_dims: Hidden dimensions for branch network
            trunk_hidden_dims: Hidden dimensions for trunk network
            latent_dim: Latent space dimension for operator representation
            activation: Activation function type
            dropout_rate: Dropout probability
        """
        
    def forward(self, trajectory: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet.
        
        Args:
            trajectory: Input trajectory, shape (batch_size, trajectory_length, state_dim)
            coordinates: Query coordinates, shape (batch_size, coordinate_dim)
            
        Returns:
            Operator output at query coordinates
        """
        
    def get_operator_matrix(self) -> np.ndarray:
        """
        Extract operator matrix from trained DeepONet.
        
        Evaluates the learned operator on a grid of coordinates
        to construct finite-dimensional matrix approximation.
        
        Returns:
            Operator matrix approximation
        """
```

## Training

### MLPTrainer

Training utilities for MLP models.

```python
class MLPTrainer:
    """
    Trainer class for MLP Koopman models.
    
    Handles training loop, validation, checkpointing, and
    hyperparameter management for MLP architectures.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize MLP trainer.
        
        Args:
            model_config: Configuration dictionary for model architecture
        """
        
    def train(self, dataset: TrajectoryDataset, training_config: Dict[str, Any],
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Train MLP model on trajectory dataset.
        
        Args:
            dataset: Training dataset
            training_config: Training hyperparameters
            resume_from: Path to checkpoint for resuming training
            
        Returns:
            Dictionary containing training results and metrics
        """
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on validation dataset.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        
    def save_checkpoint(self, epoch: int, loss: float, save_path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current training epoch
            loss: Current validation loss
            save_path: Path to save checkpoint
        """
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
```

### DeepONetTrainer

Training utilities for DeepONet models.

```python
class DeepONetTrainer:
    """
    Trainer class for DeepONet Koopman models.
    
    Implements specialized training procedures for operator learning
    including function space loss computation and operator extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepONet trainer.
        
        Args:
            config: Complete configuration including model and training parameters
        """
        
    def train(self, dataset: TrajectoryDataset) -> Dict[str, Any]:
        """
        Train DeepONet model with operator learning objectives.
        
        Args:
            dataset: Training dataset with trajectory sequences
            
        Returns:
            Training results including operator matrix and metrics
        """
        
    def compute_operator_loss(self, predictions: torch.Tensor, 
                            targets: torch.Tensor) -> torch.Tensor:
        """
        Compute specialized loss for operator learning.
        
        Combines prediction loss with operator consistency constraints.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Combined loss tensor
        """
```

## Analysis

### SpectralAnalyzer

Main class for spectral analysis of learned operators.

```python
class SpectralAnalyzer:
    """
    Spectral analysis tools for Koopman operators.
    
    Provides eigenvalue extraction, eigenfunction computation,
    and spectral property analysis for learned operators.
    """
    
    def __init__(self):
        """Initialize spectral analyzer."""
        
    def extract_eigenvalues(self, operator_matrix: np.ndarray, 
                          max_eigenvalues: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eigenvalues and eigenvectors from operator matrix.
        
        Args:
            operator_matrix: Learned operator matrix
            max_eigenvalues: Maximum number of eigenvalues to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) arrays
        """
        
    def compute_spectral_radius(self, eigenvalues: np.ndarray) -> float:
        """
        Compute spectral radius (maximum eigenvalue magnitude).
        
        Args:
            eigenvalues: Array of complex eigenvalues
            
        Returns:
            Spectral radius value
        """
        
    def analyze_stability(self, eigenvalues: np.ndarray) -> Dict[str, int]:
        """
        Analyze stability properties of eigenvalue spectrum.
        
        Args:
            eigenvalues: Array of complex eigenvalues
            
        Returns:
            Dictionary with stability statistics
        """
        
    def compute_spectral_error(self, learned_eigenvalues: np.ndarray,
                             reference_eigenvalues: np.ndarray) -> float:
        """
        Compute error between learned and reference eigenvalues.
        
        Args:
            learned_eigenvalues: Eigenvalues from neural model
            reference_eigenvalues: Reference eigenvalues (e.g., from DMD)
            
        Returns:
            Spectral approximation error
        """
```

### DMDBaseline

Dynamic Mode Decomposition baseline implementation.

```python
class DMDBaseline:
    """
    Dynamic Mode Decomposition (DMD) baseline for comparison.
    
    Implements standard DMD algorithm for computing Koopman eigenvalues
    and eigenfunctions from trajectory data.
    """
    
    def __init__(self):
        """Initialize DMD baseline."""
        
    def compute_dmd(self, states: np.ndarray, next_states: np.ndarray,
                   rank: Optional[int] = None, exact: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute DMD decomposition from trajectory data.
        
        Args:
            states: Current state data, shape (n_samples, state_dim)
            next_states: Next state data, shape (n_samples, state_dim)
            rank: Rank truncation for SVD (None for full rank)
            exact: Whether to use exact DMD algorithm
            
        Returns:
            Dictionary containing DMD eigenvalues, eigenvectors, and modes
        """
        
    def predict_trajectory(self, initial_state: np.ndarray, n_steps: int,
                          dmd_results: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict trajectory evolution using DMD model.
        
        Args:
            initial_state: Initial state vector
            n_steps: Number of prediction steps
            dmd_results: Results from compute_dmd method
            
        Returns:
            Predicted trajectory array
        """
```

### ModelComparator

Framework for comparing different model architectures.

```python
class ModelComparator:
    """
    Comprehensive model comparison framework.
    
    Provides standardized comparison protocols for evaluating
    different neural architectures on identical datasets.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        
    def compare_models(self, model_data: Dict[str, Dict[str, Any]],
                      reference_eigenvalues: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple models across various metrics.
        
        Args:
            model_data: Dictionary of model results
            reference_eigenvalues: Reference spectrum for comparison
            
        Returns:
            Comprehensive comparison results
        """
        
    def compute_performance_metrics(self, predictions: np.ndarray,
                                  targets: np.ndarray) -> Dict[str, float]:
        """
        Compute standard performance metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of performance metrics (MSE, MAE, R²)
        """
        
    def rank_models(self, comparison_results: Dict[str, Any]) -> List[str]:
        """
        Rank models based on overall performance.
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            List of model names ranked by performance
        """
```

## Visualization

### FractalVisualizer

Visualization tools for fractal attractors.

```python
class FractalVisualizer:
    """
    Visualization tools for fractal dynamical systems.
    
    Creates publication-quality plots of fractal attractors,
    trajectory evolution, and phase space analysis.
    """
    
    def __init__(self):
        """Initialize fractal visualizer."""
        
    def plot_attractor(self, states: np.ndarray, title: str = "Fractal Attractor",
                      save_path: Optional[str] = None, dpi: int = 600) -> None:
        """
        Plot fractal attractor from trajectory data.
        
        Args:
            states: Trajectory states, shape (n_points, 2)
            title: Plot title
            save_path: Path to save figure (optional)
            dpi: Figure resolution for saving
        """
        
    def plot_trajectory_evolution(self, states: np.ndarray, n_points: int = 1000,
                                save_path: Optional[str] = None) -> None:
        """
        Plot trajectory evolution with color-coded time progression.
        
        Args:
            states: Trajectory states
            n_points: Number of points to plot
            save_path: Path to save figure (optional)
        """
        
    def create_phase_space_plot(self, states: np.ndarray, next_states: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Create phase space plot (return map).
        
        Args:
            states: Current states
            next_states: Next states
            save_path: Path to save figure (optional)
        """
```

### SpectrumVisualizer

Visualization tools for eigenvalue spectra.

```python
class SpectrumVisualizer:
    """
    Visualization tools for eigenvalue spectra and spectral analysis.
    
    Creates complex plane plots, spectral comparisons, and
    eigenfunction visualizations.
    """
    
    def __init__(self):
        """Initialize spectrum visualizer."""
        
    def plot_eigenvalue_spectrum(self, eigenvalues: np.ndarray, 
                               title: str = "Eigenvalue Spectrum",
                               save_path: Optional[str] = None, dpi: int = 600) -> None:
        """
        Plot eigenvalues in complex plane.
        
        Args:
            eigenvalues: Complex eigenvalue array
            title: Plot title
            save_path: Path to save figure (optional)
            dpi: Figure resolution
        """
        
    def plot_comparative_spectrum(self, eigenvalue_data: Dict[str, np.ndarray],
                                title: str = "Spectrum Comparison",
                                save_path: Optional[str] = None, dpi: int = 600) -> None:
        """
        Plot comparative eigenvalue spectra for multiple models.
        
        Args:
            eigenvalue_data: Dictionary mapping model names to eigenvalues
            title: Plot title
            save_path: Path to save figure (optional)
            dpi: Figure resolution
        """
        
    def plot_spectral_convergence(self, eigenvalue_history: List[np.ndarray],
                                save_path: Optional[str] = None) -> None:
        """
        Plot eigenvalue convergence during training.
        
        Args:
            eigenvalue_history: List of eigenvalue arrays over training
            save_path: Path to save figure (optional)
        """
```

### TrainingVisualizer

Visualization tools for training progress and metrics.

```python
class TrainingVisualizer:
    """
    Visualization tools for training progress and performance metrics.
    
    Creates training curves, loss landscapes, and performance comparisons.
    """
    
    def __init__(self):
        """Initialize training visualizer."""
        
    def plot_training_curves(self, training_history: Dict[str, List[float]],
                           save_path: Optional[str] = None, dpi: int = 600) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            training_history: Dictionary with 'train_loss' and 'val_loss' lists
            save_path: Path to save figure (optional)
            dpi: Figure resolution
        """
        
    def plot_comparative_training_curves(self, training_histories: Dict[str, Dict[str, List[float]]],
                                       save_path: Optional[str] = None, dpi: int = 600) -> None:
        """
        Plot comparative training curves for multiple models.
        
        Args:
            training_histories: Dictionary mapping model names to training histories
            save_path: Path to save figure (optional)
            dpi: Figure resolution
        """
        
    def create_performance_dashboard(self, model_results: Dict[str, Dict[str, Any]],
                                   save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance dashboard.
        
        Args:
            model_results: Dictionary of model performance results
            save_path: Path to save figure (optional)
        """
```

## Configuration

### ConfigManager

Configuration management utilities.

```python
class ConfigManager:
    """
    Configuration management for experiments and model training.
    
    Handles loading, validation, and merging of configuration files
    with support for hierarchical configurations and parameter overrides.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema dictionary defining required fields
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        
    def merge_configs(self, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries with override precedence.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        
    def save_config(self, config: Dict[str, Any], save_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            save_path: Path to save configuration file
        """
```

## Usage Examples

### Basic Training Pipeline

```python
from src.data.generators.fractal_generator import FractalGenerator
from src.data.datasets.trajectory_dataset import TrajectoryDataset
from src.models.mlp.mlp_koopman import MLPKoopman
from src.training.trainers.mlp_trainer import MLPTrainer

# Generate fractal data
generator = FractalGenerator()
states, next_states = generator.generate_sierpinski_trajectories(20000)

# Create dataset
dataset = TrajectoryDataset(states, next_states, normalize=True)

# Configure and train model
model_config = {
    'input_dim': 2,
    'hidden_dims': [128, 256, 128, 64],
    'output_dim': 2,
    'activation': 'relu',
    'dropout_rate': 0.2
}

training_config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 200,
    'weight_decay': 0.0001
}

trainer = MLPTrainer(model_config)
results = trainer.train(dataset, training_config)
```

### Spectral Analysis Pipeline

```python
from src.analysis.spectral.spectral_analyzer import SpectralAnalyzer
from src.visualization.spectral.spectrum_visualizer import SpectrumVisualizer

# Extract operator matrix from trained model
operator_matrix = trainer.model.get_operator_matrix()

# Perform spectral analysis
analyzer = SpectralAnalyzer()
eigenvalues, eigenvectors = analyzer.extract_eigenvalues(operator_matrix)

# Visualize spectrum
visualizer = SpectrumVisualizer()
visualizer.plot_eigenvalue_spectrum(eigenvalues, save_path='spectrum.png')
```

### Model Comparison Pipeline

```python
from src.analysis.comparison.model_comparator import ModelComparator

# Compare multiple trained models
model_data = {
    'MLP': {'eigenvalues': mlp_eigenvalues, 'test_metrics': mlp_metrics},
    'DeepONet': {'eigenvalues': deeponet_eigenvalues, 'test_metrics': deeponet_metrics}
}

comparator = ModelComparator()
comparison_results = comparator.compare_models(model_data, dmd_eigenvalues)
ranking = comparator.rank_models(comparison_results)
```

This API documentation provides comprehensive coverage of all major classes and methods in the framework. For additional details and implementation examples, refer to the source code and Jupyter notebooks.