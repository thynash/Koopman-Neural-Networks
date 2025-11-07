# Installation and Usage Guide

This guide provides detailed instructions for installing, configuring, and using the Koopman Fractal Spectral Learning framework.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Environment Setup](#environment-setup)
4. [Configuration](#configuration)
5. [Quick Start Tutorial](#quick-start-tutorial)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB free space for installation and data
- **CPU**: Multi-core processor (4+ cores recommended)

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 20.04+) or macOS 11+
- **Python**: 3.9 or 3.10
- **RAM**: 32 GB or more
- **Storage**: 50 GB+ SSD storage
- **CPU**: 8+ core processor (Intel i7/i9 or AMD Ryzen 7/9)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### GPU Support (Optional)

For accelerated training with CUDA:
- **NVIDIA GPU**: GTX 1060 or better (RTX series recommended)
- **CUDA**: Version 11.0 or higher
- **cuDNN**: Version 8.0 or higher
- **GPU Memory**: 6 GB minimum, 12 GB+ recommended

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/koopman-fractal-spectral-learning.git
cd koopman-fractal-spectral-learning

# Create virtual environment
python -m venv koopman_env

# Activate virtual environment
# On Linux/macOS:
source koopman_env/bin/activate
# On Windows:
koopman_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy, scipy, matplotlib; print('Installation successful!')"
```

### Method 2: Conda Installation

```bash
# Clone the repository
git clone https://github.com/your-username/koopman-fractal-spectral-learning.git
cd koopman-fractal-spectral-learning

# Create conda environment
conda create -n koopman_env python=3.9
conda activate koopman_env

# Install dependencies via conda
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy scipy matplotlib seaborn pandas
conda install jupyter ipywidgets
conda install -c conda-forge scikit-learn pyyaml tqdm

# Install remaining dependencies via pip
pip install -r requirements-extra.txt

# Verify installation
python -c "import torch, numpy, scipy, matplotlib; print('Installation successful!')"
```

### Method 3: Docker Installation

```bash
# Clone the repository
git clone https://github.com/your-username/koopman-fractal-spectral-learning.git
cd koopman-fractal-spectral-learning

# Build Docker image
docker build -t koopman-fractal .

# Run container
docker run -it --rm -v $(pwd):/workspace koopman-fractal

# For GPU support (requires nvidia-docker)
docker run --gpus all -it --rm -v $(pwd):/workspace koopman-fractal
```

### Method 4: Development Installation

For contributors and developers:

```bash
# Clone with development dependencies
git clone https://github.com/your-username/koopman-fractal-spectral-learning.git
cd koopman-fractal-spectral-learning

# Create development environment
python -m venv koopman_dev_env
source koopman_dev_env/bin/activate  # Linux/macOS
# koopman_dev_env\Scripts\activate  # Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/ -v
```

## Environment Setup

### Virtual Environment Management

**Creating Environment:**
```bash
# Using venv
python -m venv koopman_env

# Using conda
conda create -n koopman_env python=3.9

# Using virtualenv
virtualenv koopman_env
```

**Activating Environment:**
```bash
# venv (Linux/macOS)
source koopman_env/bin/activate

# venv (Windows)
koopman_env\Scripts\activate

# conda
conda activate koopman_env
```

**Deactivating Environment:**
```bash
# venv
deactivate

# conda
conda deactivate
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
PYTHONPATH=/path/to/koopman-fractal-spectral-learning/src
CUDA_VISIBLE_DEVICES=0  # GPU device ID (if using GPU)
OMP_NUM_THREADS=8       # Number of CPU threads
MKL_NUM_THREADS=8       # Intel MKL threads
NUMBA_NUM_THREADS=8     # Numba threads

# Data directories
DATA_DIR=/path/to/data
RESULTS_DIR=/path/to/results
FIGURES_DIR=/path/to/figures

# Logging
LOG_LEVEL=INFO
LOG_FILE=/path/to/logs/koopman.log
```

Load environment variables:
```bash
# Linux/macOS
export $(cat .env | xargs)

# Or use python-dotenv
pip install python-dotenv
```

### CUDA Setup (GPU Support)

**Check CUDA Installation:**
```bash
nvidia-smi
nvcc --version
```

**Install PyTorch with CUDA:**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Configuration

### Project Structure Setup

```bash
# Create necessary directories
mkdir -p data/{raw,processed,external}
mkdir -p models/{checkpoints,trained}
mkdir -p results/{experiments,evaluation,comparison}
mkdir -p figures/{fractals,training,spectral,publication}
mkdir -p logs
mkdir -p configs/experiments

# Set permissions (Linux/macOS)
chmod +x experiments/scripts/*.py
```

### Configuration Files

**Main Configuration (`config/default.yaml`):**
```yaml
# Project settings
project:
  name: "koopman-fractal-spectral-learning"
  version: "1.0.0"
  author: "Your Name"
  
# Paths
paths:
  data_dir: "data"
  models_dir: "models"
  results_dir: "results"
  figures_dir: "figures"
  logs_dir: "logs"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/koopman.log"

# Computation
computation:
  device: "auto"  # "cpu", "cuda", or "auto"
  num_workers: 4
  random_seed: 42
  
# Visualization
visualization:
  dpi: 600
  format: "png"
  style: "seaborn-v0_8"
```

**Experiment Configuration (`experiments/configs/sierpinski_mlp.yaml`):**
```yaml
# Experiment metadata
experiment:
  name: "sierpinski_mlp_baseline"
  description: "MLP training on Sierpinski gasket data"
  tags: ["mlp", "sierpinski", "baseline"]

# Data configuration
data:
  system_type: "sierpinski"
  n_points: 20000
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  normalize: true
  augmentation: false
  seed: 42

# Model configuration
model:
  type: "mlp"
  input_dim: 2
  hidden_dims: [128, 256, 128, 64]
  output_dim: 2
  activation: "relu"
  dropout_rate: 0.2
  use_batch_norm: true
  
# Training configuration
training:
  optimizer: "adam"
  learning_rate: 0.001
  batch_size: 64
  epochs: 200
  weight_decay: 0.0001
  
  # Learning rate scheduling
  scheduler:
    type: "reduce_on_plateau"
    patience: 15
    factor: 0.5
    min_lr: 1e-6
    
  # Early stopping
  early_stopping:
    patience: 30
    min_delta: 1e-6
    
  # Checkpointing
  checkpoint:
    save_freq: 25
    save_best: true
    
# Evaluation configuration
evaluation:
  metrics: ["mse", "mae", "r2"]
  spectral_analysis: true
  max_eigenvalues: 50
  
# Output configuration
output:
  save_model: true
  save_results: true
  save_figures: true
  results_dir: "results/sierpinski_mlp"
```

### Environment Configuration

**Requirements File (`requirements.txt`):**
```txt
# Core dependencies
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Configuration and utilities
pyyaml>=6.0
tqdm>=4.62.0
click>=8.0.0
python-dotenv>=0.19.0

# Jupyter and visualization
jupyter>=1.0.0
ipywidgets>=7.6.0
plotly>=5.0.0

# Optional GPU support
# torch-audio  # Uncomment if needed
# torchtext    # Uncomment if needed
```

**Development Requirements (`requirements-dev.txt`):**
```txt
# Testing
pytest>=6.2.0
pytest-cov>=3.0.0
pytest-mock>=3.6.0

# Code quality
black>=22.0.0
flake8>=4.0.0
isort>=5.10.0
mypy>=0.950

# Documentation
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0

# Development tools
pre-commit>=2.17.0
jupyter-lab>=3.3.0
```

## Quick Start Tutorial

### Step 1: Generate Fractal Data

```bash
# Generate data for all fractal systems
python experiments/scripts/generate_data.py \
    --systems sierpinski barnsley julia \
    --output-dir data/

# Check generated data
ls -la data/
# Expected output:
# sierpinski_trajectories.npy
# barnsley_trajectories.npy  
# julia_trajectories.npy
```

### Step 2: Train a Single Model

```bash
# Train MLP on Sierpinski data
python experiments/scripts/train_mlp.py \
    --config experiments/configs/sierpinski_mlp.yaml \
    --output-dir results/mlp_sierpinski

# Monitor training progress
tail -f logs/koopman.log
```

### Step 3: Train Multiple Models

```bash
# Train all architectures on Sierpinski data
python experiments/scripts/train_all_models.py \
    --data-path data/sierpinski_trajectories.npy \
    --models mlp deeponet \
    --output-dir results/sierpinski_comparison

# Check results
ls -la results/sierpinski_comparison/
```

### Step 4: Evaluate and Compare

```bash
# Run comprehensive evaluation
python experiments/scripts/evaluate_and_compare.py \
    --results-dir results/sierpinski_comparison \
    --data-path data/sierpinski_trajectories.npy \
    --output-dir results/evaluation

# View generated figures
ls -la results/evaluation/figures/
```

### Step 5: Interactive Exploration

```bash
# Launch Jupyter notebooks
jupyter notebook experiments/notebooks/

# Open and run:
# 1. fractal_exploration.ipynb - Interactive fractal visualization
# 2. model_comparison.ipynb - Model performance analysis
# 3. results_demonstration.ipynb - Key findings summary
```

## Advanced Usage

### Custom Fractal Systems

**Implementing New Fractal Generator:**

```python
# src/data/generators/custom_fractal.py
import numpy as np
from .fractal_generator import FractalGenerator

class CustomFractalGenerator(FractalGenerator):
    def generate_custom_system(self, n_points: int, **params) -> np.ndarray:
        """Implement your custom fractal system here."""
        # Your implementation
        states = np.random.randn(n_points, 2)  # Placeholder
        next_states = self.apply_custom_dynamics(states, **params)
        return states, next_states
    
    def apply_custom_dynamics(self, states: np.ndarray, **params) -> np.ndarray:
        """Define your custom dynamics."""
        # Your custom transformation
        return states  # Placeholder
```

**Register Custom System:**

```python
# In your experiment script
from src.data.generators.custom_fractal import CustomFractalGenerator

generator = CustomFractalGenerator()
states, next_states = generator.generate_custom_system(
    n_points=10000,
    custom_param1=0.5,
    custom_param2=1.0
)
```

### Custom Neural Architectures

**Implementing New Model:**

```python
# src/models/custom/custom_koopman.py
import torch
import torch.nn as nn
from ..base.koopman_model import KoopmanModel

class CustomKoopmanModel(KoopmanModel):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        # Define your architecture
        self.layers = nn.Sequential(
            # Your custom layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def get_operator_matrix(self) -> np.ndarray:
        # Extract operator matrix
        # Your implementation
        pass
```

### Batch Processing

**Processing Multiple Datasets:**

```bash
# Create batch processing script
cat > batch_process.sh << 'EOF'
#!/bin/bash

SYSTEMS=("sierpinski" "barnsley" "julia")
MODELS=("mlp" "deeponet")

for system in "${SYSTEMS[@]}"; do
    echo "Processing $system system..."
    
    # Generate data
    python experiments/scripts/generate_data.py \
        --systems $system \
        --output-dir data/
    
    # Train models
    for model in "${MODELS[@]}"; do
        echo "Training $model on $system..."
        python experiments/scripts/train_all_models.py \
            --data-path data/${system}_trajectories.npy \
            --models $model \
            --output-dir results/${system}_${model}
    done
    
    # Evaluate
    python experiments/scripts/evaluate_and_compare.py \
        --results-dir results/${system}_comparison \
        --output-dir results/${system}_evaluation
done
EOF

chmod +x batch_process.sh
./batch_process.sh
```

### Hyperparameter Optimization

**Using Optuna for Hyperparameter Search:**

```python
# hyperparameter_search.py
import optuna
from src.training.trainers.mlp_trainer import MLPTrainer

def objective(trial):
    # Suggest hyperparameters
    config = {
        'hidden_dims': [
            trial.suggest_int('hidden_1', 64, 512),
            trial.suggest_int('hidden_2', 64, 512),
            trial.suggest_int('hidden_3', 32, 256)
        ],
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }
    
    # Train and evaluate model
    trainer = MLPTrainer(config)
    results = trainer.train(dataset, config)
    
    return results['best_val_loss']

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")
```

### Distributed Training

**Multi-GPU Training:**

```python
# distributed_training.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def train_distributed(model, dataset):
    setup_distributed()
    
    model = model.cuda()
    model = DDP(model)
    
    # Training loop with distributed data loading
    # Your implementation here
    
if __name__ == '__main__':
    # Launch with: torchrun --nproc_per_node=2 distributed_training.py
    train_distributed(model, dataset)
```

## Troubleshooting

### Common Installation Issues

**Issue: PyTorch CUDA version mismatch**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue: Memory errors during training**
```python
# Reduce batch size in config
training:
  batch_size: 32  # Reduce from 64

# Or enable gradient checkpointing
model:
  gradient_checkpointing: true
```

**Issue: Slow training on CPU**
```bash
# Check CPU utilization
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Use smaller models for CPU training
model:
  hidden_dims: [64, 128, 64]  # Reduce from [128, 256, 128, 64]
```

### Debugging Tips

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
logging:
  level: "DEBUG"
```

**Profile Memory Usage:**
```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your training code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Check Data Loading:**
```python
# Verify data shapes and ranges
print(f"States shape: {states.shape}")
print(f"States range: [{states.min():.3f}, {states.max():.3f}]")
print(f"Next states shape: {next_states.shape}")

# Check for NaN values
print(f"NaN in states: {np.isnan(states).any()}")
print(f"NaN in next_states: {np.isnan(next_states).any()}")
```

### Performance Issues

**Slow Data Loading:**
```python
# Increase number of workers
dataset.get_data_loaders(batch_size=64, num_workers=8)

# Use pin_memory for GPU training
DataLoader(dataset, batch_size=64, pin_memory=True)
```

**Training Convergence Issues:**
```yaml
# Adjust learning rate
training:
  learning_rate: 0.0001  # Reduce if loss explodes
  
# Add gradient clipping
training:
  gradient_clip_norm: 1.0
  
# Use different optimizer
training:
  optimizer: "adamw"
  weight_decay: 0.01
```

## Performance Optimization

### Hardware Optimization

**CPU Optimization:**
```bash
# Set optimal thread counts
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMBA_NUM_THREADS=8

# Use Intel MKL if available
pip install mkl

# Enable CPU optimizations in PyTorch
torch.set_num_threads(8)
```

**GPU Optimization:**
```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Memory Optimization:**
```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()

# Use smaller data types
model.half()  # Use FP16 instead of FP32
```

### Code Optimization

**Vectorization:**
```python
# Use NumPy vectorized operations
# Instead of loops:
for i in range(len(states)):
    next_states[i] = transform(states[i])

# Use vectorized operations:
next_states = np.apply_along_axis(transform, 1, states)
```

**Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(x):
    # Cache results of expensive computations
    return result
```

**Parallel Processing:**
```python
from multiprocessing import Pool
from joblib import Parallel, delayed

# Use multiprocessing for data generation
def generate_parallel(n_processes=8):
    with Pool(n_processes) as pool:
        results = pool.map(generate_chunk, chunk_params)
    return np.concatenate(results)

# Or use joblib
results = Parallel(n_jobs=8)(
    delayed(generate_chunk)(params) for params in chunk_params
)
```

This comprehensive installation and usage guide should help users get started with the framework and optimize their usage for different scenarios and hardware configurations.