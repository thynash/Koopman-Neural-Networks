# Experiment Results Summary

**Experiment:** koopman_fractal_demo
**Timestamp:** 20251106_231117
**Date:** 2025-11-06 23:12:08

## Model Performance Comparison

| Model | Final Train Loss | Final Val Loss | Best Val Loss | Convergence Epoch | Training Time (s) | Spectral Error |
|-------|------------------|----------------|---------------|-------------------|-------------------|----------------|
| MLP | 1.23e-02 | 1.56e-02 | 1.45e-02 | 78 | 195.5 | 8.90e-03 |
| DeepONet | 9.80e-03 | 1.34e-02 | 1.28e-02 | 85 | 357.2 | 6.70e-03 |
| LSTM | 1.45e-02 | 1.78e-02 | 1.65e-02 | 92 | 285.7 | 1.12e-02 |

## Key Findings

- **Best Training Loss:** DeepONet (9.80e-03)
- **Best Validation Loss:** DeepONet (1.28e-02)
- **Best Spectral Approximation:** DeepONet (6.70e-03)
- **Fastest Convergence:** MLP (78 epochs)

## Eigenvalue Analysis

- **MLP:** 5 eigenvalues, dominant lambda = 0.9500 + 0.1000i
- **DeepONet:** 5 eigenvalues, dominant lambda = 0.9200 + 0.0800i
- **LSTM:** 5 eigenvalues, dominant lambda = 0.9000 + 0.1200i

## Computational Efficiency

- **Total Training Time:** 838.4 seconds
- **Average Memory Usage:** 232.1 MB

### Training Time Breakdown

- **MLP:** 195.5s (23.3%)
- **DeepONet:** 357.2s (42.6%)
- **LSTM:** 285.7s (34.1%)
