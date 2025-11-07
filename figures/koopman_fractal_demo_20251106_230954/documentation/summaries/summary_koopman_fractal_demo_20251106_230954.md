# Experiment Results Summary

**Experiment:** koopman_fractal_demo
**Timestamp:** 20251106_230954
**Date:** 2025-11-06 23:10:46

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

