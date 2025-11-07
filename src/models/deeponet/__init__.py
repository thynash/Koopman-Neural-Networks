"""
Deep Neural Operator (DeepONet) implementation for Koopman operator learning.

This module implements the DeepONet architecture with branch-trunk structure
for learning operators between function spaces in fractal dynamical systems.
"""

from .deeponet_koopman import DeepONetKoopman

__all__ = ['DeepONetKoopman']