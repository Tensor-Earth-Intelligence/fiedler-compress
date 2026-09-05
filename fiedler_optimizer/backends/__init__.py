"""
Geometric similarity backends for Fiedler spectral compression.

Each backend takes a list of Chunk objects and returns a symmetric
similarity matrix (numpy ndarray, shape [n, n]).
"""

from fiedler_optimizer.backends.registry import get_backend, list_backends, BACKENDS

__all__ = ["get_backend", "list_backends", "BACKENDS"]
