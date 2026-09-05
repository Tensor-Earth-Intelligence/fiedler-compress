"""
Euler Characteristic Curve (ECC) computation for topological screening.

Computes the Euler characteristic chi = V - E + T at multiple filtration
thresholds on the similarity graph. The resulting curve serves as a fast
topological fingerprint for file integrity monitoring.

Reference: Turner et al. (2014) "Frechet Means for Distributions of
Persistence Diagrams".
"""

from __future__ import annotations

import numpy as np


def compute_ecc(similarity_matrix: np.ndarray, n_filtrations: int = 20) -> np.ndarray:
    """
    Compute Euler Characteristic Curve from a similarity matrix.

    At each filtration threshold (converting similarity to distance),
    builds the Rips complex and computes chi = V - E + T.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Symmetric similarity matrix, shape (n, n).
    n_filtrations : int
        Number of filtration steps.

    Returns
    -------
    np.ndarray
        ECC values at each filtration step, shape (n_filtrations,).
    """
    n = similarity_matrix.shape[0]
    if n < 2:
        return np.array([1.0] * n_filtrations)

    # Convert similarity to distance
    distance_matrix = 1.0 - np.clip(similarity_matrix, 0.0, 1.0)
    np.fill_diagonal(distance_matrix, 0.0)

    max_dist = distance_matrix.max()
    if max_dist <= 0:
        max_dist = 1.0

    filtration_values = np.linspace(0, max_dist * 1.1, n_filtrations)

    ecc = np.zeros(n_filtrations)

    for k, eps in enumerate(filtration_values):
        # Adjacency at this filtration: edge if distance <= eps
        adj = (distance_matrix <= eps).astype(np.float64)
        np.fill_diagonal(adj, 0)

        V = n
        E = int(adj.sum()) // 2

        # Triangle counting via matrix multiplication for n > 50
        if n > 50:
            T = int(np.trace(adj @ adj @ adj)) // 6
        else:
            T = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        for m in range(j + 1, n):
                            if adj[i, m] and adj[j, m]:
                                T += 1

        ecc[k] = V - E + T

    return ecc


def ecc_distance(ecc1: np.ndarray, ecc2: np.ndarray) -> float:
    """L2 distance between two Euler characteristic curves."""
    return float(np.sqrt(np.sum((ecc1 - ecc2) ** 2)))
