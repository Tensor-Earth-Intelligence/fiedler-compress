"""
Fisher-Rao information-geometric similarity backend.

Computes the Fisher-Rao geodesic distance between chunk term distributions,
which is equivalent to the Bhattacharyya angle for multinomial distributions:

    d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))

This is the fastest geometric backend -- closed-form, no iterative solver.

Reference: Rao (1945), Atkinson & Mitchell (1981).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer.graph import _compute_tfidf_matrix


def fisher_rao_similarity(chunks: Sequence[Chunk]) -> np.ndarray:
    """
    Compute Fisher-Rao similarity between chunk term distributions.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to compare.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix, shape (n, n), values in [0, 1].
    """
    n = len(chunks)
    if n < 2:
        return np.ones((n, n))

    # Get TF-IDF matrix
    tfidf = _compute_tfidf_matrix(chunks)
    tfidf = np.maximum(tfidf, 0.0)

    # Normalize rows to probability distributions (L1 norm)
    row_sums = tfidf.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    probs = tfidf / row_sums

    # Add small epsilon to avoid sqrt(0) issues
    probs = probs + 1e-12
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Compute sqrt(p) for Bhattacharyya coefficient
    sqrt_probs = np.sqrt(probs)

    # Bhattacharyya coefficient: BC(p,q) = sum(sqrt(p_i * q_i)) = sqrt_p . sqrt_q
    bc_matrix = sqrt_probs @ sqrt_probs.T

    # Clip to [0, 1] for numerical stability (rounding can push slightly above 1)
    bc_matrix = np.clip(bc_matrix, 0.0, 1.0)

    # Fisher-Rao distance = 2 * arccos(BC)
    # Similarity = 1 - distance/pi (normalize to [0, 1])
    # Or more directly: use BC itself as similarity (it's already in [0, 1])
    # BC = 1 means identical distributions, BC = 0 means completely different
    return bc_matrix
