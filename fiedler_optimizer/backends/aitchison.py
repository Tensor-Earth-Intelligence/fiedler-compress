"""
Aitchison (Compositional Data Analysis) TF-IDF backend.

TF-IDF vectors are compositional data (positive, sum-normalizable).
Cosine similarity on raw TF-IDF suffers from spurious closure correlations.
The Centered Log-Ratio (CLR) transform removes these, producing similarity
that reflects genuine term co-occurrence rather than compositional artifacts.

Reference: Aitchison (1986), Martín-Fernández et al. (2003) for zero replacement.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer.graph import _compute_tfidf_matrix


def _multiplicative_replacement(data: np.ndarray, delta_factor: float = 0.65) -> np.ndarray:
    """
    Replace zeros using multiplicative replacement (Martín-Fernández et al. 2003).

    For each row, replace zeros with delta = delta_factor * min(nonzero values),
    then renormalize the row to preserve the simplex constraint.
    """
    result = data.copy()
    for i in range(result.shape[0]):
        row = result[i]
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            result[i] = 1.0 / result.shape[1]  # uniform if all zero
            continue
        delta = delta_factor * nonzero.min()
        n_zeros = np.sum(row == 0)
        if n_zeros == 0:
            continue
        # Replace zeros with delta
        row_sum = row.sum()
        result[i, row == 0] = delta
        # Rescale nonzero entries to maintain row sum
        if row_sum > 0:
            scale = (row_sum - n_zeros * delta) / row_sum
            result[i, row > 0] *= scale
    return result


def aitchison_similarity(chunks: Sequence[Chunk]) -> np.ndarray:
    """
    Compute similarity using CLR-transformed TF-IDF (Aitchison geometry).

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to compare.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix, shape (n, n).
    """
    n = len(chunks)
    if n < 2:
        return np.ones((n, n))

    # Get raw TF-IDF matrix (already L2-normalized, undo that for CoDA)
    tfidf = _compute_tfidf_matrix(chunks)
    tfidf = np.maximum(tfidf, 0.0)

    # Reduce to non-zero columns only (CoDA requires dense compositional vectors)
    col_mask = tfidf.sum(axis=0) > 0
    tfidf_dense = tfidf[:, col_mask]

    if tfidf_dense.shape[1] < 2:
        # Too few terms for meaningful CoDA — fall back to standard cosine
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return np.clip((tfidf / norms) @ (tfidf / norms).T, 0.0, 1.0)

    # Multiplicative replacement for remaining zeros
    tfidf_replaced = _multiplicative_replacement(tfidf_dense)

    # CLR transform: log(x_i / geometric_mean(x))
    log_data = np.log(np.maximum(tfidf_replaced, 1e-30))
    clr = log_data - log_data.mean(axis=1, keepdims=True)

    # Check for NaN/Inf
    if not np.all(np.isfinite(clr)):
        # Fall back to standard cosine if CLR produces numerical issues
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return np.clip((tfidf / norms) @ (tfidf / norms).T, 0.0, 1.0)

    # L2-normalize CLR vectors
    norms = np.linalg.norm(clr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    clr_normed = clr / norms

    # Cosine similarity on CLR space
    similarity = clr_normed @ clr_normed.T

    # Clip to [0, 1] — CLR cosine can go negative, which breaks the Laplacian
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity
