"""
Wasserstein / Sinkhorn divergence similarity backend.

Computes optimal transport distance between chunk term distributions,
accounting for word-level semantic relationships via a ground metric.
Uses Sinkhorn's algorithm (entropic regularization) for O(n^2) speed
instead of O(n^3 log n) exact Wasserstein.

Reference: Cuturi (2013) "Sinkhorn Distances", Peyre & Cuturi (2019).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer.graph import _compute_tfidf_matrix


def wasserstein_similarity(
    chunks: Sequence[Chunk],
    reg: float = 0.1,
    max_vocab: int = 500,
) -> np.ndarray:
    """
    Compute Sinkhorn divergence similarity between chunk distributions.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to compare.
    reg : float
        Entropic regularization parameter. Higher = faster, less precise.
    max_vocab : int
        Maximum vocabulary size for the ground metric. Larger = more accurate
        but O(vocab^2) memory for the cost matrix.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix, shape (n, n), values in [0, 1].
    """
    import ot

    n = len(chunks)
    if n < 2:
        return np.ones((n, n))

    # Get TF-IDF matrix
    tfidf = _compute_tfidf_matrix(chunks)
    tfidf = np.maximum(tfidf, 0.0)

    # Reduce vocabulary for tractable ground metric computation
    vocab_size = tfidf.shape[1]
    if vocab_size > max_vocab:
        # Keep top terms by total weight
        col_sums = tfidf.sum(axis=0)
        top_indices = np.argsort(col_sums)[-max_vocab:]
        tfidf = tfidf[:, top_indices]
        vocab_size = max_vocab

    # Normalize rows to probability distributions
    row_sums = tfidf.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    probs = tfidf / row_sums

    # Ensure valid distributions (add small uniform for empty rows)
    for i in range(n):
        if probs[i].sum() < 1e-10:
            probs[i] = 1.0 / vocab_size

    # Ground metric: cosine distance between word TF-IDF column vectors
    # Word vectors = columns of the TF-IDF matrix (how each word distributes across docs)
    word_vectors = tfidf.T  # shape (vocab, n_docs)
    word_norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    word_norms = np.where(word_norms == 0, 1, word_norms)
    word_normed = word_vectors / word_norms
    word_cosine = word_normed @ word_normed.T
    cost_matrix = np.maximum(1.0 - word_cosine, 0.0)  # distance = 1 - similarity
    cost_matrix = cost_matrix.astype(np.float64)

    # Compute pairwise Sinkhorn divergence
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = probs[i].astype(np.float64)
            b = probs[j].astype(np.float64)
            # Ensure they sum to exactly 1
            a = a / a.sum()
            b = b / b.sum()
            try:
                d = ot.sinkhorn2(a, b, cost_matrix, reg=reg, numItermax=100)
                distances[i, j] = float(d)
                distances[j, i] = float(d)
            except Exception:
                distances[i, j] = 1.0
                distances[j, i] = 1.0

    # Convert distances to similarities
    max_dist = distances.max()
    if max_dist > 0:
        similarity = 1.0 - distances / max_dist
    else:
        similarity = np.ones((n, n))

    np.fill_diagonal(similarity, 1.0)
    return similarity
