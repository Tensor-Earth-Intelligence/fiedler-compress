"""
Hyperbolic (Poincare disk) similarity backend.

Embeds chunk term distributions into the Poincare disk model of hyperbolic
space, which naturally captures hierarchical relationships (instruction ->
context -> detail). Chunks with parent-child relationships are close in
hyperbolic space even when cosine similarity is moderate.

Uses TF-IDF nearest-neighbor pairs with positional directionality as
training data for the Poincare embedding.

Reference: Nickel & Kiela (2017) "Poincare Embeddings for Learning
Hierarchical Representations".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer.graph import _compute_tfidf_matrix, _tokenize


def _poincare_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Poincare disk distance: d(u,v) = acosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))."""
    norm_u_sq = np.dot(u, u)
    norm_v_sq = np.dot(v, v)
    diff_sq = np.dot(u - v, u - v)

    # Clip norms to stay inside the disk (numerical stability)
    denom = max((1.0 - norm_u_sq) * (1.0 - norm_v_sq), 1e-10)
    x = 1.0 + 2.0 * diff_sq / denom

    # acosh(x) = log(x + sqrt(x^2 - 1))
    return float(np.log(x + np.sqrt(max(x * x - 1.0, 0.0)) + 1e-10))


def hyperbolic_similarity(
    chunks: Sequence[Chunk],
    dim: int = 10,
    epochs: int = 50,
) -> np.ndarray:
    """
    Compute similarity using Poincare disk embeddings.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to compare.
    dim : int
        Dimensionality of the Poincare embedding.
    epochs : int
        Training epochs for the Poincare model.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix, shape (n, n).
    """
    n = len(chunks)
    if n < 2:
        return np.ones((n, n))

    # Build training pairs for Poincare embedding
    # Use TF-IDF to find nearest neighbors, with earlier chunks as "hypernyms"
    tfidf = _compute_tfidf_matrix(chunks)
    tfidf = np.maximum(tfidf, 0.0)
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    tfidf_normed = tfidf / norms
    cosine_sim = tfidf_normed @ tfidf_normed.T

    # Build (hypernym, hyponym) pairs:
    # Adjacent chunks: earlier = more general (hypernym-like)
    # High-similarity non-adjacent: earlier = hypernym
    pairs = []
    chunk_labels = [f"chunk_{i}" for i in range(n)]

    for i in range(n):
        # Adjacent pair
        if i + 1 < n:
            pairs.append((chunk_labels[i], chunk_labels[i + 1]))

        # Top-k nearest neighbors (k=3) with directionality
        sims = cosine_sim[i].copy()
        sims[i] = -1  # exclude self
        top_k = np.argsort(sims)[-3:]
        for j in top_k:
            if sims[j] > 0.1:
                if i < j:
                    pairs.append((chunk_labels[i], chunk_labels[j]))
                else:
                    pairs.append((chunk_labels[j], chunk_labels[i]))

    if len(pairs) < 2:
        # Not enough structure for Poincare -- fall back to cosine
        return np.maximum(cosine_sim, 0.0)

    # Deduplicate pairs
    pairs = list(set(pairs))

    # Train Poincare embedding
    try:
        from gensim.models.poincare import PoincareModel

        model = PoincareModel(
            pairs,
            size=dim,
            negative=2,
            epochs=epochs,
            burn_in=5,
        )
        model.train(epochs=epochs)

        # Extract embeddings
        embeddings = np.zeros((n, dim))
        for i in range(n):
            label = chunk_labels[i]
            if label in model.kv:
                embeddings[i] = model.kv[label]
            else:
                # Fallback: place at origin
                embeddings[i] = np.zeros(dim)

        # Compute pairwise hyperbolic distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _poincare_distance(embeddings[i], embeddings[j])
                distances[i, j] = d
                distances[j, i] = d

        # Convert to similarity: sim = exp(-distance)
        similarity = np.exp(-distances)
        np.fill_diagonal(similarity, 1.0)
        return similarity

    except Exception:
        # If Poincare training fails, fall back to cosine
        return np.maximum(cosine_sim, 0.0)
