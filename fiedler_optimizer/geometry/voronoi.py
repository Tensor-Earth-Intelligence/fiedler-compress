"""Voronoi density-based redundancy analysis and compression.

Estimates each chunk's Voronoi cell volume via k-nearest-neighbor
density.  Chunks with small cells (high density, crowded by similar
neighbors) are redundant.  Chunks with large cells (low density,
unique semantic territory) are essential.

This method requires:
- No graph construction
- No eigendecomposition
- No zone classification
- No iterative optimization

It is a third mathematically independent compression pathway
(after Fiedler spectral and Dymaxion information-theoretic).
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def estimate_voronoi_volumes(vectors: np.ndarray, k: int = 5,
                             metric: str = "cosine", agg: str = "kth") -> np.ndarray:
    """Estimate Voronoi cell volumes via k-nearest-neighbor distances.

    For each point, the distance to its k-th nearest neighbor serves
    as a proxy for the Voronoi cell radius.  We use the raw k-NN
    distance as the importance score (larger distance = larger cell =
    more unique = more important) rather than computing d-dimensional
    volumes, because the relative ranking is preserved and avoids
    numerical overflow in high dimensions.

    Args:
        vectors: (n_samples, n_features) array of embedded points.
        k: Number of neighbors.  Default: 5.  Automatically clamped to
           min(k, n_samples - 1).

    Returns:
        (n_samples,) array of estimated Voronoi importance scores.
        Higher = more unique (larger Voronoi cell).
        Lower = more redundant (smaller Voronoi cell).
    """
    n = vectors.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return np.ones(n)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=metric)
    nn.fit(vectors)
    distances, _ = nn.kneighbors(vectors)

    # Column 0 is self (dist ~0); columns 1..k_actual are the k neighbours.
    neigh = distances[:, 1:k_actual + 1]
    if agg == "mean":
        return neigh.mean(axis=1)          # smoother density (robust to one duplicate)
    if agg == "sum":
        return neigh.sum(axis=1)           # truer cell-volume proxy
    return distances[:, k_actual]          # 'kth' -- default, unchanged behaviour


def voronoi_compress(
    text: str,
    target: str = "2x",
    k: int = 5,
    backend: str = "tfidf",
    metric: str = "cosine",
    agg: str = "kth",
    content_prior=None,
    prior_weight: float = 10.0,
) -> dict:
    """Compress text using Voronoi density-based pruning.

    Chunks with the smallest Voronoi cells (most semantically crowded)
    are pruned first.  Chunks with the largest cells (unique semantic
    territory) are preserved.

    Args:
        text: Input text to compress.
        target: Compression target ('2x', '4x', or float ratio like 0.5).
        k: k-NN parameter for density estimation.  Default: 5.
        backend: Embedding method.
                 'tfidf' - raw TF-IDF vectors.
                 'aitchison' - CLR-transformed TF-IDF.
        content_prior: Optional class-level supervision.  None (default) leaves the
                 pure density ranking unchanged.  'identifier' boosts chunks that
                 carry identifier/secret-shaped tokens (codes, API/license keys,
                 ticket IDs, SKUs, long serials) so they survive past the point
                 where boilerplate framing lines would otherwise crowd them out.
                 May also be any callable(text)->float returning a per-chunk boost.
                 This is a content-TYPE prior only: it never sees the query or the
                 answer.  NOTE: opt-in.  On identifier-dense inputs (code, logs,
                 JSON) it flags most chunks and stops discriminating; and it does
                 nothing for a natural-language needle with no lexical signature.
        prior_weight: Strength of the content_prior boost on top of the standardized
                 density score.  Saturates by ~4 (>=4 gives hard-tier behaviour:
                 flagged chunks kept until only they remain); smaller values blend.

    Returns:
        dict with compressed text and metadata.
        NOTE: No lambda2, no fiedler_vector, no spectral certificate.
    """
    t0 = time.perf_counter()

    from fiedler_optimizer.chunker import chunk_text, ChunkingStrategy, merge_kept_spans

    chunks = chunk_text(text, strategy=ChunkingStrategy.ADAPTIVE)
    chunk_texts = [c.text for c in chunks]
    n = len(chunks)

    if n < 2:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "compressed": text,
            "original_tokens": _estimate_tokens(text),
            "compressed_tokens": _estimate_tokens(text),
            "compression_ratio": 0.0,
            "voronoi_scores": [1.0] * n,
            "chunks_removed": 0,
            "chunks_total": n,
            "time_ms": elapsed,
            "method": "voronoi",
            "k": k,
            "backend": backend,
        }

    # Embed chunks into vector space
    vectors = _embed_chunks(chunk_texts, backend)

    # Estimate Voronoi cell volumes
    voronoi_scores = estimate_voronoi_volumes(vectors, k=k, metric=metric, agg=agg)

    # Optional class-level supervision: lift identifier-bearing chunks above the
    # boilerplate.  When content_prior is None this branch is skipped and the
    # ranking is byte-identical to the unsupervised path.
    prior_name = None
    if content_prior is not None:
        prior_fn = content_prior if callable(content_prior) else _CONTENT_PRIORS.get(content_prior)
        if prior_fn is None:
            raise ValueError(
                f"unknown content_prior {content_prior!r}; "
                f"use one of {sorted(_CONTENT_PRIORS)} or a callable(text)->float")
        boosts = np.array([float(prior_fn(t)) for t in chunk_texts])
        if np.any(boosts > 0):
            scale = voronoi_scores.std() or 1.0
            voronoi_scores = voronoi_scores / scale + prior_weight * boosts
        prior_name = content_prior if isinstance(content_prior, str) else "callable"

    # Determine how many chunks to remove
    target_ratio = _parse_target(target)
    n_to_remove = int(n * target_ratio)
    n_to_remove = min(n_to_remove, n - 1)

    # Prune chunks with smallest Voronoi scores (most redundant)
    removal_order = np.argsort(voronoi_scores)  # smallest first
    remove_indices = set(removal_order[:n_to_remove].tolist())

    # Reassemble in original order
    kept_indices = [i for i in range(n) if i not in remove_indices]
    compressed = merge_kept_spans(text, chunks, kept_indices)

    elapsed = (time.perf_counter() - t0) * 1000
    orig_tokens = _estimate_tokens(text)
    comp_tokens = _estimate_tokens(compressed)

    return {
        "compressed": compressed,
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "compression_ratio": 1.0 - (comp_tokens / max(orig_tokens, 1)),
        "voronoi_scores": voronoi_scores.tolist(),
        "chunks_removed": len(remove_indices),
        "chunks_total": n,
        "time_ms": elapsed,
        "method": "voronoi",
        "k": k,
        "backend": backend,
        "metric": metric,
        "agg": agg,
        "content_prior": prior_name,
        "prior_weight": prior_weight if prior_name else None,
    }


def voronoi_anomaly_score(
    baseline_vectors: np.ndarray,
    new_vectors: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Score new points for anomaly relative to a baseline distribution.

    Points whose k-NN distance in the baseline distribution is large
    are anomalous (they occupy empty semantic territory).

    Args:
        baseline_vectors: (n_baseline, n_features) baseline point cloud.
        new_vectors: (n_new, n_features) points to score.
        k: k-NN parameter.

    Returns:
        (n_new,) array of anomaly scores.  Higher = more anomalous.
    """
    k_actual = min(k, baseline_vectors.shape[0])
    if k_actual < 1:
        return np.zeros(new_vectors.shape[0])

    nn = NearestNeighbors(n_neighbors=k_actual, metric="cosine")
    nn.fit(baseline_vectors)
    distances, _ = nn.kneighbors(new_vectors)
    return distances[:, -1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# The curated identifier detector lives in geometry/content_prior.py (single
# source of truth). Re-exported here under the private names the hook uses.
from fiedler_optimizer.geometry.content_prior import (  # noqa: E402
    identifier_prior as _identifier_prior,
    CONTENT_PRIORS as _CONTENT_PRIORS,
)


def _embed_chunks(chunk_texts: list[str], backend: str = "tfidf") -> np.ndarray:
    """Embed chunk texts into a vector space."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(chunk_texts).toarray()

    if backend == "aitchison":
        # CLR transform for compositional data
        tfidf_pos = tfidf.copy()
        tfidf_pos[tfidf_pos == 0] = 1e-10
        row_sums = tfidf_pos.sum(axis=1, keepdims=True)
        tfidf_pos = tfidf_pos / row_sums
        log_data = np.log(tfidf_pos)
        return log_data - log_data.mean(axis=1, keepdims=True)

    return tfidf


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_target(target) -> float:
    if isinstance(target, (int, float)):
        return float(target)
    t = str(target).strip().lower()
    if t.endswith("x"):
        return 1.0 - (1.0 / float(t[:-1]))
    return float(t)
