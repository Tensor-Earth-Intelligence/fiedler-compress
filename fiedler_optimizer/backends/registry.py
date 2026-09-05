"""
Backend registry for geometric similarity computation.

Maps backend names to functions that produce similarity matrices.
Each function takes Sequence[Chunk] and returns np.ndarray (n x n).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk


# Type alias for backend functions
BackendFn = Callable[[Sequence[Chunk]], np.ndarray]


def _tfidf_backend(chunks: Sequence[Chunk]) -> np.ndarray:
    """Default TF-IDF cosine similarity (existing behavior)."""
    from fiedler_optimizer.graph import _compute_tfidf_matrix
    vectors = _compute_tfidf_matrix(chunks)
    similarity = vectors @ vectors.T
    return similarity


def _neural_backend(chunks: Sequence[Chunk]) -> np.ndarray:
    """Neural sentence-transformer similarity (existing behavior)."""
    from fiedler_optimizer.graph import _compute_neural_embeddings
    vectors = _compute_neural_embeddings(chunks)
    similarity = vectors @ vectors.T
    return similarity


def _aitchison_backend(chunks: Sequence[Chunk]) -> np.ndarray:
    from fiedler_optimizer.backends.aitchison import aitchison_similarity
    return aitchison_similarity(chunks)


def _fisher_rao_backend(chunks: Sequence[Chunk]) -> np.ndarray:
    from fiedler_optimizer.backends.fisher_rao import fisher_rao_similarity
    return fisher_rao_similarity(chunks)


def _wasserstein_backend(chunks: Sequence[Chunk]) -> np.ndarray:
    from fiedler_optimizer.backends.wasserstein import wasserstein_similarity
    return wasserstein_similarity(chunks)


def _hyperbolic_backend(chunks: Sequence[Chunk]) -> np.ndarray:
    from fiedler_optimizer.backends.hyperbolic import hyperbolic_similarity
    return hyperbolic_similarity(chunks)


BACKENDS: dict[str, BackendFn] = {
    "tfidf": _tfidf_backend,
    "neural": _neural_backend,
    "aitchison": _aitchison_backend,
    "fisher-rao": _fisher_rao_backend,
    "wasserstein": _wasserstein_backend,
    "hyperbolic": _hyperbolic_backend,
}


def list_backends() -> list[str]:
    """Return sorted list of available backend names."""
    return sorted(BACKENDS)


def get_backend(name: str) -> BackendFn:
    """Get a backend function by name."""
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]
