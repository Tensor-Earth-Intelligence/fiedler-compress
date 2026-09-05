"""Compute bridge edges between polyhedral faces.

When chunks live in different faces with different local similarity
metrics, cross-face similarity must be computed using a bridging
strategy. Bridge edges connect the Dymaxion faces into a unified
structure.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer.graph import _compute_tfidf_matrix
from fiedler_optimizer.dymaxion.faces import FaceDecomposition


# ---------------------------------------------------------------------------
# Inline similarity backends (self-contained, no external backends dependency)
# ---------------------------------------------------------------------------

def _tfidf_cosine_similarity(chunk_objs: list[Chunk]) -> np.ndarray:
    """Standard TF-IDF cosine similarity."""
    vectors = _compute_tfidf_matrix(chunk_objs)
    sim = vectors @ vectors.T
    return np.clip(sim, 0.0, 1.0)


def _aitchison_similarity(chunk_objs: list[Chunk]) -> np.ndarray:
    """CLR-transformed TF-IDF cosine similarity (Aitchison geometry)."""
    tfidf = _compute_tfidf_matrix(chunk_objs)
    tfidf = np.maximum(tfidf, 0.0)

    # Reduce to non-zero columns
    col_mask = tfidf.sum(axis=0) > 0
    tfidf_dense = tfidf[:, col_mask]

    if tfidf_dense.shape[1] < 2:
        return _tfidf_cosine_similarity(chunk_objs)

    # Multiplicative replacement for zeros
    result = tfidf_dense.copy()
    for i in range(result.shape[0]):
        row = result[i]
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            result[i] = 1.0 / result.shape[1]
            continue
        delta = 0.65 * nonzero.min()
        n_zeros = np.sum(row == 0)
        if n_zeros == 0:
            continue
        row_sum = row.sum()
        result[i, row == 0] = delta
        if row_sum > 0:
            scale = (row_sum - n_zeros * delta) / row_sum
            result[i, row > 0] *= scale

    # CLR transform
    log_data = np.log(np.maximum(result, 1e-30))
    clr = log_data - log_data.mean(axis=1, keepdims=True)

    if not np.all(np.isfinite(clr)):
        return _tfidf_cosine_similarity(chunk_objs)

    # L2 normalize and cosine
    norms = np.linalg.norm(clr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    clr_normed = clr / norms
    sim = clr_normed @ clr_normed.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def _fisher_rao_similarity(chunk_objs: list[Chunk]) -> np.ndarray:
    """Bhattacharyya coefficient between chunk term distributions."""
    tfidf = _compute_tfidf_matrix(chunk_objs)
    tfidf = np.maximum(tfidf, 0.0)

    # Normalize to probability distributions (L1)
    row_sums = tfidf.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    probs = tfidf / row_sums

    probs = probs + 1e-12
    probs = probs / probs.sum(axis=1, keepdims=True)

    sqrt_probs = np.sqrt(probs)
    bc_matrix = sqrt_probs @ sqrt_probs.T
    return np.clip(bc_matrix, 0.0, 1.0)


# Hyperbolic and Wasserstein require external deps (gensim, pot).
# Fall back to TF-IDF cosine if those backends are selected but
# dependencies aren't available.

_METRIC_FUNCTIONS = {
    "tfidf": _tfidf_cosine_similarity,
    "aitchison": _aitchison_similarity,
    "fisher-rao": _fisher_rao_similarity,
    "wasserstein": _tfidf_cosine_similarity,  # fallback
    "hyperbolic": _tfidf_cosine_similarity,   # fallback
}


def _get_metric_fn(name: str):
    """Get similarity function by metric name."""
    return _METRIC_FUNCTIONS.get(name, _tfidf_cosine_similarity)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bridge_edges(
    face_decomposition: FaceDecomposition,
    all_chunks: list[str],
    bridge_method: str = "cosine_fallback",
) -> np.ndarray:
    """Compute full n x n similarity matrix with local metrics + bridges.

    Intra-face blocks use each face's recommended metric.
    Inter-face blocks use the bridge method.

    Returns:
        Full n x n similarity matrix.
    """
    n = face_decomposition.n_chunks

    # Build Chunk objects for backend calls
    chunk_objs = [
        Chunk(text=t, index=i, start_char=0, end_char=len(t),
              word_count=len(t.split()))
        for i, t in enumerate(all_chunks)
    ]

    # Start with zeros
    similarity = np.zeros((n, n))

    # Fill intra-face blocks using each face's recommended metric
    face_blocks = _build_intra_face_blocks(face_decomposition, chunk_objs)
    for face_id, (indices, sub_sim) in face_blocks.items():
        for i_local, i_global in enumerate(indices):
            for j_local, j_global in enumerate(indices):
                similarity[i_global, j_global] = sub_sim[i_local, j_local]

    # Fill inter-face blocks using bridge method
    if face_decomposition.n_faces > 1:
        _fill_bridge_edges(
            similarity, face_decomposition, chunk_objs, bridge_method
        )

    # Sanitize
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)

    return similarity


def _build_intra_face_blocks(
    face_decomposition: FaceDecomposition,
    chunk_objs: list[Chunk],
) -> dict[int, tuple[list[int], np.ndarray]]:
    """Compute similarity within each face using its recommended metric."""
    blocks = {}

    for face in face_decomposition.faces:
        indices = face.chunk_indices
        face_chunks = [chunk_objs[i] for i in indices]
        metric = face.recommended_metric

        if len(face_chunks) < 2:
            sub_sim = np.ones((len(face_chunks), len(face_chunks)))
        else:
            try:
                metric_fn = _get_metric_fn(metric)
                sub_sim = metric_fn(face_chunks)
            except Exception:
                sub_sim = _tfidf_cosine_similarity(face_chunks)

        blocks[face.face_id] = (indices, sub_sim)

    return blocks


def _fill_bridge_edges(
    similarity: np.ndarray,
    face_decomposition: FaceDecomposition,
    chunk_objs: list[Chunk],
    bridge_method: str,
) -> None:
    """Fill inter-face similarity. Modifies similarity matrix in place."""
    faces = face_decomposition.faces

    for i_face in range(len(faces)):
        for j_face in range(i_face + 1, len(faces)):
            idx_a = faces[i_face].chunk_indices
            idx_b = faces[j_face].chunk_indices

            cross_sim = _compute_cross_face_similarity(
                [chunk_objs[i] for i in idx_a],
                [chunk_objs[i] for i in idx_b],
                bridge_method,
            )

            for i_local, i_global in enumerate(idx_a):
                for j_local, j_global in enumerate(idx_b):
                    val = cross_sim[i_local, j_local]
                    similarity[i_global, j_global] = val
                    similarity[j_global, i_global] = val


def _compute_cross_face_similarity(
    chunks_a: list[Chunk],
    chunks_b: list[Chunk],
    method: str,
) -> np.ndarray:
    """Compute similarity between chunks in different faces.

    Returns:
        (len_a, len_b) similarity matrix.
    """
    combined = list(chunks_a) + list(chunks_b)
    na = len(chunks_a)

    # All bridge methods currently use TF-IDF cosine for cross-face pairs
    tfidf = _compute_tfidf_matrix(combined)
    full_sim = tfidf @ tfidf.T
    full_sim = np.clip(full_sim, 0.0, 1.0)
    return full_sim[:na, na:]
