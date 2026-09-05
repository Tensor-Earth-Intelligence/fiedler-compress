"""Polyhedral face decomposition of a text document.

Partitions chunks into semantic faces using geometric clustering.
Each face is a group of chunks with coherent geometric properties
where a locally-optimal similarity metric applies.

This module does NOT use the Fiedler vector for partitioning.
It uses geometric clustering (k-means on TF-IDF features) to
avoid dependency on spectral decomposition.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer.graph import _compute_tfidf_matrix


@dataclass
class Face:
    """A polyhedral face -- a group of geometrically coherent chunks."""
    face_id: int
    chunk_indices: list[int]
    chunks: list[str]
    geometric_properties: dict
    recommended_metric: str


@dataclass
class FaceDecomposition:
    """Complete polyhedral decomposition of a document."""
    faces: list[Face]
    n_chunks: int
    n_faces: int
    chunk_to_face: list[int]


def decompose_into_faces(
    chunks: list[str],
    max_faces: int = 5,
    min_face_size: int = 3,
    silhouette_floor: float = 0.2,
    force_k: int | None = None,
) -> FaceDecomposition:
    """Decompose text chunks into polyhedral semantic faces.

    Uses k-means clustering on TF-IDF vectors to identify groups of
    chunks with similar geometric character. The number of faces k
    is determined automatically by silhouette score (up to max_faces).
    """
    n = len(chunks)
    if n == 0:
        raise ValueError("Cannot decompose empty chunk list")

    # Convert to Chunk objects for TF-IDF computation
    chunk_objs = [
        Chunk(text=t, index=i, start_char=0, end_char=len(t), word_count=len(t.split()))
        for i, t in enumerate(chunks)
    ]

    # Compute TF-IDF matrix
    tfidf = _compute_tfidf_matrix(chunk_objs)

    # Explicit override: use exactly force_k faces (bypasses the silhouette floor).
    # Fed by the n_faces optimizer so a recommended decomposition can be applied.
    if force_k is not None:
        k = int(force_k)
        if k <= 1 or n < 2 * min_face_size:
            return _single_face_decomposition(chunks, tfidf)
        k = min(k, n // max(1, min_face_size))
        if k < 2:
            return _single_face_decomposition(chunks, tfidf)
        labels, _ = _kmeans_with_silhouette(tfidf, k)
        labels = _merge_small_faces(labels, tfidf, min_face_size)
        return _build_decomposition(chunks, labels, tfidf)

    # If too few chunks for meaningful clustering, use single face
    if n < 2 * min_face_size:
        return _single_face_decomposition(chunks, tfidf)

    # Try k=2..max_faces, pick best silhouette
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)

    max_k = min(max_faces, n // min_face_size)
    if max_k < 2:
        return _single_face_decomposition(chunks, tfidf)

    for k in range(2, max_k + 1):
        labels, score = _kmeans_with_silhouette(tfidf, k)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # If silhouette is too low, fall back to single face
    if best_score < silhouette_floor:
        return _single_face_decomposition(chunks, tfidf)

    # Merge small faces into nearest neighbor
    best_labels = _merge_small_faces(best_labels, tfidf, min_face_size)

    # Build Face objects
    return _build_decomposition(chunks, best_labels, tfidf)


def _single_face_decomposition(
    chunks: list[str], tfidf: np.ndarray
) -> FaceDecomposition:
    """Create a single-face decomposition (equivalent to standard pipeline)."""
    props = _compute_face_properties(chunks, tfidf, list(range(len(chunks))))
    metric = _select_metric(props)
    face = Face(
        face_id=0,
        chunk_indices=list(range(len(chunks))),
        chunks=list(chunks),
        geometric_properties=props,
        recommended_metric=metric,
    )
    return FaceDecomposition(
        faces=[face],
        n_chunks=len(chunks),
        n_faces=1,
        chunk_to_face=[0] * len(chunks),
    )


def _kmeans_with_silhouette(
    tfidf: np.ndarray, k: int, max_iter: int = 50, n_init: int = 3
) -> tuple[np.ndarray, float]:
    """Run k-means and compute silhouette score. Pure numpy, no sklearn."""
    n, d = tfidf.shape
    best_labels = np.zeros(n, dtype=int)
    best_inertia = float("inf")

    for _ in range(n_init):
        # K-means++ initialization
        centroids = _kmeans_pp_init(tfidf, k)

        for _ in range(max_iter):
            # Assign clusters
            dists = _pairwise_distances(tfidf, centroids)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centroids[j] = tfidf[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        inertia = sum(
            np.sum((tfidf[labels == j] - centroids[j]) ** 2)
            for j in range(k)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    # Compute silhouette score
    score = _silhouette_score(tfidf, best_labels)
    return best_labels, score


def _kmeans_pp_init(X: np.ndarray, k: int) -> np.ndarray:
    """K-means++ initialization."""
    n = X.shape[0]
    rng = np.random.RandomState(42)
    centroids = [X[rng.randint(n)]]

    for _ in range(1, k):
        dists = np.min(_pairwise_distances(X, np.array(centroids)), axis=1)
        total = dists.sum()
        if total <= 0:
            # All points are identical distance; pick randomly
            idx = rng.randint(n)
        else:
            probs = dists / total
            # Ensure exact sum to 1.0 for numpy's rng.choice
            probs = probs / probs.sum()
            idx = rng.choice(n, p=probs)
        centroids.append(X[idx])

    return np.array(centroids)


def _pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Euclidean distances between rows of X and rows of Y."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    D = XX + YY.T - 2 * X @ Y.T
    D = np.maximum(D, 0.0)
    return np.sqrt(D)


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score."""
    n = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Pairwise distance matrix
    dists = _pairwise_distances(X, X)
    silhouettes = np.zeros(n)

    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = own_mask.sum()

        # a(i): mean distance to same-cluster points
        if own_count > 1:
            a_i = dists[i, own_mask].sum() / (own_count - 1)
        else:
            a_i = 0.0

        # b(i): min mean distance to any other cluster
        b_i = float("inf")
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = labels == lbl
            other_count = other_mask.sum()
            if other_count > 0:
                mean_dist = dists[i, other_mask].sum() / other_count
                b_i = min(b_i, mean_dist)

        if b_i == float("inf"):
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i, 1e-30)

    return float(np.mean(silhouettes))


def _merge_small_faces(
    labels: np.ndarray, tfidf: np.ndarray, min_face_size: int
) -> np.ndarray:
    """Merge faces smaller than min_face_size into nearest neighbor face."""
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)

    # Compute centroids per face
    centroids = {}
    for lbl in unique:
        mask = labels == lbl
        centroids[lbl] = tfidf[mask].mean(axis=0)

    for lbl, cnt in zip(unique, counts):
        if cnt < min_face_size:
            # Find nearest face (by centroid distance)
            best_target = -1
            best_dist = float("inf")
            for other_lbl in unique:
                if other_lbl == lbl:
                    continue
                other_mask = labels == other_lbl
                if other_mask.sum() >= min_face_size or other_mask.sum() > cnt:
                    d = np.linalg.norm(centroids[lbl] - centroids[other_lbl])
                    if d < best_dist:
                        best_dist = d
                        best_target = other_lbl
            if best_target >= 0:
                labels[labels == lbl] = best_target

    # Re-index labels to be contiguous 0..n_faces-1
    unique_new = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_new)}
    return np.array([remap[l] for l in labels])


def _build_decomposition(
    chunks: list[str], labels: np.ndarray, tfidf: np.ndarray
) -> FaceDecomposition:
    """Build FaceDecomposition from chunk labels."""
    unique_labels = np.unique(labels)
    faces = []

    for face_id, lbl in enumerate(unique_labels):
        mask = labels == lbl
        indices = list(np.where(mask)[0])
        face_chunks = [chunks[i] for i in indices]
        props = _compute_face_properties(face_chunks, tfidf, indices)
        metric = _select_metric(props)
        faces.append(Face(
            face_id=face_id,
            chunk_indices=indices,
            chunks=face_chunks,
            geometric_properties=props,
            recommended_metric=metric,
        ))

    chunk_to_face = [0] * len(chunks)
    for face in faces:
        for idx in face.chunk_indices:
            chunk_to_face[idx] = face.face_id

    return FaceDecomposition(
        faces=faces,
        n_chunks=len(chunks),
        n_faces=len(faces),
        chunk_to_face=chunk_to_face,
    )


def _compute_face_properties(
    chunks: list[str],
    tfidf_matrix: np.ndarray,
    indices: list[int],
) -> dict:
    """Compute measurable geometric properties of a face.

    Properties used for metric selection:
    - vocabulary_sparsity: fraction of zero entries in the face's TF-IDF submatrix
    - compositional_balance: coefficient of variation of row sums
    - hierarchical_depth: fraction of chunks that look like headings
    - vocabulary_overlap: average Jaccard index of term sets between chunks
    - information_density: average entropy of normalized TF-IDF distributions
    - face_size: number of chunks
    """
    sub = tfidf_matrix[indices]
    n_rows, n_cols = sub.shape

    # vocabulary_sparsity
    total_entries = n_rows * n_cols
    zero_entries = np.sum(sub == 0)
    vocabulary_sparsity = float(zero_entries / max(total_entries, 1))

    # compositional_balance (CV of row sums)
    row_sums = sub.sum(axis=1)
    if row_sums.mean() > 0:
        compositional_balance = float(row_sums.std() / row_sums.mean())
    else:
        compositional_balance = 0.0

    # hierarchical_depth: fraction of chunks that look like headings
    heading_count = 0
    for chunk_text in chunks:
        stripped = chunk_text.strip()
        words = stripped.split()
        # Heading heuristic: short, no periods, mostly title-case or all-caps
        if (
            len(words) <= 8
            and "." not in stripped
            and (stripped.istitle() or stripped.isupper())
        ):
            heading_count += 1
    hierarchical_depth = heading_count / max(len(chunks), 1)

    # vocabulary_overlap: average Jaccard index of term sets
    _TOKEN_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
    term_sets = [set(_TOKEN_RE.findall(c.lower())) for c in chunks]
    jaccard_sum = 0.0
    n_pairs = 0
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            union = term_sets[i] | term_sets[j]
            if union:
                jaccard_sum += len(term_sets[i] & term_sets[j]) / len(union)
            n_pairs += 1
    vocabulary_overlap = jaccard_sum / max(n_pairs, 1)

    # information_density: average Shannon entropy of normalized TF-IDF rows
    entropies = []
    for row in sub:
        row_sum = row.sum()
        if row_sum > 0:
            p = row / row_sum
            p = p[p > 0]
            entropies.append(float(-np.sum(p * np.log2(p))))
        else:
            entropies.append(0.0)
    information_density = float(np.mean(entropies)) if entropies else 0.0

    return {
        "vocabulary_sparsity": vocabulary_sparsity,
        "compositional_balance": compositional_balance,
        "hierarchical_depth": hierarchical_depth,
        "vocabulary_overlap": vocabulary_overlap,
        "information_density": information_density,
        "face_size": len(chunks),
    }


def _select_metric(properties: dict) -> str:
    """Select the locally-optimal similarity metric for a face.

    Decision rules (order of priority):
    1. vocabulary_sparsity > 0.95 and face_size >= 3 -> 'fisher-rao'
    2. compositional_balance > 1.5 -> 'aitchison'
    3. hierarchical_depth > 0.3 -> 'hyperbolic'
    4. vocabulary_overlap < 0.1 and face_size >= 5 -> 'wasserstein'
    5. Default -> 'tfidf'
    """
    if properties["vocabulary_sparsity"] > 0.95 and properties["face_size"] >= 3:
        return "fisher-rao"
    if properties["compositional_balance"] > 1.5:
        return "aitchison"
    if properties["hierarchical_depth"] > 0.3:
        return "hyperbolic"
    if properties["vocabulary_overlap"] < 0.1 and properties["face_size"] >= 5:
        return "wasserstein"
    return "tfidf"
