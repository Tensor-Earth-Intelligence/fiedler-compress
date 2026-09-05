"""Dymaxion-Pure variant: polyhedral compression WITHOUT any form of
graph-based matrix decomposition that extracts connectivity vectors.

This variant achieves compression through polyhedral face decomposition
and information-theoretic redundancy scoring. It does NOT compute:
- The graph matrix decomposition used by the standard pipeline
- The connectivity vector or any matrix vectors
- Algebraic connectivity
- Zone-dependent protection factors

It is a mathematically independent compression method.

DESIGN-AROUND VERIFICATION CHECKLIST:
[x] No import of decomposition, graph matrix, or matrix vector extraction
[x] No computation of algebraic connectivity or any matrix value
[x] No zone classification with protection factors
[x] No connectivity-vector position used for pruning decisions
[x] All redundancy scoring is information-theoretic, not graph-based
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np

from fiedler_optimizer.chunker import Chunk, ChunkingStrategy, chunk_text
from fiedler_optimizer.dymaxion.faces import decompose_into_faces
from fiedler_optimizer.dymaxion.bridges import compute_bridge_edges


def _parse_target(target: str) -> float:
    """Convert target string ('2x', '4x', or float) to removal fraction."""
    if isinstance(target, (int, float)):
        return float(target)
    t = str(target).strip().lower()
    if t.endswith("x"):
        ratio = float(t[:-1])
        return 1.0 - (1.0 / ratio)
    return float(t)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def dymaxion_pure_compress(
    text: str,
    target: str = "2x",
    max_faces: int = 5,
    bridge_method: str = "cosine_fallback",
    silhouette_floor: float = 0.2,
    force_k: int | None = None,
) -> dict:
    """Compress text using Dymaxion-Pure method (no graph-matrix decomposition).

    Steps:
    1. Chunk the text
    2. Decompose into polyhedral faces (k-means, NOT any form of
       graph-based clustering)
    3. For each face, compute local similarity using face-optimal metric
    4. Within each face, score chunks for redundancy using
       INFORMATION-THEORETIC methods:
       a. Pairwise redundancy (max similarity to any neighbor)
       b. Information contribution (marginal entropy change)
       c. Geometric peripherality (centroid distance, NOT graph-vector distance)
    5. Determine per-face compression level based on face entropy
    6. Prune chunks with highest redundancy scores
    7. Reassemble surviving chunks
    """
    t0 = time.perf_counter()

    if not text or not text.strip():
        raise ValueError("Input text is empty or whitespace-only")

    target_ratio = _parse_target(target)

    # Step 1: Chunk
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
            "n_faces": 1,
            "face_metrics": ["tfidf"],
            "face_compression_levels": [0.0],
            "redundancy_scores": [0.0] * n,
            "time_ms": elapsed,
            "method": "dymaxion-pure",
            "face_decomposition": {},
            "face_entropies": [0.0],
            "centroid_distances": [0.0] * n,
        }

    # Step 2: Decompose into faces
    decomp = decompose_into_faces(chunk_texts, max_faces=max_faces, silhouette_floor=silhouette_floor, force_k=force_k)

    # Step 3: Build similarity matrix (for redundancy scoring, not for
    # graph-matrix decomposition)
    similarity = compute_bridge_edges(decomp, chunk_texts, bridge_method)
    np.fill_diagonal(similarity, 0.0)

    # Compute TF-IDF matrix for entropy calculations
    chunk_objs = [
        Chunk(text=t, index=i, start_char=0, end_char=len(t),
              word_count=len(t.split()))
        for i, t in enumerate(chunk_texts)
    ]
    from fiedler_optimizer.graph import _compute_tfidf_matrix
    tfidf_matrix = _compute_tfidf_matrix(chunk_objs)

    # Step 4: Score redundancy per chunk within each face
    all_redundancy_scores = np.zeros(n)
    all_centroid_distances = np.zeros(n)
    face_entropies = []

    for face in decomp.faces:
        indices = face.chunk_indices
        if len(indices) < 2:
            face_entropies.append(0.0)
            for idx in indices:
                all_redundancy_scores[idx] = 0.0
                all_centroid_distances[idx] = 0.0
            continue

        # Extract face sub-matrices
        face_sim = similarity[np.ix_(indices, indices)]
        face_tfidf = tfidf_matrix[indices]

        # Compute face entropy
        face_entropy = _compute_face_entropy(face_tfidf)
        face_entropies.append(face_entropy)

        # Compute centroid for geometric peripherality
        centroid = face_tfidf.mean(axis=0)

        # Score each chunk in this face
        for local_idx, global_idx in enumerate(indices):
            scores = _score_chunk_redundancy(
                local_idx, face_sim, face_tfidf, centroid
            )
            all_redundancy_scores[global_idx] = scores["combined_score"]
            all_centroid_distances[global_idx] = scores["geometric_peripherality"]

    # Step 5: Compute per-face compression budgets
    max_possible_entropy = math.log2(max(tfidf_matrix.shape[1], 2))
    face_compression_levels = []
    for face_idx, face in enumerate(decomp.faces):
        budget = _compute_face_compression_budget(
            face.geometric_properties,
            face_entropies[face_idx],
            max_possible_entropy,
            target_ratio,
            decomp.n_faces,
        )
        face_compression_levels.append(budget)

    # Step 6: Prune within each face based on redundancy scores and budgets
    remove_indices: set[int] = set()

    for face_idx, face in enumerate(decomp.faces):
        indices = face.chunk_indices
        if len(indices) < 2:
            continue

        budget = face_compression_levels[face_idx]
        n_to_remove = max(0, int(len(indices) * budget))

        if n_to_remove == 0:
            continue

        # Sort by redundancy score (highest = most prunable)
        scored = [(idx, all_redundancy_scores[idx]) for idx in indices]
        scored.sort(key=lambda x: x[1], reverse=True)

        for idx, _ in scored[:n_to_remove]:
            remove_indices.add(idx)

    # Step 7: Reassemble from original character spans, merging overlapping
    # windows so the ADAPTIVE chunker's overlap is not duplicated (duplication
    # inflates output and yields negative compression on short inputs).
    kept_idx = [i for i in range(n) if i not in remove_indices]
    spans = sorted((chunks[i].start_char, chunks[i].end_char) for i in kept_idx)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    compressed = "\n\n".join(text[s:e] for s, e in merged)

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "compressed": compressed,
        "original_tokens": _estimate_tokens(text),
        "compressed_tokens": _estimate_tokens(compressed),
        "compression_ratio": 1.0 - (len(compressed) / max(len(text), 1)),
        "n_faces": decomp.n_faces,
        "face_metrics": [f.recommended_metric for f in decomp.faces],
        "face_compression_levels": face_compression_levels,
        "redundancy_scores": all_redundancy_scores.tolist(),
        "time_ms": elapsed,
        "method": "dymaxion-pure",
        # Polyhedral certificate (NOT a graph-based certificate)
        "face_decomposition": {
            "n_faces": decomp.n_faces,
            "face_sizes": [len(f.chunk_indices) for f in decomp.faces],
            "face_metrics": [f.recommended_metric for f in decomp.faces],
        },
        "face_entropies": face_entropies,
        "centroid_distances": all_centroid_distances.tolist(),
    }


def _score_chunk_redundancy(
    chunk_local_idx: int,
    face_similarity: np.ndarray,
    face_tfidf: np.ndarray,
    centroid: np.ndarray,
) -> dict:
    """Score a single chunk's redundancy using three information-theoretic signals.

    Signal 1 -- Pairwise redundancy:
        max similarity to any other chunk in the face.

    Signal 2 -- Information contribution:
        Marginal entropy change when chunk is removed.

    Signal 3 -- Geometric peripherality:
        Distance from face centroid in TF-IDF space (NOT from any
        graph-based vector position).
    """
    n_face = face_similarity.shape[0]

    # Signal 1: Pairwise redundancy
    row = face_similarity[chunk_local_idx].copy()
    row[chunk_local_idx] = 0.0  # exclude self
    pairwise_redundancy = float(np.max(row)) if n_face > 1 else 0.0

    # Signal 2: Information contribution (marginal entropy)
    face_entropy_with = _compute_face_entropy(face_tfidf)

    # Entropy without this chunk
    mask = np.ones(n_face, dtype=bool)
    mask[chunk_local_idx] = False
    if mask.sum() > 0:
        face_entropy_without = _compute_face_entropy(face_tfidf[mask])
    else:
        face_entropy_without = 0.0

    marginal_info = face_entropy_with - face_entropy_without
    # Normalize: low marginal info -> high redundancy
    # Cap at reasonable range
    info_contribution = max(0.0, marginal_info)

    # Signal 3: Geometric peripherality (centroid distance in TF-IDF space)
    chunk_vec = face_tfidf[chunk_local_idx]
    dist = float(np.linalg.norm(chunk_vec - centroid))
    geometric_peripherality = dist

    # Combined score: higher = more prunable
    # Normalize signals to [0, 1] range approximately
    # pairwise_redundancy is already [0, 1]
    # info_contribution: invert (low contribution = high prunability)
    max_info = max(face_entropy_with, 1e-10)
    info_score = 1.0 - min(info_contribution / max_info, 1.0)
    # peripherality: normalize by max possible distance
    max_dist = max(
        float(np.max(np.linalg.norm(face_tfidf - centroid, axis=1))), 1e-10
    )
    periph_score = dist / max_dist

    combined = (
        0.4 * pairwise_redundancy
        + 0.3 * info_score
        + 0.3 * periph_score
    )

    return {
        "pairwise_redundancy": pairwise_redundancy,
        "information_contribution": info_contribution,
        "geometric_peripherality": geometric_peripherality,
        "combined_score": combined,
    }


def _compute_face_entropy(tfidf_submatrix: np.ndarray) -> float:
    """Compute the total information entropy of a face.

    Aggregates TF-IDF vectors into a single distribution (sum and normalize),
    then computes Shannon entropy.
    """
    if tfidf_submatrix.size == 0:
        return 0.0

    # Sum across all chunks to get aggregate term distribution
    aggregate = tfidf_submatrix.sum(axis=0)
    if isinstance(aggregate, np.matrix):
        aggregate = np.asarray(aggregate).flatten()
    aggregate = np.asarray(aggregate).flatten()

    total = aggregate.sum()
    if total <= 0:
        return 0.0

    p = aggregate / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _compute_face_compression_budget(
    face_properties: dict,
    face_entropy: float,
    max_possible_entropy: float,
    global_target: float,
    n_faces: int,
) -> float:
    """Determine how aggressively to compress each face.

    Dense faces (high entropy) get smaller budgets (less compression).
    Sparse faces (low entropy) get larger budgets (more compression).

    This replaces zone protection factors with information-theoretic
    compression budget allocation.
    """
    if max_possible_entropy <= 0:
        return global_target

    # Normalized entropy [0, 1]
    normalized_entropy = min(face_entropy / max_possible_entropy, 1.0)

    # Compressibility: inverse of normalized entropy
    # High entropy -> low compressibility -> compress less
    # Low entropy -> high compressibility -> compress more
    compressibility = 1.0 - normalized_entropy

    # Scale around the global target
    # Factor ranges from 0.5 (dense, protect) to 1.5 (sparse, compress more)
    factor = 0.5 + compressibility  # [0.5, 1.5]

    budget = global_target * factor

    # Clamp to reasonable range
    return max(0.0, min(budget, 0.9))
