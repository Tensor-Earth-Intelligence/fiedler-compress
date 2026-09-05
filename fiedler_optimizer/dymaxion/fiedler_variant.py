"""Dymaxion-Fiedler variant: polyhedral similarity graph + Fiedler decomposition.

This variant builds a Dymaxion similarity graph (locally-adaptive metrics
per face, bridge edges between faces) and then applies the standard Fiedler
spectral decomposition to the assembled graph.

This is an ENHANCEMENT to the existing Fiedler pipeline -- it still uses
the Fiedler vector, graph Laplacian, and spectral position for pruning.
"""

from __future__ import annotations

import time

import numpy as np

from fiedler_optimizer.chunker import Chunk, ChunkingStrategy, chunk_text
from fiedler_optimizer.graph import compute_fiedler_vector, compute_chunk_scores
from fiedler_optimizer.dymaxion.faces import decompose_into_faces
from fiedler_optimizer.dymaxion.bridges import compute_bridge_edges


def _parse_target(target: str) -> float:
    """Convert target string ('2x', '4x', or float) to removal fraction."""
    if isinstance(target, (int, float)):
        return float(target)
    t = str(target).strip().lower()
    if t.endswith("x"):
        ratio = float(t[:-1])
        return 1.0 - (1.0 / ratio)  # 2x -> 0.5, 4x -> 0.75
    return float(t)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def dymaxion_fiedler_compress(
    text: str,
    target: str = "2x",
    max_faces: int = 5,
    bridge_method: str = "cosine_fallback",
    silhouette_floor: float = 0.2,
    force_k: int | None = None,
) -> dict:
    """Compress text using Dymaxion-Fiedler method.

    Steps:
    1. Chunk the text (reuse existing chunker)
    2. Decompose into polyhedral faces
    3. Build Dymaxion similarity graph (local metrics + bridges)
    4. Compute Fiedler vector on the assembled graph
    5. Prune peripheral chunks based on Fiedler value
    6. Reassemble and return
    """
    t0 = time.perf_counter()

    if not text or not text.strip():
        raise ValueError("Input text is empty or whitespace-only")

    target_ratio = _parse_target(target)

    # Step 1: Chunk
    chunks = chunk_text(text, strategy=ChunkingStrategy.ADAPTIVE)
    chunk_texts = [c.text for c in chunks]
    n = len(chunks)

    if n < 4:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "compressed": text,
            "original_tokens": _estimate_tokens(text),
            "compressed_tokens": _estimate_tokens(text),
            "compression_ratio": 0.0,
            "lambda2": 0.0,
            "chunks_removed": 0,
            "chunks_total": n,
            "n_faces": 1,
            "face_metrics": ["tfidf"],
            "method": "dymaxion-fiedler",
            "time_ms": elapsed,
            "backend": "dymaxion-fiedler",
        }

    # Step 2: Decompose into faces
    decomp = decompose_into_faces(chunk_texts, max_faces=max_faces, silhouette_floor=silhouette_floor, force_k=force_k)

    # Step 3: Build Dymaxion similarity graph
    similarity = compute_bridge_edges(decomp, chunk_texts, bridge_method)

    # Zero diagonal and apply threshold (matching graph.py behavior)
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < 0.05] = 0.0

    # Ensure connectivity with positional proximity edges
    for i in range(n - 1):
        if similarity[i, i + 1] < 0.05:
            similarity[i, i + 1] = 0.1
            similarity[i + 1, i] = 0.1

    # Step 4: Fiedler vector on assembled graph
    fiedler, lambda2 = compute_fiedler_vector(similarity)

    # Step 5: Score and prune
    scores = compute_chunk_scores(chunks, fiedler, similarity)

    target_removals = max(1, int(n * target_ratio))
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])

    remove_indices: set[int] = set()
    for idx, score in indexed_scores:
        if len(remove_indices) >= target_removals:
            break
        remove_indices.add(idx)

    # Step 6: Reassemble
    # Session R&D: reassemble from original char spans, merging overlaps so the
    # ADAPTIVE chunker's window overlap is not duplicated (matches dymaxion.pure
    # so compression numbers are comparable).
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
        "lambda2": float(lambda2),
        "chunks_removed": len(remove_indices),
        "chunks_total": n,
        "n_faces": decomp.n_faces,
        "face_metrics": [f.recommended_metric for f in decomp.faces],
        "method": "dymaxion-fiedler",
        "time_ms": elapsed,
        "backend": "dymaxion-fiedler",
    }
