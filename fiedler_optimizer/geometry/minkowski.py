"""Minkowski spacetime metric for positional-semantic analysis.

Unifies semantic content distance and sequential position distance
using the Minkowski metric from special relativity:

    d_M^2(a,b) = semantic_distance^2  -  alpha * positional_distance^2

The negative sign means positionally proximate semantic similarity
is MORE redundant than positionally distant semantic similarity.

Key applications:
- Agentic context management (prune old, peripheral turns)
- Selective repetition (identify positionally vulnerable central content)
- Streaming anomaly detection (distinguish clustered vs distributed events)
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


def minkowski_distance_matrix(
    chunks: list[str],
    alpha: float = 0.1,
    semantic_metric: str = "cosine",
    max_position_distance: float | None = None,
) -> np.ndarray:
    """Compute pairwise Minkowski spacetime distances.

    Args:
        chunks: List of text chunks in document order.
        alpha: Coupling constant.  Controls relative weight of position
               vs semantics.
               - Low alpha (0.01-0.05): nearly pure semantic distance.
               - Medium alpha (0.05-0.2): moderate positional influence.
               - High alpha (0.2-1.0): strong recency bias.
        semantic_metric: 'cosine' or 'aitchison'.
        max_position_distance: Normalize positional distances to [0, max_pos].
            Default: None (raw position indices).
            Set to 1.0 to normalize to [0, 1].

    Returns:
        (n, n) Minkowski distance-squared matrix.  Values can be negative
        (timelike separation: chunks are more connected than semantic
        distance alone would suggest).
    """
    n = len(chunks)

    # Semantic distances
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(chunks)

    if semantic_metric == "aitchison":
        dense = tfidf.toarray()
        dense[dense == 0] = 1e-10
        row_sums = dense.sum(axis=1, keepdims=True)
        dense = dense / row_sums
        log_data = np.log(dense)
        clr = log_data - log_data.mean(axis=1, keepdims=True)
        sem_dist = cosine_distances(clr)
    else:
        sem_dist = cosine_distances(tfidf)

    # Positional distances
    positions = np.arange(n, dtype=float)
    if max_position_distance is not None and positions.max() > 0:
        positions = positions / positions.max() * max_position_distance
    pos_dist = np.abs(positions[:, None] - positions[None, :])

    # Minkowski metric: d^2_M = sem^2 - alpha * pos^2
    d_squared = sem_dist ** 2 - alpha * pos_dist ** 2
    return d_squared


def minkowski_similarity_matrix(
    chunks: list[str],
    alpha: float = 0.1,
    semantic_metric: str = "cosine",
) -> np.ndarray:
    """Compute Minkowski-based similarity matrix.

    Converts Minkowski distances to similarities via:
    sim = exp(-max(0, d_M))

    Returns:
        (n, n) similarity matrix with values in [0, 1].
    """
    d_sq = minkowski_distance_matrix(
        chunks,
        alpha=alpha,
        semantic_metric=semantic_metric,
        max_position_distance=1.0,
    )
    # Signed sqrt
    d = np.sign(d_sq) * np.sqrt(np.abs(d_sq))
    sim = np.exp(-np.maximum(0, d))
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0, 1)


def minkowski_compress(
    text: str,
    target: str = "2x",
    alpha: float = 0.1,
) -> dict:
    """Compress text using Minkowski positional-semantic analysis.

    Uses the Minkowski similarity matrix with the standard Fiedler
    pipeline (graph Laplacian + Fiedler vector).  The Minkowski metric
    modifies WHICH chunks appear peripheral.

    This is an enhancement to Claims 1-5, not a design-around.

    Args:
        text: Input text.
        target: Compression target.
        alpha: Minkowski coupling constant.

    Returns:
        Compression result dict including alpha and method.
    """
    t0 = time.perf_counter()

    from fiedler_optimizer.chunker import chunk_text, ChunkingStrategy, merge_kept_spans
    from fiedler_optimizer.graph import compute_fiedler_vector, compute_chunk_scores

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
            "alpha": alpha,
            "time_ms": elapsed,
            "method": "minkowski-fiedler",
        }

    # Build Minkowski similarity matrix
    similarity = minkowski_similarity_matrix(chunk_texts, alpha=alpha)

    # Zero diagonal, threshold, ensure connectivity
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < 0.05] = 0.0

    for i in range(n - 1):
        if similarity[i, i + 1] < 0.05:
            similarity[i, i + 1] = 0.1
            similarity[i + 1, i] = 0.1

    # Fiedler decomposition on Minkowski-modified graph
    fiedler, lambda2 = compute_fiedler_vector(similarity)

    # Score and prune
    scores = compute_chunk_scores(chunks, fiedler, similarity)

    target_ratio = _parse_target(target)
    target_removals = max(1, int(n * target_ratio))
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])

    remove_indices: set[int] = set()
    for idx, _score in indexed_scores:
        if len(remove_indices) >= target_removals:
            break
        remove_indices.add(idx)

    kept_indices = [i for i in range(n) if i not in remove_indices]
    compressed = merge_kept_spans(text, chunks, kept_indices)

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "compressed": compressed,
        "original_tokens": _estimate_tokens(text),
        "compressed_tokens": _estimate_tokens(compressed),
        "compression_ratio": 1.0 - (len(compressed) / max(len(text), 1)),
        "lambda2": float(lambda2),
        "chunks_removed": len(remove_indices),
        "chunks_total": n,
        "alpha": alpha,
        "time_ms": elapsed,
        "method": "minkowski-fiedler",
    }


def calibrate_alpha(
    chunks: list[str],
    document_type: str = "auto",
) -> float:
    """Calibrate the Minkowski coupling constant for a document.

    Heuristic calibration based on document structure:
    - 'agentic': high alpha (0.4) - strong recency bias
    - 'instructional': low alpha (0.03) - intentional repetition common
    - 'narrative': medium alpha (0.15) - moderate positional influence
    - 'technical': low alpha (0.08) - position matters less than content
    - 'auto': analyzes chunk patterns to auto-detect

    Returns:
        Calibrated alpha value.
    """
    defaults = {
        "agentic": 0.4,
        "instructional": 0.03,
        "narrative": 0.15,
        "technical": 0.08,
    }
    if document_type != "auto":
        return defaults.get(document_type, 0.1)

    if not chunks:
        return 0.1

    # Auto-detection heuristics
    text_joined = " ".join(chunks)
    words = text_joined.split()
    n_words = len(words)
    if n_words == 0:
        return 0.1

    # Turn markers suggest agentic conversation
    import re
    turn_patterns = re.findall(
        r"\b(Turn|Step|User|Assistant|Agent|Human|AI)\s*\d*\s*:", text_joined
    )
    if len(turn_patterns) >= len(chunks) * 0.3:
        return defaults["agentic"]

    # Many section headers suggest technical documentation
    heading_count = sum(
        1
        for c in chunks
        if c.strip() and len(c.strip().split()) <= 8 and c.strip().istitle()
    )
    if heading_count >= len(chunks) * 0.2:
        return defaults["technical"]

    # High vocabulary diversity suggests technical content
    unique_ratio = len(set(w.lower() for w in words)) / max(n_words, 1)
    if unique_ratio > 0.65:
        return defaults["technical"]

    return defaults["narrative"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_target(target) -> float:
    if isinstance(target, (int, float)):
        return float(target)
    t = str(target).strip().lower()
    if t.endswith("x"):
        return 1.0 - (1.0 / float(t[:-1]))
    return float(t)
