"""Conformal embeddings via UMAP for thematic analysis.

UMAP (Uniform Manifold Approximation and Projection) provides
approximately conformal (angle-preserving) embeddings from high-
dimensional semantic space to low-dimensional thematic space.

Key applications:
- Thematic compression (remove chunks covering already-represented themes)
- Thematic memory search (retrieve by topic, not keywords)
- Thematic anomaly detection (content in empty thematic regions)
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def compute_conformal_embedding(
    chunks: list[str],
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Compute conformal embedding of text chunks via UMAP.

    Args:
        chunks: List of text chunks.
        n_components: Embedding dimensionality (3-5 for thematic analysis).
        n_neighbors: Controls local vs global structure.
        min_dist: Controls cluster tightness.
        metric: Distance metric in original space.
        random_state: For reproducibility.

    Returns:
        (n_chunks, n_components) conformal embedding array.
    """
    import umap

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(chunks)

    n = len(chunks)
    # Small-N guard: UMAP's spectral init needs more samples than components,
    # else scipy eigsh raises "k >= N".  Clamp dims; use random init when tiny.
    n_comp = max(1, min(n_components, n - 2)) if n > 3 else 1
    n_nb = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_nb,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        init="random" if n <= n_comp + 2 else "spectral",
    )
    return reducer.fit_transform(tfidf)


def optimize_umap_params(
    chunks: list[str],
    task: str = "compression",
    random_state: int = 42,
) -> dict:
    """Grid search for optimal UMAP hyperparameters.

    Tests a compact parameter grid and selects parameters that maximize
    the silhouette score of the resulting embedding.

    Args:
        chunks: List of text chunks (need >= 10 for meaningful optimization).
        task: 'compression', 'retrieval', or 'anomaly'.
        random_state: For reproducibility.

    Returns:
        dict with best params, silhouette_score, and n_configs_tested.
    """
    import umap

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(chunks)

    grids = {
        "compression": {
            "n_components": [3, 5],
            "n_neighbors": [5, 15, 30],
            "min_dist": [0.0, 0.1, 0.3],
            "metric": ["cosine"],
        },
        "retrieval": {
            "n_components": [5, 10],
            "n_neighbors": [15, 30, 50],
            "min_dist": [0.1, 0.3, 0.5],
            "metric": ["cosine"],
        },
        "anomaly": {
            "n_components": [3, 5],
            "n_neighbors": [5, 10, 20],
            "min_dist": [0.0, 0.05, 0.1],
            "metric": ["cosine"],
        },
    }

    if task not in grids:
        raise ValueError(
            f"Unknown task '{task}'. Use 'compression', 'retrieval', or 'anomaly'."
        )

    param_grid = grids[task]
    best_score = -1.0
    best_params = None
    n_tested = 0

    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        nn_clamped = min(params["n_neighbors"], len(chunks) - 1)
        if nn_clamped < 2:
            continue

        try:
            reducer = umap.UMAP(
                n_components=params["n_components"],
                n_neighbors=nn_clamped,
                min_dist=params["min_dist"],
                metric=params["metric"],
                random_state=random_state,
            )
            embedding = reducer.fit_transform(tfidf)

            n_clusters = min(5, max(2, len(chunks) // 3))
            if len(chunks) > n_clusters + 1:
                km = KMeans(
                    n_clusters=n_clusters, random_state=random_state, n_init=3
                )
                labels = km.fit_predict(embedding)
                if len(set(labels)) > 1:
                    score = silhouette_score(embedding, labels)
                else:
                    score = -1.0
            else:
                score = -1.0

            n_tested += 1
            if score > best_score:
                best_score = score
                best_params = {**params, "n_neighbors": nn_clamped}
        except Exception:
            n_tested += 1

    if best_params is None:
        best_params = {
            "n_components": 5,
            "n_neighbors": min(15, len(chunks) - 1),
            "min_dist": 0.1,
            "metric": "cosine",
        }
        best_score = 0.0

    return {
        **best_params,
        "silhouette_score": best_score,
        "n_configs_tested": n_tested,
    }


def conformal_compress(
    text: str,
    target: str = "2x",
    optimize_params_flag: bool = False,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> dict:
    """Compress text using conformal embedding thematic analysis.

    Embeds chunks into conformal space, identifies thematic clusters,
    and retains one representative per cluster (closest to centroid)
    while removing thematically redundant chunks.

    Args:
        text: Input text.
        target: Compression target.
        optimize_params_flag: If True, run grid search for optimal UMAP params.
        n_components, n_neighbors, min_dist: UMAP params (ignored if
            optimize_params_flag is True).

    Returns:
        dict with compressed text and thematic analysis metadata.
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
            "n_clusters": 1,
            "umap_params": {},
            "time_ms": elapsed,
            "method": "conformal-umap",
        }

    # Optimize params if requested
    if optimize_params_flag and n >= 10:
        params = optimize_umap_params(chunk_texts, task="compression")
        n_components = params["n_components"]
        n_neighbors = params["n_neighbors"]
        min_dist = params["min_dist"]

    # Compute conformal embedding
    embedding = compute_conformal_embedding(
        chunk_texts,
        n_components=n_components,
        n_neighbors=min(n_neighbors, n - 1),
        min_dist=min_dist,
    )

    # Cluster in conformal space
    target_ratio = _parse_target(target)
    n_to_keep = max(1, int(n * (1 - target_ratio)))
    n_clusters = min(n_to_keep, max(2, n // 3))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    labels = km.fit_predict(embedding)

    # From each cluster, keep the chunk closest to centroid
    keep_indices: set[int] = set()
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_idx = np.where(mask)[0]
        if len(cluster_idx) == 0:
            continue
        centroid = km.cluster_centers_[cid]
        dists = np.linalg.norm(embedding[cluster_idx] - centroid, axis=1)
        keep_indices.add(int(cluster_idx[np.argmin(dists)]))

    # Fill remaining budget from largest clusters
    while len(keep_indices) < n_to_keep and len(keep_indices) < n:
        for cid in range(n_clusters):
            if len(keep_indices) >= n_to_keep:
                break
            cluster_idx = np.where(labels == cid)[0]
            for idx in cluster_idx:
                if idx not in keep_indices:
                    keep_indices.add(int(idx))
                    break

    # Reassemble in original order
    surviving = sorted(keep_indices)
    compressed = merge_kept_spans(text, chunks, surviving)

    elapsed = (time.perf_counter() - t0) * 1000
    orig_tokens = _estimate_tokens(text)
    comp_tokens = _estimate_tokens(compressed)

    return {
        "compressed": compressed,
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "compression_ratio": 1.0 - (comp_tokens / max(orig_tokens, 1)),
        "n_clusters": n_clusters,
        "umap_params": {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "optimized": optimize_params_flag,
        },
        "time_ms": elapsed,
        "method": "conformal-umap",
    }


def conformal_thematic_search(
    query: str,
    memories: list[str],
    embedding: np.ndarray | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Search memories by thematic proximity in conformal space.

    Args:
        query: Search query text.
        memories: List of memory texts.
        embedding: Pre-computed conformal embedding of memories.
            If None, computes a new one (slower).
        top_k: Number of results to return.

    Returns:
        List of dicts with index, text, and distance fields.
    """
    all_texts = memories + [query]

    if embedding is None:
        full_embedding = compute_conformal_embedding(
            all_texts,
            n_components=5,
            n_neighbors=min(15, len(all_texts) - 1),
            min_dist=0.1,
        )
    else:
        # Re-fit including query (for production, use transform())
        full_embedding = compute_conformal_embedding(
            all_texts,
            n_components=embedding.shape[1],
            n_neighbors=min(15, len(all_texts) - 1),
            min_dist=0.1,
        )

    query_point = full_embedding[-1]
    memory_points = full_embedding[:-1]

    distances = np.linalg.norm(memory_points - query_point, axis=1)
    top_indices = np.argsort(distances)[:top_k]

    return [
        {
            "index": int(idx),
            "text": memories[idx],
            "distance": float(distances[idx]),
        }
        for idx in top_indices
    ]


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
