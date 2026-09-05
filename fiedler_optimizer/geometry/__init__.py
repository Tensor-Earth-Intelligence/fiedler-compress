"""Geometric analysis methods for Fiedler spectral compression.

Three independent approaches:
- Voronoi density-based pruning (third independent compression pathway)
- Minkowski spacetime metric (positional-semantic analysis)
- Conformal UMAP embeddings (thematic analysis and memory search)

These methods need scientific-Python extras that the core compressor does not.
Install them with: ``pip install fiedler-compress[geometry]``
"""

try:
    from fiedler_optimizer.geometry.voronoi import (
        voronoi_compress,
        estimate_voronoi_volumes,
        voronoi_anomaly_score,
    )
    from fiedler_optimizer.geometry.minkowski import (
        minkowski_distance_matrix,
        minkowski_similarity_matrix,
        minkowski_compress,
        calibrate_alpha,
    )
    from fiedler_optimizer.geometry.conformal import (
        conformal_compress,
        compute_conformal_embedding,
        optimize_umap_params,
        conformal_thematic_search,
    )
except ImportError as exc:  # pragma: no cover - depends on optional extras
    raise ImportError(
        "fiedler_optimizer.geometry requires scikit-learn (and umap-learn for "
        "conformal embeddings), which are not core dependencies. Install with: "
        "pip install fiedler-compress[geometry]"
    ) from exc

__all__ = [
    "voronoi_compress",
    "estimate_voronoi_volumes",
    "voronoi_anomaly_score",
    "minkowski_distance_matrix",
    "minkowski_similarity_matrix",
    "minkowski_compress",
    "calibrate_alpha",
    "conformal_compress",
    "compute_conformal_embedding",
    "optimize_umap_params",
    "conformal_thematic_search",
]
