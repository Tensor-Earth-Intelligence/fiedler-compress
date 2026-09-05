"""
Spectral topology classification and caching.

Analyzes the spectral structure of a similarity graph and classifies it
into one of five topology types.  A cache stores topology–parameter pairs
keyed by a locality-sensitive hash of the topology feature vector, enabling
warm-start compression for new corpora that match a cached topology.

Topology types
--------------
- **monolithic**: single tightly-connected cluster (small spectral gap,
  high overall density).
- **bipartite**: two clearly separated partitions (large λ₂/λ₃ ratio,
  balanced Fiedler sign partition).
- **hierarchical**: nested community structure (moderate spectral gap,
  multi-level eigenvector distribution).
- **fragmented**: many weakly-connected components (very small λ₂,
  high component count or near-zero edge weights).
- **hub-spoke**: one or few high-degree nodes bridging many low-degree
  nodes (high eigenvector localization, skewed degree distribution).

All persistence is JSON — pickle is never used.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Topology types
# ---------------------------------------------------------------------------

@unique
class TopologyType(Enum):
    MONOLITHIC = "monolithic"
    BIPARTITE = "bipartite"
    HIERARCHICAL = "hierarchical"
    FRAGMENTED = "fragmented"
    HUB_SPOKE = "hub-spoke"


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyFeatures:
    """Spectral and structural features of a similarity graph."""

    n_nodes: int
    """Number of nodes (chunks) in the graph."""

    lambda_2: float
    """Algebraic connectivity (second-smallest Laplacian eigenvalue)."""

    lambda_3: float
    """Third-smallest Laplacian eigenvalue."""

    spectral_gap_ratio: float
    """λ₂ / λ₃ — large values indicate a clear bipartition."""

    mean_degree: float
    """Mean weighted degree across all nodes."""

    degree_cv: float
    """Coefficient of variation of degrees (std / mean).
    High values indicate hub-spoke structure."""

    fiedler_bipartition_balance: float
    """Balance of the Fiedler sign partition: min(n+, n−) / max(n+, n−).
    Values near 1.0 indicate balanced bipartite structure."""

    fiedler_localization: float
    """Inverse participation ratio of the Fiedler vector.
    High values indicate the eigenvector is localized on few nodes."""

    edge_density: float
    """Fraction of non-zero edges in the adjacency matrix."""

    n_weak_components: int
    """Number of connected components with edge weight > threshold."""

    def to_vector(self) -> list[float]:
        """Return a fixed-order numeric vector for hashing and comparison."""
        return [
            float(self.n_nodes),
            self.lambda_2,
            self.lambda_3,
            self.spectral_gap_ratio,
            self.mean_degree,
            self.degree_cv,
            self.fiedler_bipartition_balance,
            self.fiedler_localization,
            self.edge_density,
            float(self.n_weak_components),
        ]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _count_weak_components(adjacency: np.ndarray, threshold: float = 1e-10) -> int:
    """Count connected components via BFS on the thresholded adjacency."""
    n = adjacency.shape[0]
    visited = [False] * n
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if not visited[neighbor] and adjacency[node, neighbor] > threshold:
                    visited[neighbor] = True
                    queue.append(neighbor)
    return components


def _inverse_participation_ratio(v: np.ndarray) -> float:
    """IPR of a vector — measures eigenvector localization.

    IPR = sum(v_i^4) / (sum(v_i^2))^2.  For a uniformly spread vector
    of length n, IPR = 1/n.  For a vector localized on 1 node, IPR = 1.
    """
    v2 = v ** 2
    s2 = float(v2.sum())
    if s2 < 1e-15:
        return 0.0
    return float((v ** 4).sum()) / (s2 ** 2)


def extract_features(
    adjacency: np.ndarray,
    fiedler: np.ndarray,
    lambda_2: float,
) -> TopologyFeatures:
    """
    Extract topology features from a similarity graph.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric weighted adjacency matrix.
    fiedler : np.ndarray
        The Fiedler vector.
    lambda_2 : float
        Algebraic connectivity.
    """
    n = adjacency.shape[0]

    # Compute λ₃ via dense Laplacian eigendecomposition
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigenvalues = np.sort(np.linalg.eigh(laplacian)[0])

    lambda_3 = float(eigenvalues[2]) if n > 2 else 0.0
    spectral_gap_ratio = (lambda_2 / lambda_3) if lambda_3 > 1e-12 else 0.0

    # Degree statistics
    degrees = adjacency.sum(axis=1)
    mean_deg = float(degrees.mean())
    std_deg = float(degrees.std())
    degree_cv = (std_deg / mean_deg) if mean_deg > 1e-12 else 0.0

    # Fiedler partition balance
    n_pos = int(np.sum(fiedler >= 0))
    n_neg = n - n_pos
    balance = min(n_pos, n_neg) / max(n_pos, n_neg) if max(n_pos, n_neg) > 0 else 0.0

    # Eigenvector localization
    localization = _inverse_participation_ratio(fiedler)

    # Edge density (fraction of possible edges that are non-zero)
    n_edges = int(np.sum(adjacency > 1e-10))
    max_edges = n * (n - 1)
    edge_density = n_edges / max_edges if max_edges > 0 else 0.0

    # Connected components
    n_components = _count_weak_components(adjacency)

    return TopologyFeatures(
        n_nodes=n,
        lambda_2=round(float(lambda_2), 8),
        lambda_3=round(lambda_3, 8),
        spectral_gap_ratio=round(spectral_gap_ratio, 8),
        mean_degree=round(mean_deg, 8),
        degree_cv=round(degree_cv, 8),
        fiedler_bipartition_balance=round(balance, 8),
        fiedler_localization=round(localization, 8),
        edge_density=round(edge_density, 8),
        n_weak_components=n_components,
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyClassification:
    """Result of topology classification."""

    topology: TopologyType
    confidence: float
    """Confidence in the classification (0.0–1.0)."""
    features: TopologyFeatures
    """Feature vector used for classification."""


class TopologyClassifier:
    """
    Classifies a similarity graph's spectral topology.

    Classification rules (applied in priority order):

    1. **fragmented**: n_weak_components > 1 OR λ₂ < 0.001
    2. **hub-spoke**: degree_cv > 0.8 AND edge_density < 0.5
    3. **monolithic**: edge_density > 0.9 AND degree_cv < 0.1
       (fully/near-fully connected, uniform weights)
    4. **bipartite**: spectral_gap_ratio > 0.4 AND balance > 0.5
    5. **hierarchical**: edge_density > 0.3 AND fiedler_localization < 0.15
    6. **monolithic**: default
    """

    def classify(
        self,
        adjacency: np.ndarray,
        fiedler: np.ndarray,
        lambda_2: float,
    ) -> TopologyClassification:
        """Classify the topology of a similarity graph."""
        features = extract_features(adjacency, fiedler, lambda_2)
        return self.classify_from_features(features)

    def classify_from_features(
        self,
        features: TopologyFeatures,
    ) -> TopologyClassification:
        """Classify from pre-extracted features."""

        # Rule 1: Fragmented — disconnected or near-zero algebraic connectivity
        if features.n_weak_components > 1 or features.lambda_2 < 0.001:
            conf = min(1.0, 0.7 + 0.3 * (1.0 - features.lambda_2 / 0.001)
                        if features.lambda_2 < 0.001 else 0.9)
            return TopologyClassification(
                TopologyType.FRAGMENTED, round(conf, 4), features)

        # Rule 2: Hub-spoke — high degree variance, sparse graph
        if features.degree_cv > 0.8 and features.edge_density < 0.5:
            conf = min(1.0, 0.5 + 0.25 * min(features.degree_cv, 2.0) / 2.0
                       + 0.25)
            return TopologyClassification(
                TopologyType.HUB_SPOKE, round(conf, 4), features)

        # Rule 3: Monolithic (dense) — near-full connectivity, uniform degrees
        if features.edge_density > 0.9 and features.degree_cv < 0.1:
            conf = min(1.0, 0.6 + 0.4 * features.edge_density)
            return TopologyClassification(
                TopologyType.MONOLITHIC, round(conf, 4), features)

        # Rule 4: Bipartite — spectral separation with balanced partition
        if (features.spectral_gap_ratio > 0.15
                and features.fiedler_bipartition_balance > 0.5):
            conf = min(1.0, 0.5
                       + 0.25 * min(features.spectral_gap_ratio, 1.0)
                       + 0.25 * features.fiedler_bipartition_balance)
            return TopologyClassification(
                TopologyType.BIPARTITE, round(conf, 4), features)

        # Rule 5: Hierarchical — moderate density, delocalized eigenvector
        if features.edge_density > 0.3 and features.fiedler_localization < 0.15:
            conf = min(1.0, 0.6
                       + 0.2 * min(features.edge_density, 1.0)
                       + 0.2 * (1.0 - features.fiedler_localization / 0.15))
            return TopologyClassification(
                TopologyType.HIERARCHICAL, round(conf, 4), features)

        # Rule 6: Monolithic (default)
        conf = min(1.0, 0.6 + 0.4 * features.edge_density)
        return TopologyClassification(
            TopologyType.MONOLITHIC, round(conf, 4), features)


# ---------------------------------------------------------------------------
# Cached warm-start parameters
# ---------------------------------------------------------------------------

@dataclass
class CachedParameters:
    """Compression parameters associated with a topology."""

    topology: str
    target_ratio: float
    similarity_threshold: float
    protect_instructions: bool
    features: dict
    """Serialized TopologyFeatures."""


# ---------------------------------------------------------------------------
# Locality-sensitive hash
# ---------------------------------------------------------------------------

def _feature_hash(features: TopologyFeatures, precision: int = 4) -> str:
    """
    Locality-sensitive hash of a topology feature vector.

    Rounds each feature to ``precision`` decimal places before hashing,
    so nearby feature vectors produce the same hash.
    """
    rounded = [round(v, precision) for v in features.to_vector()]
    canonical = repr(rounded).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:32]


def _feature_distance(a: TopologyFeatures, b: TopologyFeatures) -> float:
    """Normalized L2 distance between two feature vectors."""
    va = np.array(a.to_vector(), dtype=np.float64)
    vb = np.array(b.to_vector(), dtype=np.float64)
    # Normalize by the magnitude of a to get a relative distance
    norm = np.linalg.norm(va)
    if norm < 1e-12:
        norm = 1.0
    return float(np.linalg.norm(va - vb) / norm)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_MAX_CACHE_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class TopologyCache:
    """
    Persistent cache of topology classifications and compression parameters.

    Stored as JSON in a configurable directory.  Never uses pickle.
    Cache entries are keyed by a locality-sensitive hash of the topology
    feature vector.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for cache files.  Created if it does not exist.
    invalidation_threshold : float
        Maximum feature-distance between a new corpus and a cached entry
        for the cache hit to be valid.  Default 0.15 (15% relative).
    """

    def __init__(
        self,
        cache_dir: str | Path = ".fiedler_cache",
        invalidation_threshold: float = 0.15,
    ) -> None:
        self._dir = Path(cache_dir)
        self._threshold = invalidation_threshold

        if ".." in self._dir.parts:
            raise ValueError(
                f"Path traversal not allowed in cache_dir: {cache_dir}"
            )

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        # Sanitize key to safe filename characters
        safe = "".join(c for c in key if c.isalnum())[:64]
        return self._dir / f"topo_{safe}.json"

    def get(self, features: TopologyFeatures) -> CachedParameters | None:
        """
        Look up cached parameters for a topology feature vector.

        Returns ``None`` on cache miss or if the cached entry has drifted
        beyond the invalidation threshold.
        """
        key = _feature_hash(features)
        path = self._cache_path(key)

        if not path.exists():
            return None

        try:
            file_size = path.stat().st_size
            if file_size > _MAX_CACHE_FILE_BYTES:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        # Validate top-level schema before accessing nested fields.
        # A crafted cache file must not crash or inject unexpected values.
        required_keys = {"topology", "target_ratio", "similarity_threshold",
                         "protect_instructions", "features"}
        if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
            return None

        # Validate topology is a known type
        valid_topologies = {t.value for t in TopologyType}
        if data["topology"] not in valid_topologies:
            return None

        # Validate numeric fields are within sane bounds
        try:
            tr = float(data["target_ratio"])
            st = float(data["similarity_threshold"])
        except (ValueError, TypeError):
            return None
        if not (0.0 < tr < 1.0) or not (0.0 <= st <= 1.0):
            return None
        if not isinstance(data["protect_instructions"], bool):
            return None

        # Reconstruct cached features and check distance
        try:
            cached_features = TopologyFeatures(**data["features"])
        except (KeyError, TypeError):
            return None

        distance = _feature_distance(features, cached_features)
        if distance > self._threshold:
            return None  # invalidated — too far from cached topology

        return CachedParameters(
            topology=data["topology"],
            target_ratio=tr,
            similarity_threshold=st,
            protect_instructions=data["protect_instructions"],
            features=data["features"],
        )

    def put(
        self,
        features: TopologyFeatures,
        topology: TopologyType,
        target_ratio: float,
        similarity_threshold: float = 0.05,
        protect_instructions: bool = True,
    ) -> None:
        """Store a topology classification and its compression parameters."""
        self._ensure_dir()
        key = _feature_hash(features)
        path = self._cache_path(key)

        data = {
            "topology": topology.value,
            "target_ratio": target_ratio,
            "similarity_threshold": similarity_threshold,
            "protect_instructions": protect_instructions,
            "features": asdict(features),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def invalidate(self, features: TopologyFeatures) -> bool:
        """Remove a cache entry.  Returns True if an entry was removed."""
        key = _feature_hash(features)
        path = self._cache_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Remove all cache entries.  Returns the count of files removed."""
        if not self._dir.exists():
            return 0
        count = 0
        for f in self._dir.glob("topo_*.json"):
            f.unlink()
            count += 1
        return count
