"""
Post-compression ligature generation.

Ligatures are semantic bridge annotations between adjacent compressed zones
that preserve inter-chunk coherence during decompression.  They are derived
purely from abstract spectral properties -- cluster membership indices,
spectral gap values, semantic coherence scores, and Fiedler eigenvalues --
never from raw user text.

The strict ``ChunkLigature`` schema allowlists only permitted control fields
so that ligature payloads can be trusted downstream without additional
sanitisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

_ALLOWED_BRIDGE_TYPES = frozenset({"intra-cluster", "cross-cluster"})

_ALLOWED_FIELDS = frozenset({
    "source_index",
    "target_index",
    "cluster_membership",
    "spectral_gap",
    "coherence_score",
    "fiedler_eigenvalue",
    "bridge_type",
})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkLigature:
    """
    Strict schema for a single ligature between adjacent compressed zones.

    Every field is an abstract spectral property -- no raw text is stored.
    Validation runs automatically on construction via ``__post_init__``.

    Parameters
    ----------
    source_index : int
        Position of the source chunk in the *compressed* output (0-based).
    target_index : int
        Position of the target chunk in the *compressed* output (0-based).
    cluster_membership : tuple[int, int]
        Fiedler-sign partition label for (source, target).  ``0`` means the
        chunk's Fiedler component is non-negative, ``1`` means negative.
    spectral_gap : float
        Absolute difference in Fiedler vector values between source and
        target.  Larger gaps indicate a stronger spectral boundary.
    coherence_score : float
        Edge weight between source and target normalised to [0, 1].
    fiedler_eigenvalue : float
        Algebraic connectivity (lambda-2) of the graph.
    bridge_type : str
        ``"intra-cluster"`` if both chunks share a Fiedler partition,
        ``"cross-cluster"`` otherwise.
    """

    source_index: int
    target_index: int
    cluster_membership: tuple[int, int]
    spectral_gap: float
    coherence_score: float
    fiedler_eigenvalue: float
    bridge_type: str

    def __post_init__(self) -> None:
        if self.bridge_type not in _ALLOWED_BRIDGE_TYPES:
            raise ValueError(
                f"bridge_type must be one of {sorted(_ALLOWED_BRIDGE_TYPES)}, "
                f"got {self.bridge_type!r}"
            )
        if not (0.0 <= self.coherence_score <= 1.0):
            raise ValueError(
                f"coherence_score must be in [0, 1], got {self.coherence_score}"
            )
        if self.spectral_gap < 0.0:
            raise ValueError(
                f"spectral_gap must be >= 0, got {self.spectral_gap}"
            )
        if self.fiedler_eigenvalue < 0.0:
            raise ValueError(
                f"fiedler_eigenvalue must be >= 0, got {self.fiedler_eigenvalue}"
            )
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError("source_index and target_index must be >= 0")


def validate_ligature(ligature: ChunkLigature) -> bool:
    """
    Validate that *ligature* conforms to the allowlisted schema.

    Raises ``ValueError`` if any field name is not in the allowlist or if
    any value violates its constraint.  Returns ``True`` on success.
    """
    for f in fields(ligature):
        if f.name not in _ALLOWED_FIELDS:
            raise ValueError(f"Disallowed field in ligature schema: {f.name}")

    # Re-run the dataclass post-init checks for completeness
    ChunkLigature(**asdict(ligature))
    return True


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_ligatures(
    kept_indices: Sequence[int],
    fiedler_vector: np.ndarray,
    lambda2: float,
    adjacency: np.ndarray,
) -> list[ChunkLigature]:
    """
    Generate ligature annotations for adjacent chunks in compressed output.

    Parameters
    ----------
    kept_indices : Sequence[int]
        Indices (into the *original* chunk list) of chunks that survived
        compression, in document order.
    fiedler_vector : np.ndarray
        The full Fiedler vector over all original chunks.
    lambda2 : float
        Algebraic connectivity of the similarity graph.
    adjacency : np.ndarray
        The (possibly ligature-enhanced) adjacency matrix.

    Returns
    -------
    list[ChunkLigature]
        One ligature per adjacent pair in the compressed output.
    """
    if len(kept_indices) < 2:
        return []

    max_weight = float(adjacency.max()) if adjacency.size > 0 and adjacency.max() > 0 else 1.0
    safe_lambda2 = max(0.0, float(lambda2))

    ligatures: list[ChunkLigature] = []

    for pos in range(len(kept_indices) - 1):
        src_orig = kept_indices[pos]
        tgt_orig = kept_indices[pos + 1]

        src_cluster = 0 if fiedler_vector[src_orig] >= 0 else 1
        tgt_cluster = 0 if fiedler_vector[tgt_orig] >= 0 else 1

        gap = abs(float(fiedler_vector[src_orig]) - float(fiedler_vector[tgt_orig]))
        raw_coherence = float(adjacency[src_orig, tgt_orig]) / max_weight
        coherence = max(0.0, min(1.0, raw_coherence))

        bridge = "intra-cluster" if src_cluster == tgt_cluster else "cross-cluster"

        lig = ChunkLigature(
            source_index=pos,
            target_index=pos + 1,
            cluster_membership=(src_cluster, tgt_cluster),
            spectral_gap=round(gap, 6),
            coherence_score=round(coherence, 6),
            fiedler_eigenvalue=round(safe_lambda2, 6),
            bridge_type=bridge,
        )
        validate_ligature(lig)
        ligatures.append(lig)

    return ligatures
