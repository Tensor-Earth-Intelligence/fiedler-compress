"""Unified Dymaxion compression entry point."""

from __future__ import annotations

from typing import Literal


def dymaxion_compress(
    text: str,
    variant: Literal["fiedler", "pure"] = "pure",
    target: str = "2x",
    max_faces: int = 5,
    bridge_method: str = "cosine_fallback",
    silhouette_floor: float = 0.2,
    force_k: int | None = None,
) -> dict:
    """Compress text using the Dymaxion polyhedral method.

    Args:
        text: Input text.
        variant: 'fiedler' (Dymaxion graph + standard decomposition) or
                 'pure' (Dymaxion faces + information-theoretic scoring,
                 no graph-matrix decomposition).
        target: Compression target ('2x', '4x', or float ratio).
        max_faces: Maximum number of polyhedral faces.
        bridge_method: Cross-face similarity computation method.

    Returns:
        Compression result dict (format depends on variant).
    """
    if variant == "fiedler":
        from fiedler_optimizer.dymaxion.fiedler_variant import (
            dymaxion_fiedler_compress,
        )
        return dymaxion_fiedler_compress(text, target, max_faces, bridge_method, silhouette_floor, force_k)
    elif variant == "pure":
        from fiedler_optimizer.dymaxion.pure_variant import (
            dymaxion_pure_compress,
        )
        return dymaxion_pure_compress(text, target, max_faces, bridge_method, silhouette_floor, force_k)
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Use 'fiedler' or 'pure'."
        )
