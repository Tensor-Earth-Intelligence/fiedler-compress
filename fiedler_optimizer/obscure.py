"""
Spectral obscure encoding.

Replaces natural language fragments in compressed output with spectral
zone references and control metadata only — zone IDs, Fiedler eigenvalue
indices, coherence scores, and cluster membership.  No original user text
appears in the obscured output.

This is **practical obscurity through spectral abstraction**, not
encryption.  No cryptographic primitives are used.  The original text can
be reconstructed from the obscured output plus a **zone map** that the
user retains locally and never transmits.

Obscured format (one line per kept chunk)::

    [ZONE:0|type=INSTRUCTION|fiedler=0.8231|score=1.42|cluster=0|coherence=0.65|words=15]
    [ZONE:1|type=CONTEXT|fiedler=-0.3012|score=0.78|cluster=1|coherence=0.41|words=22]
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk

# Regex to parse a ZONE reference line produced by spectral_obscure.
_ZONE_RE = re.compile(r"^\[ZONE:(\d+)\|")


def spectral_obscure(
    kept_indices: Sequence[int],
    chunks: Sequence[Chunk],
    fiedler_vector: np.ndarray,
    lambda2: float,
    adjacency: np.ndarray,
    zoned: Sequence,
    scores: Sequence[float],
) -> tuple[str, dict[str, str]]:
    """
    Replace chunk text with spectral zone references.

    Parameters
    ----------
    kept_indices : Sequence[int]
        Original-chunk indices that survived compression, in order.
    chunks : Sequence[Chunk]
        All original chunks.
    fiedler_vector : np.ndarray
        Full Fiedler vector over all original chunks.
    lambda2 : float
        Algebraic connectivity.
    adjacency : np.ndarray
        Adjacency matrix.
    zoned : Sequence[ZonedChunk]
        Zone classifications for all original chunks.
    scores : Sequence[float]
        Connectivity scores for all original chunks.

    Returns
    -------
    obscured : str
        Human-unreadable spectral encoding — one reference line per zone.
    zone_map : dict[str, str]
        Mapping from zone ID (str) to original chunk text.  Retained
        locally by the caller; required for decoding.
    """
    if not kept_indices:
        return "", {}

    max_weight = float(adjacency.max()) if adjacency.size > 0 and adjacency.max() > 0 else 1.0

    lines: list[str] = []
    zone_map: dict[str, str] = {}

    for pos, orig_idx in enumerate(kept_indices):
        chunk = chunks[orig_idx]
        zc = zoned[orig_idx]
        fval = float(fiedler_vector[orig_idx])
        score = float(scores[orig_idx])
        cluster = 0 if fval >= 0 else 1

        # Coherence to next kept chunk (if exists)
        if pos + 1 < len(kept_indices):
            next_idx = kept_indices[pos + 1]
            coherence = float(adjacency[orig_idx, next_idx]) / max_weight
            coherence = max(0.0, min(1.0, coherence))
        else:
            coherence = 0.0

        ref = (
            f"[ZONE:{pos}"
            f"|type={zc.zone.name}"
            f"|fiedler={fval:.4f}"
            f"|score={score:.4f}"
            f"|cluster={cluster}"
            f"|coherence={coherence:.4f}"
            f"|words={chunk.word_count}]"
        )
        lines.append(ref)
        zone_map[str(pos)] = chunk.text

    return "\n".join(lines), zone_map


def spectral_decode(
    obscured: str,
    zone_map: dict[str, str],
) -> str:
    """
    Reconstruct compressed text from an obscured output and its zone map.

    Parameters
    ----------
    obscured : str
        Obscured output produced by :func:`spectral_obscure`.
    zone_map : dict[str, str]
        Mapping from zone ID (str) to original chunk text.

    Returns
    -------
    str
        Reconstructed compressed text (chunks joined by double newlines).

    Raises
    ------
    ValueError
        If a zone ID referenced in the obscured output is missing from
        the zone map.
    """
    if not obscured.strip():
        return ""

    texts: list[str] = []
    for line in obscured.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _ZONE_RE.match(line)
        if not m:
            raise ValueError(f"Malformed zone reference line: {line!r}")
        zone_id = m.group(1)
        if zone_id not in zone_map:
            raise ValueError(
                f"Zone ID {zone_id} not found in zone map. "
                f"The zone map is required for decoding."
            )
        texts.append(zone_map[zone_id])

    return "\n\n".join(texts)


def validate_zone_map(obscured: str, zone_map: dict[str, str]) -> bool:
    """
    Check that all zone IDs in *obscured* are present in *zone_map*.

    Returns ``True`` if the map is complete, ``False`` otherwise.
    """
    if not obscured.strip():
        return True
    for line in obscured.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _ZONE_RE.match(line)
        if not m:
            return False
        if m.group(1) not in zone_map:
            return False
    return True
