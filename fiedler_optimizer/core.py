"""
Core compression pipeline.

This is the main entry point for Fiedler spectral compression. The
optimize() function takes text in and returns compressed text out,
orchestrating the full pipeline: chunk → graph → Fiedler → score → prune.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from fiedler_optimizer.chunker import (
    Chunk,
    ChunkingStrategy,
    chunk_text,
    merge_kept_spans,
)
from fiedler_optimizer.graph import (
    build_similarity_graph,
    compute_fiedler_vector,
    compute_chunk_scores,
)
from fiedler_optimizer.zones import Zone, ZonedChunk, detect_zones


# --- Pin-pattern safety limits (ReDoS / DoS mitigation; see SECURITY_AUDIT.md P4-1) ---
MAX_PIN_PATTERNS = 100
"""Maximum number of pin patterns accepted by optimize()."""

MAX_PIN_PATTERN_LENGTH = 1000
"""Maximum length (characters) of a single pin pattern."""

# Conservative signature for catastrophic-backtracking ("nested quantifier")
# regexes: a single quantified atom/class inside a group that is itself
# repeated by an unbounded quantifier, e.g. (a+)+, (.*)*, (\d+)+, ([a-z]+)*.
# Cheap and low-false-positive; it does NOT catch every possible ReDoS form.
_REDOS_NESTED_QUANTIFIER = re.compile(r'\((?:\[[^\]]*\]|\\?[^()\[\]])[*+]\)[*+]')


def validate_pin_patterns(pin_patterns: Sequence[str]) -> None:
    """Validate user-supplied pin patterns before any of them are compiled.

    This is the single choke point through which every caller passes — the
    :func:`optimize` ``pin_patterns`` argument, the CLI ``--pin-regex`` flag,
    and the CLI ``--pin-regex`` flag — so the same caps apply everywhere. It guards against
    denial-of-service from oversized or too-numerous regex inputs and against
    catastrophic-backtracking (ReDoS) patterns. See SECURITY_AUDIT.md P4-1.

    Parameters
    ----------
    pin_patterns : Sequence[str]
        The raw, user-supplied pin patterns to validate.

    Raises
    ------
    ValueError
        If more than :data:`MAX_PIN_PATTERNS` patterns are supplied, if any
        single pattern exceeds :data:`MAX_PIN_PATTERN_LENGTH` characters, or
        if a pattern contains a nested unbounded quantifier (e.g. ``(a+)+``)
        prone to catastrophic backtracking. The message names the cap that
        was exceeded and its limit.
    """
    if len(pin_patterns) > MAX_PIN_PATTERNS:
        raise ValueError(
            f"Too many pin patterns: {len(pin_patterns)} "
            f"(maximum is {MAX_PIN_PATTERNS})."
        )
    for pat in pin_patterns:
        if len(pat) > MAX_PIN_PATTERN_LENGTH:
            raise ValueError(
                f"Pin pattern too long: {len(pat)} characters "
                f"(maximum is {MAX_PIN_PATTERN_LENGTH})."
            )
        if _REDOS_NESTED_QUANTIFIER.search(pat):
            raise ValueError(
                "Pin pattern rejected: it contains a nested quantifier "
                "(e.g. '(a+)+') that can cause catastrophic backtracking "
                "(ReDoS). Rewrite it without nested unbounded quantifiers."
            )


@dataclass(frozen=True)
class FiedlerResult:
    """Result of Fiedler spectral compression."""

    compressed: str
    """The compressed text with low-connectivity chunks removed."""

    original_text: str
    """The original input text."""

    compression_ratio: float
    """Fraction of text removed (0.0 = nothing removed, 1.0 = everything removed)."""

    tokens_saved: int
    """Approximate tokens saved (estimated at ~4 chars/token)."""

    algebraic_connectivity: float
    """λ₂ of the original graph — higher means tighter semantic structure."""

    chunks_total: int
    """Number of chunks in the original text."""

    chunks_removed: int
    """Number of chunks removed."""

    removed_chunks: list[str] = field(default_factory=list)
    """The text of chunks that were removed (for inspection/debugging).

    Bare text, so duplicate chunks are indistinguishable: if two chunks share
    the same text and one is removed, you cannot tell which from this list. Use
    the chunk offsets from ``chunk_text`` if you need to identify the exact
    source span.
    """

    chunk_scores: list[float] = field(default_factory=list)
    """Connectivity score for each original chunk (for visualization)."""

    ligatures: list = field(default_factory=list)
    """Post-compression ligature annotations (list of ChunkLigature dicts)."""

    reasoning_template: dict | None = field(default=None)
    """Reasoning template built for the compressed text (set by ``template=``)."""

    certificate: dict | None = field(default=None)
    """Signed attestation over the spectrum and output (set by ``certify=``)."""

    signing_key: str | None = field(default=None)
    """Hex HMAC key used for ``certificate``. Auto-generated when not supplied."""

    provenance: dict | None = field(default=None)
    """Attestation additionally committing to the source prompt (``provenance=``)."""

    provenance_key: str | None = field(default=None)
    """Hex HMAC key used for ``provenance``. Auto-generated when not supplied."""

    topology: dict | None = field(default=None)
    """Topology classification and cache status (set by ``topology_cache=``)."""

    distillation: dict | None = field(default=None)
    """Distillation report: backend, zones, improvement (``distill_backend=``)."""

    obscured: str | None = field(default=None)
    """Spectrally obscured rendering of the output (set by ``obscure=``)."""

    zone_map: dict | None = field(default=None)
    """Zone map needed to interpret ``obscured`` (set by ``obscure=``)."""


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


COVERAGE_AUTO_DOMINANCE = 0.6
"""Auto-mode turns the coverage floor on only when one cluster holds at least
this fraction of all chunks (the signature of a redundant bulk with sparse
outliers)."""


def _coverage_clusters(adjacency) -> list[int]:
    """Topical islands for the coverage floor: connected components of the
    similarity graph thresholded at the mean of its positive edge weights.
    Chunks on distinct topics (weakly connected) land in separate components, so
    a per-cluster keep-floor guarantees each topic keeps at least one survivor."""
    from scipy.sparse import csr_matrix as _csr
    from scipy.sparse.csgraph import connected_components as _cc

    A = np.asarray(adjacency, dtype=float)
    pos = A[A > 0]
    thr = float(pos.mean()) if pos.size else 0.0
    _, labels = _cc(_csr((A > thr).astype(int)), directed=False)
    return labels.tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(
    text: str,
    target_ratio: float = 0.20,
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
    protect_instructions: bool = True,
    min_chunks: int = 4,
    vectors: np.ndarray | None = None,
    use_neural: bool = False,
    ligature_rules: str | list | None = None,
    emit_ligatures: bool = False,
    certify: bool | str = False,
    provenance: bool | str = False,
    template: str | None = None,
    obscure: bool = False,
    topology_cache: object | None = None,
    distill_backend: str | None = None,
    pin_patterns: list[str] | None = None,
    content_prior: str | Callable[[str], "list[str] | None"] | None = None,
    backend: str | None = None,
    min_keep_per_cluster: int = 0,
    cluster_labels: list[int] | None = None,
    coverage_auto: bool = False,
) -> FiedlerResult:
    """
    Compress text using Fiedler spectral decomposition.

    The algorithm:
    1. Chunk the text into semantically meaningful segments.
    2. Build a similarity graph over chunks (TF-IDF cosine similarity).
    3. Compute the graph Laplacian and extract the Fiedler vector.
    4. Score each chunk by its spectral connectivity.
    5. Optionally classify chunks into instruction/context zones.
    6. Remove the lowest-scoring context chunks up to the target ratio.
    7. Return the compressed text with metadata.

    Parameters
    ----------
    text : str
        The input text to compress.
    target_ratio : float
        Target fraction of text to remove. Default 0.20 (20%). The actual
        removal may be less if too many chunks are protected.
    strategy : ChunkingStrategy
        How to split the text. ADAPTIVE auto-selects.
    protect_instructions : bool
        If True, instruction-zone chunks get 2–3x protection weight,
        making them much harder to remove.
    min_chunks : int
        Minimum number of chunks required for spectral analysis. If the
        text produces fewer chunks, it's returned unmodified.
    vectors : np.ndarray, optional
        Pre-computed feature vectors for the chunks. If None, TF-IDF
        vectors are computed. Pass pre-computed embeddings here to use a
        custom similarity space.
    pin_patterns : list[str], optional
        Regex patterns or literal strings.  Any chunk whose text matches
        at least one pattern is *pinned* — it is never removed during
        compression.  The Fiedler vector is still computed over all
        chunks (including pinned ones) so the spectral topology stays
        accurate.  Default ``None`` means no pinning.
    content_prior : str or callable, optional
        Shortcut for pinning by content TYPE instead of hand-writing patterns.
        Either a callable ``fn(text) -> list[str] | None`` (equivalent to
        ``optimize(text, pin_patterns=fn(text))``), or one of the curated preset
        names ``"identifier"``, ``"salience"``, ``"observation"``, which resolve
        against :mod:`fiedler_optimizer.geometry.content_prior` and therefore
        need the ``geometry`` extra.  Raises ``ValueError`` for an unknown
        preset.  Composes with an explicit ``pin_patterns``: the two lists are
        merged rather than one overriding the other.
    backend : str, optional
        Named similarity backend (e.g. ``'aitchison'``, ``'fisher-rao'``,
        ``'wasserstein'``, ``'hyperbolic'``).  Overrides ``use_neural`` and
        ``vectors``.  Raises ``ValueError`` for an unknown backend.
    min_keep_per_cluster : int
        Coverage floor: every topical cluster retains at least this many chunks, so
        evidence sitting alone on a distinct topic is not removed just for
        scoring low.  Default 0 disables the floor entirely.
    cluster_labels : list[int], optional
        Precomputed cluster label per chunk for the coverage floor.  When
        ``None`` the clusters are derived from the similarity graph.
    coverage_auto : bool
        Apply a coverage floor of 1 automatically, but only when one cluster
        dominates the input (see :data:`COVERAGE_AUTO_DOMINANCE`) — the pattern
        of a redundant bulk with sparse outliers.  On inputs made of distinct
        topics the floor stays off, so it never strangles compression.

    Returns
    -------
    FiedlerResult
        Compressed text and metadata.
    """
    # --- Guard: empty or very short text ---
    if not text or not text.strip():
        return FiedlerResult(
            compressed=text,
            original_text=text,
            compression_ratio=0.0,
            tokens_saved=0,
            algebraic_connectivity=0.0,
            chunks_total=0,
            chunks_removed=0,
        )

    if backend is not None:
        from fiedler_optimizer.backends.registry import BACKENDS
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {sorted(BACKENDS)}"
            )

    # --- Step 1: Chunk ---
    chunks = chunk_text(text, strategy=strategy)

    if len(chunks) < min_chunks:
        return FiedlerResult(
            compressed=text,
            original_text=text,
            compression_ratio=0.0,
            tokens_saved=0,
            algebraic_connectivity=0.0,
            chunks_total=len(chunks),
            chunks_removed=0,
        )

    # --- Step 2: Build similarity graph ---
    adjacency = build_similarity_graph(
        chunks, vectors=vectors, use_neural=use_neural, backend=backend
    )
    # --- Step 2b: Optional graph enrichment ---
    if ligature_rules is not None:
        from fiedler_optimizer.ligatures import apply_ligatures, RULE_SETS
        if isinstance(ligature_rules, str):
            if ligature_rules not in RULE_SETS:
                raise ValueError(
                    f"Unknown rule set: {ligature_rules}. "
                    f"Available: {list(RULE_SETS.keys())}"
                )
            rules = RULE_SETS[ligature_rules]
        else:
            rules = ligature_rules
        adjacency, _ligature_result = apply_ligatures(adjacency, chunks, rules)

    # --- Step 3: Fiedler vector ---
    fiedler, lambda_2 = compute_fiedler_vector(adjacency)

    # --- Step 3b: Optional graph caching ---
    topology_data: dict | None = None
    if topology_cache is not None:
        from fiedler_optimizer.topology import (
            TopologyClassifier,
            TopologyCache,
            extract_features,
        )
        features = extract_features(adjacency, fiedler, lambda_2)
        cached = topology_cache.get(features)
        if cached is not None:
            # Warm-start: use cached parameters
            target_ratio = cached.target_ratio
            protect_instructions = cached.protect_instructions
        # Always classify and store
        classifier = TopologyClassifier()
        classification = classifier.classify_from_features(features)
        topology_cache.put(
            features, classification.topology,
            target_ratio=target_ratio,
            protect_instructions=protect_instructions,
        )
        topology_data = {
            "type": classification.topology.value,
            "confidence": classification.confidence,
            "cache_hit": cached is not None,
        }

    # --- Step 4: Score chunks ---
    scores = compute_chunk_scores(chunks, fiedler, adjacency)

    # --- Step 5: Zone detection ---
    if protect_instructions:
        zoned = detect_zones(chunks)
        # Apply zone protection weights to scores
        weighted_scores = [
            score * zc.protection_weight
            for score, zc in zip(scores, zoned)
        ]
    else:
        zoned = [ZonedChunk(c, Zone.UNKNOWN, 0.0) for c in chunks]
        weighted_scores = scores

    # --- Step 5b: Identify pinned chunks ---
    # content_prior is resolved into pin_patterns first, so generated patterns go
    # through the same validate_pin_patterns() choke point as user-supplied ones.
    # The preset path imports geometry lazily, so the extra is only needed when a
    # preset name is actually requested.
    if content_prior is not None:
        if isinstance(content_prior, str):
            from fiedler_optimizer.geometry.content_prior import (
                CONTENT_PRIOR_PIN_PATTERNS,
            )
            if content_prior not in CONTENT_PRIOR_PIN_PATTERNS:
                raise ValueError(
                    f"Unknown content_prior '{content_prior}'. "
                    f"Available: {list(CONTENT_PRIOR_PIN_PATTERNS.keys())}"
                )
            prior_patterns = CONTENT_PRIOR_PIN_PATTERNS[content_prior](text)
        else:
            prior_patterns = content_prior(text)
        if prior_patterns:
            pin_patterns = (
                [*pin_patterns, *prior_patterns]
                if pin_patterns
                else list(prior_patterns)
            )

    pinned_indices: set[int] = set()
    if pin_patterns:
        # Guard against oversized / too-many / ReDoS-prone user input before
        # compiling anything (centralized in validate_pin_patterns).
        validate_pin_patterns(pin_patterns)
        compiled = []
        for pat in pin_patterns:
            try:
                compiled.append(re.compile(pat, re.IGNORECASE | re.DOTALL))
            except re.error:
                # Treat as a literal string if it's not a valid regex
                compiled.append(re.compile(re.escape(pat), re.IGNORECASE))
        for i, chunk in enumerate(chunks):
            for rx in compiled:
                if rx.search(chunk.text):
                    pinned_indices.add(i)
                    break

    # --- Step 6: Determine which chunks to remove ---
    n = len(chunks)
    target_removals = max(1, int(n * target_ratio))

    # Coverage floor: never empty a topical cluster. Each cluster keeps at least
    # `effective_min` survivors, so evidence sitting alone on a distinct topic is
    # protected even when it scores low. Enabled explicitly by
    # min_keep_per_cluster > 0, or automatically by coverage_auto when one cluster
    # dominates (a redundant bulk with sparse outlier needles). On mostly-singleton
    # inputs (distinct topics / RAG passages) auto keeps the floor off, where it
    # would otherwise strangle compression.
    cov_labels = None
    cluster_kept = None
    effective_min = min_keep_per_cluster
    if min_keep_per_cluster > 0 or coverage_auto:
        from collections import Counter as _Counter

        cov_labels = (
            cluster_labels if cluster_labels is not None
            else _coverage_clusters(adjacency)
        )
        cluster_kept = dict(_Counter(cov_labels))
        n_clusters = len(cluster_kept)
        dominance = (max(cluster_kept.values()) / n) if n else 0.0
        if coverage_auto and min_keep_per_cluster == 0:
            effective_min = 1 if dominance >= COVERAGE_AUTO_DOMINANCE else 0
        # Disable when the floor cannot permit meaningful removal (all singletons,
        # or auto declined): fall back to ordinary removal.
        if effective_min == 0 or n_clusters >= n:
            cov_labels = None
            cluster_kept = None
            effective_min = 0

    # Build (index, weighted_score) pairs and sort by score ascending
    indexed_scores = sorted(enumerate(weighted_scores), key=lambda x: x[1])

    # Remove the lowest-scoring chunks, but never remove more than target
    remove_indices: set[int] = set()
    total_chars = sum(c.word_count for c in chunks)
    removed_chars = 0

    for idx, wscore in indexed_scores:
        if len(remove_indices) >= target_removals:
            break
        # Don't remove pinned chunks (explicit user protection)
        if idx in pinned_indices:
            continue
        # Don't remove chunks that are highly protected instructions
        if protect_instructions and zoned[idx].zone == Zone.INSTRUCTION and zoned[idx].confidence > 0.7:
            continue
        # Coverage floor: don't remove the last survivor(s) of a topical cluster
        if cluster_kept is not None:
            lbl = cov_labels[idx]
            if cluster_kept[lbl] - 1 < effective_min:
                continue
        # Don't remove if it would exceed target ratio by character count
        chunk_chars = chunks[idx].word_count
        if total_chars > 0 and (removed_chars + chunk_chars) / total_chars > target_ratio * 1.5:
            continue
        remove_indices.add(idx)
        removed_chars += chunk_chars
        if cluster_kept is not None:
            cluster_kept[cov_labels[idx]] -= 1

    # --- Step 7: Reconstruct compressed text ---
    kept_indices = sorted(i for i in range(n) if i not in remove_indices)
    removed_chunks_text = [chunks[i].text for i in sorted(remove_indices)]
    # Reassemble from original character spans, merging overlapping windows.
    # Joining chunk text directly would duplicate the ADAPTIVE chunker's
    # overlap, inflating the output into negative "compression".
    compressed = merge_kept_spans(text, chunks, kept_indices)

    original_tokens = _estimate_tokens(text)
    compressed_tokens = _estimate_tokens(compressed)
    tokens_saved = max(0, original_tokens - compressed_tokens)

    char_ratio = 1.0 - (len(compressed) / max(len(text), 1))

    # --- Step 7b: Optional post-processing ---
    distillation_data: dict | None = None
    if distill_backend is not None:
        from dataclasses import asdict as _asdict_dist
        from fiedler_optimizer.distillation import Distiller, get_backend
        from fiedler_optimizer.ligature import generate_ligatures as _gen_lig_early

        # Pre-compute annotations so post-processing can preserve anchors
        early_ligatures = _gen_lig_early(kept_indices, fiedler, lambda_2, adjacency)
        early_lig_dicts = [_asdict_dist(lig) for lig in early_ligatures]

        # Accept either a backend name (str) or a pre-constructed instance.
        # Named distinctly from the `backend` parameter, which selects the
        # similarity metric and must not be clobbered here.
        if isinstance(distill_backend, str):
            distiller_backend = get_backend(distill_backend)
        else:
            distiller_backend = distill_backend  # already a backend instance
        distiller = Distiller(distiller_backend)
        dist_result = distiller.distill(
            kept_indices, chunks, zoned, fiedler, scores,
            ligatures=early_lig_dicts,
        )
        # Replace compressed text with the distilled version
        compressed = dist_result.distilled_text
        compressed_tokens = _estimate_tokens(compressed)
        tokens_saved = max(0, original_tokens - compressed_tokens)
        char_ratio = 1.0 - (len(compressed) / max(len(text), 1))
        distillation_data = {
            "backend": dist_result.backend_name,
            "total_compression_improvement": dist_result.total_compression_improvement,
            "zone_count": len(dist_result.zones),
            "zones": [
                {
                    "position": dz.zone_position,
                    "anchors": list(dz.anchors),
                    "compression_improvement": dz.compression_improvement,
                }
                for dz in dist_result.zones
            ],
        }

    # --- Step 8: Optional spectral obscured output ---
    obscured_text: str | None = None
    zone_map_data: dict | None = None
    if obscure:
        from fiedler_optimizer.obscure import spectral_obscure
        obscured_text, zone_map_data = spectral_obscure(
            kept_indices, chunks, fiedler, lambda_2, adjacency, zoned, scores,
        )

    # --- Step 9: Optional reasoning template ---
    template_data: dict | None = None
    if template is not None:
        from dataclasses import asdict as _asdict_tmpl
        from fiedler_optimizer.templates import (
            TemplateRegistry,
            compute_spectral_profile,
        )
        profile = compute_spectral_profile(
            kept_indices, fiedler, lambda_2, adjacency, zoned,
        )
        tmpl = TemplateRegistry.build(template, profile)
        template_data = _asdict_tmpl(tmpl)

    # --- Step 10: Optional post-compression ligatures ---
    ligature_annotations: list = []
    if emit_ligatures:
        from dataclasses import asdict as _asdict_lig
        from fiedler_optimizer.ligature import generate_ligatures
        raw_ligatures = generate_ligatures(kept_indices, fiedler, lambda_2, adjacency)
        ligature_annotations = [_asdict_lig(lig) for lig in raw_ligatures]

    # --- Step 11: Optional spectral certificate ---
    certificate_data: dict | None = None
    cert_key: str | None = None
    if certify:
        from dataclasses import asdict as _asdict_cert
        from fiedler_optimizer.certificate import generate_certificate
        key_arg = certify if isinstance(certify, str) else None
        cert, cert_key = generate_certificate(
            fiedler, adjacency, zoned, compressed, signing_key=key_arg,
        )
        certificate_data = _asdict_cert(cert)

    # --- Step 12: Optional provenance certificate ---
    provenance_data: dict | None = None
    provenance_key: str | None = None
    if provenance:
        from dataclasses import asdict as _asdict_prov
        from fiedler_optimizer.certificate import generate_provenance_certificate
        prov_key_arg = provenance if isinstance(provenance, str) else None
        prov_cert, provenance_key = generate_provenance_certificate(
            text, fiedler, adjacency, zoned, compressed, signing_key=prov_key_arg,
        )
        provenance_data = _asdict_prov(prov_cert)

    return FiedlerResult(
        compressed=compressed,
        original_text=text,
        compression_ratio=char_ratio,
        tokens_saved=tokens_saved,
        algebraic_connectivity=lambda_2,
        chunks_total=n,
        chunks_removed=len(remove_indices),
        removed_chunks=removed_chunks_text,
        chunk_scores=scores,
        reasoning_template=template_data,
        ligatures=ligature_annotations,
        certificate=certificate_data,
        signing_key=cert_key,
        provenance=provenance_data,
        provenance_key=provenance_key,
        topology=topology_data,
        distillation=distillation_data,
        obscured=obscured_text,
        zone_map=zone_map_data,
    )
