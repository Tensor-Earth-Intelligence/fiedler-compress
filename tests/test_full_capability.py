"""Coverage for the capabilities opened up in 0.4.0.

These paths previously raised CommercialTierError in the public package, so
nothing here was reachable before. The point of these tests is narrow: each
feature must actually execute and populate its result field, and the span-based
reassembly introduced alongside them must not duplicate overlapping text.
"""
from __future__ import annotations

import pytest

from fiedler_optimizer import optimize
from fiedler_optimizer.chunker import (
    Chunk,
    ChunkingStrategy,
    chunk_text,
    merge_kept_spans,
)


SAMPLE = (
    "The cat sat on the mat. The dog barked loudly at noon. " * 8
    + "\n\n"
    + "Quantum chromodynamics describes the strong interaction. "
      "Gluons mediate color charge. " * 8
)


def test_ligature_annotations_emitted():
    result = optimize(SAMPLE, target_ratio=0.5, emit_ligatures=True)
    assert result.ligatures
    assert all(isinstance(lig, dict) for lig in result.ligatures)


def test_ligature_rule_sets_apply():
    from fiedler_optimizer.ligatures import RULE_SETS

    assert "rag" in RULE_SETS
    result = optimize(SAMPLE, target_ratio=0.5, ligature_rules="rag")
    assert result.compressed


def test_unknown_ligature_rule_set_rejected():
    with pytest.raises(ValueError, match="Unknown rule set"):
        optimize(SAMPLE, target_ratio=0.5, ligature_rules="does-not-exist")


def test_reasoning_template_built():
    result = optimize(SAMPLE, target_ratio=0.5, template="summarization")
    assert result.reasoning_template["template_type"] == "summarization"
    assert result.reasoning_template["sections"]


def test_unknown_template_rejected():
    with pytest.raises(ValueError, match="Unknown template type"):
        optimize(SAMPLE, target_ratio=0.5, template="does-not-exist")


def test_certificate_is_signed():
    result = optimize(SAMPLE, target_ratio=0.5, certify=True)
    assert result.certificate["signature"]
    # Auto-generated key is secrets.token_hex(32) -> 64 hex characters.
    assert len(result.signing_key) == 64


def test_certificate_signature_depends_on_key():
    # Signing keys are hex-encoded HMAC keys (see certificate.generate_certificate).
    a = optimize(SAMPLE, target_ratio=0.5, certify="aa" * 32)
    b = optimize(SAMPLE, target_ratio=0.5, certify="bb" * 32)
    assert a.certificate["compressed_hash"] == b.certificate["compressed_hash"]
    assert a.certificate["signature"] != b.certificate["signature"]


def test_provenance_commits_to_source_prompt():
    result = optimize(SAMPLE, target_ratio=0.5, provenance=True)
    assert result.provenance["prompt_commitment"]
    assert result.provenance["signature"]


def test_obscure_emits_text_and_zone_map():
    result = optimize(SAMPLE, target_ratio=0.5, obscure=True)
    assert result.obscured
    assert isinstance(result.zone_map, dict)


def test_topology_cache_round_trips(tmp_path):
    from fiedler_optimizer.topology import TopologyCache

    # TopologyCache persists to disk (default ".fiedler_cache" in the CWD), so
    # the test must point it at a scratch directory to stay order-independent.
    cache = TopologyCache(cache_dir=tmp_path / "topo")
    first = optimize(SAMPLE, target_ratio=0.5, topology_cache=cache)
    assert first.topology["cache_hit"] is False
    second = optimize(SAMPLE, target_ratio=0.5, topology_cache=cache)
    assert second.topology["cache_hit"] is True


def test_topology_cache_rejects_path_traversal():
    from fiedler_optimizer.topology import TopologyCache

    with pytest.raises(ValueError, match="Path traversal"):
        TopologyCache(cache_dir="../escaped")


# ---------------------------------------------------------------------------
# Similarity backends and content priors
# ---------------------------------------------------------------------------

def test_tfidf_backend_selectable_by_name():
    result = optimize(SAMPLE, target_ratio=0.5, backend="tfidf")
    assert result.compressed


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        optimize(SAMPLE, target_ratio=0.5, backend="does-not-exist")


def test_content_prior_callable_pins_matching_chunks():
    result = optimize(SAMPLE, target_ratio=0.8, content_prior=lambda t: ["Gluons"])
    assert "Gluons" in result.compressed


def test_unknown_content_prior_rejected():
    with pytest.raises(ValueError, match="Unknown content_prior"):
        optimize(SAMPLE, target_ratio=0.5, content_prior="does-not-exist")


def test_content_prior_patterns_are_validated():
    """Generated patterns must pass the same ReDoS guard as user-supplied ones."""
    with pytest.raises(ValueError, match="nested quantifier"):
        optimize(SAMPLE, target_ratio=0.5, content_prior=lambda t: ["(a+)+"])


# ---------------------------------------------------------------------------
# Coverage floor
# ---------------------------------------------------------------------------

# A redundant bulk (one dominant cluster) plus a single distinct needle.
_BULK = "The server logs show a routine health check completed successfully. " * 30
_NEEDLE = "The armadillo migration pattern across Patagonia shifted in 1987."
NEEDLE_DOC = _BULK + "\n\n" + _NEEDLE


def test_coverage_floor_protects_an_isolated_topic():
    """The needle scores low and is dropped by default; the floor keeps it,
    without spending any additional budget."""
    plain = optimize(NEEDLE_DOC, target_ratio=0.8)
    floored = optimize(NEEDLE_DOC, target_ratio=0.8, min_keep_per_cluster=1)

    assert _NEEDLE not in plain.compressed
    assert _NEEDLE in floored.compressed
    assert floored.chunks_removed == plain.chunks_removed


def test_coverage_auto_engages_on_dominant_cluster():
    auto = optimize(NEEDLE_DOC, target_ratio=0.8, coverage_auto=True)
    assert _NEEDLE in auto.compressed


def test_coverage_auto_declines_on_balanced_topics():
    """With no dominant cluster the floor stays off, so compression is unchanged."""
    plain = optimize(SAMPLE, target_ratio=0.8)
    auto = optimize(SAMPLE, target_ratio=0.8, coverage_auto=True)
    assert auto.chunks_removed == plain.chunks_removed


# ---------------------------------------------------------------------------
# Span reassembly
# ---------------------------------------------------------------------------

def test_merge_kept_spans_does_not_duplicate_overlap():
    """Overlapping windows must be merged, not concatenated.

    Joining chunk text directly would emit the shared region twice, which is
    what previously turned sliding-window compression negative.
    """
    text = "abcdefghij"
    chunks = [
        Chunk(text="abcdef", index=0, start_char=0, end_char=6, word_count=1),
        Chunk(text="defghij", index=1, start_char=3, end_char=10, word_count=1),
    ]
    assert merge_kept_spans(text, chunks, [0, 1]) == "abcdefghij"


def test_merge_kept_spans_separates_disjoint_regions():
    text = "abcdefghij"
    chunks = [
        Chunk(text="abc", index=0, start_char=0, end_char=3, word_count=1),
        Chunk(text="hij", index=1, start_char=7, end_char=10, word_count=1),
    ]
    assert merge_kept_spans(text, chunks, [0, 1]) == "abc\n\nhij"


def test_compression_is_never_negative_on_overlapping_chunks():
    result = optimize(SAMPLE, target_ratio=0.5, strategy=ChunkingStrategy.SLIDING_WINDOW)
    assert len(result.compressed) <= len(SAMPLE)


# ---------------------------------------------------------------------------
# CODE chunking strategy
# ---------------------------------------------------------------------------

CODE_SAMPLE = """{
  "name": "example",
  "version": [1, 2, 3],
  "nested": {"a": 1; "b": 2},
  "flag": true
}"""


def test_code_text_routes_to_line_chunking():
    from fiedler_optimizer.chunker import _choose_strategy

    assert _choose_strategy(CODE_SAMPLE) is ChunkingStrategy.CODE
    # Chunks follow line boundaries (very short lines such as a bare brace are
    # merged), so no chunk should span the whole document.
    chunks = chunk_text(CODE_SAMPLE, strategy=ChunkingStrategy.ADAPTIVE)
    assert 1 < len(chunks) <= len([ln for ln in CODE_SAMPLE.splitlines() if ln.strip()])


def test_prose_is_not_misread_as_code():
    """Instruction prose contains ':' and ';' but must stay on the prose path."""
    prose = (
        "First: review the document carefully.\n"
        "Second: summarize the key findings.\n"
        "Third: list any open questions; be specific.\n"
        "Finally: recommend a course of action.\n"
    )
    chunks = chunk_text(prose, strategy=ChunkingStrategy.ADAPTIVE)
    assert all(c.text in prose for c in chunks)
    from fiedler_optimizer.chunker import _looks_like_code

    assert not _looks_like_code(prose)


def test_code_chunk_spans_are_exact_substrings():
    chunks = chunk_text(CODE_SAMPLE, strategy=ChunkingStrategy.CODE)
    for c in chunks:
        assert CODE_SAMPLE[c.start_char:c.end_char] == c.text
