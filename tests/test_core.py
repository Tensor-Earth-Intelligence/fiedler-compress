"""
Tests for Fiedler Optimizer core pipeline.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from fiedler_optimizer import optimize, FiedlerResult
from fiedler_optimizer.chunker import chunk_text, ChunkingStrategy, Chunk
from fiedler_optimizer.graph import (
    build_similarity_graph,
    compute_fiedler_vector,
    compute_chunk_scores,
)
from fiedler_optimizer.zones import detect_zones, Zone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_PROMPT = (
    "You are an expert Python developer. Always use type hints. "
    "Follow PEP 8 conventions strictly.\n\n"
    "Context: The project uses FastAPI for the web framework. "
    "The database is PostgreSQL with SQLAlchemy ORM. "
    "Authentication is handled via JWT tokens. "
    "The codebase follows a repository pattern for data access. "
    "Tests use pytest with fixtures for database setup.\n\n"
    "Task: Write a function to fetch a user by email address."
)

REDUNDANT_PROMPT = (
    "The Eiffel Tower is 330 meters tall. It was built in 1889. "
    "The tower is located in Paris, France.\n\n"
    "The Eiffel Tower, standing at 330 meters, was constructed in 1889. "
    "It is situated in Paris.\n\n"
    "Paris is home to the Eiffel Tower, a 330-meter structure from 1889.\n\n"
    "Other Paris landmarks include the Louvre and Notre-Dame Cathedral. "
    "The city sees 30 million tourists annually.\n\n"
    "Question: How tall is the Eiffel Tower?"
)

SHORT_TEXT = "Hello world."


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_sentence_chunking(self):
        chunks = chunk_text(SIMPLE_PROMPT, strategy=ChunkingStrategy.SENTENCE)
        assert len(chunks) >= 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.word_count >= 1 for c in chunks)

    def test_paragraph_chunking(self):
        chunks = chunk_text(SIMPLE_PROMPT, strategy=ChunkingStrategy.PARAGRAPH)
        assert len(chunks) >= 2  # at least instruction + context paragraphs

    def test_adaptive_selects_strategy(self):
        chunks = chunk_text(SIMPLE_PROMPT, strategy=ChunkingStrategy.ADAPTIVE)
        assert len(chunks) >= 2

    def test_short_text_produces_chunks(self):
        chunks = chunk_text("One. Two. Three.", strategy=ChunkingStrategy.SENTENCE)
        assert len(chunks) >= 1

    def test_empty_text(self):
        chunks = chunk_text("", strategy=ChunkingStrategy.SENTENCE)
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_adjacency_matrix_shape(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        adj = build_similarity_graph(chunks)
        n = len(chunks)
        assert adj.shape == (n, n)

    def test_adjacency_is_symmetric(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        adj = build_similarity_graph(chunks)
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_no_self_loops(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        adj = build_similarity_graph(chunks)
        np.testing.assert_array_equal(np.diag(adj), 0.0)

    def test_fiedler_vector_length(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        adj = build_similarity_graph(chunks)
        fiedler, lambda_2 = compute_fiedler_vector(adj)
        assert len(fiedler) == len(chunks)
        assert lambda_2 >= 0.0

    def test_fiedler_normalized_range(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        adj = build_similarity_graph(chunks)
        fiedler, _ = compute_fiedler_vector(adj)
        assert np.max(np.abs(fiedler)) <= 1.0 + 1e-10

    def test_chunk_scores_length(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        adj = build_similarity_graph(chunks)
        fiedler, _ = compute_fiedler_vector(adj)
        scores = compute_chunk_scores(chunks, fiedler, adj)
        assert len(scores) == len(chunks)
        assert all(0.0 <= s <= 3.5 for s in scores)  # account for zone weights


# ---------------------------------------------------------------------------
# Zone detection tests
# ---------------------------------------------------------------------------

class TestZones:
    def test_instruction_detection(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        zoned = detect_zones(chunks)
        # The first chunk ("You are an expert...Always use type hints...")
        # should be detected as instruction
        instruction_zones = [z for z in zoned if z.zone == Zone.INSTRUCTION]
        assert len(instruction_zones) >= 1

    def test_context_detection(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        zoned = detect_zones(chunks)
        context_zones = [z for z in zoned if z.zone == Zone.CONTEXT]
        assert len(context_zones) >= 1

    def test_protection_weights(self):
        chunks = chunk_text(SIMPLE_PROMPT)
        zoned = detect_zones(chunks)
        for z in zoned:
            if z.zone == Zone.INSTRUCTION:
                assert z.protection_weight >= 2.0
            else:
                assert z.protection_weight == 1.0

    def test_task_directive_detected(self):
        """Regression test: 'Task:' lines must be classified as INSTRUCTION."""
        from fiedler_optimizer.chunker import Chunk
        task_chunk = Chunk(
            text="Task: Write a function to fetch a user by email address.",
            index=0, start_char=0, end_char=56, word_count=11,
        )
        zoned = detect_zones([task_chunk])
        assert zoned[0].zone == Zone.INSTRUCTION

    def test_question_directive_detected(self):
        from fiedler_optimizer.chunker import Chunk
        q_chunk = Chunk(
            text="Question: How tall is the Eiffel Tower?",
            index=0, start_char=0, end_char=39, word_count=8,
        )
        zoned = detect_zones([q_chunk])
        assert zoned[0].zone == Zone.INSTRUCTION



# ---------------------------------------------------------------------------
# Core optimize() tests
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_basic_compression(self):
        result = optimize(SIMPLE_PROMPT)
        assert isinstance(result, FiedlerResult)
        assert result.compression_ratio >= 0.0
        assert result.compressed  # not empty
        assert len(result.compressed) <= len(SIMPLE_PROMPT)

    def test_redundant_prompt_compression(self):
        result = optimize(REDUNDANT_PROMPT, target_ratio=0.30)
        # Redundant content should compress well
        assert result.chunks_removed >= 1
        assert result.compression_ratio > 0.0

    def test_short_text_passthrough(self):
        result = optimize(SHORT_TEXT)
        # Too short for meaningful compression
        assert result.compressed == SHORT_TEXT
        assert result.compression_ratio == 0.0
        assert result.chunks_removed == 0

    def test_empty_text_passthrough(self):
        result = optimize("")
        assert result.compressed == ""
        assert result.compression_ratio == 0.0

    def test_target_ratio_respected(self):
        result = optimize(REDUNDANT_PROMPT, target_ratio=0.10)
        # Should not remove more than ~15% (target * 1.5 safety margin)
        assert result.compression_ratio <= 0.50  # generous bound

    def test_instruction_protection(self):
        result = optimize(SIMPLE_PROMPT, protect_instructions=True)
        # The instruction text should survive compression
        assert "type hints" in result.compressed or "PEP 8" in result.compressed

    def test_no_instruction_protection(self):
        result = optimize(SIMPLE_PROMPT, protect_instructions=False)
        # Should still produce valid output
        assert isinstance(result.compressed, str)

    def test_result_metadata(self):
        result = optimize(SIMPLE_PROMPT)
        assert result.chunks_total >= result.chunks_removed
        assert result.algebraic_connectivity >= 0.0
        assert result.tokens_saved >= 0
        assert isinstance(result.chunk_scores, list)

    def test_json_serializable(self):
        """Result fields should be JSON-serializable for the CLI."""
        import json
        result = optimize(SIMPLE_PROMPT)
        data = {
            "compressed": result.compressed,
            "compression_ratio": result.compression_ratio,
            "tokens_saved": result.tokens_saved,
            "algebraic_connectivity": result.algebraic_connectivity,
            "chunks_total": result.chunks_total,
            "chunks_removed": result.chunks_removed,
        }
        serialized = json.dumps(data)
        assert serialized  # no exception


# ---------------------------------------------------------------------------
# Pin patterns tests
# ---------------------------------------------------------------------------

# A prompt with clearly distinct sections — some should be pinned, others not.
PIN_TEST_PROMPT = (
    "## Rules\n"
    "1. Always verify the user identity before proceeding.\n"
    "2. Never share confidential data with unauthorized parties.\n"
    "3. You must log all interactions for audit purposes.\n"
    "4. Always respond in the specified output format.\n\n"
    "## Background\n"
    "The company was founded in 2010 in San Francisco. "
    "It provides cloud infrastructure services to enterprise clients. "
    "The customer base spans 40 countries across North America and Europe.\n\n"
    "The company processes over 10 million API requests daily. "
    "Infrastructure runs on AWS with multi-region failover. "
    "The support team operates 24/7 across three time zones.\n\n"
    "Additional context about the company history and market position "
    "that provides useful but non-essential background information "
    "for the assistant to reference when needed.\n\n"
    "## Safety\n"
    "Do not provide medical or legal advice.\n"
    "Always recommend consulting a qualified professional.\n\n"
    "Question: What are the key rules I need to follow?"
)


class TestPinPatterns:
    def test_pinned_chunks_never_removed(self):
        """Chunks matching pin_patterns must survive compression."""
        # Pin any chunk containing "must" or "never" or numbered rules
        result = optimize(
            PIN_TEST_PROMPT,
            target_ratio=0.50,
            pin_patterns=[r"\b(?:must|never)\b", r"^\d+\.\s"],
        )
        # All numbered rules and constraint keywords must be retained
        assert "Always verify" in result.compressed
        assert "Never share" in result.compressed
        assert "must log" in result.compressed
        # Some compression should still happen on the background content
        assert result.chunks_removed >= 1
        assert result.compression_ratio > 0.0

    def test_fiedler_computed_over_all_chunks(self):
        """Fiedler vector should be computed over all chunks including pinned."""
        # With aggressive pinning, we should still get valid scores for all chunks
        result_pinned = optimize(
            REDUNDANT_PROMPT,
            target_ratio=0.30,
            pin_patterns=[r"Eiffel Tower"],
        )
        result_unpinned = optimize(
            REDUNDANT_PROMPT,
            target_ratio=0.30,
        )
        # Both should have the same total chunks (Fiedler runs on all)
        assert result_pinned.chunks_total == result_unpinned.chunks_total
        # Both should have valid chunk scores for ALL chunks
        assert len(result_pinned.chunk_scores) == result_pinned.chunks_total
        assert len(result_unpinned.chunk_scores) == result_unpinned.chunks_total
        # Pinned version should remove fewer chunks
        assert result_pinned.chunks_removed <= result_unpinned.chunks_removed

    def test_pin_patterns_none_backward_compatible(self):
        """pin_patterns=None should produce identical results to default."""
        result_default = optimize(REDUNDANT_PROMPT, target_ratio=0.30)
        result_none = optimize(REDUNDANT_PROMPT, target_ratio=0.30, pin_patterns=None)
        assert result_default.compressed == result_none.compressed
        assert result_default.chunks_removed == result_none.chunks_removed

    def test_pin_patterns_empty_list(self):
        """pin_patterns=[] should behave like None."""
        result_default = optimize(REDUNDANT_PROMPT, target_ratio=0.30)
        result_empty = optimize(REDUNDANT_PROMPT, target_ratio=0.30, pin_patterns=[])
        assert result_default.compressed == result_empty.compressed

    def test_pin_pattern_redos_rejected(self):
        """A nested-quantifier (ReDoS) pin pattern is rejected, not executed.

        The guard runs before re.compile, so the catastrophic regex never
        actually runs — this test is instant and does not hang.
        """
        from fiedler_optimizer.core import MAX_PIN_PATTERNS, MAX_PIN_PATTERN_LENGTH

        with pytest.raises(ValueError, match="catastrophic backtracking"):
            optimize(REDUNDANT_PROMPT, target_ratio=0.30, pin_patterns=["(a+)+$"])

        # Too many patterns is rejected with a clear error.
        with pytest.raises(ValueError, match="Too many pin patterns"):
            optimize(
                REDUNDANT_PROMPT, target_ratio=0.30,
                pin_patterns=["x"] * (MAX_PIN_PATTERNS + 1),
            )

        # An over-long single pattern is rejected with a clear error.
        with pytest.raises(ValueError, match="too long"):
            optimize(
                REDUNDANT_PROMPT, target_ratio=0.30,
                pin_patterns=["a" * (MAX_PIN_PATTERN_LENGTH + 1)],
            )

        # A normal pin pattern is unaffected (no false positive, still works).
        ok = optimize(REDUNDANT_PROMPT, target_ratio=0.30, pin_patterns=["verify"])
        assert isinstance(ok, FiedlerResult)

    def test_validate_pin_patterns_helper(self):
        """The centralized validate_pin_patterns() helper enforces the caps.

        Tests the single shared validator directly (it backs optimize(), the
        CLI --pin-regex flag, and the paid API), independent of compression.
        """
        from fiedler_optimizer.core import (
            validate_pin_patterns,
            MAX_PIN_PATTERNS,
            MAX_PIN_PATTERN_LENGTH,
        )

        # Over-length single pattern -> rejected, message names the cap.
        with pytest.raises(ValueError, match="too long"):
            validate_pin_patterns(["a" * (MAX_PIN_PATTERN_LENGTH + 1)])

        # Too many patterns -> rejected, message names the cap.
        with pytest.raises(ValueError, match="Too many pin patterns"):
            validate_pin_patterns(["x"] * (MAX_PIN_PATTERNS + 1))

        # Catastrophic-backtracking pattern -> rejected before it is ever run.
        with pytest.raises(ValueError, match="catastrophic backtracking"):
            validate_pin_patterns(["(a+)+$"])

        # Normal, in-bounds patterns pass validation unchanged (returns None,
        # raises nothing) — including a pattern at exactly the length cap and
        # the maximum allowed count.
        validate_pin_patterns(["verify", r"\bRULES?\b", "a" * MAX_PIN_PATTERN_LENGTH])
        validate_pin_patterns(["x"] * MAX_PIN_PATTERNS)
        validate_pin_patterns([])

    def test_section_aware_pinning(self):
        """Section-aware pinning should pin chunks under matching headers."""
        from fiedler_optimizer.pinning import section_pin_patterns
        patterns = section_pin_patterns(["Rules", "Safety"], PIN_TEST_PROMPT)
        assert len(patterns) > 0  # Should find patterns from Rules and Safety sections

        result = optimize(
            PIN_TEST_PROMPT,
            target_ratio=0.50,
            pin_patterns=patterns,
        )
        # Rules section content should survive
        assert "verify" in result.compressed
        assert "confidential" in result.compressed
        # Safety section should survive
        assert "medical" in result.compressed or "legal" in result.compressed
        # Should still achieve some compression from Background
        assert result.chunks_removed >= 1

    def test_instruction_preset_matches(self):
        """Built-in INSTRUCTION_PRESET should match expected patterns."""
        import re
        from fiedler_optimizer.pinning import INSTRUCTION_PRESET

        test_cases = [
            ("1. Always verify identity.", True),   # numbered rule
            ("2. Never share secrets.", True),       # numbered rule
            ("You must follow protocol.", True),     # constraint keyword
            ("Never bypass security.", True),        # constraint keyword
            ("Do not share credentials.", True),     # constraint keyword
            ("## Output Format", True),              # markdown header
            ("The company was founded in 2010.", False),  # plain text
            ("Hello world.", False),                 # plain text
        ]

        for text, should_match in test_cases:
            matched = any(
                re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                for pat in INSTRUCTION_PRESET
            )
            assert matched == should_match, (
                f"INSTRUCTION_PRESET {'should' if should_match else 'should not'} "
                f"match: {text!r}"
            )

    def test_pin_all_chunks_prevents_compression(self):
        """If every chunk is pinned, no chunks should be removed."""
        result = optimize(
            REDUNDANT_PROMPT,
            target_ratio=0.50,
            pin_patterns=[r"."],  # matches everything
        )
        assert result.chunks_removed == 0
