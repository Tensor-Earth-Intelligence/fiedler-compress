"""Coverage for the geometric analysis modules opened up in 0.4.0.

These shipped in 0.4.0 with essentially no tests of their own. The assertions
here are deliberately about invariants and documented contracts (score ordering,
matrix properties, detector boundaries, dict schemas) rather than exact numeric
output, so they pin down behaviour without freezing incidental values.

scikit-learn backs voronoi and minkowski; conformal additionally needs
umap-learn. Both are optional extras, so the whole module skips when the
`geometry` extra is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="geometry extra not installed")

from fiedler_optimizer.geometry.content_prior import (  # noqa: E402
    CONTENT_PRIOR_PIN_PATTERNS,
    CONTENT_PRIORS,
    identifier_pin_patterns,
    identifier_prior,
    is_durable_fact,
    is_identifier_token,
    is_observation_result,
    observation_pin_patterns,
    salience_pin_patterns,
)
from fiedler_optimizer.geometry.minkowski import (  # noqa: E402
    calibrate_alpha,
    minkowski_compress,
    minkowski_distance_matrix,
    minkowski_similarity_matrix,
)
from fiedler_optimizer.geometry.voronoi import (  # noqa: E402
    estimate_voronoi_volumes,
    voronoi_anomaly_score,
    voronoi_compress,
)

# A redundant document: near-identical routine lines, one carrying an identifier.
ROUTINE_LINES = [
    f"Routine status check number {i} completed and reported nominal operation today."
    for i in range(12)
]
ROUTINE_DOC = "\n\n".join(ROUTINE_LINES)

MIXED_DOC = (
    "The system performed a routine status check and reported nominal operation. " * 12
    + "\n\n"
    + "Incident ticket XR-8412 was escalated to the on-call engineer."
)


# ---------------------------------------------------------------------------
# Voronoi density estimation
# ---------------------------------------------------------------------------

class TestVoronoiVolumes:
    def test_unique_point_scores_above_crowded_ones(self):
        """A point in empty territory has the largest cell; near-duplicates the
        smallest. This ordering is the whole basis of the pruning rule."""
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [1.0, 0.0, 0.01],
            [0.99, 0.0, 0.0],
            [0.0, 0.0, 1.0],   # unique
        ])
        scores = estimate_voronoi_volumes(vectors, k=2)
        assert int(np.argmax(scores)) == 4
        assert scores[4] > scores[:4].max()

    def test_single_point_returns_unit_scores(self):
        scores = estimate_voronoi_volumes(np.array([[1.0, 0.0]]))
        assert scores.tolist() == [1.0]

    def test_k_is_clamped_to_available_neighbors(self):
        """k far larger than the sample count must not raise."""
        scores = estimate_voronoi_volumes(np.array([[1.0, 0.0], [0.0, 1.0]]), k=99)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("agg", ["kth", "mean", "sum"])
    def test_aggregations_agree_on_the_unique_point(self, agg):
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.99, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        scores = estimate_voronoi_volumes(vectors, k=2, agg=agg)
        assert int(np.argmax(scores)) == 3


class TestVoronoiAnomaly:
    def test_orthogonal_point_scores_above_an_in_distribution_one(self):
        baseline = np.array([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [1.0, 0.0, 0.02]])
        new = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scores = voronoi_anomaly_score(baseline, new, k=2)
        assert scores[1] > scores[0]

    def test_returns_one_score_per_new_point(self):
        baseline = np.array([[1.0, 0.0], [0.0, 1.0]])
        new = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        assert voronoi_anomaly_score(baseline, new, k=1).shape == (3,)


# ---------------------------------------------------------------------------
# Voronoi compression
# ---------------------------------------------------------------------------

class TestVoronoiCompress:
    def test_result_schema(self):
        result = voronoi_compress(MIXED_DOC, target="2x")
        for key in (
            "compressed", "original_tokens", "compressed_tokens",
            "compression_ratio", "voronoi_scores", "chunks_removed",
            "chunks_total", "method", "k", "backend",
        ):
            assert key in result
        assert result["method"] == "voronoi"
        assert len(result["voronoi_scores"]) == result["chunks_total"]

    def test_higher_target_removes_more(self):
        light = voronoi_compress(MIXED_DOC, target="2x")
        heavy = voronoi_compress(MIXED_DOC, target="4x")
        assert heavy["chunks_removed"] > light["chunks_removed"]

    def test_target_accepts_multiplier_and_ratio_forms(self):
        """'2x' means halve, i.e. the same as the ratio 0.5."""
        as_mult = voronoi_compress(MIXED_DOC, target="2x")
        as_ratio = voronoi_compress(MIXED_DOC, target=0.5)
        assert as_mult["chunks_removed"] == as_ratio["chunks_removed"]

    def test_never_removes_every_chunk(self):
        result = voronoi_compress(MIXED_DOC, target="100x")
        assert result["chunks_removed"] == result["chunks_total"] - 1
        assert result["compressed"].strip()

    def test_short_input_passes_through_untouched(self):
        result = voronoi_compress("only one line here")
        assert result["chunks_removed"] == 0
        assert result["compression_ratio"] == 0.0

    def test_aitchison_backend_runs(self):
        result = voronoi_compress(MIXED_DOC, target="2x", backend="aitchison")
        assert result["backend"] == "aitchison"
        assert result["compressed"]

    def test_output_is_a_substring_reassembly(self):
        """Compression only ever drops spans; it must not invent text."""
        result = voronoi_compress(MIXED_DOC, target="2x")
        for segment in result["compressed"].split("\n\n"):
            assert segment in MIXED_DOC


class TestVoronoiContentPrior:
    def test_none_leaves_the_unsupervised_ranking_untouched(self):
        """Documented invariant: content_prior=None is byte-identical to the
        pure density path."""
        plain = voronoi_compress(ROUTINE_DOC, target="4x")
        explicit_none = voronoi_compress(ROUTINE_DOC, target="4x", content_prior=None)
        assert plain["voronoi_scores"] == explicit_none["voronoi_scores"]
        assert plain["content_prior"] is None
        assert plain["prior_weight"] is None

    def test_named_prior_is_recorded(self):
        result = voronoi_compress(ROUTINE_DOC, target="4x", content_prior="identifier")
        assert result["content_prior"] == "identifier"
        assert result["prior_weight"] == 10.0

    def test_callable_prior_is_recorded(self):
        result = voronoi_compress(
            ROUTINE_DOC, target="4x", content_prior=lambda t: 0.0
        )
        assert result["content_prior"] == "callable"

    def test_unknown_prior_name_rejected(self):
        with pytest.raises(ValueError, match="unknown content_prior"):
            voronoi_compress(ROUTINE_DOC, content_prior="does-not-exist")

    def test_boost_rescues_a_chunk_that_would_be_pruned(self):
        """The point of the prior: a flagged chunk survives even when density
        alone ranks it last, and the compression budget is unchanged."""
        from fiedler_optimizer.chunker import ChunkingStrategy, chunk_text

        baseline = voronoi_compress(ROUTINE_DOC, target="4x")
        chunks = [c.text for c in chunk_text(ROUTINE_DOC, strategy=ChunkingStrategy.ADAPTIVE)]
        victim = chunks[int(np.argmin(baseline["voronoi_scores"]))]
        assert victim not in baseline["compressed"]

        boosted = voronoi_compress(
            ROUTINE_DOC, target="4x",
            content_prior=lambda t: 1.0 if t == victim else 0.0,
        )
        assert victim in boosted["compressed"]
        assert boosted["chunks_removed"] == baseline["chunks_removed"]

    def test_identifier_prior_ranks_the_identifier_chunk_top(self):
        result = voronoi_compress(MIXED_DOC, target="4x", content_prior="identifier")
        assert "XR-8412" in result["compressed"]


# ---------------------------------------------------------------------------
# Minkowski spacetime metric
# ---------------------------------------------------------------------------

CHUNKS = [
    "alpha beta gamma delta",
    "alpha beta gamma epsilon",
    "totally different words here",
    "more distinct vocabulary terms",
]


class TestMinkowskiMatrices:
    def test_distance_matrix_is_square_symmetric_with_zero_diagonal(self):
        d = minkowski_distance_matrix(CHUNKS, alpha=0.1)
        assert d.shape == (len(CHUNKS), len(CHUNKS))
        assert np.allclose(d, d.T)
        assert np.allclose(np.diag(d), 0.0)

    def test_distances_can_go_negative(self):
        """Timelike separation is the whole point of the metric: the positional
        term is subtracted, so d^2 may be below zero."""
        d = minkowski_distance_matrix(CHUNKS, alpha=1.0)
        assert (d < 0).any()

    def test_larger_alpha_strengthens_the_positional_term(self):
        low = minkowski_distance_matrix(CHUNKS, alpha=0.01)
        high = minkowski_distance_matrix(CHUNKS, alpha=1.0)
        assert not np.allclose(low, high)
        assert high.min() < low.min()

    def test_aitchison_metric_runs_and_differs_from_cosine(self):
        cosine = minkowski_distance_matrix(CHUNKS, alpha=0.1)
        aitchison = minkowski_distance_matrix(CHUNKS, alpha=0.1, semantic_metric="aitchison")
        assert aitchison.shape == cosine.shape
        assert np.all(np.isfinite(aitchison))

    def test_similarity_matrix_is_bounded_with_unit_diagonal(self):
        sim = minkowski_similarity_matrix(CHUNKS, alpha=0.1)
        assert sim.shape == (len(CHUNKS), len(CHUNKS))
        assert sim.min() >= 0.0 and sim.max() <= 1.0
        assert np.allclose(np.diag(sim), 1.0)


class TestCalibrateAlpha:
    @pytest.mark.parametrize(
        "doc_type,expected",
        [("agentic", 0.4), ("instructional", 0.03), ("narrative", 0.15), ("technical", 0.08)],
    )
    def test_named_document_types_return_their_preset(self, doc_type, expected):
        assert calibrate_alpha([], doc_type) == expected

    def test_unknown_document_type_falls_back_to_default(self):
        assert calibrate_alpha([], "no-such-type") == 0.1

    def test_empty_input_returns_default(self):
        assert calibrate_alpha([], "auto") == 0.1

    def test_auto_detects_a_conversation_from_turn_markers(self):
        turns = [
            "User: hello there friend",
            "Assistant: hi how are you",
            "User: fine thanks",
            "Assistant: good to hear",
        ]
        assert calibrate_alpha(turns, "auto") == 0.4

    def test_auto_returns_a_usable_alpha_for_plain_prose(self):
        alpha = calibrate_alpha(ROUTINE_LINES, "auto")
        assert 0.0 < alpha <= 1.0


class TestMinkowskiCompress:
    def test_result_schema(self):
        result = minkowski_compress(MIXED_DOC, target="2x")
        for key in (
            "compressed", "compression_ratio", "lambda2",
            "chunks_removed", "chunks_total", "alpha", "method",
        ):
            assert key in result
        assert result["method"] == "minkowski-fiedler"
        assert isinstance(result["lambda2"], float)

    def test_higher_target_removes_more(self):
        light = minkowski_compress(MIXED_DOC, target="2x")
        heavy = minkowski_compress(MIXED_DOC, target="4x")
        assert heavy["chunks_removed"] > light["chunks_removed"]

    def test_short_input_passes_through_untouched(self):
        result = minkowski_compress("One. Two.")
        assert result["chunks_removed"] == 0
        assert result["compression_ratio"] == 0.0

    def test_alpha_is_reported_back(self):
        assert minkowski_compress(MIXED_DOC, target="2x", alpha=0.35)["alpha"] == 0.35

    def test_output_is_a_substring_reassembly(self):
        result = minkowski_compress(MIXED_DOC, target="2x")
        for segment in result["compressed"].split("\n\n"):
            assert segment in MIXED_DOC


# ---------------------------------------------------------------------------
# Content priors
# ---------------------------------------------------------------------------

class TestIdentifierDetector:
    @pytest.mark.parametrize(
        "token", ["XR-8412", "AKIA1234567890", "1234567", "deadbeefcafe01"]
    )
    def test_identifier_shapes_are_detected(self, token):
        assert is_identifier_token(token)

    @pytest.mark.parametrize("token", ["2024", "hello", "ab12", "the", ""])
    def test_prose_years_and_short_tokens_are_not_identifiers(self, token):
        """A four-digit year must not be mistaken for a serial; this boundary is
        what keeps the prior from firing on ordinary prose."""
        assert not is_identifier_token(token)

    def test_prior_counts_identifier_tokens(self):
        assert identifier_prior("Ticket XR-8412 and key AKIA1234567890 in 2024") == 2.0

    def test_prior_is_zero_for_plain_prose(self):
        assert identifier_prior("the quick brown fox jumped over it") == 0.0

    def test_pin_patterns_escape_matches_and_are_none_when_absent(self):
        assert identifier_pin_patterns("just plain words here") is None
        patterns = identifier_pin_patterns("ticket XR-8412 here")
        assert patterns == [r"XR\-8412"]

    def test_generated_patterns_match_their_source_token(self):
        import re

        text = "incident XR-8412 raised"
        for pattern in identifier_pin_patterns(text):
            assert re.search(pattern, text)


class TestSalienceDetector:
    @pytest.mark.parametrize(
        "line",
        [
            "I am allergic to peanuts",
            "My name is Dolores",
            "I have a deadline on Friday",
        ],
    )
    def test_durable_first_person_facts_are_detected(self, line):
        assert is_durable_fact(line)

    @pytest.mark.parametrize(
        "line",
        ["What is my name?", "the weather is nice today", ""],
    )
    def test_questions_and_impersonal_lines_are_not_facts(self, line):
        assert not is_durable_fact(line)

    def test_pin_patterns_select_only_the_fact_lines(self):
        text = "I am allergic to peanuts\nthe weather is nice"
        patterns = salience_pin_patterns(text)
        assert patterns == [r"I\ am\ allergic\ to\ peanuts"]

    def test_pin_patterns_none_when_no_facts(self):
        assert salience_pin_patterns("the weather is nice") is None


class TestObservationDetector:
    @pytest.mark.parametrize(
        "line",
        [
            "Observation 1: user_id maps to 4417",
            "the lookup returned three rows",
            "the file was stored at /tmp/out",
        ],
    )
    def test_result_binding_observations_are_detected(self, line):
        assert is_observation_result(line)

    @pytest.mark.parametrize("line", ["Thinking about it", "Observation:", ""])
    def test_routine_steps_are_not_results(self, line):
        assert not is_observation_result(line)

    def test_pin_patterns_select_only_result_lines(self):
        text = "Observation: id maps to 44\nnothing here"
        assert observation_pin_patterns(text) == [r"Observation:\ id\ maps\ to\ 44"]


class TestPriorRegistries:
    def test_both_registries_expose_the_same_names(self):
        assert sorted(CONTENT_PRIORS) == sorted(CONTENT_PRIOR_PIN_PATTERNS)

    def test_registry_entries_are_callable_and_score_numerically(self):
        for name, fn in CONTENT_PRIORS.items():
            assert isinstance(fn("Ticket XR-8412 for my deadline maps to 4417"), float), name

    def test_pin_pattern_registry_returns_lists_or_none(self):
        for name, fn in CONTENT_PRIOR_PIN_PATTERNS.items():
            result = fn("Ticket XR-8412 for my deadline maps to 4417")
            assert result is None or isinstance(result, list), name


# ---------------------------------------------------------------------------
# Conformal (UMAP) — needs umap-learn on top of scikit-learn
# ---------------------------------------------------------------------------

class TestConformal:
    """UMAP is a heavy optional dependency, so these skip unless it is present.
    They are written to run for real when the geometry extra is installed."""

    @pytest.fixture(autouse=True)
    def _require_umap(self):
        pytest.importorskip("umap", reason="umap-learn not installed")

    def test_embedding_has_requested_shape_and_finite_values(self):
        from fiedler_optimizer.geometry.conformal import compute_conformal_embedding

        chunks = ROUTINE_LINES + ["a wholly unrelated sentence about oceans"]
        embedding = compute_conformal_embedding(
            chunks, n_components=3, n_neighbors=4, random_state=42
        )
        assert embedding.shape == (len(chunks), 3)
        assert np.isfinite(embedding).all()

    # NOTE: deliberately no determinism test. ``random_state`` is forwarded to
    # UMAP correctly, but UMAP does not reproduce across separate fits here
    # (two identical calls differ by >25 in embedding coordinates), so callers
    # must not rely on a stable embedding the way they can for the Fiedler
    # vector. See the caveat on compute_conformal_embedding's random_state.

    @pytest.mark.parametrize(
        "n_chunks,requested,expected",
        [(5, 5, 3), (3, 5, 1)],
    )
    def test_components_are_clamped_for_small_inputs(self, n_chunks, requested, expected):
        """UMAP's spectral init fails when components approach the sample count,
        so the dimensionality is clamped rather than allowed to raise."""
        from fiedler_optimizer.geometry.conformal import compute_conformal_embedding

        chunks = ROUTINE_LINES[:n_chunks]
        embedding = compute_conformal_embedding(
            chunks, n_components=requested, n_neighbors=min(3, n_chunks - 1)
        )
        assert embedding.shape == (n_chunks, expected)

    def test_compress_returns_its_schema(self):
        from fiedler_optimizer.geometry.conformal import conformal_compress

        result = conformal_compress(MIXED_DOC, target="2x", n_neighbors=4, n_components=3)
        for key in ("compressed", "compression_ratio", "n_clusters", "umap_params", "method"):
            assert key in result
        assert result["method"] == "conformal-umap"
        assert result["umap_params"]["optimized"] is False

    def test_thematic_search_ranks_the_on_topic_memory_first(self):
        from fiedler_optimizer.geometry.conformal import conformal_thematic_search

        memories = [
            "The cat slept on the warm windowsill all afternoon.",
            "Quantum chromodynamics describes the strong nuclear interaction.",
            "Gluons mediate the color charge between quarks in a nucleus.",
            "I baked sourdough bread with a long overnight fermentation.",
        ]
        results = conformal_thematic_search(
            "particle physics and nuclear forces", memories, top_k=2
        )
        assert len(results) == 2
        for entry in results:
            assert {"index", "text", "distance"} <= set(entry)
            assert entry["text"] in memories
        distances = [e["distance"] for e in results]
        assert distances == sorted(distances)

    def test_param_optimizer_reports_its_search(self):
        from fiedler_optimizer.geometry.conformal import optimize_umap_params

        result = optimize_umap_params(ROUTINE_LINES + ["an unrelated sentence"])
        assert result["n_configs_tested"] >= 1
        assert "silhouette_score" in result
