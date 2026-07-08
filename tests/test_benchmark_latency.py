"""
Tests for the latency profiling harness.

Uses small input sizes and few runs to verify profiler correctness
without requiring large memory or long execution times.
"""

from __future__ import annotations

import json

import pytest

from fiedler_optimizer.benchmarks.latency import (
    DEFAULT_SIZES,
    LatencyProfiler,
    LatencyReport,
    SizeResult,
    StageTimings,
    _percentile,
    format_latency_table,
    generate_corpus,
    report_to_json,
)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

class TestGenerateCorpus:
    def test_generates_nonempty_text(self):
        text = generate_corpus(100)
        assert len(text) > 0

    def test_approximate_token_count(self):
        text = generate_corpus(500)
        # ~4 chars per token, allow ±50% tolerance
        approx_tokens = len(text) / 4
        assert 250 < approx_tokens < 1000

    def test_realistic_content(self):
        """Generated text contains real English words, not random chars."""
        text = generate_corpus(200)
        words = text.split()
        # Should have recognizable English words
        assert any(w.lower() in ("the", "and", "for", "with", "using")
                    for w in words)

    def test_deterministic_with_seed(self):
        t1 = generate_corpus(200, seed=42)
        t2 = generate_corpus(200, seed=42)
        assert t1 == t2

    def test_different_seeds_different_output(self):
        t1 = generate_corpus(200, seed=1)
        t2 = generate_corpus(200, seed=2)
        assert t1 != t2

    def test_scales_to_target(self):
        small = generate_corpus(100)
        large = generate_corpus(1000)
        assert len(large) > len(small)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_median_of_sorted(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_p95(self):
        data = list(range(100))
        p95 = _percentile(data, 95)
        assert 93 < p95 < 96

    def test_p99(self):
        data = list(range(100))
        p99 = _percentile(data, 99)
        assert 97 < p99 < 100

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 95) == 42.0


# ---------------------------------------------------------------------------
# StageTimings
# ---------------------------------------------------------------------------

class TestStageTimings:
    def test_mean(self):
        st = StageTimings("test", (10.0, 20.0, 30.0))
        assert st.mean == 20.0

    def test_median(self):
        st = StageTimings("test", (10.0, 20.0, 100.0))
        assert st.median == 20.0

    def test_empty(self):
        st = StageTimings("test", ())
        assert st.mean == 0.0
        assert st.median == 0.0
        assert st.p95 == 0.0


# ---------------------------------------------------------------------------
# Profiler execution — small inputs only
# ---------------------------------------------------------------------------

class TestLatencyProfiler:
    def test_runs_on_small_input(self):
        profiler = LatencyProfiler(sizes=[100], runs=2)
        report = profiler.run()

        assert isinstance(report, LatencyReport)
        assert len(report.results) == 1
        assert report.runs_per_size == 2
        assert report.backend == "tfidf"

    def test_result_has_all_stages(self):
        profiler = LatencyProfiler(sizes=[100], runs=2)
        report = profiler.run()
        r = report.results[0]

        stage_names = {s.name for s in r.stages}
        assert "chunking" in stage_names
        assert "similarity" in stage_names
        assert "eigendecomposition" in stage_names
        assert "scoring" in stage_names
        assert "zone_detection" in stage_names
        assert "compression" in stage_names

    def test_timings_are_positive(self):
        profiler = LatencyProfiler(sizes=[100], runs=3)
        report = profiler.run()
        r = report.results[0]

        for stage in r.stages:
            assert len(stage.times_ms) == 3
            for t in stage.times_ms:
                assert t >= 0

        assert r.total.median > 0

    def test_multiple_sizes(self):
        profiler = LatencyProfiler(sizes=[100, 500], runs=2)
        report = profiler.run()

        assert len(report.results) == 2
        assert report.results[0].target_tokens == 100
        assert report.results[1].target_tokens == 500

    def test_chunk_count_recorded(self):
        profiler = LatencyProfiler(sizes=[200], runs=1)
        report = profiler.run()
        r = report.results[0]
        assert r.chunk_count > 0

    def test_memory_peak_recorded(self):
        profiler = LatencyProfiler(sizes=[100], runs=1)
        report = profiler.run()
        r = report.results[0]
        assert r.memory_peak_bytes > 0

    def test_actual_chars_recorded(self):
        profiler = LatencyProfiler(sizes=[100], runs=1)
        report = profiler.run()
        r = report.results[0]
        assert r.actual_chars > 0

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            LatencyProfiler(sizes=[100], backend="invalid")

    def test_invalid_runs_rejected(self):
        with pytest.raises(ValueError, match="runs"):
            LatencyProfiler(sizes=[100], runs=0)

    def test_larger_input_takes_longer(self):
        """Sanity check: 1000 tokens should be slower than 100 tokens."""
        profiler = LatencyProfiler(sizes=[100, 1000], runs=3)
        report = profiler.run()

        small = report.results[0].total.median
        large = report.results[1].total.median
        # Large should generally take at least as long (allow some noise)
        assert large >= small * 0.5  # very relaxed bound for CI stability


# ---------------------------------------------------------------------------
# Eigendecomposition warnings
# ---------------------------------------------------------------------------

class TestEigenWarnings:
    def test_no_warning_for_fast_eigen(self):
        profiler = LatencyProfiler(sizes=[100], runs=1)
        report = profiler.run()
        r = report.results[0]
        # 100 tokens should be well under 1s
        assert len(r.warnings) == 0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    @pytest.fixture()
    def report(self):
        profiler = LatencyProfiler(sizes=[100, 500], runs=2)
        return profiler.run()

    def test_format_table_nonempty(self, report):
        table = format_latency_table(report)
        assert len(table) > 100
        assert "FIEDLER LATENCY PROFILE" in table
        assert "Tokens" in table

    def test_format_table_contains_sizes(self, report):
        table = format_latency_table(report)
        assert "100" in table
        assert "500" in table

    def test_format_table_has_breakdown(self, report):
        table = format_latency_table(report)
        assert "PER-STAGE BREAKDOWN" in table
        assert "Eigen" in table

    def test_report_to_json_serializable(self, report):
        data = report_to_json(report)
        serialized = json.dumps(data)
        assert serialized
        parsed = json.loads(serialized)
        assert parsed["backend"] == "tfidf"
        assert len(parsed["results"]) == 2

    def test_report_to_json_has_statistics(self, report):
        data = report_to_json(report)
        r0 = data["results"][0]
        assert "stages" in r0
        eigen = r0["stages"]["eigendecomposition"]
        assert "mean" in eigen
        assert "median" in eigen
        assert "p95" in eigen
        assert "p99" in eigen
        assert "times_ms" in eigen
        assert len(eigen["times_ms"]) == 2

    def test_report_to_json_has_total(self, report):
        data = report_to_json(report)
        r0 = data["results"][0]
        assert "total" in r0
        assert "mean" in r0["total"]

    def test_report_to_json_has_memory(self, report):
        data = report_to_json(report)
        r0 = data["results"][0]
        assert "memory_peak_bytes" in r0
        assert r0["memory_peak_bytes"] > 0


# ---------------------------------------------------------------------------
# Default sizes
# ---------------------------------------------------------------------------

class TestDefaultSizes:
    def test_default_sizes_tuple(self):
        assert isinstance(DEFAULT_SIZES, tuple)
        assert len(DEFAULT_SIZES) == 8

    def test_default_sizes_sorted(self):
        assert list(DEFAULT_SIZES) == sorted(DEFAULT_SIZES)

    def test_default_sizes_positive(self):
        assert all(s > 0 for s in DEFAULT_SIZES)
