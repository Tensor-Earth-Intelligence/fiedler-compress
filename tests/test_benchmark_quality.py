"""
Tests for the compression quality evaluation harness.

Uses a mock LLM backend and mock dataset loaders to verify harness
mechanics without requiring real API calls or dataset downloads.
"""

from __future__ import annotations

import json

import pytest

from fiedler_optimizer.benchmarks.quality import (
    DATASETS,
    BenchmarkReport,
    BenchmarkRunner,
    LLMClient,
    SampleResult,
    _multiplier_to_target_ratio,
    exact_match,
    exact_match_number,
    f1_score,
    format_summary_table,
    report_to_json,
    rouge_l_score,
)


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Deterministic mock LLM that returns canned responses.

    If the prompt contains a key from ``responses``, returns its value.
    Otherwise returns ``default_response``.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "42",
    ) -> None:
        self._responses = responses or {}
        self._default = default_response
        self.call_count = 0
        self._model = "mock-model"

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        for key, value in self._responses.items():
            if key in prompt:
                return value
        return self._default


# ---------------------------------------------------------------------------
# Mock dataset samples
# ---------------------------------------------------------------------------

MOCK_GSM8K_SAMPLES = [
    {"id": "gsm8k_0", "prompt": "What is 2 + 3?", "ground_truth": "5"},
    {"id": "gsm8k_1", "prompt": "If a train travels 60 miles per hour for 2 hours, how far does it go?", "ground_truth": "120"},
    {"id": "gsm8k_2", "prompt": "A store has 15 apples and sells 7. How many are left?", "ground_truth": "8"},
]

MOCK_BBH_SAMPLES = [
    {"id": "bbh_0", "prompt": "Is (True and False) True or False?", "ground_truth": "False"},
    {"id": "bbh_1", "prompt": "Is (True or False) True or False?", "ground_truth": "True"},
    {"id": "bbh_2", "prompt": "Not (True) is?", "ground_truth": "False"},
]

MOCK_NQ_SAMPLES = [
    {"id": "nq_0", "prompt": "What is the capital of France?", "ground_truth": "Paris"},
    {"id": "nq_1", "prompt": "Who wrote Romeo and Juliet?", "ground_truth": "William Shakespeare"},
    {"id": "nq_2", "prompt": "What is the largest planet?", "ground_truth": "Jupiter"},
]


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------

class TestExactMatchNumber:
    def test_correct_integer(self):
        assert exact_match_number("The answer is 42", "42") == 1.0

    def test_correct_with_explanation(self):
        assert exact_match_number("I think it's about 120 miles total.", "120") == 1.0

    def test_wrong_number(self):
        assert exact_match_number("The answer is 43", "42") == 0.0

    def test_no_number_in_prediction(self):
        assert exact_match_number("I don't know", "42") == 0.0

    def test_no_number_in_truth(self):
        assert exact_match_number("42", "no number here") == 0.0

    def test_decimal(self):
        assert exact_match_number("It costs $3.50", "3.50") == 1.0

    def test_negative(self):
        assert exact_match_number("The result is -5", "-5") == 1.0

    def test_comma_separated(self):
        assert exact_match_number("Population: 1,000", "1000") == 1.0

    def test_last_number_used(self):
        assert exact_match_number("Step 1: 10, Step 2: 20, Final: 30", "30") == 1.0


class TestExactMatch:
    def test_exact(self):
        assert exact_match("True", "True") == 1.0

    def test_case_insensitive(self):
        assert exact_match("true", "True") == 1.0

    def test_whitespace_stripped(self):
        assert exact_match("  False  ", "False") == 1.0

    def test_wrong(self):
        assert exact_match("True", "False") == 0.0


class TestF1Score:
    def test_perfect(self):
        assert f1_score("Paris", "Paris") == 1.0

    def test_partial_overlap(self):
        score = f1_score("Paris is the capital of France", "Paris France")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert f1_score("London", "Paris") == 0.0

    def test_empty_both(self):
        assert f1_score("", "") == 1.0

    def test_empty_prediction(self):
        assert f1_score("", "Paris") == 0.0


class TestRougeLScore:
    def test_identical(self):
        assert rouge_l_score("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_partial(self):
        score = rouge_l_score("the cat sat on a mat", "the cat sat on the mat")
        assert 0.5 < score < 1.0

    def test_no_overlap(self):
        assert rouge_l_score("hello world", "foo bar baz") == 0.0

    def test_empty_reference(self):
        assert rouge_l_score("hello", "") == 0.0

    def test_empty_prediction(self):
        assert rouge_l_score("", "hello") == 0.0


# ---------------------------------------------------------------------------
# Multiplier conversion
# ---------------------------------------------------------------------------

class TestMultiplierConversion:
    def test_2x(self):
        assert _multiplier_to_target_ratio(2) == 0.5

    def test_4x(self):
        assert _multiplier_to_target_ratio(4) == 0.75

    def test_8x(self):
        assert _multiplier_to_target_ratio(8) == 0.875

    def test_1x(self):
        assert _multiplier_to_target_ratio(1) == 0.0

    def test_below_1_raises(self):
        with pytest.raises(ValueError, match="must be >= 1.0"):
            _multiplier_to_target_ratio(0.5)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class TestLLMClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            client._get_api_key()

    def test_custom_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "test-key-123")
        client = LLMClient(api_key_env="MY_CUSTOM_KEY")
        assert client._get_api_key() == "test-key-123"

    def test_api_key_never_in_source(self):
        """No API key is hardcoded in the quality module source."""
        import re
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "fiedler_optimizer" / "benchmarks" / "quality.py"
        content = src.read_text(encoding="utf-8")
        # Match an actual OpenAI-style key, not the substring "sk-" (which
        # legitimately appears inside words like "risk-adjusted").
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", content)
        assert "api_key = \"" not in content


# ---------------------------------------------------------------------------
# Benchmark runner with mocks
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    @pytest.fixture(autouse=True)
    def _mock_loader(self, monkeypatch):
        """Replace dataset loaders with mock data."""
        monkeypatch.setitem(DATASETS["gsm8k"], "loader",
                            lambda limit=None: MOCK_GSM8K_SAMPLES[:limit])
        monkeypatch.setitem(DATASETS["bbh"], "loader",
                            lambda limit=None: MOCK_BBH_SAMPLES[:limit])
        monkeypatch.setitem(DATASETS["natural_questions"], "loader",
                            lambda limit=None: MOCK_NQ_SAMPLES[:limit])

    def test_run_produces_report(self):
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=client)
        report = runner.run()

        assert isinstance(report, BenchmarkReport)
        assert report.dataset == "gsm8k"
        assert report.metric == "exact_match_number"
        assert report.n_samples == 3

    def test_results_per_sample_per_ratio(self):
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2, 4], llm_client=client)
        report = runner.run()

        # 3 samples × 2 ratios = 6 results
        assert len(report.results) == 6

    def test_limit_respected(self):
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=client, limit=1)
        report = runner.run()

        assert report.n_samples == 1
        assert len(report.results) == 1

    def test_correct_answer_scores_1(self):
        """Mock returns '5', ground truth for sample 0 is '5' → score 1.0."""
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=client, limit=1)
        report = runner.run()

        r = report.results[0]
        assert r.original_score == 1.0
        assert r.compressed_score == 1.0

    def test_wrong_answer_scores_0(self):
        """Mock returns '999', ground truth is '5' → score 0.0."""
        client = MockLLMClient(default_response="999")
        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=client, limit=1)
        report = runner.run()

        r = report.results[0]
        assert r.original_score == 0.0

    def test_summary_has_ratio_keys(self):
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2, 4], llm_client=client)
        report = runner.run()

        assert "2x" in report.summary
        assert "4x" in report.summary
        assert "n_samples" in report.summary["2x"]
        assert "mean_original_score" in report.summary["2x"]

    def test_llm_failure_does_not_abort(self):
        """An LLM that raises still produces results with score=0.0."""
        class FailingLLM:
            _model = "failing"
            def complete(self, prompt):
                raise RuntimeError("API down!")

        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=FailingLLM(), limit=1)
        with pytest.warns(match="LLM call failed"):
            report = runner.run()
        assert report.n_samples == 1

    def test_unknown_dataset_rejected(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            BenchmarkRunner("nonexistent", ratios=[2], llm_client=MockLLMClient())

    def test_bbh_exact_match(self):
        client = MockLLMClient(default_response="False")
        runner = BenchmarkRunner("bbh", ratios=[2], llm_client=client, limit=1)
        report = runner.run()
        assert report.metric == "exact_match"
        assert report.results[0].original_score == 1.0

    def test_nq_f1(self):
        client = MockLLMClient(default_response="Paris")
        runner = BenchmarkRunner("natural_questions", ratios=[2],
                                 llm_client=client, limit=1)
        report = runner.run()
        assert report.metric == "f1"
        assert report.results[0].original_score == 1.0

    def test_compression_metadata(self):
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=client, limit=1)
        report = runner.run()

        r = report.results[0]
        assert r.compression_multiplier == 2
        assert 0.0 <= r.compression_achieved <= 1.0
        assert r.compress_time_ms >= 0
        assert isinstance(r.tokens_saved, int)

    def test_llm_called_for_both_original_and_compressed(self):
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2], llm_client=client, limit=1)
        runner.run()
        # 1 original call + 1 compressed call = 2
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    @pytest.fixture()
    def sample_report(self, monkeypatch):
        monkeypatch.setitem(DATASETS["gsm8k"], "loader",
                            lambda limit=None: MOCK_GSM8K_SAMPLES[:limit])
        client = MockLLMClient(default_response="5")
        runner = BenchmarkRunner("gsm8k", ratios=[2, 4], llm_client=client, limit=2)
        return runner.run()

    def test_format_summary_table_nonempty(self, sample_report):
        table = format_summary_table(sample_report)
        assert len(table) > 100
        assert "FIEDLER" in table
        assert "gsm8k" in table
        assert "2x" in table

    def test_report_to_json_serializable(self, sample_report):
        data = report_to_json(sample_report)
        serialized = json.dumps(data)
        assert serialized
        parsed = json.loads(serialized)
        assert parsed["dataset"] == "gsm8k"
        assert "summary" in parsed
        assert "results" in parsed

    def test_report_to_json_structure(self, sample_report):
        data = report_to_json(sample_report)
        assert data["model"] == "mock-model"
        assert data["metric"] == "exact_match_number"
        assert len(data["results"]) == 4  # 2 samples × 2 ratios


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

class TestDatasetRegistry:
    def test_all_datasets_have_required_keys(self):
        for name, entry in DATASETS.items():
            assert "loader" in entry, f"{name} missing loader"
            assert "scorer" in entry, f"{name} missing scorer"
            assert "metric" in entry, f"{name} missing metric"
            assert callable(entry["loader"]), f"{name} loader not callable"
            assert callable(entry["scorer"]), f"{name} scorer not callable"

    def test_expected_datasets_present(self):
        expected = {
            "gsm8k", "bbh", "natural_questions", "meetingbank",
            "system_prompts", "agentic_contexts", "adversarial",
        }
        assert set(DATASETS.keys()) == expected
