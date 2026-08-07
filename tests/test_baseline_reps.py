"""The uncompressed baseline must be averaged over reps, not a single draw.

Every compressed arm is scored against the same uncompressed score, so error in
that score is common-mode across ratios: a lucky draw shifts all of them the
same way and looks like a clean dose-response rather than noise.

This is not theoretical. On GSM8K, qwen2.5:7b scored 0.913 and then 0.853 on an
identical uncompressed re-run -- a 6-point swing with no compression involved,
the same size as its apparent 2x effect. Two of its three "significant" cells
did not survive correcting for it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fiedler_optimizer.benchmarks.quality")

from fiedler_optimizer.benchmarks.quality import BenchmarkRunner  # noqa: E402


class FlakyClient:
    """Returns a scripted sequence of answers, cycling once exhausted."""

    def __init__(self, answers, truncate_on=()):
        self._answers = list(answers)
        self._truncate_on = set(truncate_on)
        self.calls = 0
        self.last_truncated = False

    def complete(self, prompt: str) -> str:
        i = self.calls
        self.calls += 1
        self.last_truncated = i in self._truncate_on
        return self._answers[i % len(self._answers)]


def _runner(client, reps):
    r = BenchmarkRunner(
        dataset="gsm8k", ratios=[2], llm_client=client, baseline_reps=reps,
    )
    return r


def test_rejects_zero_reps():
    with pytest.raises(ValueError):
        BenchmarkRunner(dataset="gsm8k", ratios=[2],
                        llm_client=FlakyClient(["1"]), baseline_reps=0)


def test_default_is_three_reps():
    r = _runner(FlakyClient(["1"]), 3)
    assert r._baseline_reps == 3


def test_baseline_is_averaged_not_last_draw():
    # Baseline answers: right, wrong, right -> mean 2/3. Compressed: right.
    client = FlakyClient(["#### 7", "#### 0", "#### 7", "#### 7"])
    results = _runner(client, 3)._evaluate_sample(
        {"id": "s", "prompt": "Q: x\nA:", "ground_truth": "7"},
        lambda text, **kw: type("R", (), {
            "compressed": text, "tokens_saved": 0, "chunks_total": 4,
            "chunks_removed": 1, "compression_ratio": 0.5})(),
    )
    assert results[0].original_reps == 3
    # SampleResult rounds scores to 4dp, so match at that resolution.
    assert results[0].original_score == pytest.approx(2 / 3, abs=1e-4)


def test_single_rep_reproduces_old_behaviour():
    client = FlakyClient(["#### 7", "#### 0"])
    results = _runner(client, 1)._evaluate_sample(
        {"id": "s", "prompt": "Q: x\nA:", "ground_truth": "7"},
        lambda text, **kw: type("R", (), {
            "compressed": text, "tokens_saved": 0, "chunks_total": 4,
            "chunks_removed": 1, "compression_ratio": 0.5})(),
    )
    assert results[0].original_reps == 1
    assert results[0].original_score == pytest.approx(1.0)


def test_truncated_reps_are_dropped_not_averaged_as_wrong():
    # Second baseline rep truncates; it must not drag the mean toward zero.
    client = FlakyClient(["#### 7", "", "#### 7", "#### 7"], truncate_on={1})
    results = _runner(client, 3)._evaluate_sample(
        {"id": "s", "prompt": "Q: x\nA:", "ground_truth": "7"},
        lambda text, **kw: type("R", (), {
            "compressed": text, "tokens_saved": 0, "chunks_total": 4,
            "chunks_removed": 1, "compression_ratio": 0.5})(),
    )
    assert results[0].original_reps == 2, "truncated rep should be dropped"
    assert results[0].original_score == pytest.approx(1.0)
    assert results[0].original_truncated is True


def test_reps_cost_the_expected_number_of_calls():
    client = FlakyClient(["#### 7"])
    _runner(client, 3)._evaluate_sample(
        {"id": "s", "prompt": "Q: x\nA:", "ground_truth": "7"},
        lambda text, **kw: type("R", (), {
            "compressed": text, "tokens_saved": 0, "chunks_total": 4,
            "chunks_removed": 1, "compression_ratio": 0.5})(),
    )
    # 3 baseline reps + 1 compressed arm
    assert client.calls == 4
