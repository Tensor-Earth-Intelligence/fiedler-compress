"""Truncated generations must be excluded from scoring, not counted as wrong.

A generation that hits the model's token cap returns a truncated or empty
string. The scorer reads the last number in the output, so such a generation
scores as a wrong answer and is indistinguishable from a genuine failure. This
silently charged compression for failures it did not cause, three times in this
project's history.

Truncation does not fall equally on the two arms -- the compressed prompt is
shorter, so it truncates less often -- which means the error does not cancel
between baseline and compressed. Dropping the pair is the only treatment that
keeps the comparison paired.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fiedler_optimizer.benchmarks.quality")

from fiedler_optimizer.benchmarks.quality import (  # noqa: E402
    BenchmarkRunner,
    SampleResult,
)


def _result(sample_id, ratio, orig, comp, *, ot=False, ct=False):
    return SampleResult(
        sample_id=sample_id,
        compression_multiplier=ratio,
        target_ratio=0.5,
        compression_achieved=0.5,
        tokens_saved=10,
        original_score=orig,
        compressed_score=comp,
        score_delta=comp - orig,
        relative_quality=1.0,
        compress_time_ms=1.0,
        original_truncated=ot,
        compressed_truncated=ct,
    )


def _summarize(results):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)  # no LLM needed
    return runner._compute_summary(results)


def test_defaults_to_not_truncated():
    r = _result("a", 2, 1.0, 1.0)
    assert r.original_truncated is False
    assert r.compressed_truncated is False


def test_clean_results_are_all_kept():
    s = _summarize([_result(f"s{i}", 2, 1.0, 1.0) for i in range(4)])
    assert s["2x"]["n_samples"] == 4
    assert s["2x"]["n_excluded_truncated"] == 0


@pytest.mark.parametrize("ot,ct", [(True, False), (False, True), (True, True)])
def test_truncated_pair_is_excluded_either_arm(ot, ct):
    results = [_result(f"s{i}", 2, 1.0, 1.0) for i in range(3)]
    results.append(_result("bad", 2, 1.0, 0.0, ot=ot, ct=ct))
    s = _summarize(results)
    assert s["2x"]["n_excluded_truncated"] == 1
    assert s["2x"]["n_samples"] == 3
    # The truncated pair scored 0 compressed; excluding it must leave the mean clean.
    assert s["2x"]["mean_compressed_score"] == pytest.approx(1.0)
    assert s["2x"]["mean_score_delta"] == pytest.approx(0.0)


def test_exclusion_changes_the_reported_delta():
    """The whole point: including a truncated pair biases the delta downward."""
    clean = [_result(f"s{i}", 2, 1.0, 1.0) for i in range(3)]
    with_trunc = clean + [_result("bad", 2, 1.0, 0.0, ct=True)]

    kept = _summarize(clean)["2x"]["mean_score_delta"]
    excluded = _summarize(with_trunc)["2x"]["mean_score_delta"]
    assert kept == pytest.approx(excluded), (
        "excluding the truncated pair should reproduce the clean delta"
    )

    # And if it were NOT excluded, the delta would be visibly worse.
    naive = sum(r.score_delta for r in with_trunc) / len(with_trunc)
    assert naive < excluded, "test fixture no longer demonstrates the bias"


def test_exclusion_is_per_ratio():
    results = [
        _result("s0", 2, 1.0, 1.0),
        _result("s1", 2, 1.0, 0.0, ct=True),
        _result("s0", 8, 1.0, 1.0),
        _result("s1", 8, 1.0, 1.0),
    ]
    s = _summarize(results)
    assert s["2x"]["n_excluded_truncated"] == 1
    assert s["8x"]["n_excluded_truncated"] == 0


def test_all_truncated_falls_back_rather_than_reporting_zero_samples():
    """If everything truncated, reporting n=0 would look like a clean empty run."""
    results = [_result(f"s{i}", 2, 1.0, 0.0, ct=True) for i in range(3)]
    s = _summarize(results)
    assert s["2x"]["n_samples"] == 3
    assert s["2x"]["n_excluded_truncated"] == 0
