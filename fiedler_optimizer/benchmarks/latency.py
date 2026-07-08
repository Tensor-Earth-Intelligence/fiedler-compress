"""
Latency profiling harness for the Fiedler compression pipeline.

Measures wall-clock time and memory usage per pipeline stage across
configurable input sizes.  Identifies the O(n³) eigendecomposition
bottleneck with explicit warnings.

Usage::

    from fiedler_optimizer.benchmarks.latency import LatencyProfiler

    profiler = LatencyProfiler(sizes=[100, 1000, 10000], runs=5)
    report = profiler.run()
    print(format_latency_table(report))
"""

from __future__ import annotations

import math
import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from fiedler_optimizer.chunker import ChunkingStrategy, Chunk, chunk_text
from fiedler_optimizer.graph import (
    MAX_CHUNKS,
    _compute_tfidf_matrix,
    build_similarity_graph,
    compute_chunk_scores,
    compute_fiedler_vector,
)
from fiedler_optimizer.zones import detect_zones


# ---------------------------------------------------------------------------
# Built-in sample corpus (realistic English paragraphs)
# ---------------------------------------------------------------------------

_SAMPLE_PARAGRAPHS = [
    (
        "Machine learning models require careful evaluation across multiple "
        "dimensions including accuracy, latency, memory consumption, and "
        "fairness. A model that achieves high accuracy but consumes excessive "
        "memory may be unsuitable for edge deployment scenarios."
    ),
    (
        "The database schema uses a normalized design with foreign key "
        "constraints ensuring referential integrity. Indexes are placed on "
        "frequently queried columns to minimize scan operations during peak "
        "traffic periods."
    ),
    (
        "Authentication tokens expire after 24 hours and must be refreshed "
        "using the refresh token endpoint. Rate limiting is applied at 100 "
        "requests per minute per user to prevent abuse and ensure fair "
        "resource allocation across tenants."
    ),
    (
        "The deployment pipeline consists of three stages: continuous "
        "integration with automated testing, staging environment validation "
        "with smoke tests, and production rollout using blue-green deployment "
        "strategy to minimize downtime."
    ),
    (
        "Graph spectral methods decompose the adjacency matrix into "
        "eigenvalues and eigenvectors. The Fiedler vector, corresponding to "
        "the second-smallest Laplacian eigenvalue, provides an optimal "
        "bipartition of the graph for community detection."
    ),
    (
        "Natural language processing pipelines typically involve tokenization, "
        "part-of-speech tagging, named entity recognition, and dependency "
        "parsing. Modern transformer architectures handle these tasks with "
        "a single model trained on large corpora."
    ),
    (
        "Financial regulations require quarterly reporting of all derivative "
        "positions exceeding the notional threshold. Counterparty credit risk "
        "must be assessed using standardized approaches or internal models "
        "approved by the regulator."
    ),
    (
        "The microservices architecture uses an API gateway for request "
        "routing, circuit breakers for fault tolerance, and distributed "
        "tracing for observability. Service mesh provides mutual TLS "
        "authentication between all internal services."
    ),
    (
        "Climate models simulate atmospheric dynamics using partial "
        "differential equations discretized on a three-dimensional grid. "
        "Resolution improvements increase computational cost cubically, "
        "requiring high-performance computing clusters."
    ),
    (
        "User experience research methods include contextual inquiry, "
        "usability testing, card sorting, and A/B testing. Quantitative "
        "metrics such as task completion rate and time on task complement "
        "qualitative insights from think-aloud protocols."
    ),
    (
        "Supply chain optimization involves demand forecasting, inventory "
        "management, and logistics planning. Stochastic models account for "
        "uncertainty in lead times and demand variability to minimize "
        "stockout risk while controlling carrying costs."
    ),
    (
        "The compiler performs lexical analysis, parsing, semantic analysis, "
        "intermediate code generation, optimization, and target code "
        "generation. Each phase transforms the program representation "
        "bringing it closer to executable machine instructions."
    ),
]


def generate_corpus(target_tokens: int, seed: int = 42) -> str:
    """Generate realistic English text of approximately *target_tokens* tokens.

    Uses a built-in corpus of technical paragraphs, repeated and shuffled
    to achieve the target size.  One token is approximately 4 characters.

    Parameters
    ----------
    target_tokens : int
        Approximate number of tokens in the generated text.
    seed : int
        Random seed for reproducibility.
    """
    rng = random.Random(seed)
    target_chars = target_tokens * 4

    paragraphs = list(_SAMPLE_PARAGRAPHS)
    result_parts: list[str] = []
    total_chars = 0

    while total_chars < target_chars:
        rng.shuffle(paragraphs)
        for para in paragraphs:
            result_parts.append(para)
            total_chars += len(para) + 2  # account for \n\n separator
            if total_chars >= target_chars:
                break

    return "\n\n".join(result_parts)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageTimings:
    """Wall-clock timings for a single pipeline stage across runs."""

    name: str
    times_ms: tuple[float, ...]
    """Individual run times in milliseconds."""

    @property
    def mean(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def p95(self) -> float:
        return _percentile(self.times_ms, 95)

    @property
    def p99(self) -> float:
        return _percentile(self.times_ms, 99)


@dataclass(frozen=True)
class SizeResult:
    """Profiling result for one input size."""

    target_tokens: int
    actual_chars: int
    chunk_count: int
    stages: tuple[StageTimings, ...]
    total: StageTimings
    memory_peak_bytes: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LatencyReport:
    """Full latency profiling report."""

    sizes: tuple[int, ...]
    runs_per_size: int
    backend: str
    results: tuple[SizeResult, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(data: Sequence[float], pct: float) -> float:
    """Compute the *pct*-th percentile of *data*."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _ms(seconds: float) -> float:
    """Convert seconds to milliseconds, rounded."""
    return round(seconds * 1000, 3)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

# Default sizes to profile (in tokens)
DEFAULT_SIZES = (100, 500, 1000, 5000, 10000, 25000, 50000, 100000)


class LatencyProfiler:
    """Profiles each pipeline stage across configurable input sizes.

    Parameters
    ----------
    sizes : Sequence[int]
        Input sizes in tokens to profile.
    runs : int
        Number of repetitions per size for statistical robustness.
    backend : str
        Similarity backend (``"tfidf"``).
    target_ratio : float
        Compression target ratio (fraction to remove).
    strategy : ChunkingStrategy
        Chunking strategy.
    """

    def __init__(
        self,
        sizes: Sequence[int] = DEFAULT_SIZES,
        runs: int = 10,
        backend: str = "tfidf",
        target_ratio: float = 0.20,
        strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
    ) -> None:
        if backend != "tfidf":
            raise ValueError(f"backend must be 'tfidf', got {backend!r}")
        if runs < 1:
            raise ValueError("runs must be >= 1")

        self._sizes = tuple(sizes)
        self._runs = runs
        self._backend = backend
        self._target_ratio = target_ratio
        self._strategy = strategy

    def run(self) -> LatencyReport:
        """Run the latency profiling harness."""
        results: list[SizeResult] = []

        for size in self._sizes:
            # Skip sizes that would exceed MAX_CHUNKS for realistic chunking.
            # Estimate: ~50 chars/chunk for sentence strategy, ~200 for paragraph.
            # Conservative: skip if token count would produce > MAX_CHUNKS chunks.
            result = self._profile_size(size)
            results.append(result)

        return LatencyReport(
            sizes=self._sizes,
            runs_per_size=self._runs,
            backend=self._backend,
            results=tuple(results),
        )

    def _profile_size(self, target_tokens: int) -> SizeResult:
        """Profile all stages for a single input size."""
        text = generate_corpus(target_tokens)

        # Pre-run once to get chunk count and check feasibility
        chunks = chunk_text(text, strategy=self._strategy)
        chunk_count = len(chunks)

        if chunk_count > MAX_CHUNKS:
            # Skip profiling — too many chunks. Record as a warning.
            return SizeResult(
                target_tokens=target_tokens,
                actual_chars=len(text),
                chunk_count=chunk_count,
                stages=(),
                total=StageTimings("total", ()),
                memory_peak_bytes=0,
                warnings=(
                    f"Skipped: {chunk_count} chunks exceed MAX_CHUNKS={MAX_CHUNKS}",
                ),
            )

        # Collect timings per stage across runs
        chunking_times: list[float] = []
        similarity_times: list[float] = []
        eigen_times: list[float] = []
        scoring_times: list[float] = []
        zone_times: list[float] = []
        compression_times: list[float] = []
        total_times: list[float] = []

        # Measure peak memory across all runs
        tracemalloc.start()
        mem_baseline = tracemalloc.get_traced_memory()[0]

        for _ in range(self._runs):
            t_total_start = time.perf_counter()

            # Stage 1: Chunking
            t0 = time.perf_counter()
            chunks = chunk_text(text, strategy=self._strategy)
            chunking_times.append(_ms(time.perf_counter() - t0))

            # Stage 2: Similarity computation
            t0 = time.perf_counter()
            adjacency = build_similarity_graph(chunks)
            similarity_times.append(_ms(time.perf_counter() - t0))

            # Stage 3: Eigendecomposition
            t0 = time.perf_counter()
            fiedler, lambda_2 = compute_fiedler_vector(adjacency)
            eigen_times.append(_ms(time.perf_counter() - t0))

            # Stage 4: Scoring
            t0 = time.perf_counter()
            scores = compute_chunk_scores(chunks, fiedler, adjacency)
            scoring_times.append(_ms(time.perf_counter() - t0))

            # Stage 5: Zone detection
            t0 = time.perf_counter()
            zoned = detect_zones(chunks)
            zone_times.append(_ms(time.perf_counter() - t0))

            # Stage 6: Compression (pruning + reconstruction)
            t0 = time.perf_counter()
            n = len(chunks)
            weighted_scores = [
                s * zc.protection_weight for s, zc in zip(scores, zoned)
            ]
            target_removals = max(1, int(n * self._target_ratio))
            indexed = sorted(enumerate(weighted_scores), key=lambda x: x[1])
            remove = set()
            for idx, _ in indexed:
                if len(remove) >= target_removals:
                    break
                remove.add(idx)
            kept = sorted(i for i in range(n) if i not in remove)
            compressed = "\n\n".join(chunks[i].text for i in kept)
            compression_times.append(_ms(time.perf_counter() - t0))

            total_times.append(_ms(time.perf_counter() - t_total_start))

        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Build stage results
        stages = (
            StageTimings("chunking", tuple(chunking_times)),
            StageTimings("similarity", tuple(similarity_times)),
            StageTimings("eigendecomposition", tuple(eigen_times)),
            StageTimings("scoring", tuple(scoring_times)),
            StageTimings("zone_detection", tuple(zone_times)),
            StageTimings("compression", tuple(compression_times)),
        )
        total = StageTimings("total", tuple(total_times))

        # Check for eigendecomposition warnings
        warnings: list[str] = []
        eigen_median_s = statistics.median(eigen_times) / 1000
        if eigen_median_s > 30.0:
            warnings.append(
                f"CRITICAL: eigendecomposition median {eigen_median_s:.1f}s "
                f"exceeds 30s — O(n³) bottleneck is dominant"
            )
        elif eigen_median_s > 5.0:
            warnings.append(
                f"WARNING: eigendecomposition median {eigen_median_s:.1f}s "
                f"exceeds 5s — O(n³) bottleneck is significant"
            )
        elif eigen_median_s > 1.0:
            warnings.append(
                f"NOTE: eigendecomposition median {eigen_median_s:.1f}s "
                f"exceeds 1s — O(n³) bottleneck is becoming material"
            )

        return SizeResult(
            target_tokens=target_tokens,
            actual_chars=len(text),
            chunk_count=chunk_count,
            stages=stages,
            total=total,
            memory_peak_bytes=mem_peak,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_latency_table(report: LatencyReport) -> str:
    """Format a latency report as a human-readable table."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  FIEDLER LATENCY PROFILE")
    lines.append("=" * 90)
    lines.append(f"  Backend: {report.backend}    Runs: {report.runs_per_size}")
    lines.append("")
    lines.append(
        f"  {'Tokens':>8s}  {'Chunks':>6s}  "
        f"{'Eigen (ms)':>10s}  {'Total (ms)':>10s}  "
        f"{'Mem (MB)':>8s}  Notes"
    )
    lines.append(
        f"  {'--------':>8s}  {'------':>6s}  "
        f"{'----------':>10s}  {'----------':>10s}  "
        f"{'--------':>8s}  -----"
    )

    for r in report.results:
        if not r.stages:
            # Skipped size
            lines.append(
                f"  {r.target_tokens:>8d}  {r.chunk_count:>6d}  "
                f"{'SKIPPED':>10s}  {'SKIPPED':>10s}  "
                f"{'---':>8s}  {'; '.join(r.warnings)}"
            )
            continue

        eigen_stage = next(
            (s for s in r.stages if s.name == "eigendecomposition"), None
        )
        eigen_med = eigen_stage.median if eigen_stage else 0.0
        total_med = r.total.median
        mem_mb = r.memory_peak_bytes / (1024 * 1024)

        notes = "; ".join(r.warnings) if r.warnings else ""
        lines.append(
            f"  {r.target_tokens:>8d}  {r.chunk_count:>6d}  "
            f"{eigen_med:>10.1f}  {total_med:>10.1f}  "
            f"{mem_mb:>8.1f}  {notes}"
        )

    lines.append("=" * 90)

    # Per-stage breakdown for each size
    lines.append("")
    lines.append("  PER-STAGE BREAKDOWN (median ms)")
    lines.append("  " + "-" * 86)
    lines.append(
        f"  {'Tokens':>8s}  {'Chunk':>7s}  {'Simil':>7s}  "
        f"{'Eigen':>7s}  {'Score':>7s}  {'Zones':>7s}  "
        f"{'Compr':>7s}  {'Total':>8s}"
    )

    for r in report.results:
        if not r.stages:
            continue
        stage_meds = {s.name: s.median for s in r.stages}
        lines.append(
            f"  {r.target_tokens:>8d}  "
            f"{stage_meds.get('chunking', 0):>7.1f}  "
            f"{stage_meds.get('similarity', 0):>7.1f}  "
            f"{stage_meds.get('eigendecomposition', 0):>7.1f}  "
            f"{stage_meds.get('scoring', 0):>7.1f}  "
            f"{stage_meds.get('zone_detection', 0):>7.1f}  "
            f"{stage_meds.get('compression', 0):>7.1f}  "
            f"{r.total.median:>8.1f}"
        )

    lines.append("  " + "-" * 86)
    return "\n".join(lines)


def report_to_json(report: LatencyReport) -> dict:
    """Convert a latency report to a JSON-serializable dict."""
    results = []
    for r in report.results:
        stages_data = {}
        for s in r.stages:
            stages_data[s.name] = {
                "times_ms": list(s.times_ms),
                "mean": round(s.mean, 3),
                "median": round(s.median, 3),
                "p95": round(s.p95, 3),
                "p99": round(s.p99, 3),
            }
        results.append({
            "target_tokens": r.target_tokens,
            "actual_chars": r.actual_chars,
            "chunk_count": r.chunk_count,
            "stages": stages_data,
            "total": {
                "times_ms": list(r.total.times_ms),
                "mean": round(r.total.mean, 3),
                "median": round(r.total.median, 3),
                "p95": round(r.total.p95, 3),
                "p99": round(r.total.p99, 3),
            },
            "memory_peak_bytes": r.memory_peak_bytes,
            "warnings": list(r.warnings),
        })

    return {
        "sizes": list(report.sizes),
        "runs_per_size": report.runs_per_size,
        "backend": report.backend,
        "results": results,
    }
