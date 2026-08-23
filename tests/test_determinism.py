"""
Determinism regression guard for the Fiedler pipeline.

The Fiedler vector is the second eigenvector of the graph Laplacian, computed
with ARPACK (`scipy.sparse.linalg.eigsh`). ARPACK seeds its start vector `v0`
randomly unless told otherwise, which historically made the Fiedler vector -- and
therefore every chunk score and removal decision -- vary run to run. The fix seeds
`v0` and sign-canonicalizes the result (see graph.compute_fiedler_vector).

These tests lock that guarantee so a regression is caught in CI: identical input
must always yield the identical Fiedler vector and the identical compressed output.

Run with: pytest tests/test_determinism.py -v
"""
import numpy as np

from fiedler_optimizer import optimize
from fiedler_optimizer.chunker import chunk_text
from fiedler_optimizer.graph import (
    build_similarity_graph,
    compute_fiedler_vector,
    compute_chunk_scores,
)

_DOC = (
    "The mitochondrion is the powerhouse of the cell. It generates ATP through "
    "oxidative phosphorylation.\n\n"
    "Photosynthesis occurs in the chloroplast. Light reactions produce oxygen and "
    "ATP; the Calvin cycle fixes carbon.\n\n"
    "The nucleus stores genetic material. DNA is transcribed into RNA, which is "
    "translated into protein at the ribosome.\n\n"
    "Cell membranes are selectively permeable. Ion channels and pumps maintain the "
    "electrochemical gradient across the bilayer.\n\n"
    "Enzymes lower activation energy. Their activity depends on temperature, pH, "
    "and substrate concentration."
)


def test_fiedler_vector_is_deterministic():
    """Same adjacency matrix -> identical Fiedler vector across repeated calls."""
    chunks = chunk_text(_DOC)
    adjacency = build_similarity_graph(chunks)
    first, lam0 = compute_fiedler_vector(adjacency)
    for _ in range(8):
        v, lam = compute_fiedler_vector(adjacency)
        assert np.array_equal(v, first), "Fiedler vector changed run to run"
        assert lam == lam0


def test_chunk_scores_are_deterministic():
    """Scores derived from the Fiedler vector are stable run to run."""
    chunks = chunk_text(_DOC)
    adjacency = build_similarity_graph(chunks)
    fiedler, _ = compute_fiedler_vector(adjacency)
    baseline = compute_chunk_scores(chunks, fiedler, adjacency)
    for _ in range(8):
        f, _ = compute_fiedler_vector(adjacency)
        assert compute_chunk_scores(chunks, f, adjacency) == baseline


def test_optimize_output_is_deterministic():
    """optimize() returns byte-identical compressed text across reps and targets."""
    for target in (0.33, 0.50, 0.67, 0.80):
        outputs = {optimize(_DOC, target_ratio=target).compressed for _ in range(6)}
        assert len(outputs) == 1, (
            f"compressed output not stable at target_ratio={target}"
        )
