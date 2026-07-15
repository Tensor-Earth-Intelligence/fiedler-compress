"""
Fiedler Optimizer — Spectral graph-theoretic prompt compression.

Uses the Fiedler vector (second-smallest eigenvector of the graph Laplacian)
to identify and remove semantically disconnected chunks from LLM prompts,
reducing token count while preserving information fidelity.

Basic usage:
    from fiedler_optimizer import optimize

    result = optimize("Your long prompt text here...")
    print(result.compressed)
    print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.1%})")

This is the open-core distribution: the TF-IDF + single-eigenvector (k=1)
spectral compression pipeline. Additional capabilities are available in a
separate commercial tier.
"""

__version__ = "0.1.1"

from fiedler_optimizer.core import optimize, FiedlerResult
from fiedler_optimizer.graph import build_similarity_graph, compute_fiedler_vector
from fiedler_optimizer.chunker import chunk_text, ChunkingStrategy
from fiedler_optimizer.zones import detect_zones, Zone

__all__ = [
    "optimize",
    "FiedlerResult",
    "build_similarity_graph",
    "compute_fiedler_vector",
    "chunk_text",
    "ChunkingStrategy",
    "detect_zones",
    "Zone",
]
