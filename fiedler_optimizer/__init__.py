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

As of 0.4.0 this is the complete capability, with no held-back commercial
tier: spectral compression plus ligatures, topology caching, distillation,
spectral obscuring, reasoning templates, and signed certificates. Optional
extras pull in heavier scientific dependencies -- see ``pyproject.toml``.
"""

__version__ = "0.4.0"

from fiedler_optimizer.core import optimize, FiedlerResult
from fiedler_optimizer.graph import build_similarity_graph, compute_fiedler_vector
from fiedler_optimizer.chunker import chunk_text, ChunkingStrategy, merge_kept_spans
from fiedler_optimizer.zones import detect_zones, Zone

__all__ = [
    "optimize",
    "FiedlerResult",
    "build_similarity_graph",
    "compute_fiedler_vector",
    "chunk_text",
    "ChunkingStrategy",
    "merge_kept_spans",
    "detect_zones",
    "Zone",
]
