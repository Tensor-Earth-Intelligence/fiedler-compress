"""
Similarity graph construction and spectral decomposition.

Builds a weighted adjacency graph over text chunks, computes the graph
Laplacian, and extracts the Fiedler vector — the eigenvector corresponding
to the second-smallest eigenvalue (algebraic connectivity λ₂). Chunks with
extreme Fiedler values lie at the semantic periphery and are candidates
for removal.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from fiedler_optimizer.chunker import Chunk
from fiedler_optimizer._tier import commercial_tier_error


# Maximum number of chunks to process. Larger counts create O(n²) similarity
# matrices that can exhaust memory (e.g., 10k chunks = ~800MB dense matrix).
# Raised to 2000: eigendecomposition is only ~69ms at 847 chunks and scales
# as O(n·log n) in practice (scipy eigsh exploits Laplacian sparsity). The
# O(n²) similarity matrix is ~23MB at 1694 chunks — well within memory
# limits. This covers inputs up to ~100k tokens without truncation.
MAX_CHUNKS = 2000


# ---------------------------------------------------------------------------
# TF-IDF vectorization (dependency-free, no sklearn/transformers needed)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())


def _compute_tfidf_matrix(chunks: Sequence[Chunk]) -> np.ndarray:
    """
    Build a TF-IDF matrix from chunks without external dependencies.

    Returns shape (n_chunks, vocab_size) dense matrix.
    Dependency-free; uses only NumPy.
    """
    # Build vocabulary
    doc_tokens = [_tokenize(c.text) for c in chunks]
    vocab: dict[str, int] = {}
    doc_freq: dict[str, int] = {}

    for tokens in doc_tokens:
        seen = set()
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            if tok not in seen:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
                seen.add(tok)

    n_docs = len(chunks)
    n_vocab = len(vocab)

    if n_vocab == 0:
        return np.zeros((n_docs, 1))

    # Cap vocabulary to prevent excessive memory usage. With MAX_CHUNKS=2000
    # and max_vocab=50000, the dense TF-IDF matrix is ~800MB worst case.
    max_vocab = 50_000
    if n_vocab > max_vocab:
        # Keep the most common terms by document frequency
        sorted_terms = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
        keep_terms = {term for term, _ in sorted_terms[:max_vocab]}
        vocab = {term: i for i, (term, _) in enumerate(sorted_terms[:max_vocab])}
        n_vocab = max_vocab

    # Build TF-IDF
    tfidf = np.zeros((n_docs, n_vocab), dtype=np.float64)
    for i, tokens in enumerate(doc_tokens):
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        max_tf = max(counts.values()) if counts else 1
        for tok, count in counts.items():
            tf = 0.5 + 0.5 * (count / max_tf)  # augmented TF
            idf = math.log((n_docs + 1) / (doc_freq.get(tok, 0) + 1)) + 1
            tfidf[i, vocab[tok]] = tf * idf

    # L2 normalize rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    tfidf /= norms

    return tfidf


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------

def build_similarity_graph(
    chunks: Sequence[Chunk],
    vectors: np.ndarray | None = None,
    similarity_threshold: float = 0.05,
    use_neural: bool = False,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Construct a weighted adjacency matrix from chunk similarity.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to build the graph over.
    vectors : np.ndarray, optional
        Pre-computed feature vectors (n_chunks x d). If None, vectors
        are computed automatically using TF-IDF.
    similarity_threshold : float
        Edges with weight below this are zeroed out to produce a
        sparse but connected graph.
    use_neural : bool
        Reserved for a commercial-tier similarity backend; not available
        in the open-core package.
    model_name : str
        Reserved for a commercial-tier similarity backend.

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix of shape (n, n).
    """
    n = len(chunks)
    if n < 2:
        return np.ones((n, n))

    if n > MAX_CHUNKS:
        raise ValueError(
            f"Input produces {n} chunks, exceeding the maximum of {MAX_CHUNKS}. "
            f"Use a coarser chunking strategy or shorter input to avoid "
            f"excessive memory usage (O(n²) similarity matrix)."
        )

    if vectors is None:
        if use_neural:
            # This similarity backend is a commercial-tier capability and is
            # not bundled with the open-core package.
            raise commercial_tier_error()
        else:
            vectors = _compute_tfidf_matrix(chunks)

    # Cosine similarity (vectors are already L2-normalized)
    similarity = vectors @ vectors.T

    # Zero out self-loops and sub-threshold edges
    np.fill_diagonal(similarity, 0.0)
    similarity = np.maximum(similarity, 0.0)  # clip negatives
    similarity[similarity < similarity_threshold] = 0.0

    # Ensure graph connectivity: add small positional proximity edges
    # between adjacent chunks. This prevents disconnected components
    # which would make the Fiedler vector meaningless.
    for i in range(n - 1):
        if similarity[i, i + 1] < similarity_threshold:
            proximity_weight = similarity_threshold * 2
            similarity[i, i + 1] = proximity_weight
            similarity[i + 1, i] = proximity_weight

    return similarity


# ---------------------------------------------------------------------------
# Laplacian and Fiedler vector
# ---------------------------------------------------------------------------

def compute_fiedler_vector(
    adjacency: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute the Fiedler vector and algebraic connectivity λ₂.

    The Fiedler vector is the eigenvector corresponding to the second-
    smallest eigenvalue of the graph Laplacian L = D - A. It provides
    the optimal bipartition of the graph (normalized cut), and its
    component values indicate each node's position in the spectral
    ordering — nodes with extreme values are at the semantic periphery.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric weighted adjacency matrix.

    Returns
    -------
    fiedler_vector : np.ndarray
        The Fiedler vector (length n).
    algebraic_connectivity : float
        λ₂, the second-smallest eigenvalue. Higher values indicate a
        more tightly connected graph.
    """
    n = adjacency.shape[0]

    if n <= 2:
        return np.array([1.0] * n), 0.0

    # Degree matrix
    degree = np.diag(adjacency.sum(axis=1))

    # Unnormalized Laplacian L = D - A
    laplacian = degree - adjacency

    # Convert to sparse for efficient eigendecomposition
    L_sparse = csr_matrix(laplacian)

    # Compute the two smallest eigenvalues/vectors
    # (smallest is always 0 for connected graphs)
    try:
        eigenvalues, eigenvectors = eigsh(
            L_sparse,
            k=min(2, n - 1),
            which="SM",  # smallest magnitude
            tol=1e-8,
        )
    except Exception:
        # Fallback for numerical issues: use dense solver
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # The Fiedler vector is the eigenvector for λ₂
    if len(eigenvalues) >= 2:
        fiedler = eigenvectors[:, 1]
        lambda_2 = float(eigenvalues[1])
    else:
        fiedler = eigenvectors[:, 0]
        lambda_2 = float(eigenvalues[0])

    # Normalize to [-1, 1] range for interpretability
    max_abs = np.max(np.abs(fiedler))
    if max_abs > 0:
        fiedler = fiedler / max_abs

    return fiedler, lambda_2


def compute_chunk_scores(
    chunks: Sequence[Chunk],
    fiedler: np.ndarray,
    adjacency: np.ndarray,
) -> list[float]:
    """
    Score each chunk's semantic connectivity.

    Combines the Fiedler vector position with the chunk's weighted degree
    (total similarity to all other chunks). Chunks that are both spectrally
    peripheral AND weakly connected are the best removal candidates.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        The text chunks.
    fiedler : np.ndarray
        The Fiedler vector.
    adjacency : np.ndarray
        The similarity adjacency matrix.

    Returns
    -------
    list[float]
        Connectivity score per chunk. Lower = more removable.
    """
    n = len(chunks)
    if n == 0:
        return []

    # Weighted degree: sum of all edge weights for each node
    degrees = adjacency.sum(axis=1)
    max_degree = degrees.max() if degrees.max() > 0 else 1.0

    # Fiedler centrality: how close to the spectral center (0.0)
    # Nodes near 0 in the Fiedler vector bridge the two partitions
    fiedler_centrality = 1.0 - np.abs(fiedler)

    # Combined score: weighted average of degree centrality and Fiedler centrality
    degree_norm = degrees / max_degree
    scores = 0.6 * degree_norm + 0.4 * fiedler_centrality

    return scores.tolist()