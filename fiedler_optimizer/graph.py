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


# ---------------------------------------------------------------------------
# Neural model cache — persists across calls within the same process
# ---------------------------------------------------------------------------

_NEURAL_MODEL_CACHE: dict[str, object] = {}

# Allowlist of trusted sentence-transformer models pinned to specific
# Hugging Face revision hashes.  This prevents both loading of arbitrary
# (potentially malicious) models and supply-chain attacks via silent
# model updates on the Hub.
#
# Each entry maps model_name -> (revision_sha1, expected_embedding_dim).
# After loading, the embedding dimension is verified to detect model
# tampering or cache corruption.
_ALLOWED_MODELS: dict[str, tuple[str, int]] = {
    "all-MiniLM-L6-v2": ("c9745ed1d9f207416be6d2e6f8de32d1f16199bf", 384),
    "all-MiniLM-L12-v2": ("a50ef00143b4d5391434df20ae11632588ac25be", 384),
    "all-mpnet-base-v2": ("e8c3b32edf5434bc2275fc9bab85f82640a19130", 768),
    "paraphrase-MiniLM-L6-v2": ("c9a2bfebc254878aee8c3aca9e6844d5bbb102d1", 384),
    "paraphrase-multilingual-MiniLM-L12-v2": ("e8f8c211226b894fcb81acc59f3b34ba3efd5f42", 384),
}


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


def _compute_neural_embeddings(
    chunks: Sequence[Chunk], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Compute neural embeddings using sentence-transformers.

    Returns shape (n_chunks, embedding_dim) L2-normalized matrix.
    Requires the 'embeddings' extra: pip install fiedler-compress[embeddings]

    Uses a module-level cache so the model is only loaded once per process.
    Only models in _ALLOWED_MODELS can be loaded to prevent arbitrary code
    execution via malicious Hugging Face models.

    Post-load integrity check: verifies the embedding dimension matches
    the expected value to detect model tampering or cache corruption.
    """
    if model_name not in _ALLOWED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not in the allowed models list. "
            f"Allowed: {sorted(_ALLOWED_MODELS)}"
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for neural embeddings. "
            "Install with: pip install fiedler-compress[embeddings]"
        )

    revision, expected_dim = _ALLOWED_MODELS[model_name]

    if model_name not in _NEURAL_MODEL_CACHE:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        model = SentenceTransformer(model_name, revision=revision, device=device)

        # Post-load integrity check: verify embedding dimension matches
        # the pinned expectation to catch cache corruption or tampering.
        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"Model integrity check failed for '{model_name}': "
                f"expected embedding dim {expected_dim}, got {actual_dim}. "
                f"The cached model may be corrupted or tampered with."
            )

        _NEURAL_MODEL_CACHE[model_name] = model

    model = _NEURAL_MODEL_CACHE[model_name]
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------

def build_similarity_graph(
    chunks: Sequence[Chunk],
    vectors: np.ndarray | None = None,
    similarity_threshold: float = 0.05,
    use_neural: bool = False,
    model_name: str = "all-MiniLM-L6-v2",
    backend: str | None = None,
) -> np.ndarray:
    """
    Construct a weighted adjacency matrix from chunk similarity.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to build the graph over.
    vectors : np.ndarray, optional
        Pre-computed feature vectors (n_chunks x d). If None, vectors
        are computed automatically using TF-IDF or neural embeddings.
    similarity_threshold : float
        Edges with weight below this are zeroed out to produce a
        sparse but connected graph.
    use_neural : bool
        If True, use sentence-transformers for similarity instead of TF-IDF.
        Ignored if ``backend`` is set.
    model_name : str
        Which sentence-transformers model to use (only when use_neural=True).
    backend : str, optional
        Named similarity backend (e.g. 'tfidf', 'aitchison', 'fisher-rao',
        'wasserstein', 'hyperbolic', 'neural'). If set, overrides
        ``use_neural`` and ``vectors``.

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

    # If a named backend is specified, use it directly to produce similarity
    if backend is not None:
        from fiedler_optimizer.backends.registry import get_backend
        backend_fn = get_backend(backend)
        similarity = backend_fn(chunks)
    elif vectors is None:
        if use_neural:
            vectors = _compute_neural_embeddings(chunks, model_name)
        else:
            vectors = _compute_tfidf_matrix(chunks)
        # Cosine similarity (vectors are already L2-normalized)
        similarity = vectors @ vectors.T
    else:
        similarity = vectors @ vectors.T

    # Sanitize: alternative backends can emit NaN/Inf, which would silently
    # poison the Laplacian. Replace them and clip to [0, 1].
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
    similarity = np.clip(similarity, 0.0, 1.0)

    # Zero out self-loops and sub-threshold edges
    np.fill_diagonal(similarity, 0.0)
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
    # Fixed start vector: ARPACK otherwise seeds v0 randomly, which makes the
    # Fiedler vector (and every downstream score) nondeterministic run-to-run.
    v0 = np.random.default_rng(0).standard_normal(n)
    try:
        eigenvalues, eigenvectors = eigsh(
            L_sparse,
            k=min(2, n - 1),
            which="SM",  # smallest magnitude
            tol=1e-8,
            v0=v0,
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

    # Eigenvectors are defined only up to sign; the sparse and dense paths (and
    # different LAPACK builds) can return either orientation. Canonicalize so the
    # Fiedler vector is stable: make the largest-magnitude component positive.
    if fiedler[np.argmax(np.abs(fiedler))] < 0:
        fiedler = -fiedler

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