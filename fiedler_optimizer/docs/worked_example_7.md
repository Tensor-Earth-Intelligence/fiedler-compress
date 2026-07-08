# Example 7: Pipeline Latency Profile Across Input Scales

## Field of Application

This worked example characterizes the computational performance of the Fiedler spectral compression pipeline across a range of input sizes representative of production large language model workloads. The example establishes that the full pipeline operates within sub-second latency on consumer-grade hardware for inputs up to 50,000 tokens, validating its suitability as a transparent middleware layer in real-time LLM inference pipelines.

## Method

Synthetic English text inputs were generated at seven target sizes: 100, 500, 1,000, 5,000, 10,000, 25,000, and 50,000 tokens. Each input consists of coherent multi-paragraph prose with varied vocabulary and sentence structure, ensuring realistic chunking and similarity graph properties.

Each input was processed through the complete Fiedler compression pipeline using the following configuration:

- **Similarity backend:** TF-IDF cosine similarity (no neural embeddings, no GPU)
- **Chunking strategy:** ADAPTIVE (automatically selects between sentence, paragraph, and sliding-window segmentation)
- **Target compression ratio:** 0.20 (20% text removal)
- **Instruction protection:** Enabled (zone-aware compression with 2x--3x score multipliers for instruction-classified chunks)

Each input size was profiled across 5 independent runs. Per-stage timing instrumentation measured six pipeline phases independently: chunking, similarity matrix construction, eigendecomposition, chunk scoring and zone assignment, compression (chunk selection and text reconstruction), and total end-to-end time. Median values across the 5 runs are reported to reduce variance from system scheduling effects. Peak memory usage was measured using process-level resident set size monitoring.

All experiments were conducted on a workstation equipped with an Intel Core i5-8500T processor (6 cores, 3.0 GHz base frequency) and 32 GB of DDR4 RAM, running a standard Python 3.12 environment. No GPU was used.

## Results

### Per-Stage Latency Profile (Median of 5 Runs)

| Tokens | Chunks | Chunk (ms) | Similarity (ms) | Eigen (ms) | Score + Zone (ms) | Compress (ms) | Total (ms) | Memory (MB) |
|-------:|-------:|-----------:|-----------------:|-----------:|------------------:|--------------:|-----------:|------------:|
|    100 |      5 |        0.8 |              0.3 |        0.5 |               0.3 |           0.1 |        2.4 |         0.1 |
|    500 |     12 |        2.1 |              0.9 |        1.2 |               0.8 |           0.2 |        5.8 |         0.2 |
|  1,000 |     22 |        4.3 |              2.1 |        2.8 |               1.5 |           0.3 |       12.1 |         0.3 |
|  5,000 |     85 |       18.6 |             15.2 |       10.4 |               5.8 |           0.5 |       52.8 |         0.8 |
| 10,000 |    170 |       37.3 |             36.3 |       23.0 |              11.6 |           0.8 |      109.6 |         1.5 |
| 25,000 |    424 |       94.5 |            118.4 |       40.9 |              29.6 |           1.9 |      293.4 |         5.6 |
| 50,000 |    847 |      192.4 |            219.8 |       69.3 |              58.6 |           4.1 |      552.6 |        21.3 |

### Latency Scaling Summary

| Tokens | Chunks | Eigen (ms) | Total (ms) | Eigen % of Total |
|-------:|-------:|-----------:|-----------:|-----------------:|
|    100 |      5 |        0.5 |        2.4 |            20.8% |
|  1,000 |     22 |        2.8 |       12.1 |            23.1% |
| 10,000 |    170 |       23.0 |      109.6 |            21.0% |
| 50,000 |    847 |       69.3 |      552.6 |            12.5% |

## Key Findings

### Finding 1: Eigendecomposition Scales as O(n log n) in Practice

The eigendecomposition step, which computes the Fiedler vector from the graph Laplacian, has a theoretical worst-case complexity of O(n^3) for dense matrices of dimension n. However, the observed scaling is substantially more favorable. As the number of chunks increases from 170 to 424 to 847, eigendecomposition time increases from 23.0 ms to 40.9 ms to 69.3 ms --- a growth pattern consistent with approximately O(n log n) scaling rather than cubic.

This favorable scaling is attributable to two properties of the graph Laplacian in the TF-IDF similarity context. First, the Laplacian is symmetric positive semi-definite, enabling the use of Lanczos-based iterative methods rather than full matrix factorization. Second, the similarity matrix exhibits effective sparsity: most chunk pairs have low cosine similarity, and the TF-IDF vectors are themselves sparse, resulting in a Laplacian whose spectral structure is amenable to rapid convergence of the ARPACK-based eigensolver used by SciPy's `eigsh` function. Only the second-smallest eigenvalue and its corresponding eigenvector are required, further reducing computation relative to a full eigendecomposition.

At the largest input tested (50,000 tokens, 847 chunks), eigendecomposition required only 69.3 ms, representing 12.5% of total pipeline time. The eigendecomposition step is therefore not the computational bottleneck of the system at any tested scale.

### Finding 2: Similarity Computation and Chunking Dominate at Scale

As input size increases, the dominant latency contributors shift from eigendecomposition to similarity matrix construction and text chunking. At 50,000 tokens (847 chunks):

- **Similarity matrix construction:** 219.8 ms (39.8% of total pipeline time). The O(n^2) pairwise cosine similarity computation between TF-IDF vectors is the single largest contributor. However, the sparse nature of TF-IDF vectors (most terms have zero weight in any given chunk) enables optimized inner-product computation, keeping the absolute time modest.

- **Chunking:** 192.4 ms (34.8% of total pipeline time). Sentence boundary detection via regular expression matching and the subsequent chunk merging pass scale linearly in input length but have a substantial constant factor due to Unicode-aware text processing.

- **Score and zone assignment:** 58.6 ms (10.6% of total). Zone detection applies pattern matching heuristics to classify each chunk, with cost linear in the number of chunks and the number of patterns.

Together, similarity computation and chunking account for 74.6% of total pipeline time at 50,000 tokens. This indicates that optimization efforts for larger inputs should focus on these stages rather than on the eigendecomposition.

### Finding 3: Memory Usage Remains Within Consumer Hardware Constraints

Peak memory usage at 50,000 tokens is 21.3 MB, driven primarily by the O(n^2) dense similarity matrix (847 x 847 float64 = approximately 5.7 MB) and the TF-IDF feature matrix. This is well within the memory constraints of consumer hardware and imposes negligible overhead relative to the memory requirements of the LLM inference process that follows compression.

The MAX_CHUNKS limit of 2,000 (corresponding to inputs of approximately 100,000 tokens) bounds the maximum similarity matrix size to approximately 32 MB (2,000 x 2,000 float64), ensuring that memory usage remains predictable and bounded even for the largest supported inputs.

## Conclusion

The Fiedler spectral compression pipeline operates in sub-second time for inputs up to 50,000 tokens on consumer-grade hardware (Intel Core i5-8500T, 32 GB RAM) without GPU acceleration. At the most common input sizes for production LLM workloads (1,000 to 10,000 tokens), total pipeline latency ranges from 12 to 110 milliseconds, adding negligible overhead to the typical LLM inference latency of 500 ms to 5,000 ms.

The favorable practical scaling of the eigendecomposition step --- O(n log n) rather than the theoretical O(n^3) --- combined with the sub-quadratic growth of the dominant similarity computation stage, establishes that spectral compression is computationally tractable as a transparent middleware layer in production LLM pipelines. No specialized hardware is required, and the pipeline can be deployed as a synchronous preprocessing step without introducing perceptible latency to the end user.
