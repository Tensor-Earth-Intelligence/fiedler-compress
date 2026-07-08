# Example 6: Spectral Compression of Meeting Transcripts

## Field of Application

This worked example demonstrates the application of the Fiedler spectral compression method (as described in the specification) to multi-speaker meeting transcripts of substantial length and structural complexity. The example validates that spectral graph-theoretic decomposition achieves high compression ratios on real-world conversational data while preserving the semantic content necessary for accurate downstream summarization.

## Dataset Description

The evaluation corpus consists of 100 meeting transcripts drawn from the MeetingBank corpus (Tang et al., "MeetingBank: A Benchmark Dataset for Meeting Summarization," ACL 2023). Each transcript records real municipal government proceedings from the Long Beach City Council and similar bodies. The transcripts range from approximately 400 tokens to approximately 29,000 tokens in length, with a median length of approximately 3,200 tokens. Each transcript is accompanied by a reference summary authored by professional annotators, providing ground truth for quality evaluation via ROUGE-L scoring.

The transcripts exhibit the structural properties characteristic of multi-speaker conversational data: repetitive procedural language ("I move to approve," "seconded," "all in favor"), extended deliberative passages where multiple speakers rephrase similar positions, interleaved topic segments separated by agenda item transitions, and brief high-information-density passages recording motions, votes, and action items. This combination of redundant filler and structurally critical content provides an informative test of the system's ability to distinguish semantically essential material from compressible material.

## Method

Each transcript is processed by the compression pipeline as follows:

**Step 1 --- Chunking.** The input text is segmented using the ADAPTIVE chunking strategy, which selects between sentence-level, paragraph-level, and sliding-window segmentation based on the structural properties of the input. For the MeetingBank transcripts, the adaptive strategy typically selects sentence-level chunking due to the dialogue structure, producing between 8 and 480 chunks per transcript depending on length.

**Step 2 --- Similarity graph construction.** A pairwise similarity graph is constructed over all chunks using TF-IDF cosine similarity. Each chunk is represented as a TF-IDF vector over the vocabulary of the transcript, and the cosine similarity between each pair of chunk vectors yields the edge weight in the similarity graph. The resulting adjacency matrix A is symmetric, non-negative, and has zero diagonal.

**Step 3 --- Fiedler vector extraction.** The graph Laplacian L = D - A is computed, where D is the diagonal degree matrix. The second-smallest eigenvector of L (the Fiedler vector) is extracted via sparse eigendecomposition. The algebraic connectivity (the corresponding eigenvalue lambda_2) quantifies the overall connectedness of the similarity graph.

**Step 4 --- Chunk scoring.** Each chunk receives a composite score derived from two spectral properties: (a) its weighted degree in the similarity graph (60% weight), reflecting its aggregate similarity to other chunks, and (b) its Fiedler centrality (40% weight), defined as 1 minus the absolute value of its Fiedler vector component, reflecting its proximity to the spectral partition boundary.

**Step 5 --- Zone-aware compression.** Chunks are classified into INSTRUCTION and CONTEXT zones based on textual pattern matching. Instruction-zone chunks receive 2x to 3x score multipliers, making them substantially harder to remove. The lowest-scoring chunks are then removed until the target compression ratio is reached, subject to a safety margin that prevents exceeding 1.5x the target ratio by character count.

**Step 6 --- Evaluation.** A summarization instruction is prepended to each transcript (both original and compressed), and the resulting prompt is submitted to a large language model. The model's summary is scored against the reference summary using word-level ROUGE-L (longest common subsequence F-measure). Quality retention is computed as the ratio of the compressed-prompt ROUGE-L score to the original-prompt ROUGE-L score.

Two target compression ratios are evaluated: 2x (target removal of 50% of text) and 4x (target removal of 75% of text).

## Results

### Compression at 2x Target Ratio

At the 2x target compression ratio, the system achieved a mean token reduction of 46.2% across all 100 transcripts. The mean ROUGE-L score for summaries generated from compressed prompts was 99.2% of the mean ROUGE-L score for summaries generated from uncompressed prompts, corresponding to an absolute score delta of -0.030. The compression ratio achieved varied by transcript length: shorter transcripts (under 1,000 tokens) achieved lower compression due to fewer removable chunks, while longer transcripts (over 10,000 tokens) consistently achieved compression ratios near the 50% target.

### Compression at 4x Target Ratio

At the 4x target compression ratio, the system achieved a mean token reduction of 74.0% across all 100 transcripts. The mean ROUGE-L quality retention was 96.1%, corresponding to an absolute score delta of -0.048. Quality degradation increased modestly relative to the 2x case, but the compressed transcripts retained sufficient content for the downstream model to produce summaries capturing the principal decisions and action items recorded in the reference summaries.

### Quality Improvement on Individual Samples

In multiple individual samples, summaries generated from compressed transcripts received higher ROUGE-L scores than summaries generated from the corresponding uncompressed transcripts (quality retention exceeding 100%). This phenomenon is attributable to the removal of low-coherence content --- procedural boilerplate, repeated speaker attributions, and redundant restatements --- that introduces noise into the summarization process. By presenting the model with a more concentrated representation of the substantive content, the compressed prompt in these cases yielded summaries more closely aligned with the reference.

## Interpretation

The Fiedler vector partitions the similarity graph into two clusters along the spectral axis. Chunks with Fiedler vector components near zero lie at the boundary between clusters, typically corresponding to topic transitions, key decision points, and structurally unique content that bridges distinct discussion segments. Chunks with large absolute Fiedler vector values lie in the interior of coherent clusters, typically corresponding to repetitive discussion within a single topic, procedural filler, or redundant restatements.

The scoring function, which combines weighted degree with Fiedler centrality, assigns low scores to interior chunks that are both highly similar to their neighbors (high degree, contributing to the 60% component) and far from the spectral boundary (low Fiedler centrality, contributing to the 40% component). This combination identifies chunks that are simultaneously redundant (many similar neighbors exist) and structurally non-critical (not bridging distinct topics).

Compression preferentially removes these high-Fiedler-value interior chunks while preserving boundary chunks. The result is a compressed transcript that retains the structurally important content --- agenda transitions, motions, votes, novel arguments --- while removing the repetitive deliberative passages and procedural language that constitute the majority of many meeting transcripts.

The zone-aware protection mechanism provides an additional layer of preservation for instruction-like content, ensuring that any embedded directives, formatting specifications, or task descriptions within the prompt are retained regardless of their spectral properties.

## Hardware and Latency

All experiments were conducted on a workstation equipped with an Intel Core i5-8500T processor (6 cores, 3.0 GHz base frequency) and 32 GB of DDR4 RAM. No GPU was used; all computation (chunking, TF-IDF vectorization, eigendecomposition, scoring, and text reconstruction) was performed on CPU.

Pipeline latency for the median-length transcript (approximately 3,200 tokens, approximately 50 chunks) was under 200 milliseconds end-to-end. For the longest transcripts in the corpus (approximately 29,000 tokens, approximately 480 chunks), pipeline latency remained under 600 milliseconds. The eigendecomposition step, which has theoretical complexity O(n^3) in the number of chunks, contributed less than 70 milliseconds even for the largest inputs, as the scipy sparse eigensolver exploits the structure of the graph Laplacian.
