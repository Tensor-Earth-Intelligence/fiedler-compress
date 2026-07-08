# fiedler-compress

**Spectral graph-theoretic prompt compression — a transparent middleware layer for LLM and agentic pipelines.**

fiedler-compress uses the Fiedler vector — the second-smallest eigenvector of the graph Laplacian — to identify and remove semantically disconnected content from LLM prompts. The result: shorter prompts that preserve meaning, save tokens, and fit more useful context into your model's window. It runs as a lightweight, CPU-only preprocessing step, making it a drop-in middleware stage in front of any LLM.

## Why Spectral Compression?

Existing prompt compression tools (LLMLingua, TOON, selective summarization) treat text as a linear sequence and use statistical or neural methods to decide what to cut. They work, but they can't see **structural relationships** — they don't know that paragraph 3 is semantically load-bearing because it bridges two otherwise disconnected topics, or that paragraphs 7 and 8 say the same thing in different words.

fiedler-compress builds a **similarity graph** over your text chunks, computes the spectral decomposition, and uses the Fiedler vector to find the natural semantic partitions. Chunks at the spectral periphery — weakly connected to the rest of the content — are the ones that can be safely removed. Chunks that bridge partitions are preserved, because removing them would fragment the prompt's information structure.

This isn't summarization. It's **graph surgery**.

## Why Middleware

The pipeline is designed to sit transparently between your application and the model: text goes in, a smaller functionally-equivalent prompt comes out, and the downstream LLM call proceeds unchanged. It's CPU-only (no GPU, no neural model required for the core path) and fast enough to run synchronously in a request path without perceptible overhead.

## For Agentic and Long-Context Workflows

Agentic systems accumulate context fast — tool outputs, retrieved documents, prior reasoning, and multi-turn history all compete for a finite context window. fiedler-compress is built for this setting: compress retrieved documents and accumulated context to fit more useful material into the window. **Zone-aware protection** automatically detects instruction content (directives, constraints, format specs) and shields it from removal, so the parts of the prompt that steer the model survive compression while redundant context is pruned. This makes it a practical component for RAG pipelines, long-document processing, and multi-step agent memory.

## Installation

```bash
pip install fiedler-compress
```

The open-core package runs entirely on NumPy and SciPy. A separate commercial tier offers additional capabilities (see below).

## Quick Start

### Python API

```python
from fiedler_optimizer import optimize

result = optimize("""
You are an expert financial analyst. Always respond in JSON.

Context: The company was founded in 2015. Revenue was $45M in Q3,
up 23% YoY. Operating expenses were $38M. Cash reserves: $120M.
Their competitor reported $52M but is losing mid-market share.
Industry analysts predict 15% sector growth. The CFO noted
international expansion costs impacted margins. Retention is 94%.
Average contract value up 18% to $85K.

Task: Analyze Q3 performance.
""")

print(result.compressed)
print(f"Saved ~{result.tokens_saved} tokens ({result.compression_ratio:.0%} reduction)")
print(f"Algebraic connectivity λ₂ = {result.algebraic_connectivity:.4f}")
```

### CLI

```bash
# Compress a prompt
fiedler optimize "Your long prompt here..."

# Compress from file
fiedler optimize --file my_prompt.txt

# Aggressive compression (30% target removal)
fiedler optimize --file my_prompt.txt --target 0.30

# JSON output for piping
fiedler optimize --file my_prompt.txt --json

# Show what was removed
fiedler optimize --file my_prompt.txt --verbose

# Run the built-in benchmark
fiedler benchmark
```

## How It Works

1. **Chunk** the input text into semantically meaningful segments (sentences, paragraphs, or adaptive)
2. **Build a similarity graph** — each chunk is a node, edge weights are cosine similarity (TF-IDF by default, neural embeddings optional)
3. **Compute the Fiedler vector** — the eigenvector for λ₂ of the graph Laplacian
4. **Score chunks** by combining spectral centrality with weighted degree
5. **Zone-aware protection** — instruction content (directives, constraints, format specs) gets 2–3× protection weight; context content is the compression target
6. **Prune** the lowest-scoring context chunks up to the target ratio

## Performance

The core pipeline is CPU-only and fast enough to run inline. Representative end-to-end latency (TF-IDF backend, single CPU core, no GPU):

| Input size | End-to-end latency |
|-----------:|-------------------:|
| 1,000 tokens | ~12 ms |
| 10,000 tokens | ~110 ms |
| 50,000 tokens | ~550 ms |

At typical production prompt sizes (1,000–10,000 tokens) the pipeline adds on the order of tens of milliseconds, negligible against typical LLM inference latency. Compression is a lossy operation — recall of fine-grained facts decreases as the target ratio increases — so keep must-keep content in protected instruction zones and tune the target ratio to your task. Detailed benchmarks will be published in a dedicated benchmarks document.

## Key Features

- **Dependency-free core** — only NumPy and SciPy required. No neural models, no API calls, runs entirely local
- **Zone-aware compression** — distinguishes instructions from context, applies differential protection so directives survive
- **Algebraic connectivity metric** — λ₂ tells you how tightly structured your prompt is *before* compression
- **Transparent middleware** — smaller prompt in, unchanged LLM call out; runs inline with negligible latency
- **JSON output** — pipe results into your existing toolchain

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_ratio` | `0.20` | Fraction of text to remove (0.0–1.0) |
| `strategy` | `adaptive` | Chunking: `adaptive`, `sentence`, `paragraph`, `window` |
| `protect_instructions` | `True` | Shield instruction zones from removal |
| `min_chunks` | `4` | Minimum chunks required for spectral analysis |
| `vectors` | `None` | Pre-computed embeddings (n_chunks × d) |

## Project Structure

```
fiedler-compress/
├── fiedler_optimizer/
│   ├── __init__.py      # Public API
│   ├── core.py          # optimize() pipeline
│   ├── chunker.py       # Text segmentation strategies
│   ├── graph.py         # Similarity graph + Fiedler vector
│   ├── zones.py         # Instruction/context zone detection
│   └── cli.py           # Command-line interface
├── tests/
│   └── test_core.py     # Test suite
├── pyproject.toml       # Package config
├── LICENSE              # FSL-1.1-ALv2
└── README.md
```

## Commercial Tier

This package is the open-core distribution: the TF-IDF + single-eigenvector (k=1) spectral compression pipeline. Additional capabilities — including **content attestation / certification** — are available as a commercial add-on.

For commercial licensing or attestation inquiries:
**Tensor Earth Intelligence (TEI), LLC** — tensor.earth.intelligence@gmail.com (Mark Chappell).

## Roadmap

- [x] Core spectral compression with TF-IDF similarity
- [x] Zone-aware instruction protection
- [x] CLI with JSON output
- [ ] VS Code extension with semantic density visualization
- [ ] Additional capabilities available in the commercial tier

## Background

The Fiedler vector was originally conceived by Miroslav Fiedler (1973) for graph partitioning. This implementation applies it to natural language, treating text chunks as nodes in a semantic similarity graph. The spectral decomposition reveals the natural information structure of a prompt — which parts are tightly interconnected and which are peripheral — enabling principled, structure-aware compression.

## License

Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License (FSL-1.1-ALv2). See [LICENSE](LICENSE) for full terms.