# fiedler-compress

**Spectral graph-theoretic prompt compression — a transparent middleware layer for LLM and agentic pipelines.**

fiedler-compress uses the Fiedler vector — the second-smallest eigenvector of the graph Laplacian — to identify and remove semantically disconnected content from LLM prompts. The result: shorter prompts that preserve meaning, save tokens, and fit more useful context into your model's window. It runs as a lightweight, CPU-only preprocessing step, making it a drop-in middleware stage in front of any LLM.

## Why Spectral Compression?

fiedler-compress builds a **similarity graph** over your text chunks, computes the spectral decomposition, and uses the Fiedler vector to find the natural semantic partitions. Chunks at the spectral periphery — weakly connected to the rest of the content — are the ones that can be safely removed. Chunks that bridge partitions are preserved, because removing them would fragment the prompt's information structure.

This isn't summarization. It's **graph surgery** — and it's extractive, so surviving text is a verbatim subset of the input, in original order.

### How it compares — the honest version

The main alternative is a *learned* compressor such as **LLMLingua-2**, which trains a transformer to classify each token keep/drop. We benchmarked against it directly on identical data, and the trade is clear in both directions:

- **LLMLingua-2 retains more answers at the same compression ratio** — by roughly 17–29 exact-match points at 1.5–2.5× on our multi-passage QA benchmark. A supervised model trained for this task should win on quality, and it does. We do not claim otherwise.
- **fiedler-compress is ~1,205× cheaper to run** — 1.9 ms vs 2.34 s per document on the same CPU, with no model download, no tokenizer coupling, and no GPU.
- **When you know what to protect, the chunk-based pin wins decisively** — given the same target span, pinning the containing chunk retained answers 25–45 points better than LLMLingua-2's token-level `force_tokens` list, because a span plus its surrounding context survives as one contiguous unit.

So this is a **cost-and-dependency trade, not a quality claim**. Choose it when compression must be cheap, local, synchronous, and dependency-light; choose a learned compressor when maximum retention at a fixed token budget is what matters most.

Full numbers, confidence intervals, raw per-item scores, and the script that regenerates every table: **[`benchmarks/`](benchmarks/)** (see [`benchmarks/SWEEP.md`](benchmarks/SWEEP.md)).

## Why Middleware

The pipeline is designed to sit transparently between your application and the model: text goes in, a smaller functionally-equivalent prompt comes out, and the downstream LLM call proceeds unchanged. It's CPU-only (no GPU, no neural model required for the core path) and fast enough to run synchronously in a request path without perceptible overhead.

## For Agentic and Long-Context Workflows

Agentic systems accumulate context fast — tool outputs, retrieved documents, prior reasoning, and multi-turn history all compete for a finite context window. fiedler-compress is built for this setting: compress retrieved documents and accumulated context to fit more useful material into the window. **Zone-aware protection** automatically detects instruction content (directives, constraints, format specs) and shields it from removal, so the parts of the prompt that steer the model survive compression while redundant context is pruned. This makes it a practical component for RAG pipelines, long-document processing, and multi-step agent memory.

## Installation

```bash
pip install fiedler-compress
```

The package runs entirely on NumPy and SciPy — no neural models, no API calls, no GPU.

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

The core pipeline is CPU-only and fast enough to run inline. Measured end-to-end latency (TF-IDF backend, default settings, single CPU core, no GPU; mean of 5 runs on concatenated real prose):

| Input size | End-to-end latency |
|-----------:|-------------------:|
| ~1,800 tokens | ~2 ms |
| ~10,000 tokens | ~13 ms |
| ~50,000 tokens | ~71 ms |

At typical production prompt sizes the pipeline adds single-digit to low-tens of milliseconds — negligible against LLM inference latency.

Compression is **lossy, and the loss is task-dependent**: recall of fine-grained facts decreases as the target ratio increases. On our multi-passage QA benchmark, exact-match answer recall falls from ~84–89% uncompressed to ~15–19% at 5× compression when the compressor is given no hint about what matters — and returns to ~81–85% when the relevant span is pinned. Keep must-keep content in protected instruction zones or pin it explicitly, and tune the target ratio to your task rather than assuming a default is safe.

Full benchmark — 150 multi-passage SQuAD documents, a 1.5–5× compression sweep, five open-weight models (2B–30B), two deterministic scorers, bootstrap 95% confidence intervals, and a no-context control for training-memory leakage — is in **[`benchmarks/SWEEP.md`](benchmarks/SWEEP.md)**, with raw per-item scores and the regenerating script alongside it.

Compression itself is deterministic — identical input yields byte-identical output. This is locked by **[`tests/test_determinism.py`](tests/test_determinism.py)** (CI regression guard) and demonstrated end-to-end by **[`benchmarks/reproduce_determinism.py`](benchmarks/reproduce_determinism.py)**.

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
├── LICENSE              # Apache-2.0
└── README.md
```

## Scope

As of 0.4.0 there is no commercial tier and nothing is held back. Everything the
library can do ships here under Apache-2.0: spectral compression, ligatures,
topology caching, distillation, spectral obscuring, reasoning templates, signed
certificates, alternative similarity backends, and the geometric analysis
modules. Earlier versions raised `CommercialTierError` from these paths; that
error and the tier behind it are gone.

The heavier capabilities are optional installs rather than optional licences:

```bash
pip install fiedler-compress                 # core: NumPy + SciPy only
pip install fiedler-compress[embeddings]     # neural similarity
pip install fiedler-compress[geometry]       # Voronoi / Minkowski / conformal
pip install fiedler-compress[distill]        # LLM-backed distillation
pip install fiedler-compress[backends]       # Wasserstein / hyperbolic metrics
```

Enquiries: **Tensor Earth Intelligence (TEI), LLC**,
founder@tensorearthintelligence.com (Mark Chappell).

## Roadmap

- [x] Core spectral compression with TF-IDF similarity
- [x] Zone-aware instruction protection
- [x] CLI with JSON output
- [x] Full capability released open source (0.4.0)
- [ ] VS Code extension with semantic density visualization

## Background

The Fiedler vector was originally conceived by Miroslav Fiedler (1973) for graph partitioning. This implementation applies it to natural language, treating text chunks as nodes in a semantic similarity graph. The spectral decomposition reveals the natural information structure of a prompt — which parts are tightly interconnected and which are peripheral — enabling principled, structure-aware compression.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for full terms.