# Changelog

All notable changes to `fiedler-compress` are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/),
and the project uses [semantic versioning](https://semver.org/).

## [0.4.0] — 2026-09-04

The previously commercial capability is now part of the open package under the
same Apache-2.0 licence. There is no held-back tier: what the library can do is
what ships here.

### Changed
- **`CommercialTierError` is gone**, along with the `_tier` module that raised
  it. Eight code paths in `optimize()` and the neural similarity backend in
  `build_similarity_graph()` previously raised it when their implementation was
  absent. They now run. Code that caught `CommercialTierError` should drop the
  handler; nothing else in the calling convention changed.
- Heavier scientific dependencies stay optional. The core install is still just
  NumPy and SciPy; the new extras are `embeddings`, `geometry`, `distill` and
  `backends`. Importing `fiedler_optimizer.geometry` without its extra raises an
  `ImportError` naming the install command rather than failing on `sklearn`.

### Added
- **Ligatures** (`ligature_rules=`, `emit_ligatures=`): similarity-graph
  enrichment from rule sets (`project_management`, `legal`, `rag`) and
  post-compression annotations describing what was joined.
- **Topology caching** (`topology_cache=`): classifies prompt topology and
  warm-starts compression parameters from a previous run. The cache is JSON on
  disk (never pickle) and defaults to `.fiedler_cache` in the working directory.
- **Distillation** (`distill_backend=`): LLM-backed rewriting of the kept text,
  anchored on ligatures so structure survives. Needs an API key in
  `FIEDLER_DISTILL_API_KEY`; the key is never logged.
- **Spectral obscuring** (`obscure=`) and **reasoning templates** (`template=`).
- **Signed certificates** (`certify=`, `provenance=`): HMAC attestation over the
  spectrum, the compressed output, and, for provenance, a commitment to the
  source prompt. Signing keys are hex-encoded; one is generated with
  `secrets.token_hex(32)` when not supplied.
- **Neural similarity** (`use_neural=True`): sentence-transformers embeddings.
  Models are restricted to an allowlist pinned to specific Hugging Face revision
  hashes, and the embedding dimension is verified after loading, so a tampered
  or silently updated model is rejected rather than used.
- **Alternative similarity backends** (`backend=`): Aitchison, Fisher-Rao,
  Wasserstein and hyperbolic metrics, selected by name. Wasserstein needs the
  `backends` extra; the rest are pure NumPy.
- **Content priors** (`content_prior=`): pin by content type rather than by
  hand-written regex. Accepts a callable, or the curated preset names
  `identifier`, `salience` and `observation` (which need the `geometry` extra).
  Generated patterns pass through the same `validate_pin_patterns()` guard as
  user-supplied ones, and compose with an explicit `pin_patterns` rather than
  replacing it.
- **Coverage floor** (`min_keep_per_cluster=`, `coverage_auto=`,
  `cluster_labels=`): guarantees each topical cluster keeps at least one
  survivor, so a fact sitting alone on its own topic is not dropped merely for
  scoring low. On a redundant document with a single distinct needle this
  retains the needle at the same compression that would otherwise discard it.
  `coverage_auto` applies the floor only when one cluster dominates the input,
  leaving balanced inputs (documents, RAG passages) untouched.
- **Geometric analysis** (`fiedler_optimizer.geometry`): Voronoi density
  pruning, Minkowski positional-semantic metrics, and conformal UMAP embeddings.
  Voronoi is an independent compression pathway rather than a variation on the
  Fiedler one.
- **`fiedler_optimizer.dymaxion`** and **`euler_curves`**: geodesic and
  curve-based experiments. These are research code, exposed as-is.
- **`ChunkingStrategy.CODE`**: line-level chunking for source and configuration
  text, selected automatically by `ADAPTIVE`. Structured input previously had no
  structure the sentence and paragraph splitters recognised, so compression
  barely engaged. Instruction prose that merely contains `:` and `;` still
  routes to the prose splitters.

### Fixed
- **Overlapping chunks were emitted twice.** Compressed output is now
  reassembled from the original character spans (`merge_kept_spans`), merging
  overlapping windows. Joining chunk text directly duplicated the region shared
  by adjacent windows, which inflated output size and could make the reported
  compression negative under `SLIDING_WINDOW` and `ADAPTIVE`.
- Similarity matrices are sanitised for `NaN`/`Inf` before the Laplacian is
  built. The alternative backends can produce them, and they would otherwise
  propagate silently into the eigendecomposition.

### Notes
- The ReDoS and input-size guards added for 0.3.0 (`validate_pin_patterns`,
  `MAX_PIN_PATTERNS`, `MAX_CHUNKS`) are unchanged and still apply to every
  caller, including the newly opened paths.
- The distillation backend is the least exercised part of this release: it
  requires a live API key, so it is not covered by the test suite.

## [0.3.0] — 2026-08-24

### Fixed
- **Faithful chunk offsets.** `chunk_text` now carries `(start_char, end_char)`
  spans through splitting and merging instead of recovering them by anchor
  search, so the invariant `normalized[chunk.start_char:chunk.end_char] ==
  chunk.text` holds for every strategy. Previously the offsets could point at
  the wrong span, breaking any caller that sliced the original text by them.
- **Deterministic Fiedler vector.** `compute_fiedler_vector` now seeds ARPACK's
  start vector (`v0`) and sign-canonicalizes the result. Previously ARPACK's
  random start made the Fiedler vector — and therefore every chunk score and
  removal decision — vary run to run. Algebraic connectivity (λ₂) is unchanged;
  scoring already used `|v_i|`, so only stability changes, not the eigenvalue.
- `quality.py` truncation now slices the Unicode-normalized text, matching the
  text the chunker actually operates on.
- Version string was inconsistent (`__init__.py` reported `0.1.1` while the
  package built as `0.2.0`); both now report `0.3.0`.

### Added
- `benchmarks/reproduce_determinism.py` and `tests/test_determinism.py`: a
  reproducibility artifact + regression guard for the determinism fix. Verified
  that pre-fix and seeded code produce byte-identical compressed output on the
  Paper-1 SQuAD-150 sweep (all 7 conditions, 0/1050 differences at 24 reps/case),
  so published benchmark numbers reproduce exactly under the seeded release.

### Changed
- Line endings normalized to LF via `.gitattributes` (`* text=auto eol=lf`).

## [0.2.0] — 2026 (tagged, not published to PyPI)

### Fixed
- GSM8K benchmark harness: average the uncompressed baseline over repetitions
  (default 3) for stable comparisons.

## [0.1.2] — 2026 (tagged, not published to PyPI)

- Apache-2.0 licensing housekeeping.

## [0.1.1] — 2026-07-15

- Published to PyPI. Open-core TF-IDF + single-eigenvector (k=1) pipeline.

## [0.1.0] — 2026-07-08

- Initial open-core release.
