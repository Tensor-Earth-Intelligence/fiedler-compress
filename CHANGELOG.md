# Changelog

All notable changes to `fiedler-compress` are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/),
and the project uses [semantic versioning](https://semver.org/).

## [0.3.0] — unreleased

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
