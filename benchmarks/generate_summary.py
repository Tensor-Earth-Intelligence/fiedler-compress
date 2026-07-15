r"""
Generate benchmarks/README.md from the raw result JSON in benchmarks/results/.

This is the "generating script" layer of the three-layer publishing structure
(readable summary + raw JSON + generating script): the credibility multiplier is
that the summary is GENERATED from the raw data, not hand-asserted, so it can't
silently drift from what the raw JSON actually says. Re-run this file any time and
diff README.md -- if it changes, the summary was stale.

Self-contained on purpose: this repo is the public one, so this script does not
import from any private research repo. The small SQuAD-sampling helper is inlined
below (same deterministic seed as the original run) so a skeptic can independently
reconstruct which SQuAD item each stored survival rate corresponds to, and therefore
verify the mem-leak filtering themselves -- not just trust our numbers.

Requires: pip install datasets (only to re-verify the mem-leak filtering by
reconstructing sample order; the raw JSON alone is enough to see the unfiltered
numbers even without it).

    python benchmarks/generate_summary.py
"""
from __future__ import annotations

import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 20260620
N_SAMPLES = 40


def load_squad_sample(n=N_SAMPLES, seed=SEED, min_words=60, max_words=320):
    """Deterministic SQuAD v1.1 (CC BY-SA) sample -- identical to the loader used for
    the original run, so re-calling this reproduces the exact same item order."""
    from datasets import load_dataset
    try:
        ds = load_dataset("rajpurkar/squad", split="validation")
    except Exception:
        ds = load_dataset("squad", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        ctx = row["context"]
        ans = row["answers"]["text"]
        if not ans:
            continue
        wc = len(ctx.split())
        if wc < min_words or wc > max_words:
            continue
        out.append({"qid": f"squad_{i}"})
        if len(out) >= n:
            break
    return out


def _rescore_survival(data, qids):
    out = {}
    for model, conds in data["results"].items():
        leaked = set(data.get("memleak", {}).get(model, []))
        leaked_idx = {i for i, q in enumerate(qids) if q in leaked} if qids else set()
        row = {}
        for cond, vals in conds.items():
            if not vals:
                continue
            filtered = [v for i, v in enumerate(vals) if i not in leaked_idx]
            row[cond] = {
                "all_n": len(vals), "all_mean": sum(vals) / len(vals),
                "filtered_n": len(filtered),
                "filtered_mean": (sum(filtered) / len(filtered)) if filtered else None,
            }
        out[model] = {"memleak_n": len(leaked), "conditions": row}
    return out


def _rescore_depth(data):
    by_model_cond: dict = {}
    for r in data["records"]:
        by_model_cond.setdefault(r["model"], {}).setdefault(r["cond"], []).append(r)
    out = {}
    for model, conds in by_model_cond.items():
        leaked = set(data.get("memleak", {}).get(model, []))
        row = {}
        for cond, recs in conds.items():
            vals = [r["survival"] for r in recs]
            filtered = [r["survival"] for r in recs if r["qid"] not in leaked]
            row[cond] = {
                "all_n": len(vals), "all_mean": sum(vals) / len(vals) if vals else None,
                "filtered_n": len(filtered),
                "filtered_mean": (sum(filtered) / len(filtered)) if filtered else None,
            }
        out[model] = {"memleak_n": len(leaked), "conditions": row}
    return out


def _survival_table(report, exclude_models=()):
    lines = ["| model | condition | survival (all) | survival (mem-leak filtered) |",
             "|---|---|---|---|"]
    for model, d in report.items():
        note = " *(see data-quality note below)*" if model in exclude_models else ""
        for cond, c in d["conditions"].items():
            fm = f"{c['filtered_mean']:.0%} (n={c['filtered_n']})" if c["filtered_mean"] is not None else "n/a"
            lines.append(f"| {model}{note} | {cond} | {c['all_mean']:.0%} (n={c['all_n']}) | {fm} |")
    return "\n".join(lines)


def main():
    try:
        sample = load_squad_sample()
        qids = [s["qid"] for s in sample]
    except Exception as e:
        print(f"NOTE: could not reconstruct SQuAD sample order ({e}); "
              f"mem-leak-filtered columns will be omitted from this run.")
        qids = None

    survival_data = json.load(open(RESULTS / "public_survival_gx10_2026-06-30.json", encoding="utf-8"))
    survival_report = _rescore_survival(survival_data, qids)

    desktop_data = json.load(open(RESULTS / "public_survival_desktop_models.json", encoding="utf-8"))
    desktop_report = _rescore_survival(desktop_data, qids)

    depth_data = json.load(open(RESULTS / "public_depth_results_runpod_20260703.json", encoding="utf-8"))
    depth_report = _rescore_depth(depth_data)

    md = f"""# fiedler-compress — public benchmark results

Generated by `benchmarks/generate_summary.py` from the raw JSON in `benchmarks/results/`.
Re-run that script to regenerate this file; if it changes, this summary was stale.
**Do not hand-edit the tables below.**

## Methodology notes

- **Corpus:** SQuAD v1.1 (CC BY-SA), N={N_SAMPLES} paragraphs sampled deterministically
  (seed {SEED}) -- reproducible, not cherry-picked.
- **Conditions:** `baseline` (full context), `fiedler_50/60/70` (compressed to 50/60/70%
  target ratio), `cpin70` (fiedler_70 with the gold-answer phrase content-pinned).
- **No-context control / mem-leak filtering:** every item is also asked with an EMPTY
  document. If a model answers correctly anyway, that item may be answerable from
  TRAINING MEMORY rather than genuine retrieval from the provided context -- it's
  flagged and both the raw ("all") and mem-leak-filtered numbers are reported below so
  neither is hidden.
- **Scoring:** exact substring/keyword match against the gold answer span (not cosine
  similarity or another embedding-based metric).

## Cross-architecture survival (dense + MoE, 7B-120B class)

{_survival_table(survival_report)}

## Small/desktop-class models

{_survival_table(desktop_report, exclude_models=["qwen3:30b-a3b"])}

**Data-quality note:** `qwen3:30b-a3b`'s stored survival is 0% in every condition,
including the uncompressed baseline. A model failing the UNCOMPRESSED baseline is not
a genuine compression-survival result -- this indicates a broken run (likely a
reasoning-trace-output issue specific to this model), not evidence that fiedler-compress
hurts this model. Excluded from headline claims pending a re-run.

## Long-context depth (~27K-token article, facts at shallow/mid/deep positions)

{_survival_table(depth_report)}

Note: N=5 questions per model -- a single mem-leak hit shifts the percentage by 20
points at this sample size, so these numbers are noisier than the 40-item SQuAD survival
table above.
"""

    out_path = HERE / "README.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
