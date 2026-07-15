"""
Command-line interface for Fiedler Optimizer.

Usage:
    fiedler optimize "Your long prompt here..."
    fiedler optimize --file prompt.txt
    fiedler optimize --file prompt.txt --target 0.30
    fiedler benchmark
    fiedler benchmark --suite custom_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fiedler_optimizer.core import optimize, FiedlerResult
from fiedler_optimizer.caveman import caveman_compress, CavemanResult, _count_tokens
from fiedler_optimizer.chunker import ChunkingStrategy

# Maximum input file size (10 MB) to prevent resource exhaustion
_MAX_FILE_BYTES = 10 * 1024 * 1024


def _safe_resolve(user_path: str, label: str = "file") -> Path:
    """
    Resolve a user-supplied path with security guards.

    Rejects path-traversal attempts (``..`` components) and enforces the
    global file-size limit.  Returns the resolved :class:`Path` on success
    or calls ``sys.exit(1)`` on failure.
    """
    raw = Path(user_path)

    # Reject paths containing '..' to prevent directory traversal.
    # Check the raw (pre-resolution) path so that an attacker cannot
    # use symlinks or casing tricks to hide the traversal.
    if ".." in raw.parts:
        print(f"Error: path traversal not allowed in {label}: {user_path}",
              file=sys.stderr)
        sys.exit(1)

    path = raw.resolve()

    if not path.exists():
        print(f"Error: {label} not found: {user_path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_file():
        print(f"Error: not a regular file: {user_path}", file=sys.stderr)
        sys.exit(1)

    file_size = path.stat().st_size
    if file_size > _MAX_FILE_BYTES:
        print(
            f"Error: {label} too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Maximum is {_MAX_FILE_BYTES // 1024 // 1024} MB.",
            file=sys.stderr,
        )
        sys.exit(1)

    return path


def _format_result(result: FiedlerResult, verbose: bool = False) -> str:
    """Format a FiedlerResult for terminal output."""
    lines = []
    lines.append("=" * 60)
    lines.append("  FIEDLER SPECTRAL COMPRESSION RESULT")
    lines.append("=" * 60)
    lines.append(f"  Compression ratio:      {result.compression_ratio:.1%}")
    lines.append(f"  Tokens saved (approx):  {result.tokens_saved}")
    lines.append(f"  Chunks: {result.chunks_total} total, {result.chunks_removed} removed")
    lines.append(f"  Algebraic connectivity: {result.algebraic_connectivity:.6f}")
    lines.append("=" * 60)

    if verbose and result.removed_chunks:
        lines.append("\n  REMOVED CHUNKS:")
        lines.append("  " + "-" * 56)
        for i, chunk in enumerate(result.removed_chunks, 1):
            preview = chunk[:100].replace("\n", " ")
            if len(chunk) > 100:
                preview += "..."
            lines.append(f"  [{i}] {preview}")
        lines.append("")

    lines.append("\n  COMPRESSED OUTPUT:")
    lines.append("  " + "-" * 56)
    lines.append(result.compressed)
    lines.append("")

    return "\n".join(lines)


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run Fiedler compression on input text."""
    # Get input text
    if args.file:
        path = _safe_resolve(args.file, label="input file")
        text = path.read_text(encoding="utf-8")
    elif args.text:
        text = " ".join(args.text)
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("Reading from stdin (Ctrl+D to finish):", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("Error: no input text provided.", file=sys.stderr)
        sys.exit(1)

    # Validate target ratio
    if not (0.0 < args.target < 1.0):
        print("Error: --target must be between 0.0 and 1.0 (exclusive).", file=sys.stderr)
        sys.exit(1)

    # Parse strategy
    strategy_map = {
        "adaptive": ChunkingStrategy.ADAPTIVE,
        "sentence": ChunkingStrategy.SENTENCE,
        "paragraph": ChunkingStrategy.PARAGRAPH,
        "window": ChunkingStrategy.SLIDING_WINDOW,
    }
    strategy = strategy_map.get(args.strategy, ChunkingStrategy.ADAPTIVE)

    # Build pin_patterns from flags
    pin_patterns: list[str] | None = None
    raw_pins: list[str] = []
    if getattr(args, "pin_instructions", False):
        from fiedler_optimizer.pinning import INSTRUCTION_PRESET
        raw_pins.extend(INSTRUCTION_PRESET)
    if getattr(args, "pin_regex", None):
        raw_pins.extend(args.pin_regex)
    if getattr(args, "pin_sections", None):
        from fiedler_optimizer.pinning import section_pin_patterns
        keywords = [k.strip() for k in args.pin_sections.split(",") if k.strip()]
        raw_pins.extend(section_pin_patterns(keywords, text))
    if raw_pins:
        pin_patterns = raw_pins

    # Determine pipeline mode
    caveman_pre = getattr(args, "pre_caveman", None)
    caveman_post = getattr(args, "post_caveman", None)
    caveman_only = getattr(args, "caveman_only", None)

    # Run compression pipeline
    t0 = time.perf_counter()
    stages: list[dict] = []
    total_input_tokens = _count_tokens(text)
    result = None  # FiedlerResult, set if Fiedler runs
    current_text = text

    # --- Stage: pre-caveman ---
    if caveman_pre:
        cave_result = caveman_compress(current_text, level=caveman_pre)
        stages.append({
            "name": "caveman",
            "input_tokens": cave_result.original_tokens,
            "output_tokens": cave_result.compressed_tokens,
            "ratio": round(cave_result.compression_ratio, 6),
        })
        current_text = cave_result.text

    # --- Stage: Fiedler (unless --caveman-only) ---
    if caveman_only:
        cave_result = caveman_compress(current_text, level=caveman_only)
        stages.append({
            "name": "caveman",
            "input_tokens": cave_result.original_tokens,
            "output_tokens": cave_result.compressed_tokens,
            "ratio": round(cave_result.compression_ratio, 6),
        })
        current_text = cave_result.text
    else:
        fiedler_input_tokens = _count_tokens(current_text)
        try:
            result = optimize(
                current_text,
                target_ratio=args.target,
                strategy=strategy,
                protect_instructions=not args.no_protect,
                pin_patterns=pin_patterns,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        fiedler_output_tokens = _count_tokens(result.compressed)
        fiedler_ratio = (1.0 - fiedler_output_tokens / fiedler_input_tokens
                         if fiedler_input_tokens > 0 else 0.0)
        stages.append({
            "name": "fiedler",
            "input_tokens": fiedler_input_tokens,
            "output_tokens": fiedler_output_tokens,
            "ratio": round(fiedler_ratio, 6),
        })
        current_text = result.compressed

    # --- Stage: post-caveman ---
    if caveman_post:
        cave_result = caveman_compress(current_text, level=caveman_post)
        stages.append({
            "name": "caveman",
            "input_tokens": cave_result.original_tokens,
            "output_tokens": cave_result.compressed_tokens,
            "ratio": round(cave_result.compression_ratio, 6),
        })
        current_text = cave_result.text

    elapsed = time.perf_counter() - t0
    total_output_tokens = _count_tokens(current_text)
    total_ratio = (1.0 - total_output_tokens / total_input_tokens
                   if total_input_tokens > 0 else 0.0)

    # Build stage metadata (included when any caveman flag is active)
    has_caveman = caveman_pre or caveman_post or caveman_only
    pipeline_meta = {
        "stages": stages,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_compression_ratio": round(total_ratio, 6),
    } if has_caveman else None

    if args.json:
        if result is not None:
            # Fiedler ran — full output
            output = {
                "compressed": current_text,
                "compression_ratio": result.compression_ratio,
                "tokens_saved": result.tokens_saved,
                "algebraic_connectivity": result.algebraic_connectivity,
                "chunks_total": result.chunks_total,
                "chunks_removed": result.chunks_removed,
                "elapsed_seconds": round(elapsed, 4),
            }
            if args.verbose:
                output["removed_chunks"] = result.removed_chunks
                output["chunk_scores"] = result.chunk_scores
        else:
            # Caveman-only — minimal output
            output = {
                "compressed": current_text,
                "elapsed_seconds": round(elapsed, 4),
            }
        if pipeline_meta:
            output["pipeline"] = pipeline_meta
        print(json.dumps(output, indent=2))
    else:
        if result is not None:
            # Fiedler ran — use standard formatter (shows Fiedler's compressed
            # text; if post-caveman was applied we override the displayed text)
            if caveman_post:
                # Override only the displayed text with the post-caveman result;
                # dataclasses.replace copies all other fields so this can't drift
                # out of sync when FiedlerResult gains fields.
                import dataclasses
                patched = dataclasses.replace(result, compressed=current_text)
                print(_format_result(patched, verbose=args.verbose))
            else:
                print(_format_result(result, verbose=args.verbose))
        else:
            # Caveman-only text output
            print("=" * 60)
            print("  CAVEMAN GRAMMAR-STRIP RESULT")
            print("=" * 60)
            print(f"  Level:              {caveman_only}")
            print(f"  Input tokens:       {total_input_tokens}")
            print(f"  Output tokens:      {total_output_tokens}")
            print(f"  Compression ratio:  {total_ratio:.1%}")
            print("=" * 60)
            print("\n  COMPRESSED OUTPUT:")
            print("  " + "-" * 56)
            print(current_text)
            print("")
        if pipeline_meta:
            print(f"\n  PIPELINE STAGES:")
            for stg in pipeline_meta["stages"]:
                print(f"    {stg['name']:10s}  {stg['input_tokens']} → {stg['output_tokens']} tokens  ({stg['ratio']:.1%})")
            print(f"    {'total':10s}  {pipeline_meta['total_input_tokens']} → "
                  f"{pipeline_meta['total_output_tokens']} tokens  "
                  f"({pipeline_meta['total_compression_ratio']:.1%})")
        print(f"  Completed in {elapsed:.3f}s")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the built-in benchmark suite."""
    # Built-in test prompts covering different domains and structures
    suite = [
        {
            "name": "System prompt with context",
            "text": (
                "You are an expert financial analyst. Always respond in JSON format. "
                "Never include personal opinions. Use precise numerical data.\n\n"
                "Context: The company was founded in 2015 in San Francisco. It has "
                "grown to 500 employees across 3 offices. Their primary product is a "
                "SaaS platform for supply chain management. In Q3 2025, revenue was "
                "$45M, up 23% year-over-year. Operating expenses were $38M. The company "
                "has $120M in cash reserves. Their main competitor, SupplyTrack, reported "
                "$52M in Q3 revenue but has been losing market share in the mid-market "
                "segment. Industry analysts predict 15% sector growth in 2026. The CFO "
                "noted that international expansion costs impacted margins. Customer "
                "retention rate is 94%. Average contract value increased 18% to $85K.\n\n"
                "Task: Analyze Q3 performance and provide a forward-looking assessment."
            ),
        },
        {
            "name": "RAG context with redundancy",
            "text": (
                "Based on the following documents, answer the user's question.\n\n"
                "Document 1: The Eiffel Tower is a wrought-iron lattice tower in Paris, "
                "France. It was constructed from 1887 to 1889 as the centerpiece of the "
                "1889 World's Fair. The tower is 330 metres tall and was the tallest "
                "man-made structure in the world until 1930.\n\n"
                "Document 2: The Eiffel Tower, located on the Champ de Mars in Paris, "
                "is one of the most recognizable structures in the world. Built by "
                "Gustave Eiffel's engineering company, the iron tower stands at a height "
                "of 330 meters. It was completed in 1889 for the World's Fair.\n\n"
                "Document 3: Paris landmarks include the Eiffel Tower (built 1889, "
                "330m tall), the Louvre Museum, Notre-Dame Cathedral, and the Arc de "
                "Triomphe. The city attracts over 30 million tourists annually.\n\n"
                "Document 4: Gustave Eiffel also designed the internal structural "
                "elements of the Statue of Liberty. His engineering firm was known "
                "for bridge and viaduct construction across Europe before the tower "
                "project. The Garabit Viaduct in southern France was a notable "
                "predecessor project.\n\n"
                "User question: How tall is the Eiffel Tower?"
            ),
        },
        {
            "name": "Multi-shot examples with instructions",
            "text": (
                "You are a sentiment classifier. Classify each review as POSITIVE, "
                "NEGATIVE, or NEUTRAL. Respond with only the label.\n\n"
                "Example 1: 'This product changed my life! Best purchase ever.' → POSITIVE\n"
                "Example 2: 'Terrible quality. Broke after one day.' → NEGATIVE\n"
                "Example 3: 'It's okay. Does what it says.' → NEUTRAL\n"
                "Example 4: 'Amazing customer service and fast shipping!' → POSITIVE\n"
                "Example 5: 'Would not recommend. Waste of money.' → NEGATIVE\n"
                "Example 6: 'Product arrived on time. Packaging was standard.' → NEUTRAL\n"
                "Example 7: 'Five stars! Exceeded all my expectations.' → POSITIVE\n"
                "Example 8: 'Completely useless. Returning immediately.' → NEGATIVE\n\n"
                "Now classify: 'The flavor was interesting but the texture was off.'"
            ),
        },
    ]

    if args.suite:
        path = _safe_resolve(args.suite, label="suite file")
        suite = []
        for line_num, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines(), 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error: invalid JSON on line {line_num}: {e}", file=sys.stderr)
                sys.exit(1)
            if not isinstance(entry, dict) or "text" not in entry:
                print(
                    f"Error: line {line_num} missing required 'text' field. "
                    f"Expected format: {{\"name\": \"...\", \"text\": \"...\"}}",
                    file=sys.stderr,
                )
                sys.exit(1)
            suite.append(entry)

    print("=" * 60)
    print("  FIEDLER OPTIMIZER BENCHMARK")
    print("=" * 60)
    print()

    total_saved = 0
    total_original = 0

    for item in suite:
        name = item.get("name", "Unnamed")
        text = item["text"]

        t0 = time.perf_counter()
        result = optimize(text, target_ratio=args.target)
        elapsed = time.perf_counter() - t0

        orig_tokens = len(text) // 4
        total_original += orig_tokens
        total_saved += result.tokens_saved

        print(f"  {name}")
        print(f"    Compression: {result.compression_ratio:.1%}  "
              f"Tokens saved: ~{result.tokens_saved}  "
              f"λ₂: {result.algebraic_connectivity:.4f}  "
              f"Time: {elapsed:.3f}s")
        print(f"    Chunks: {result.chunks_total} → {result.chunks_total - result.chunks_removed}")
        print()

    print("-" * 60)
    avg_ratio = total_saved / max(total_original, 1)
    print(f"  AGGREGATE: ~{total_saved} tokens saved across {len(suite)} prompts "
          f"({avg_ratio:.1%} average)")
    print("=" * 60)


def cmd_benchmark_quality(args: argparse.Namespace) -> None:
    """Run compression quality benchmark against NLP datasets."""
    try:
        from fiedler_optimizer.benchmarks.quality import (
            BenchmarkRunner,
            DATASETS,
            GeminiLLMClient,
            LLMClient,
            format_summary_table,
            report_to_json,
        )
    except ImportError:
        print(
            "Error: the 'benchmark' extra is required.\n"
            "Install with: pip install fiedler-optimizer[benchmark]",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dataset not in DATASETS:
        print(f"Error: unknown dataset {args.dataset!r}. "
              f"Available: {sorted(DATASETS)}", file=sys.stderr)
        sys.exit(1)

    # Parse ratios
    try:
        ratios = [float(r) for r in args.ratios.split(",")]
    except ValueError:
        print("Error: --ratios must be comma-separated numbers (e.g. 2,4,8)",
              file=sys.stderr)
        sys.exit(1)

    for r in ratios:
        if r < 1.0:
            print(f"Error: compression ratio {r} must be >= 1.0", file=sys.stderr)
            sys.exit(1)

    limit = args.limit
    if args.dry_run:
        limit = min(limit or 10, 10)

    provider = getattr(args, "provider", "openai")
    if provider == "gemini":
        llm_client = GeminiLLMClient(
            model=args.model,
            api_key_env=args.api_key_env,
        )
    else:
        llm_client = LLMClient(
            model=args.model,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
        )

    few_shot = getattr(args, "few_shot", 8)
    pin_tool_results = getattr(args, "pin_tool_results", False)
    bench_pin_instructions = getattr(args, "pin_instructions", False)

    bench_pin_patterns: list[str] | None = None
    if bench_pin_instructions:
        from fiedler_optimizer.pinning import INSTRUCTION_PRESET
        bench_pin_patterns = list(INSTRUCTION_PRESET)

    runner = BenchmarkRunner(
        dataset=args.dataset,
        ratios=ratios,
        llm_client=llm_client,
        limit=limit,
        loader_kwargs={"few_shot": few_shot},
        pin_tool_results=pin_tool_results,
        pin_patterns=bench_pin_patterns,
    )

    flag_info = ""
    if pin_tool_results:
        flag_info += ", pin_tool_results=True"
    if bench_pin_instructions:
        flag_info += ", pin_instructions=True"
    print(f"Running quality benchmark: {args.dataset} "
          f"(ratios={args.ratios}, model={args.model}, provider={provider}, "
          f"few_shot={few_shot}{flag_info})",
          file=sys.stderr)
    if limit:
        print(f"  Limit: {limit} samples", file=sys.stderr)

    report = runner.run()
    print(format_summary_table(report))

    if args.output:
        out_path = Path(args.output)
        if ".." in out_path.parts:
            print("Error: path traversal not allowed in --output",
                  file=sys.stderr)
            sys.exit(1)
        out_path.write_text(
            json.dumps(report_to_json(report), indent=2),
            encoding="utf-8",
        )
        print(f"  Results saved to: {args.output}", file=sys.stderr)


def cmd_benchmark_latency(args: argparse.Namespace) -> None:
    """Run latency profiling across input sizes."""
    from fiedler_optimizer.benchmarks.latency import (
        LatencyProfiler,
        format_latency_table,
        report_to_json,
    )

    # Parse sizes
    try:
        sizes = [int(s) for s in args.sizes.split(",")]
    except ValueError:
        print("Error: --sizes must be comma-separated integers", file=sys.stderr)
        sys.exit(1)

    for s in sizes:
        if s < 1:
            print(f"Error: size {s} must be positive", file=sys.stderr)
            sys.exit(1)

    if args.runs < 1:
        print("Error: --runs must be >= 1", file=sys.stderr)
        sys.exit(1)

    strategy_map = {
        "adaptive": ChunkingStrategy.ADAPTIVE,
        "sentence": ChunkingStrategy.SENTENCE,
        "paragraph": ChunkingStrategy.PARAGRAPH,
        "window": ChunkingStrategy.SLIDING_WINDOW,
    }

    profiler = LatencyProfiler(
        sizes=sizes,
        runs=args.runs,
        backend=args.backend,
        target_ratio=args.target,
        strategy=strategy_map.get(args.strategy, ChunkingStrategy.ADAPTIVE),
    )

    print(f"Running latency profile: sizes={args.sizes} "
          f"runs={args.runs} backend={args.backend}", file=sys.stderr)

    report = profiler.run()

    print(format_latency_table(report))

    # Print warnings
    for r in report.results:
        for w in r.warnings:
            print(f"  ⚠ {r.target_tokens} tokens: {w}", file=sys.stderr)

    if args.output:
        out_path = Path(args.output)
        if ".." in out_path.parts:
            print("Error: path traversal not allowed in --output",
                  file=sys.stderr)
            sys.exit(1)
        out_path.write_text(
            json.dumps(report_to_json(report), indent=2),
            encoding="utf-8",
        )
        print(f"  Results saved to: {args.output}", file=sys.stderr)


def main() -> None:
    """CLI entry point."""
    # Ensure UTF-8 output so non-ASCII characters (->, lambda2, warning signs)
    # don't crash on Windows consoles defaulting to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="fiedler",
        description="fiedler-compress — spectral graph-theoretic prompt compression",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- optimize ---
    p_opt = subparsers.add_parser("optimize", help="Compress a prompt")
    p_opt.add_argument("text", nargs="*", help="Text to compress (or use --file / stdin)")
    p_opt.add_argument("--file", "-f", help="Read input from file")
    p_opt.add_argument("--target", "-t", type=float, default=0.20,
                       help="Target removal ratio (default: 0.20)")
    p_opt.add_argument("--strategy", "-s", default="adaptive",
                       choices=["adaptive", "sentence", "paragraph", "window"],
                       help="Chunking strategy")
    p_opt.add_argument("--no-protect", action="store_true",
                       help="Disable instruction zone protection")
    p_opt.add_argument("--pin-regex", action="append", default=None, dest="pin_regex",
                       metavar="PATTERN",
                       help="Pin chunks matching this regex (repeatable)")
    p_opt.add_argument("--pin-sections", default=None, dest="pin_sections",
                       metavar="KEYWORDS",
                       help="Pin all chunks under markdown headers containing "
                            "these comma-separated keywords (e.g. 'Rules,Safety')")
    p_opt.add_argument("--pin-instructions", action="store_true", dest="pin_instructions",
                       help="Pin common instruction patterns (numbered rules, "
                            "constraint keywords, JSON schemas, headers)")
    p_opt.add_argument("--json", "-j", action="store_true",
                       help="Output as JSON")
    p_opt.add_argument("--verbose", "-v", action="store_true",
                       help="Show removed chunks and scores")

    # Caveman pipeline flags (mutually exclusive)
    # SAFETY NOTE (all three): at full/ultra level, Caveman's removal word lists
    # ("a"/"an"/"the", "is"/"are"/"was"/"were", "and"/"or") collide with Python (and
    # similar-language) syntax. Code indented under a def/class body is protected
    # incidentally by the 4+-space preserve rule, but bare top-level statements or
    # single-line snippets are NOT -- always wrap code in a markdown fence
    # (```lang ... ```) before compressing text that may contain unindented code.
    caveman_group = p_opt.add_mutually_exclusive_group()
    caveman_group.add_argument(
        "--pre-caveman", nargs="?", const="full", default=None,
        choices=["lite", "full", "ultra"], dest="pre_caveman",
        help="Apply Caveman grammar-stripping BEFORE Fiedler (default level: full). "
             "At full/ultra, fence any unindented code (```lang ... ```) first -- "
             "see caveman_compress() docstring.")
    caveman_group.add_argument(
        "--post-caveman", nargs="?", const="full", default=None,
        choices=["lite", "full", "ultra"], dest="post_caveman",
        help="Apply Caveman grammar-stripping AFTER Fiedler (default level: full). "
             "At full/ultra, fence any unindented code (```lang ... ```) first -- "
             "see caveman_compress() docstring.")
    caveman_group.add_argument(
        "--caveman-only", nargs="?", const="full", default=None,
        choices=["lite", "full", "ultra"], dest="caveman_only",
        help="Apply Caveman grammar-stripping WITHOUT Fiedler (default level: full). "
             "At full/ultra, fence any unindented code (```lang ... ```) first -- "
             "see caveman_compress() docstring.")

    p_opt.set_defaults(func=cmd_optimize)

    # --- benchmark (with nested subcommands) ---
    p_bench = subparsers.add_parser("benchmark",
                                    help="Run benchmarks (suite, quality, latency)")
    bench_sub = p_bench.add_subparsers(dest="bench_command")

    # benchmark suite (default / legacy)
    p_bs = bench_sub.add_parser("suite", help="Run built-in compression benchmark")
    p_bs.add_argument("--suite", help="Custom benchmark file (JSONL)")
    p_bs.add_argument("--target", "-t", type=float, default=0.20,
                      help="Target removal ratio (default: 0.20)")
    p_bs.set_defaults(func=cmd_benchmark)

    # benchmark quality
    p_bq = bench_sub.add_parser("quality",
                                help="Evaluate compression quality on NLP datasets")
    p_bq.add_argument("--dataset", "-d", required=True,
                      choices=["gsm8k", "bbh", "natural_questions", "meetingbank", "system_prompts", "agentic_contexts", "adversarial"],
                      help="Dataset to evaluate")
    p_bq.add_argument("--ratios", "-r", default="2,4,8",
                      help="Comma-separated compression ratios (default: 2,4,8)")
    p_bq.add_argument("--model", "-m", required=True,
                      help="LLM model name (e.g. gpt-4o-mini)")
    p_bq.add_argument("--endpoint", default="https://api.openai.com/v1/chat/completions",
                      help="Chat completions API endpoint")
    p_bq.add_argument("--api-key-env", default="OPENAI_API_KEY",
                      help="Environment variable holding API key (default: OPENAI_API_KEY)")
    p_bq.add_argument("--output", "-o", help="Save results JSON to file")
    p_bq.add_argument("--dry-run", action="store_true",
                      help="Process only 10 samples for quick validation")
    p_bq.add_argument("--limit", type=int, default=None,
                      help="Maximum samples to evaluate")
    p_bq.add_argument("--provider", choices=["openai", "gemini"], default="openai",
                      help="LLM provider: openai or gemini (default: openai)")
    p_bq.add_argument("--few-shot", type=int, default=8, dest="few_shot",
                      help="Number of few-shot exemplars to prepend (default: 8, "
                           "0 for bare prompts). Currently used by gsm8k.")
    p_bq.add_argument("--pin-tool-results", action="store_true", dest="pin_tool_results",
                      help="Pin [tool_result] blocks as non-compressible. "
                           "Useful for agentic_contexts to preserve API response data.")
    p_bq.add_argument("--pin-instructions", action="store_true", dest="pin_instructions",
                      help="Pin instruction patterns (numbered rules, constraint "
                           "keywords, JSON schemas, headers) during compression")
    p_bq.set_defaults(func=cmd_benchmark_quality)

    # benchmark latency
    p_bl = bench_sub.add_parser("latency",
                                help="Profile pipeline latency across input sizes")
    p_bl.add_argument("--sizes", default="100,500,1000,5000,10000",
                      help="Comma-separated input sizes in tokens (default: 100,...,10000)")
    p_bl.add_argument("--runs", type=int, default=10,
                      help="Repetitions per size (default: 10)")
    p_bl.add_argument("--backend", default="tfidf",
                      choices=["tfidf"],
                      help="Similarity backend (default: tfidf)")
    p_bl.add_argument("--target", "-t", type=float, default=0.20,
                      help="Compression target ratio (default: 0.20)")
    p_bl.add_argument("--strategy", "-s", default="adaptive",
                      choices=["adaptive", "sentence", "paragraph", "window"],
                      help="Chunking strategy")
    p_bl.add_argument("--output", "-o", help="Save results JSON to file")
    p_bl.set_defaults(func=cmd_benchmark_latency)

    # Default: if just `fiedler benchmark` with no subcommand, run suite
    p_bench.add_argument("--suite", help="Custom benchmark file (JSONL)", default=None)
    p_bench.add_argument("--target", "-t", type=float, default=0.20,
                         help="Target removal ratio (default: 0.20)")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
