"""
Text chunking strategies for Fiedler spectral decomposition.

The chunker breaks input text into segments that become nodes in the
similarity graph. Chunk granularity directly affects compression quality:
too fine (word-level) creates noisy graphs; too coarse (paragraph-level)
loses the ability to surgically remove low-connectivity content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""
    SENTENCE = auto()
    PARAGRAPH = auto()
    SLIDING_WINDOW = auto()
    ADAPTIVE = auto()  # Picks strategy based on text structure
    CODE = auto()      # Line-level, for structured / source / config text


@dataclass(frozen=True)
class Chunk:
    """A text segment that becomes a node in the similarity graph.

    Invariant: ``normalized[start_char:end_char] == text`` for every chunk,
    where ``normalized`` is ``_normalize_unicode(input)`` -- the string
    ``chunk_text`` actually splits. This equals the caller's raw input except
    where Unicode normalization changed length (e.g. an em dash becoming ``--``),
    so use these offsets against the normalized text for exact provenance.
    """
    text: str
    index: int
    start_char: int
    end_char: int
    word_count: int

    @property
    def is_trivial(self) -> bool:
        """Chunks with very few words carry little semantic signal."""
        return self.word_count < 3


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Regex handles common abbreviations and decimal numbers to avoid false splits.
_SENTENCE_BOUNDARY = re.compile(
    r'(?<![A-Z])[.!?]'
    r'(?=\s+[A-Z"\']|\s*$)',
    re.MULTILINE,
)

def _strip_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink a half-open span to drop leading/trailing whitespace.

    Returns None if the span is empty after stripping. Stripping by moving the
    bounds (instead of calling .strip() on a copy) keeps text[start:end] an
    exact substring of the source, which is what the offset invariant needs.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode characters for consistent cross-platform behavior."""
    import unicodedata
    # Normalize to NFC form (canonical decomposition + canonical composition)
    text = unicodedata.normalize("NFC", text)
    # Replace common Unicode variants with ASCII equivalents
    replacements = {
        '\u2014': '--',   # em dash
        '\u2013': '-',    # en dash
        '\u2018': "'",    # left single quote
        '\u2019': "'",    # right single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',    # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Split on sentence boundaries, returning (start, end) spans into *text*."""
    spans: list[tuple[int, int]] = []
    last = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        end = m.end()
        span = _strip_span(text, last, end)
        if span:
            spans.append(span)
        last = end
    # Trailing content that didn't end with punctuation
    span = _strip_span(text, last, len(text))
    if span:
        spans.append(span)
    return spans


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Split on double newlines, returning (start, end) spans into *text*."""
    spans: list[tuple[int, int]] = []
    last = 0
    for m in re.finditer(r'\n\s*\n', text):
        span = _strip_span(text, last, m.start())
        if span:
            spans.append(span)
        last = m.end()
    span = _strip_span(text, last, len(text))
    if span:
        spans.append(span)
    return spans


def _split_sliding_window(text: str, window_words: int = 50, stride_words: int = 25) -> list[tuple[int, int]]:
    """Overlapping sliding window over words, returning (start, end) spans into *text*."""
    # Word spans, so a window maps to the contiguous source slice covering it
    # (inter-word whitespace, newlines included) rather than a rejoined copy.
    words = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    if len(words) <= window_words:
        span = _strip_span(text, 0, len(text))
        return [span] if span else []
    spans: list[tuple[int, int]] = []
    for i in range(0, len(words) - window_words + 1, stride_words):
        win = words[i : i + window_words]
        spans.append((win[0][0], win[-1][1]))
    # Capture any trailing words not covered by the last full window
    if i + window_words < len(words):
        tail = words[i + stride_words :]
        spans.append((tail[0][0], tail[-1][1]))
    return spans


_STRUCT_CHARS = "{}[]=;:"


def _looks_like_code(text: str) -> bool:
    """Heuristic: structured / source / config text (JSON, code) rather than prose.
    Keys on structural-char density AND a low fraction of lines that end like a
    prose sentence.  Rule/instruction prose (lines ending in '.') is NOT code, so
    it still routes to the sentence splitter."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    struct = sum(text.count(c) for c in _STRUCT_CHARS)
    prose_end = sum(1 for ln in lines if ln.rstrip().endswith((".", "!", "?")))
    # prose_frac is the reliable discriminator (code ~0, prose >= 0.5); keep the
    # threshold conservative so document/RAG prompts are never misread as code.
    return (struct / len(lines)) >= 1.0 and (prose_end / len(lines)) < 0.3


def _split_lines(text: str) -> list[tuple[int, int]]:
    """Line-level spans for code / structured text (each non-blank line a chunk).

    Leading indentation is kept (it is real content for code); only the trailing
    newline/whitespace is dropped, so text[start:end] stays an exact substring.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            spans.append((pos, pos + len(line.rstrip())))
        pos += len(line)
    return spans


# ---------------------------------------------------------------------------
# Adaptive strategy selector
# ---------------------------------------------------------------------------

def _choose_strategy(text: str) -> ChunkingStrategy:
    """Pick the best chunking strategy based on text structure."""
    # Structured / code text has no prose structure the sentence/paragraph
    # splitters understand; split it by lines so compression can engage at all.
    if _looks_like_code(text):
        return ChunkingStrategy.CODE

    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(text)

    # If we have well-formed paragraphs of reasonable size, use them
    if len(paragraphs) >= 4:
        avg_words = sum(len(text[a:b].split()) for a, b in paragraphs) / len(paragraphs)
        if 20 <= avg_words <= 200:
            return ChunkingStrategy.PARAGRAPH

    # If we have enough sentences, use sentence-level
    if len(sentences) >= 6:
        return ChunkingStrategy.SENTENCE

    # Fall back to sliding window for unstructured blobs
    return ChunkingStrategy.SLIDING_WINDOW


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
    min_chunk_words: int = 3,
    window_words: int = 50,
    stride_words: int = 25,
) -> list[Chunk]:
    """
    Break text into chunks for graph construction.

    Parameters
    ----------
    text : str
        The input text to chunk.
    strategy : ChunkingStrategy
        Which splitting strategy to use. ADAPTIVE auto-selects.
    min_chunk_words : int
        Chunks with fewer words are merged into neighbors.
    window_words : int
        Word count per window (SLIDING_WINDOW strategy only).
    stride_words : int
        Stride between windows (SLIDING_WINDOW strategy only).

    Returns
    -------
    list[Chunk]
        Ordered list of text chunks with positional metadata.
    """
    text = _normalize_unicode(text)
    
    if strategy == ChunkingStrategy.ADAPTIVE:
        strategy = _choose_strategy(text)

    if strategy == ChunkingStrategy.SENTENCE:
        raw_spans = _split_sentences(text)
    elif strategy == ChunkingStrategy.PARAGRAPH:
        raw_spans = _split_paragraphs(text)
    elif strategy == ChunkingStrategy.SLIDING_WINDOW:
        raw_spans = _split_sliding_window(text, window_words, stride_words)
    elif strategy == ChunkingStrategy.CODE:
        raw_spans = _split_lines(text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Merge tiny chunks into their successor. Spans are carried, not text, so a
    # merge is the union (buffer_start, successor_end) -- always a contiguous
    # slice of the source, which keeps text[start:end] == chunk.text exact.
    merged: list[tuple[int, int]] = []
    buffer: tuple[int, int] | None = None
    for start, end in raw_spans:
        if buffer:
            start = buffer[0]
        if len(text[start:end].split()) < min_chunk_words:
            buffer = (start, end)
        else:
            merged.append((start, end))
            buffer = None
    if buffer:
        if merged:
            merged[-1] = (merged[-1][0], buffer[1])
        else:
            merged.append(buffer)

    # Offsets come straight from the spans -- no search, no reconstruction.
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(merged):
        segment = text[start:end]
        chunks.append(Chunk(
            text=segment,
            index=i,
            start_char=start,
            end_char=end,
            word_count=len(segment.split()),
        ))

    return chunks


def merge_kept_spans(
    text: str,
    chunks: Sequence[Chunk],
    kept_indices: Sequence[int],
) -> str:
    """Reassemble kept chunks from their original character spans.

    Overlapping windows are merged so the ADAPTIVE/SLIDING_WINDOW chunker's
    overlap is not emitted twice (duplication inflates the output and can turn
    "compression" negative). Slicing the source text directly also keeps the
    result an exact substring reassembly, which anchor-searching did not
    guarantee for passages containing embedded newlines.
    """
    spans = sorted((chunks[i].start_char, chunks[i].end_char) for i in kept_indices)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return "\n\n".join(text[s:e] for s, e in merged)
