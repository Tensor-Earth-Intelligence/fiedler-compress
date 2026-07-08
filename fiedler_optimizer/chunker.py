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


@dataclass(frozen=True)
class Chunk:
    """A text segment that becomes a node in the similarity graph."""
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

_WHITESPACE_NORMALIZE = re.compile(r'\s+')

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


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, keeping the delimiter with the sentence."""
    parts: list[str] = []
    last = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        end = m.end()
        sentence = text[last:end].strip()
        if sentence:
            parts.append(sentence)
        last = end
    # Trailing content that didn't end with punctuation
    remainder = text[last:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def _split_paragraphs(text: str) -> list[str]:
    """Split on double newlines (standard paragraph breaks)."""
    raw = re.split(r'\n\s*\n', text)
    return [p.strip() for p in raw if p.strip()]


def _split_sliding_window(text: str, window_words: int = 50, stride_words: int = 25) -> list[str]:
    """Overlapping sliding window over words."""
    words = text.split()
    if len(words) <= window_words:
        return [text.strip()]
    chunks = []
    for i in range(0, len(words) - window_words + 1, stride_words):
        chunk = " ".join(words[i : i + window_words])
        chunks.append(chunk)
    # Capture any trailing words not covered by the last full window
    if i + window_words < len(words):
        chunks.append(" ".join(words[i + stride_words :]))
    return chunks


# ---------------------------------------------------------------------------
# Adaptive strategy selector
# ---------------------------------------------------------------------------

def _choose_strategy(text: str) -> ChunkingStrategy:
    """Pick the best chunking strategy based on text structure."""
    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(text)

    # If we have well-formed paragraphs of reasonable size, use them
    if len(paragraphs) >= 4:
        avg_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
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
        raw_chunks = _split_sentences(text)
    elif strategy == ChunkingStrategy.PARAGRAPH:
        raw_chunks = _split_paragraphs(text)
    elif strategy == ChunkingStrategy.SLIDING_WINDOW:
        raw_chunks = _split_sliding_window(text, window_words, stride_words)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Merge tiny chunks into their successor
    merged: list[str] = []
    buffer = ""
    for raw in raw_chunks:
        combined = f"{buffer} {raw}".strip() if buffer else raw
        if len(combined.split()) < min_chunk_words:
            buffer = combined
        else:
            merged.append(combined)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}"
        else:
            merged.append(buffer)

    # Build Chunk objects with character offsets
    chunks: list[Chunk] = []
    search_start = 0
    for i, segment in enumerate(merged):
        # Find the approximate start position in the original text
        normalized_seg = _WHITESPACE_NORMALIZE.sub(" ", segment).strip()
        # Use first 40 chars as anchor for position search
        anchor = normalized_seg[:40]
        pos = text.find(anchor, search_start)
        if pos == -1:
            pos = search_start  # fallback

        chunks.append(Chunk(
            text=segment,
            index=i,
            start_char=pos,
            end_char=pos + len(segment),
            word_count=len(segment.split()),
        ))
        search_start = pos + 1

    return chunks
