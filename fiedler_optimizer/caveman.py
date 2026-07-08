"""
Caveman compression — local grammar-stripping text compressor.

Heuristic, non-LLM text compression inspired by the caveman-compression
project (github.com/wilpel/caveman-compression). Uses spaCy POS tagging
when available, falling back to regex-based processing.

Three intensity levels:
    LITE  — drop filler words, pleasantries, hedging phrases
    FULL  — LITE + drop articles, most adverbs, optional conjunctions,
             convert to sentence fragments
    ULTRA — FULL + telegraphic abbreviation, drop context-recoverable
             pronouns, drop copulas where predicate is parseable

Usage:
    from fiedler_optimizer.caveman import caveman_compress

    result = caveman_compress(text, level="full")
    print(result.text)
    print(f"{result.compression_ratio:.1%} compression")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class CavemanLevel(Enum):
    LITE = "lite"
    FULL = "full"
    ULTRA = "ultra"


@dataclass
class CavemanResult:
    """Result of caveman compression."""

    text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    preserved_blocks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants — word / phrase lists
# ---------------------------------------------------------------------------

FILLER_WORDS = {
    "just",
    "really",
    "basically",
    "actually",
    "simply",
    "certainly",
    "essentially",
    "generally",
    "typically",
    "obviously",
    "definitely",
}

PLEASANTRIES = [
    "i'd be glad to",
    "happy to",
    "of course",
    "no problem",
    "sure",
    "certainly",
]

HEDGING_PHRASES = [
    "it might be worth considering",
    "you could potentially",
    "it's possible that",
    "perhaps",
]

ARTICLES = {"a", "an", "the"}

COPULAS = {"is", "are", "was", "were"}

REMOVABLE_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}

REMOVABLE_ADVERBS_REGEX = re.compile(
    r"\b(?:very|quite|rather|somewhat|extremely|incredibly|remarkably"
    r"|particularly|especially|absolutely|completely|entirely|totally"
    r"|utterly|highly|slightly|barely|merely|nearly|almost|fairly"
    r"|pretty|so|too|most|least|largely|mostly|mainly|chiefly"
    r"|primarily|predominantly|frequently|occasionally|rarely"
    r"|seldom|often|sometimes|always|never|usually|normally"
    r"|regularly|constantly|continually|continuously|gradually"
    r"|suddenly|immediately|eventually|finally|ultimately"
    r"|apparently|evidently|clearly|plainly|seemingly)\b",
    re.IGNORECASE,
)

REMOVABLE_CONJUNCTIONS = {"and", "but", "or", "yet", "so", "for", "nor"}

# ---------------------------------------------------------------------------
# Preserved-block extraction
# ---------------------------------------------------------------------------

# Order matters — earlier patterns are extracted first.
_PRESERVE_PATTERNS = [
    # Fenced code blocks (``` ... ```)
    re.compile(r"```[\s\S]*?```", re.MULTILINE),
    # Indented code blocks (4+ spaces at line start, consecutive lines)
    re.compile(r"(?:^[ ]{4,}\S.*$\n?)+", re.MULTILINE),
    # Inline code
    re.compile(r"`[^`]+`"),
    # URLs
    re.compile(r"https?://\S+"),
    # File paths (Unix or Windows style)
    re.compile(r"(?:[A-Za-z]:\\|/)[\w./_\\-]+"),
    # Quoted strings (double)
    re.compile(r'"[^"]*"'),
    # Quoted strings (single)
    re.compile(r"'[^']*'"),
    # Version strings  e.g. v1.2.3, 1.2.3-beta
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b"),
    # ISO-ish dates  e.g. 2024-01-15
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # Standalone numbers (integers / floats)
    re.compile(r"\b\d+(?:\.\d+)?\b"),
]

_PLACEHOLDER_PREFIX = "zqxCAVEMAN"
_PLACEHOLDER_SUFFIX = "MANCxqz"


def _extract_preserved(text: str) -> tuple[str, list[str]]:
    """Replace preserved spans with placeholders, return (masked_text, blocks)."""
    blocks: list[str] = []

    for pattern in _PRESERVE_PATTERNS:
        def _replacer(m: re.Match, _blocks: list = blocks) -> str:
            matched = m.group(0)
            # Refuse to match across or inside an existing placeholder: if the
            # span contains a placeholder marker, leave it untouched so a later
            # pattern (e.g. the apostrophe rule spanning possessives) cannot
            # swallow an already-placed placeholder.
            if _PLACEHOLDER_PREFIX in matched:
                return matched
            idx = len(_blocks)
            _blocks.append(matched)
            idx_letters = "".join(chr(ord("a") + int(d)) for d in str(idx))
            return f"{_PLACEHOLDER_PREFIX}{idx_letters}{_PLACEHOLDER_SUFFIX}"

        text = pattern.sub(_replacer, text)

    return text, blocks


def _restore_preserved(text: str, blocks: list[str]) -> str:
    """Re-insert preserved blocks in a single pass.

    Uses one regex sweep with a callback so that a restored block's content
    can never corrupt a not-yet-restored placeholder (the sequential
    str.replace approach collided when block contents were short numbers
    that re-matched other placeholders' letter indices).
    """
    if not blocks:
        return text

    # Map each placeholder's letter-index back to its block index.
    def _idx_to_letters(idx: int) -> str:
        return "".join(chr(ord("a") + int(d)) for d in str(idx))

    letters_to_block = {
        _idx_to_letters(i): block for i, block in enumerate(blocks)
    }

    pattern = re.compile(
        re.escape(_PLACEHOLDER_PREFIX) + r"([a-j]+)" + re.escape(_PLACEHOLDER_SUFFIX)
    )

    def _sub(m: re.Match) -> str:
        letters = m.group(1)
        return letters_to_block.get(letters, m.group(0))

    return pattern.sub(_sub, text)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder: Optional[object] = None


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base, falling back to whitespace split."""
    global _encoder
    if _TIKTOKEN_AVAILABLE:
        if _encoder is None:
            _encoder = tiktoken.get_encoding("cl100k_base")
        return len(_encoder.encode(text))  # type: ignore[union-attr]
    # Rough fallback
    return len(text.split())


# ---------------------------------------------------------------------------
# spaCy loader
# ---------------------------------------------------------------------------

_nlp: Optional[object] = None


def _get_nlp():
    """Load spaCy model, return None if unavailable."""
    global _nlp
    if not _SPACY_AVAILABLE:
        return None
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None
    return _nlp


# ---------------------------------------------------------------------------
# Phrase-level removal (works in both spaCy and regex paths)
# ---------------------------------------------------------------------------


def _remove_phrases(text: str, phrases: list[str]) -> str:
    for phrase in sorted(phrases, key=len, reverse=True):
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text


def _remove_filler_words(text: str) -> str:
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# LITE level
# ---------------------------------------------------------------------------


def _apply_lite(text: str) -> str:
    text = _remove_filler_words(text)
    text = _remove_phrases(text, PLEASANTRIES)
    text = _remove_phrases(text, HEDGING_PHRASES)
    return text


# ---------------------------------------------------------------------------
# FULL level — regex fallback
# ---------------------------------------------------------------------------


def _apply_full_regex(text: str) -> str:
    text = _apply_lite(text)
    # Drop articles
    text = re.sub(r"\b(?:a|an|the)\b", "", text, flags=re.IGNORECASE)
    # Drop common adverbs
    text = REMOVABLE_ADVERBS_REGEX.sub("", text)
    # Drop conjunctions at sentence boundaries (", and", ", but", etc.)
    text = re.sub(
        r",\s*\b(?:" + "|".join(REMOVABLE_CONJUNCTIONS) + r")\b",
        ",",
        text,
        flags=re.IGNORECASE,
    )
    return text


# ---------------------------------------------------------------------------
# FULL level — spaCy path
# ---------------------------------------------------------------------------


def _apply_full_spacy(text: str, nlp) -> str:
    text = _apply_lite(text)
    doc = nlp(text)
    remove_indices: set[int] = set()
    for token in doc:
        # Articles (determiners that are a/an/the)
        if token.pos_ == "DET" and token.lower_ in ARTICLES:
            remove_indices.add(token.i)
        # Adverbs — keep negation and sentence adverbs that serve as conjunctions
        elif token.pos_ == "ADV" and token.dep_ not in ("neg", "cc", "mark"):
            remove_indices.add(token.i)
        # Non-structural conjunctions
        elif (
            token.pos_ == "CCONJ"
            and token.dep_ not in ("cc",)
            and token.lower_ in REMOVABLE_CONJUNCTIONS
        ):
            remove_indices.add(token.i)

    tokens_out = [t.text_with_ws for i, t in enumerate(doc) if i not in remove_indices]
    return "".join(tokens_out)


# ---------------------------------------------------------------------------
# ULTRA level — regex fallback
# ---------------------------------------------------------------------------


def _apply_ultra_regex(text: str) -> str:
    text = _apply_full_regex(text)
    # Drop copulas
    text = re.sub(
        r"\b(?:is|are|was|were)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Drop pronouns at sentence / clause start
    text = re.sub(
        r"(?:^|(?<=\.\s)|(?<=,\s))\b(?:I|you|he|she|it|we|they)\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


# ---------------------------------------------------------------------------
# ULTRA level — spaCy path
# ---------------------------------------------------------------------------


def _apply_ultra_spacy(text: str, nlp) -> str:
    text = _apply_lite(text)
    doc = nlp(text)
    remove_indices: set[int] = set()
    for token in doc:
        # Articles
        if token.pos_ == "DET" and token.lower_ in ARTICLES:
            remove_indices.add(token.i)
        # Adverbs (non-negation)
        elif token.pos_ == "ADV" and token.dep_ != "neg":
            remove_indices.add(token.i)
        # Non-structural conjunctions
        elif token.pos_ == "CCONJ" and token.lower_ in REMOVABLE_CONJUNCTIONS:
            remove_indices.add(token.i)
        # Copulas
        elif token.pos_ == "AUX" and token.lower_ in COPULAS:
            # Only remove if the predicate (head) survives
            if token.head.i not in remove_indices:
                remove_indices.add(token.i)
        # Pronouns (subject) recoverable from context
        elif (
            token.pos_ == "PRON"
            and token.dep_ in ("nsubj", "nsubjpass")
            and token.lower_ in REMOVABLE_PRONOUNS
        ):
            remove_indices.add(token.i)

    tokens_out = [t.text_with_ws for i, t in enumerate(doc) if i not in remove_indices]
    return "".join(tokens_out)


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    # Collapse runs of spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove space before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Remove trailing spaces per line
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    # Remove leading spaces per line (unless inside a preserved block placeholder)
    text = re.sub(r"^[ \t]+(?!" + re.escape(_PLACEHOLDER_PREFIX) + ")", "", text, flags=re.MULTILINE)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def caveman_compress(
    text: str,
    level: str = "full",
    *,
    technical_terms: Optional[list[str]] = None,
) -> CavemanResult:
    """Compress *text* using grammar-stripping heuristics.

    Parameters
    ----------
    text : str
        The input text to compress.
    level : str
        Compression intensity: ``"lite"``, ``"full"`` (default), or ``"ultra"``.
    technical_terms : list[str] | None
        Optional allowlist of terms that must never be altered.

    Returns
    -------
    CavemanResult
    """
    level_enum = CavemanLevel(level.lower())
    original_tokens = _count_tokens(text)

    # --- protect preserved blocks ----------------------------------------
    masked, preserved_blocks = _extract_preserved(text)

    # --- protect technical terms -----------------------------------------
    term_map: dict[str, str] = {}
    if technical_terms:
        for idx, term in enumerate(technical_terms):
            placeholder = f"\x00TERM{idx}\x00"
            term_map[placeholder] = term
            masked = re.sub(re.escape(term), placeholder, masked)

    # --- apply compression -----------------------------------------------
    nlp = _get_nlp()

    if level_enum == CavemanLevel.LITE:
        compressed = _apply_lite(masked)
    elif level_enum == CavemanLevel.FULL:
        if nlp is not None:
            compressed = _apply_full_spacy(masked, nlp)
        else:
            compressed = _apply_full_regex(masked)
    elif level_enum == CavemanLevel.ULTRA:
        if nlp is not None:
            compressed = _apply_ultra_spacy(masked, nlp)
        else:
            compressed = _apply_ultra_regex(masked)
    else:
        compressed = masked  # unreachable

    # --- normalize whitespace --------------------------------------------
    compressed = _normalize_whitespace(compressed)

    # --- restore technical terms -----------------------------------------
    for placeholder, term in term_map.items():
        compressed = compressed.replace(placeholder, term)

    # --- restore preserved blocks ----------------------------------------
    compressed = _restore_preserved(compressed, preserved_blocks)

    # --- token counts ----------------------------------------------------
    compressed_tokens = _count_tokens(compressed)
    ratio = 1.0 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0

    return CavemanResult(
        text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
        preserved_blocks=preserved_blocks,
    )
