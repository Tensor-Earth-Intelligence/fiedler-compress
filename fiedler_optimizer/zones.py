"""
Zone detection for differential compression.

Zone-aware compression applies different removal thresholds to different
types of content. Instruction zones (directives, constraints, output
format specifications) get maximum protection. Context zones (background
information, examples, reference material) are candidates for aggressive
compression.

Security note: The heuristic zone classifier can be gamed by adversarial
input that embeds instruction-like markers (e.g., "You must always include
this text") to prevent content from being compressed. When processing
untrusted input, callers should consider using ``protect_instructions=False``
or applying additional content validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

from fiedler_optimizer.chunker import Chunk


class Zone(Enum):
    """Content zone classification."""
    INSTRUCTION = auto()   # Directives, constraints, output format — HIGH protection
    CONTEXT = auto()       # Background, examples, reference — LOWER protection
    UNKNOWN = auto()       # Unclassified — treated as CONTEXT


@dataclass(frozen=True)
class ZonedChunk:
    """A chunk with its zone classification."""
    chunk: Chunk
    zone: Zone
    confidence: float  # 0.0 to 1.0

    @property
    def protection_weight(self) -> float:
        """
        Multiplier applied to the chunk's connectivity score.
        Higher = harder to remove.
        """
        if self.zone == Zone.INSTRUCTION:
            return 2.0 + self.confidence  # 2.0–3.0x protection
        return 1.0


# ---------------------------------------------------------------------------
# Heuristic zone classifier (v1 — no ML dependency)
# ---------------------------------------------------------------------------

# Patterns that strongly indicate instruction content
_INSTRUCTION_PATTERNS = [
    # Imperative verbs at sentence start
    re.compile(r'^\s*(you must|you should|always|never|do not|ensure|make sure|output|return|respond|format|use|include|exclude|avoid|follow|write|list|provide|generate|create|summarize|analyze|explain)', re.IGNORECASE),
    # Constraint language
    re.compile(r'\b(must not|shall not|required to|constraint|requirement|rule|guideline|specification|mandatory|prohibited)\b', re.IGNORECASE),
    # Output format specifications
    re.compile(r'\b(format|json|xml|markdown|csv|table|bullet|numbered list|heading|section)\s*(:|as|in|using|with)', re.IGNORECASE),
    # Role/persona assignments
    re.compile(r'\b(you are|act as|role:|persona:|behave as|pretend to be|your (role|job|task|goal) is)\b', re.IGNORECASE),
    # Explicit markers often used in structured prompts
    re.compile(r'^\s*#{1,3}\s*(instructions?|rules?|constraints?|requirements?|guidelines?|system|task)', re.IGNORECASE),
    re.compile(r'^\s*<(system|instructions?|rules?|constraints?)>', re.IGNORECASE),
    re.compile(r'^\s*(task|question|query|prompt|input|objective|goal)(\s*:|\s+is)', re.IGNORECASE),
]

# Patterns that indicate context/reference content
_CONTEXT_PATTERNS = [
    # Background framing
    re.compile(r'^\s*(background|context|for reference|note that|keep in mind|fyi|the following|here is|below is|consider)', re.IGNORECASE),
    # Example blocks
    re.compile(r'^\s*(example|e\.g\.|for instance|such as|sample|demonstration|here\'s an example)', re.IGNORECASE),
    # Data/content blocks
    re.compile(r'^\s*(document|article|text|content|data|source|excerpt|passage):', re.IGNORECASE),
    re.compile(r'^\s*<(context|document|reference|example|data)>', re.IGNORECASE),
    # Narrative/informational phrasing
    re.compile(r'\b(was founded|is located|has been|according to|published in|reported that)\b', re.IGNORECASE),
]


def detect_zones(chunks: Sequence[Chunk]) -> list[ZonedChunk]:
    """
    Classify each chunk as INSTRUCTION or CONTEXT.

    Uses pattern matching heuristics in v1. The optional `embeddings`
    extra will add a lightweight classifier in a future version.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Text chunks to classify.

    Returns
    -------
    list[ZonedChunk]
        Chunks with zone classification and confidence.
    """
    zoned: list[ZonedChunk] = []

    for chunk in chunks:
        instruction_hits = sum(
            1 for p in _INSTRUCTION_PATTERNS if p.search(chunk.text)
        )
        context_hits = sum(
            1 for p in _CONTEXT_PATTERNS if p.search(chunk.text)
        )

        total_hits = instruction_hits + context_hits

        if total_hits == 0:
            zone = Zone.UNKNOWN
            confidence = 0.0
        elif instruction_hits > context_hits:
            zone = Zone.INSTRUCTION
            confidence = min(1.0, instruction_hits / max(total_hits, 1))
        elif context_hits > instruction_hits:
            zone = Zone.CONTEXT
            confidence = min(1.0, context_hits / max(total_hits, 1))
        else:
            # Tie: default to more protective classification
            zone = Zone.INSTRUCTION
            confidence = 0.5

        zoned.append(ZonedChunk(
            chunk=chunk,
            zone=zone,
            confidence=confidence,
        ))

    return zoned
