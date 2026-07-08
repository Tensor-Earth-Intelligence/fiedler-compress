"""
Pin pattern utilities for instruction-aware compression.

Provides built-in pattern presets and section-aware helpers that
generate ``pin_patterns`` lists for :func:`~fiedler_optimizer.core.optimize`.
"""

from __future__ import annotations

import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Built-in instruction preset patterns
# ---------------------------------------------------------------------------

INSTRUCTION_PRESET: tuple[str, ...] = (
    # Numbered rules  (1. ... 2. ... )
    r"^\d+\.\s",
    # Imperative constraint keywords (full word boundary)
    r"\b(?:must|always|never|required|do not|shall not|mandatory|prohibited)\b",
    # JSON schema blocks
    r"\{[^{}]*(?:\"type\"|\"properties\"|\"required\"|\"enum\")[^{}]*\}",
    # Markdown section headers
    r"^#{1,3}\s+\S",
)


def section_pin_patterns(
    headers: Sequence[str],
    text: str,
) -> list[str]:
    """Build pin patterns that match all content under matching markdown headers.

    Parameters
    ----------
    headers : Sequence[str]
        Header keywords to match (case-insensitive).  A header line
        ``## Safety Guardrails`` matches if any keyword appears in it.
    text : str
        The full prompt text.  Used to extract the literal text of each
        matching section so it can be compiled into a pin pattern.

    Returns
    -------
    list[str]
        Regex patterns, one per matching section, that will pin all chunks
        whose text overlaps that section.
    """
    if not headers or not text:
        return []

    header_rx = re.compile(
        r"^(#{1,6})\s+(.+)$", re.MULTILINE
    )

    # Find all markdown headers and their positions
    header_positions: list[tuple[int, int, str, int]] = []
    for m in header_rx.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        header_positions.append((m.start(), m.end(), title, level))

    # Find sections whose header matches any keyword
    patterns: list[str] = []
    kw_lower = [h.lower() for h in headers]
    for i, (start, _end, title, _level) in enumerate(header_positions):
        if not any(kw in title.lower() for kw in kw_lower):
            continue
        # Section runs from this header to the next header of same or higher level (or EOF)
        section_end = len(text)
        for j in range(i + 1, len(header_positions)):
            if header_positions[j][3] <= _level:
                section_end = header_positions[j][0]
                break
        section_text = text[start:section_end].strip()
        if section_text:
            # Build a pattern that matches any line from this section
            # Use the first 60 chars of each line as literal patterns
            for line in section_text.splitlines():
                line = line.strip()
                if len(line) > 10:  # skip trivially short lines
                    patterns.append(re.escape(line[:60]))

    return patterns
