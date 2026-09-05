"""Curated content-type priors.

A *content prior* is a per-chunk importance signal keyed on WHAT a span is
(identifier / secret / code), independent of the query or any answer key.  It
complements the pin mechanism in two layers:

    * The extension point: callers may pass their own ``pin_patterns`` to
      :func:`fiedler_optimizer.core.optimize`.
    * This module: a maintained detector and pin-pattern generator, so callers
      do not have to hand-build token lists.

Usage
-----
Any base method gains identifier-preservation via the existing pin hook::

    from fiedler_optimizer import optimize
    from fiedler_optimizer.geometry.content_prior import identifier_pin_patterns
    optimize(text, target_ratio=0.9,
             pin_patterns=identifier_pin_patterns(text))

``geometry.voronoi.voronoi_compress`` consumes the same detector through its
``content_prior="identifier"`` argument.
"""
from __future__ import annotations

import re

_IDENT_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


def is_identifier_token(tok: str) -> bool:
    """Generic identifier/secret signature -- NOT a format-specific regex.

    Fires on code / API-or-license-key / ticket-ID / SKU / long-serial shapes
    while leaving years, quantities and ordinary prose alone.
    """
    core = tok.strip(".,;:()[]{}\"'")
    if len(core) < 4:
        return False
    has_u = any(c.isupper() for c in core)
    has_d = any(c.isdigit() for c in core)
    if not has_d:                                    # identifiers essentially always carry a digit
        return False
    if has_u:                                        # uppercase + digit (XR-8412, AKIA..., ghp_..F2)
        return True
    if re.search(r"\d{6,}", core):                   # long numeric serial / PIN (skips 4-digit years)
        return True
    alnum = sum(c.isalnum() for c in core)
    if len(core) >= 12 and alnum >= len(core) - 2:   # long lowercase hex / uuid / token
        return True
    return False


def identifier_prior(text: str) -> float:
    """Count of identifier-shaped tokens in a chunk (a content-TYPE prior)."""
    return float(sum(1 for m in _IDENT_WORD.finditer(text)
                     if is_identifier_token(m.group(0))))


def identifier_pin_patterns(text: str) -> list[str] | None:
    """Escaped-regex pin patterns for every identifier-shaped token in ``text``.

    Composes with the free ``pin_patterns`` hook so any base method preserves
    identifiers without a new core parameter.  Returns None when none are found.
    """
    toks = {m.group(0) for m in _IDENT_WORD.finditer(text)
            if is_identifier_token(m.group(0))}
    return [re.escape(t) for t in sorted(toks)] or None


# ---------------------------------------------------------------------------
# Durable-fact salience prior (heuristic v1)
# ---------------------------------------------------------------------------
# Natural-language analog of the identifier prior: flags a chunk that STATES a
# durable fact about the user's world (an allergy, a name, a preference, a
# deadline).  Validated on a held-out, non-template set at ~88% recall on
# well-formed facts with 0% false-positives on first-person chit-chat / filler.
# Structurally blind to signature-less phrasings (no first-person marker AND no
# cue), e.g. "peanuts make my throat close up" -- that residual needs a trained
# classifier or summarisation, NOT prototype embeddings (which underdelivered).

_FP = re.compile(r"\b(i|i'm|i've|i'd|my|mine|we|we're|our)\b", re.I)
_HEDGE = re.compile(
    r"\b(think|might|maybe|probably|guess|going to|gonna|should|honestly|"
    r"not sure|we'll see|hope|feels?)\b", re.I)
_SALIENCE_CUE = re.compile(
    r"\b(allerg\w+|vegetarian|vegan|prefer\w*|favou?rite|hate|can't stand|name|"
    r"named|call(?:ed)?|go by|drive|live|born|married|lease|deadline|due|budget|"
    r"decided?|choos\w+|chose|signed|booked|account|address|number)\b", re.I)
_PROPER = re.compile(r"(?<=[a-z,;:]\s)([A-Z][a-zA-Z]{2,})")


def _strip_speaker(text: str) -> str:
    return re.sub(r"^\s*(User|Assistant)\s*:\s*", "", text.strip())


def is_durable_fact(text: str) -> bool:
    """Heuristic: a first-person DECLARATIVE naming a specific entity or carrying a
    stative/possession/preference cue, gated by a hedge filter."""
    body = _strip_speaker(text)
    if not body or body.endswith("?"):
        return False
    if not _FP.search(body):
        return False
    has = bool(_SALIENCE_CUE.search(body) or _PROPER.search(body))
    if _HEDGE.search(body) and not has:
        return False
    return has


def salience_prior(text: str) -> float:
    """1.0 if the chunk states a durable user fact, else 0.0 (a content-TYPE prior)."""
    return 1.0 if is_durable_fact(text) else 0.0


def salience_pin_patterns(text: str):
    """Escaped-regex pin patterns for each line that states a durable fact.
    Composes with the free pin hook so any base method preserves durable facts."""
    pats = [re.escape(ln) for ln in text.splitlines()
            if ln.strip() and is_durable_fact(ln)]
    return pats or None


# ---------------------------------------------------------------------------
# Observation-result salience prior (agent tool-loop analog of `salience`)
# ---------------------------------------------------------------------------
# The chat salience prior above keys on FIRST-PERSON declaratives ("I/my"), which
# tool observations never use -- confirmed NOT to transfer to agent loops
# (MEMO_multiturn_chat_20260714.md: voronoi+salience = 0.00 on agent-loop semantic
# observations).  This is the THIRD-PERSON analog: a tool observation that BINDS a
# result to something (a lookup/action outcome), as opposed to a routine/no-op step.
# Same shape as identifier/salience: cheap lexical heuristic, not a trained model.

_RESULT_CUE = re.compile(
    r"\b(maps?\s+to|mapped\s+to|returns?|returned|resolv\w*\s+to|filed\s+under|"
    r"found\s+(?:at|in)|located\s+at|identified\s+as|assigned\s+to|bound\s+to|"
    r"stored\s+(?:at|in|under)|saved\s+(?:to|as)|equals|yields?|corresponds?\s+to|"
    r"belongs?\s+to|tagged\s+as|classified\s+as|set\s+to)\b", re.I)


def is_observation_result(text: str) -> bool:
    """Heuristic: a tool-observation line that binds a lookup/action result to a
    value, as opposed to a routine/no-op step.  Structurally blind to result
    phrasings outside this cue list -- same honest limitation as `is_durable_fact`."""
    body = re.sub(r"^\s*Observation\s*\d*\s*:\s*", "", text.strip())
    if not body:
        return False
    return bool(_RESULT_CUE.search(body))


def observation_prior(text: str) -> float:
    """1.0 if the chunk is a result-binding tool observation, else 0.0."""
    return 1.0 if is_observation_result(text) else 0.0


def observation_pin_patterns(text: str):
    """Escaped-regex pin patterns for each line that binds an observation result.
    Composes with the free pin hook so any base method preserves tool results."""
    pats = [re.escape(ln) for ln in text.splitlines()
            if ln.strip() and is_observation_result(ln)]
    return pats or None


# Registry of named content priors (consumed by geometry.voronoi).  Named priors
# require the ``geometry`` extra; user-supplied callables work everywhere via
# the pin_patterns mechanism, which has no extra dependencies.
CONTENT_PRIORS = {"identifier": identifier_prior, "salience": salience_prior,
                  "observation": observation_prior}

# Registry of named content priors as PIN-PATTERN generators (consumed by
# core.optimize()'s content_prior= keyword -- see core.py).  Same three curated
# detectors, exposed the other way: instead of a per-chunk score (CONTENT_PRIORS,
# used by voronoi's importance ranking), each of these returns escaped-regex patterns
# for optimize()'s pin_patterns mechanism, so callers can write
# ``optimize(text, content_prior="identifier")`` instead of manually composing
# ``optimize(text, pin_patterns=identifier_pin_patterns(text))``.
CONTENT_PRIOR_PIN_PATTERNS = {"identifier": identifier_pin_patterns,
                              "salience": salience_pin_patterns,
                              "observation": observation_pin_patterns}
