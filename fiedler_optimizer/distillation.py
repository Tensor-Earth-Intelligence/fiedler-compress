"""
Semantic cluster distillation.

Produces concise summaries of compressed zones via an auxiliary LLM while
preserving ligature anchors — tagged spans that mark where a chunk
participates in a cross-zone spectral relationship.

Distillation is **optional** and runs after zone-aware compression but
before spectral obscurity encoding.  Two backends are supported:

* **local**: loads a model from a local file path (e.g., GGUF, Hugging Face).
* **cloud**: sends requests to a configurable API endpoint.  The API key
  is read from the ``FIEDLER_DISTILL_API_KEY`` environment variable —
  it is **never** hardcoded or logged.

For testing, a ``mock`` backend returns deterministic summaries without
any model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)

import numpy as np

from fiedler_optimizer.chunker import Chunk


# ---------------------------------------------------------------------------
# Ligature anchor marker
# ---------------------------------------------------------------------------

# Anchors are inserted into distilled text to mark where a chunk
# participates in a ligature.  Format: «ANCHOR:position»
_ANCHOR_RE = re.compile(r"«ANCHOR:(\d+)»")

ANCHOR_FORMAT = "«ANCHOR:{pos}»"


# ---------------------------------------------------------------------------
# Distillation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistilledZone:
    """A single zone after distillation."""

    zone_position: int
    """Position of this chunk in the compressed output (0-based)."""

    original_text: str
    """Original chunk text before distillation."""

    distilled_text: str
    """Distilled summary produced by the LLM backend."""

    anchors: tuple[int, ...]
    """Ligature anchor positions preserved in the distilled text."""

    compression_improvement: float
    """Additional character reduction from distillation (0.0–1.0)."""


@dataclass(frozen=True)
class DistillationResult:
    """Result of distilling all kept zones."""

    zones: tuple[DistilledZone, ...]
    """One entry per kept chunk, in order."""

    distilled_text: str
    """Fully reconstructed distilled output (zones joined by double newlines)."""

    total_compression_improvement: float
    """Overall additional character reduction from distillation."""

    backend_name: str
    """Which backend produced the distillations."""


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class DistillBackend(ABC):
    """Abstract base for distillation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @abstractmethod
    def distill(self, text: str, context: dict) -> str:
        """
        Produce a concise summary of *text*.

        Parameters
        ----------
        text : str
            The chunk text to distill.
        context : dict
            Spectral context: ``zone_type``, ``fiedler_value``,
            ``connectivity_score``, ``anchors`` (list of ints).

        Returns
        -------
        str
            Distilled summary.  Must preserve any ``«ANCHOR:N»`` markers
            that were injected into *text*.
        """


# ---------------------------------------------------------------------------
# Mock backend (for testing)
# ---------------------------------------------------------------------------

class MockBackend(DistillBackend):
    """
    Deterministic mock backend for testing.

    Produces summaries by extracting the first sentence of each chunk
    and preserving all anchor markers.
    """

    @property
    def name(self) -> str:
        return "mock"

    def distill(self, text: str, context: dict) -> str:
        # Preserve anchors by extracting them first
        anchors_found = _ANCHOR_RE.findall(text)
        text_without_anchors = _ANCHOR_RE.sub("", text).strip()

        # Extract first sentence as a deterministic "summary"
        sentences = re.split(r'(?<=[.!?])\s+', text_without_anchors)
        summary = sentences[0] if sentences else text_without_anchors

        # Truncate if longer than 60% of original
        max_len = max(10, int(len(text_without_anchors) * 0.6))
        if len(summary) > max_len:
            summary = summary[:max_len].rstrip() + "..."

        # Re-inject anchors at the end
        anchor_suffix = " ".join(
            ANCHOR_FORMAT.format(pos=a) for a in anchors_found
        )
        if anchor_suffix:
            summary = f"{summary} {anchor_suffix}"

        return summary


# ---------------------------------------------------------------------------
# Local model backend
# ---------------------------------------------------------------------------

class LocalBackend(DistillBackend):
    """
    Local model backend.

    Loads a model from a local path and uses it for distillation.
    Requires the model to be a Hugging Face pipeline-compatible model.

    Parameters
    ----------
    model_path : str
        Path to the local model directory or file.
    max_tokens : int
        Maximum tokens in the distilled output.
    """

    def __init__(self, model_path: str, max_tokens: int = 128) -> None:
        from pathlib import Path as _Path
        if ".." in _Path(model_path).parts:
            raise ValueError(f"Path traversal not allowed: {model_path}")
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._pipeline = None

    @property
    def name(self) -> str:
        return "local"

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for local distillation. "
                "Install with: pip install transformers"
            )
        self._pipeline = hf_pipeline(
            "summarization", model=self._model_path,
        )

    def distill(self, text: str, context: dict) -> str:
        self._load()
        # Preserve anchors
        anchors_found = _ANCHOR_RE.findall(text)
        clean_text = _ANCHOR_RE.sub("", text).strip()

        if not clean_text:
            return ""

        result = self._pipeline(
            clean_text, max_length=self._max_tokens, min_length=10,
            do_sample=False,
        )
        summary = result[0]["summary_text"]

        anchor_suffix = " ".join(
            ANCHOR_FORMAT.format(pos=a) for a in anchors_found
        )
        if anchor_suffix:
            summary = f"{summary} {anchor_suffix}"
        return summary


# ---------------------------------------------------------------------------
# Cloud API backend
# ---------------------------------------------------------------------------

class CloudBackend(DistillBackend):
    """
    Cloud API backend.

    Sends distillation requests to a configurable API endpoint.
    The API key is loaded from the ``FIEDLER_DISTILL_API_KEY``
    environment variable — **never hardcoded or logged**.

    Parameters
    ----------
    endpoint : str
        Full URL of the distillation API endpoint.
    model : str
        Model identifier to include in the request.
    max_tokens : int
        Maximum tokens in the distilled output.
    """

    def __init__(
        self,
        endpoint: str = "https://api.anthropic.com/v1/messages",
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 128,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "cloud"

    def _get_api_key(self) -> str:
        key = os.environ.get("FIEDLER_DISTILL_API_KEY")
        if not key:
            raise RuntimeError(
                "FIEDLER_DISTILL_API_KEY environment variable is not set. "
                "Set it to your API key for cloud distillation."
            )
        return key

    def distill(self, text: str, context: dict) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for cloud distillation. "
                "Install with: pip install httpx"
            )

        anchors_found = _ANCHOR_RE.findall(text)
        clean_text = _ANCHOR_RE.sub("", text).strip()
        if not clean_text:
            return ""

        api_key = self._get_api_key()

        prompt = (
            f"Summarize the following text concisely in 1-2 sentences. "
            f"Preserve the core meaning.\n\n{clean_text}"
        )

        # Never log api_key
        response = httpx.post(
            self._endpoint,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"API returned non-JSON response: {exc}")

        # Validate response structure before accessing nested fields.
        # Malformed or adversarial responses must not propagate.
        if not isinstance(data, dict):
            raise RuntimeError("API response is not a JSON object")
        content = data.get("content")
        if not isinstance(content, list) or len(content) == 0:
            raise RuntimeError("API response missing 'content' array")
        first = content[0]
        if not isinstance(first, dict) or "text" not in first:
            raise RuntimeError("API response content entry missing 'text'")
        summary = str(first["text"])

        anchor_suffix = " ".join(
            ANCHOR_FORMAT.format(pos=a) for a in anchors_found
        )
        if anchor_suffix:
            summary = f"{summary} {anchor_suffix}"
        return summary


# ---------------------------------------------------------------------------
# Ollama backend (local LLM inference)
# ---------------------------------------------------------------------------

class OllamaBackend(DistillBackend):
    """
    Ollama backend for local LLM-based distillation.

    Sends each zone's content to a local Ollama instance for
    summarization.  Designed for CPU inference with large models
    (e.g. Gemma 4 E4B) where each call may take 15--60 seconds.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. ``"gemma4:e4b"``).
    endpoint : str
        Ollama API URL.
    timeout : float
        HTTP timeout in seconds (default 300 for slow CPU inference).
    max_ratio : float
        Maximum length of the summary as a fraction of the original
        text length.  Summaries exceeding this are truncated to ensure
        distillation actually compresses.
    """

    def __init__(
        self,
        model: str = "gemma4:e4b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 300.0,
        max_ratio: float = 0.40,
    ) -> None:
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout
        self._max_ratio = max_ratio

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def distill(self, text: str, context: dict) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for the Ollama backend. "
                "Install with: pip install httpx"
            )

        # Preserve anchors
        anchors_found = _ANCHOR_RE.findall(text)
        clean_text = _ANCHOR_RE.sub("", text).strip()

        if not clean_text:
            return text

        prompt = (
            "Summarize the following text concisely, preserving all "
            "key facts, names, numbers, and decisions:\n\n"
            f"{clean_text}"
        )

        try:
            response = httpx.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self._timeout,
            )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._endpoint}. "
                f"Start Ollama with: ollama serve"
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama request timed out after {self._timeout}s. "
                f"The model {self._model!r} may be too slow for this "
                f"hardware. Try a smaller model or increase --timeout."
            )

        response.raise_for_status()

        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Ollama returned non-JSON response: {exc}")

        summary = data.get("response", "").strip()
        if not summary:
            # Fallback: return truncated original
            summary = clean_text

        # Enforce max length ratio to ensure compression
        max_len = max(10, int(len(clean_text) * self._max_ratio))
        if len(summary) > max_len:
            # Truncate at a word boundary
            truncated = summary[:max_len]
            last_space = truncated.rfind(" ")
            if last_space > max_len // 2:
                truncated = truncated[:last_space]
            summary = truncated.rstrip() + "..."

        # Re-inject anchors
        anchor_suffix = " ".join(
            ANCHOR_FORMAT.format(pos=a) for a in anchors_found
        )
        if anchor_suffix:
            summary = f"{summary} {anchor_suffix}"

        return summary


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[DistillBackend]] = {
    "mock": MockBackend,
    "local": LocalBackend,
    "cloud": CloudBackend,
    "ollama": OllamaBackend,
}


def get_backend(name: str, **kwargs) -> DistillBackend:
    """
    Instantiate a distillation backend by name.

    Parameters
    ----------
    name : str
        One of "mock", "local", "cloud".
    **kwargs
        Passed to the backend constructor.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown distillation backend: {name!r}. "
            f"Available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name](**kwargs)


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

class Distiller:
    """
    Distills compressed zones via an auxiliary LLM backend.

    Preserves ligature anchors: if a ligature connects chunk at
    compressed-position A to chunk at compressed-position B, the
    distilled summaries retain ``«ANCHOR:A»`` and ``«ANCHOR:B»``
    markers so downstream tools can re-link the ligatures.

    Parameters
    ----------
    backend : DistillBackend
        The LLM backend to use for distillation.
    """

    def __init__(self, backend: DistillBackend) -> None:
        self._backend = backend

    def distill(
        self,
        kept_indices: Sequence[int],
        chunks: Sequence[Chunk],
        zoned: Sequence,
        fiedler_vector: np.ndarray,
        scores: Sequence[float],
        ligatures: Sequence[dict] | None = None,
    ) -> DistillationResult:
        """
        Distill all kept chunks.

        Parameters
        ----------
        kept_indices : Sequence[int]
            Original-chunk indices that survived compression, in order.
        chunks : Sequence[Chunk]
            All original chunks.
        zoned : Sequence[ZonedChunk]
            Zone classifications for all original chunks.
        fiedler_vector : np.ndarray
            Full Fiedler vector.
        scores : Sequence[float]
            Connectivity scores for all original chunks.
        ligatures : Sequence[dict], optional
            Post-compression ligature annotations (list of dicts with
            ``source_index`` and ``target_index`` fields).

        Returns
        -------
        DistillationResult
        """
        # Build a map: compressed-position → set of anchor positions
        anchor_map: dict[int, set[int]] = {}
        if ligatures:
            for lig in ligatures:
                src = lig["source_index"]
                tgt = lig["target_index"]
                anchor_map.setdefault(src, set()).add(src)
                anchor_map.setdefault(src, set()).add(tgt)
                anchor_map.setdefault(tgt, set()).add(src)
                anchor_map.setdefault(tgt, set()).add(tgt)

        distilled_zones: list[DistilledZone] = []
        original_total = 0
        distilled_total = 0

        for pos, orig_idx in enumerate(kept_indices):
            chunk = chunks[orig_idx]
            zc = zoned[orig_idx]
            original_text = chunk.text
            original_total += len(original_text)

            # Inject anchor markers into the text before distillation
            anchors_for_chunk = sorted(anchor_map.get(pos, set()))
            text_with_anchors = original_text
            if anchors_for_chunk:
                markers = " ".join(
                    ANCHOR_FORMAT.format(pos=a) for a in anchors_for_chunk
                )
                text_with_anchors = f"{original_text} {markers}"

            context = {
                "zone_type": zc.zone.name,
                "fiedler_value": round(float(fiedler_vector[orig_idx]), 6),
                "connectivity_score": round(float(scores[orig_idx]), 6),
                "anchors": anchors_for_chunk,
            }

            try:
                distilled = self._backend.distill(text_with_anchors, context)
            except Exception as exc:
                # Per-zone fallback: mock-style 60% truncation
                warnings.warn(
                    f"Distillation failed for zone {pos} "
                    f"({type(exc).__name__}: {exc}). "
                    f"Falling back to truncation.",
                    stacklevel=2,
                )
                logger.warning(
                    "Zone %d distill fallback: %s: %s",
                    pos, type(exc).__name__, exc,
                )
                _anchors = _ANCHOR_RE.findall(text_with_anchors)
                _clean = _ANCHOR_RE.sub("", text_with_anchors).strip()
                _max = max(10, int(len(_clean) * 0.6))
                _sentences = re.split(r'(?<=[.!?])\s+', _clean)
                _summary = _sentences[0] if _sentences else _clean
                if len(_summary) > _max:
                    _summary = _summary[:_max].rstrip() + "..."
                _asuf = " ".join(
                    ANCHOR_FORMAT.format(pos=a) for a in _anchors
                )
                distilled = f"{_summary} {_asuf}".strip() if _asuf else _summary

            distilled_total += len(distilled)

            orig_len = len(original_text)
            dist_len = len(re.sub(r"«ANCHOR:\d+»\s*", "", distilled))
            improvement = 1.0 - (dist_len / max(orig_len, 1))

            distilled_zones.append(DistilledZone(
                zone_position=pos,
                original_text=original_text,
                distilled_text=distilled,
                anchors=tuple(anchors_for_chunk),
                compression_improvement=round(max(0.0, improvement), 4),
            ))

        distilled_text = "\n\n".join(dz.distilled_text for dz in distilled_zones)
        total_improvement = 1.0 - (distilled_total / max(original_total, 1))

        return DistillationResult(
            zones=tuple(distilled_zones),
            distilled_text=distilled_text,
            total_compression_improvement=round(max(0.0, total_improvement), 4),
            backend_name=self._backend.name,
        )
