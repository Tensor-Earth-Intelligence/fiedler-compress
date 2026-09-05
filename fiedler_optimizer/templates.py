"""
Spectrally-derived reasoning templates.

Reasoning templates use the spectral topology of the Fiedler partition to
generate structured reasoning scaffolds that downstream LLMs can use to
process compressed content more effectively.

Templates are parameterised entirely by spectral properties -- zone counts,
cluster densities, spectral gap distributions, inter-zone coherence scores,
and Fiedler eigenvalues.  **No raw user text is ever interpolated** into a
template; only zone IDs and numeric spectral metadata are referenced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Allowlisted template types
# ---------------------------------------------------------------------------

@unique
class TemplateType(Enum):
    """Fixed set of allowlisted reasoning template types."""

    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    CAUSAL_CHAIN = "causal-chain"
    ENTITY_EXTRACTION = "entity-extraction"
    QUESTION_ANSWERING = "question-answering"


_TEMPLATE_TYPE_NAMES: frozenset[str] = frozenset(t.value for t in TemplateType)


# ---------------------------------------------------------------------------
# Spectral profile — computed from pipeline data, never from text
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralProfile:
    """
    Spectral topology of a compressed output.

    Every field is derived from the Fiedler vector, adjacency matrix, or
    zone classifications -- never from raw chunk text.
    """

    num_zones: int
    """Number of kept chunks (zones) in the compressed output."""

    zone_types: tuple[str, ...]
    """Zone classification label per kept chunk (e.g. INSTRUCTION, CONTEXT)."""

    cluster_sizes: tuple[int, int]
    """(cluster_0_count, cluster_1_count) from Fiedler sign partition."""

    spectral_gap_mean: float
    """Mean |fiedler[i] - fiedler[j]| across adjacent kept-chunk pairs."""

    spectral_gap_max: float
    """Maximum spectral gap across adjacent kept-chunk pairs."""

    cluster_density: tuple[float, float]
    """Mean intra-cluster edge weight for Fiedler partition 0 and 1."""

    inter_zone_coherence: float
    """Mean edge weight between kept chunks in different Fiedler partitions."""

    algebraic_connectivity: float
    """Lambda-2 of the graph Laplacian."""


def compute_spectral_profile(
    kept_indices: Sequence[int],
    fiedler_vector: np.ndarray,
    lambda2: float,
    adjacency: np.ndarray,
    zoned: Sequence,
) -> SpectralProfile:
    """
    Compute a :class:`SpectralProfile` from pipeline spectral data.

    Parameters
    ----------
    kept_indices : Sequence[int]
        Original-chunk indices that survived compression, in order.
    fiedler_vector : np.ndarray
        Full Fiedler vector over *all* original chunks.
    lambda2 : float
        Algebraic connectivity.
    adjacency : np.ndarray
        Adjacency matrix (possibly ligature-enhanced).
    zoned : Sequence[ZonedChunk]
        Zone classifications for *all* original chunks.
    """
    if not kept_indices:
        return SpectralProfile(
            num_zones=0,
            zone_types=(),
            cluster_sizes=(0, 0),
            spectral_gap_mean=0.0,
            spectral_gap_max=0.0,
            cluster_density=(0.0, 0.0),
            inter_zone_coherence=0.0,
            algebraic_connectivity=max(0.0, float(lambda2)),
        )

    # Zone types for kept chunks
    zone_types = tuple(zoned[i].zone.name for i in kept_indices)

    # Fiedler sign partition over kept chunks
    kept_fiedler = [float(fiedler_vector[i]) for i in kept_indices]
    cluster_0 = [i for i, f in zip(kept_indices, kept_fiedler) if f >= 0]
    cluster_1 = [i for i, f in zip(kept_indices, kept_fiedler) if f < 0]

    # Spectral gaps between adjacent kept chunks
    gaps: list[float] = []
    for pos in range(len(kept_indices) - 1):
        gap = abs(kept_fiedler[pos] - kept_fiedler[pos + 1])
        gaps.append(gap)
    gap_mean = float(np.mean(gaps)) if gaps else 0.0
    gap_max = float(np.max(gaps)) if gaps else 0.0

    # Intra-cluster density
    def _mean_intra_weight(indices: list[int]) -> float:
        if len(indices) < 2:
            return 0.0
        weights = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                weights.append(float(adjacency[indices[a], indices[b]]))
        return float(np.mean(weights)) if weights else 0.0

    density_0 = _mean_intra_weight(cluster_0)
    density_1 = _mean_intra_weight(cluster_1)

    # Inter-cluster coherence
    cross_weights: list[float] = []
    for i in cluster_0:
        for j in cluster_1:
            cross_weights.append(float(adjacency[i, j]))
    inter_coherence = float(np.mean(cross_weights)) if cross_weights else 0.0

    return SpectralProfile(
        num_zones=len(kept_indices),
        zone_types=zone_types,
        cluster_sizes=(len(cluster_0), len(cluster_1)),
        spectral_gap_mean=round(gap_mean, 6),
        spectral_gap_max=round(gap_max, 6),
        cluster_density=(round(density_0, 6), round(density_1, 6)),
        inter_zone_coherence=round(inter_coherence, 6),
        algebraic_connectivity=round(max(0.0, float(lambda2)), 6),
    )


# ---------------------------------------------------------------------------
# Template section and template dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateSection:
    """
    A single section within a reasoning scaffold.

    All content is derived from spectral control information.
    ``zone_refs`` are positional indices into the *compressed* output's
    kept-chunk list (0-based).
    """

    heading: str
    zone_refs: tuple[int, ...]
    instruction: str
    spectral_context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningTemplate:
    """
    A structured reasoning scaffold for LLM consumption.

    Templates reference zones by positional ID and spectral metrics only,
    never by raw text content.
    """

    template_type: str
    profile: dict
    sections: tuple[TemplateSection, ...]


# ---------------------------------------------------------------------------
# Template builders — one per TemplateType
# ---------------------------------------------------------------------------

def _build_summarization(profile: SpectralProfile) -> ReasoningTemplate:
    """Summarization scaffold: group by cluster, summarise each."""
    c0_zones = [i for i in range(profile.num_zones) if profile.zone_types[i] != "UNKNOWN"
                and i < profile.cluster_sizes[0]]
    c1_zones = [i for i in range(profile.num_zones) if i not in c0_zones]

    # Partition zone refs by Fiedler cluster membership
    # Use cluster_sizes to split sequentially
    c0_refs = tuple(range(profile.cluster_sizes[0]))
    c1_refs = tuple(range(profile.cluster_sizes[0], profile.num_zones))

    sections = []
    if c0_refs:
        sections.append(TemplateSection(
            heading="Core cluster (Fiedler partition 0)",
            zone_refs=c0_refs,
            instruction=(
                f"Summarise the content in zones {list(c0_refs)}. "
                f"These zones form a cohesive cluster with intra-density "
                f"{profile.cluster_density[0]:.4f}."
            ),
            spectral_context={
                "cluster_id": 0,
                "density": profile.cluster_density[0],
                "size": profile.cluster_sizes[0],
            },
        ))
    if c1_refs:
        sections.append(TemplateSection(
            heading="Secondary cluster (Fiedler partition 1)",
            zone_refs=c1_refs,
            instruction=(
                f"Summarise the content in zones {list(c1_refs)}. "
                f"These zones form a secondary cluster with intra-density "
                f"{profile.cluster_density[1]:.4f}."
            ),
            spectral_context={
                "cluster_id": 1,
                "density": profile.cluster_density[1],
                "size": profile.cluster_sizes[1],
            },
        ))
    sections.append(TemplateSection(
        heading="Synthesis",
        zone_refs=tuple(range(profile.num_zones)),
        instruction=(
            f"Synthesise across both clusters. Inter-zone coherence is "
            f"{profile.inter_zone_coherence:.4f}; spectral gap max is "
            f"{profile.spectral_gap_max:.4f}."
        ),
        spectral_context={
            "inter_zone_coherence": profile.inter_zone_coherence,
            "spectral_gap_max": profile.spectral_gap_max,
        },
    ))

    return ReasoningTemplate(
        template_type=TemplateType.SUMMARIZATION.value,
        profile=asdict(profile),
        sections=tuple(sections),
    )


def _build_comparison(profile: SpectralProfile) -> ReasoningTemplate:
    """Comparison scaffold: contrast the two Fiedler partitions."""
    c0_refs = tuple(range(profile.cluster_sizes[0]))
    c1_refs = tuple(range(profile.cluster_sizes[0], profile.num_zones))

    sections = [
        TemplateSection(
            heading="Partition A (Fiedler >= 0)",
            zone_refs=c0_refs,
            instruction=(
                f"Identify key properties of zones {list(c0_refs)}. "
                f"Cluster density: {profile.cluster_density[0]:.4f}."
            ),
            spectral_context={
                "cluster_id": 0,
                "density": profile.cluster_density[0],
                "size": profile.cluster_sizes[0],
            },
        ),
        TemplateSection(
            heading="Partition B (Fiedler < 0)",
            zone_refs=c1_refs,
            instruction=(
                f"Identify key properties of zones {list(c1_refs)}. "
                f"Cluster density: {profile.cluster_density[1]:.4f}."
            ),
            spectral_context={
                "cluster_id": 1,
                "density": profile.cluster_density[1],
                "size": profile.cluster_sizes[1],
            },
        ),
        TemplateSection(
            heading="Contrast",
            zone_refs=tuple(range(profile.num_zones)),
            instruction=(
                f"Compare and contrast partitions A and B. "
                f"Inter-zone coherence: {profile.inter_zone_coherence:.4f}. "
                f"Mean spectral gap: {profile.spectral_gap_mean:.4f}."
            ),
            spectral_context={
                "inter_zone_coherence": profile.inter_zone_coherence,
                "spectral_gap_mean": profile.spectral_gap_mean,
            },
        ),
    ]

    return ReasoningTemplate(
        template_type=TemplateType.COMPARISON.value,
        profile=asdict(profile),
        sections=tuple(sections),
    )


def _build_causal_chain(profile: SpectralProfile) -> ReasoningTemplate:
    """Causal-chain scaffold: follow spectral ordering as causal flow."""
    all_refs = tuple(range(profile.num_zones))
    sections = []

    # Create a section for each adjacent pair, using spectral gap as
    # transition-strength signal
    for pos in range(profile.num_zones - 1):
        # Approximate per-transition gap from profile averages
        gap_label = "strong" if profile.spectral_gap_mean > 0.3 else "moderate" if profile.spectral_gap_mean > 0.1 else "weak"
        sections.append(TemplateSection(
            heading=f"Transition zone {pos} -> zone {pos + 1}",
            zone_refs=(pos, pos + 1),
            instruction=(
                f"Trace the causal or logical link from zone {pos} to "
                f"zone {pos + 1}. Spectral gap indicates {gap_label} "
                f"transition (mean gap: {profile.spectral_gap_mean:.4f})."
            ),
            spectral_context={
                "from_zone": pos,
                "to_zone": pos + 1,
                "spectral_gap_mean": profile.spectral_gap_mean,
            },
        ))

    sections.append(TemplateSection(
        heading="Full causal chain",
        zone_refs=all_refs,
        instruction=(
            f"Reconstruct the complete causal chain across all {profile.num_zones} "
            f"zones. Algebraic connectivity: {profile.algebraic_connectivity:.4f}."
        ),
        spectral_context={
            "algebraic_connectivity": profile.algebraic_connectivity,
            "num_zones": profile.num_zones,
        },
    ))

    return ReasoningTemplate(
        template_type=TemplateType.CAUSAL_CHAIN.value,
        profile=asdict(profile),
        sections=tuple(sections),
    )


def _build_entity_extraction(profile: SpectralProfile) -> ReasoningTemplate:
    """Entity-extraction scaffold: group high-coherence zones for co-ref."""
    c0_refs = tuple(range(profile.cluster_sizes[0]))
    c1_refs = tuple(range(profile.cluster_sizes[0], profile.num_zones))

    sections = [
        TemplateSection(
            heading="Entity scan: cluster 0",
            zone_refs=c0_refs,
            instruction=(
                f"Extract entities from zones {list(c0_refs)}. "
                f"Intra-cluster density {profile.cluster_density[0]:.4f} "
                f"suggests {'strong' if profile.cluster_density[0] > 0.1 else 'weak'} "
                f"co-reference within this cluster."
            ),
            spectral_context={
                "cluster_id": 0,
                "density": profile.cluster_density[0],
            },
        ),
        TemplateSection(
            heading="Entity scan: cluster 1",
            zone_refs=c1_refs,
            instruction=(
                f"Extract entities from zones {list(c1_refs)}. "
                f"Intra-cluster density {profile.cluster_density[1]:.4f} "
                f"suggests {'strong' if profile.cluster_density[1] > 0.1 else 'weak'} "
                f"co-reference within this cluster."
            ),
            spectral_context={
                "cluster_id": 1,
                "density": profile.cluster_density[1],
            },
        ),
        TemplateSection(
            heading="Cross-cluster entity resolution",
            zone_refs=tuple(range(profile.num_zones)),
            instruction=(
                f"Resolve entities across clusters. Inter-zone coherence "
                f"{profile.inter_zone_coherence:.4f} indicates "
                f"{'significant' if profile.inter_zone_coherence > 0.05 else 'limited'} "
                f"cross-cluster entity overlap."
            ),
            spectral_context={
                "inter_zone_coherence": profile.inter_zone_coherence,
            },
        ),
    ]

    return ReasoningTemplate(
        template_type=TemplateType.ENTITY_EXTRACTION.value,
        profile=asdict(profile),
        sections=tuple(sections),
    )


def _build_question_answering(profile: SpectralProfile) -> ReasoningTemplate:
    """QA scaffold: separate instruction (directive) zones from evidence."""
    instruction_refs = tuple(
        i for i, zt in enumerate(profile.zone_types) if zt == "INSTRUCTION"
    )
    evidence_refs = tuple(
        i for i, zt in enumerate(profile.zone_types) if zt != "INSTRUCTION"
    )

    sections = []
    if instruction_refs:
        sections.append(TemplateSection(
            heading="Directive zones",
            zone_refs=instruction_refs,
            instruction=(
                f"Identify the question or task from zones {list(instruction_refs)}. "
                f"These are classified as INSTRUCTION with spectral connectivity "
                f"lambda-2 = {profile.algebraic_connectivity:.4f}."
            ),
            spectral_context={
                "zone_classification": "INSTRUCTION",
                "count": len(instruction_refs),
            },
        ))
    if evidence_refs:
        sections.append(TemplateSection(
            heading="Evidence zones",
            zone_refs=evidence_refs,
            instruction=(
                f"Gather evidence from zones {list(evidence_refs)}. "
                f"Mean spectral gap: {profile.spectral_gap_mean:.4f}. "
                f"Higher gaps indicate weaker connections between evidence zones."
            ),
            spectral_context={
                "zone_classification": "CONTEXT/UNKNOWN",
                "count": len(evidence_refs),
                "spectral_gap_mean": profile.spectral_gap_mean,
            },
        ))
    sections.append(TemplateSection(
        heading="Answer synthesis",
        zone_refs=tuple(range(profile.num_zones)),
        instruction=(
            f"Synthesise the answer using directive and evidence zones. "
            f"Total zones: {profile.num_zones}. "
            f"Inter-zone coherence: {profile.inter_zone_coherence:.4f}."
        ),
        spectral_context={
            "num_zones": profile.num_zones,
            "inter_zone_coherence": profile.inter_zone_coherence,
            "algebraic_connectivity": profile.algebraic_connectivity,
        },
    ))

    return ReasoningTemplate(
        template_type=TemplateType.QUESTION_ANSWERING.value,
        profile=asdict(profile),
        sections=tuple(sections),
    )


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

class TemplateRegistry:
    """
    Registry of allowlisted reasoning template builders.

    Only the fixed set of template types defined in :class:`TemplateType`
    can be requested.  Unknown types raise ``ValueError``.
    """

    _BUILDERS: dict[str, Callable[[SpectralProfile], ReasoningTemplate]] = {
        TemplateType.SUMMARIZATION.value: _build_summarization,
        TemplateType.COMPARISON.value: _build_comparison,
        TemplateType.CAUSAL_CHAIN.value: _build_causal_chain,
        TemplateType.ENTITY_EXTRACTION.value: _build_entity_extraction,
        TemplateType.QUESTION_ANSWERING.value: _build_question_answering,
    }

    @classmethod
    def list_types(cls) -> list[str]:
        """Return the sorted list of allowlisted template type names."""
        return sorted(cls._BUILDERS.keys())

    @classmethod
    def get(cls, template_type: str) -> Callable[[SpectralProfile], ReasoningTemplate]:
        """
        Return the builder function for *template_type*.

        Raises ``ValueError`` if the type is not in the allowlist.
        """
        if template_type not in cls._BUILDERS:
            raise ValueError(
                f"Unknown template type: {template_type!r}. "
                f"Allowed: {cls.list_types()}"
            )
        return cls._BUILDERS[template_type]

    @classmethod
    def build(
        cls,
        template_type: str,
        profile: SpectralProfile,
    ) -> ReasoningTemplate:
        """Build a reasoning template of the given type."""
        builder = cls.get(template_type)
        return builder(profile)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_template(template: ReasoningTemplate) -> str:
    """
    Render a :class:`ReasoningTemplate` into a string scaffold.

    The output is safe for direct inclusion in an LLM prompt.  It
    contains only zone references and spectral metadata, never raw
    user text.
    """
    prof = template.profile
    lines: list[str] = []

    lines.append(f"[REASONING SCAFFOLD: {template.template_type}]")
    lines.append(
        f"Spectral topology: {prof['num_zones']} zones, "
        f"lambda_2={prof['algebraic_connectivity']:.4f}, "
        f"gap_max={prof['spectral_gap_max']:.4f}"
    )
    c0, c1 = prof["cluster_sizes"]
    lines.append(f"Cluster balance: {c0}/{c1}")
    lines.append("")

    for section in template.sections:
        lines.append(f"## {section.heading}")
        lines.append(f"Zones: {list(section.zone_refs)}")
        lines.append(section.instruction)
        lines.append("")

    return "\n".join(lines)
