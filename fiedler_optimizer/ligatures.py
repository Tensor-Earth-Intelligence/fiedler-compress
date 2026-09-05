"""
Context-ligature preprocessing for enhanced spectral decomposition.

Ligatures inject domain-specific structural edges into the similarity graph
before Fiedler decomposition. These edges encode relationships that
embedding similarity is blind to: causal dependencies, temporal precedence,
structural importance, and domain-specific protection rules.

Analogy: In print writing, each letter is an isolated glyph. In cursive,
ligatures between letters encode sequential constraints absent in the
disconnected forms. The letters haven't changed, but the connections
carry structural information that changes how the whole word is processed.

Similarly, context ligatures transform a weakly connected similarity graph
into a structurally enriched graph where edges additionally encode known
domain relationships — making implicit dependencies explicit and
spectrally visible to the Fiedler vector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Sequence

import numpy as np

from fiedler_optimizer.chunker import Chunk


class LigatureType(Enum):
    """Categories of structural relationships that ligatures can encode."""
    DEPENDENCY = auto()      # Chunk A depends on chunk B (directional)
    PROTECTION = auto()      # Chunk should be protected from removal
    REDUNDANCY = auto()      # Chunks are known duplicates (strengthen removal signal)
    TEMPORAL = auto()        # Chunks have temporal ordering constraints
    REFERENCE = auto()       # Chunk references another chunk


@dataclass
class LigatureRule:
    """
    A rule that detects structural relationships between chunks
    and injects edges into the similarity graph.

    Parameters
    ----------
    name : str
        Human-readable name for this rule.
    ligature_type : LigatureType
        What kind of structural relationship this rule detects.
    detector : Callable
        Function(chunk_i, chunk_j) -> float. Returns the ligature
        weight (0.0 = no ligature, higher = stronger connection).
        For PROTECTION type, Function(chunk) -> float instead.
    weight : float
        Base multiplier applied to detected ligatures.
    description : str
        What this rule does, for documentation.
    """
    name: str
    ligature_type: LigatureType
    detector: Callable
    weight: float = 1.0
    description: str = ""


@dataclass
class LigatureResult:
    """Result of applying ligatures to a similarity graph."""
    original_lambda2: float
    enhanced_lambda2: float
    edges_added: int
    rules_triggered: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in ligature detectors
# ---------------------------------------------------------------------------

def _detect_blocker_dependency(chunk_i: Chunk, chunk_j: Chunk) -> float:
    """Detect blocker/dependency relationships between chunks."""
    blocker_patterns = re.compile(
        r'\b(block(?:ed|er|ing|s)?|depend(?:s|ency|encies|ent)|'
        r'wait(?:ing|s)?\s+(?:on|for)|prerequisite|requires?|'
        r'cannot\s+proceed|contingent)\b',
        re.IGNORECASE,
    )
    action_patterns = re.compile(
        r'\b(next\s+step|action\s+item|todo|follow[- ]up|'
        r'task|milestone|deliverable|deadline|cutover|'
        r'deploy(?:ment)?|release|launch)\b',
        re.IGNORECASE,
    )

    i_has_blocker = bool(blocker_patterns.search(chunk_i.text))
    j_has_action = bool(action_patterns.search(chunk_j.text))
    j_has_blocker = bool(blocker_patterns.search(chunk_j.text))
    i_has_action = bool(action_patterns.search(chunk_i.text))

    # Bidirectional: blockers connect to action items
    if (i_has_blocker and j_has_action) or (j_has_blocker and i_has_action):
        return 1.0
    # Blockers connect to other blockers (cluster them)
    if i_has_blocker and j_has_blocker:
        return 0.5
    return 0.0


def _detect_risk_stakeholder(chunk_i: Chunk, chunk_j: Chunk) -> float:
    """Detect risk-to-stakeholder relationships."""
    risk_patterns = re.compile(
        r'\b(risk|threat|vulnerabilit(?:y|ies)|exposure|'
        r'concern|warning|caveat|limitation|audit)\b',
        re.IGNORECASE,
    )
    stakeholder_patterns = re.compile(
        r'\b(stakeholder|manager|director|VP|CTO|CEO|CFO|'
        r'executive|leadership|board|client|customer|'
        r'approval|sign[- ]off)\b',
        re.IGNORECASE,
    )

    i_risk = bool(risk_patterns.search(chunk_i.text))
    j_stake = bool(stakeholder_patterns.search(chunk_j.text))
    j_risk = bool(risk_patterns.search(chunk_j.text))
    i_stake = bool(stakeholder_patterns.search(chunk_i.text))

    if (i_risk and j_stake) or (j_risk and i_stake):
        return 0.8
    return 0.0


def _detect_data_reference(chunk_i: Chunk, chunk_j: Chunk) -> float:
    """Detect when one chunk references data defined in another."""
    # Look for specific identifiers (ticket numbers, document numbers, etc.)
    id_pattern = re.compile(
        r'\b([A-Z]{2,}-\d{3,}|'           # PLAT-1234, KB-1042
        r'[Dd]ocument\s+\d+|'              # Document 1
        r'[Ss]ection\s+\d+[\.\d]*|'        # Section 4.2
        r'[Ss]tep\s+\d+|'                  # Step 1
        r'[Qq]\d\s+\d{4}|'                 # Q1 2024
        r'[Tt]able\s+\d+|'                 # Table 1
        r'[Ff]igure\s+\d+)\b',             # Figure 1
    )

    ids_i = set(id_pattern.findall(chunk_i.text))
    ids_j = set(id_pattern.findall(chunk_j.text))

    if not ids_i or not ids_j:
        return 0.0

    # Check for shared identifiers
    shared = ids_i & ids_j
    if shared:
        return min(1.0, len(shared) * 0.3)

    return 0.0


# Forward causal connectors (premise SIGNALS a following consequence) and consequence
# openers (conclusion STARTS with a consequence marker).
# Premise-signaling forward connectors: mid/end-of-clause markers that a CONCLUSION
# follows.  Deliberately EXCLUDES sentence-initial openers like "As a result" that a
# standalone conclusion (or a mimicking distractor) would carry -- keying on those
# links promiscuously.  A causal ligature must be anchored by the PREMISE.
_CAUSE_FORWARD = re.compile(
    r'\b(therefore|thus|hence|consequently|accordingly|so that|which means|'
    r'leading to|resulting in)\b', re.IGNORECASE)


def _detect_causal_chain(chunk_i: Chunk, chunk_j: Chunk) -> float:
    """Selective causal DEPENDENCY: link a premise to its IMMEDIATE conclusion only.

    Fires when chunk_j is the immediate successor of chunk_i (directional, index+1) AND
    the premise signals a forthcoming consequence (forward connector) OR the conclusion
    opens with a consequence marker.  This replaces the earlier promiscuous rule
    (link any pair within +-2 positions where EITHER chunk had causal words), which
    connected causal chunks to unrelated neighbours and hurt retention.
    """
    if chunk_j.index - chunk_i.index != 1:          # immediate successor, forward only
        return 0.0
    if _CAUSE_FORWARD.search(chunk_i.text):         # anchored by the PREMISE only
        return 1.0
    return 0.0


def _detect_protection_critical(chunk: Chunk) -> float:
    """Detect chunks that should be strongly protected from removal."""
    critical_patterns = re.compile(
        r'\b(blocker|blocked|critical|urgent|must|required|'
        r'deadline|SLA|compliance|security|risk|P[01]\s|'
        r'production\s+(?:down|outage|incident)|'
        r'data\s+(?:loss|breach|leak))\b',
        re.IGNORECASE,
    )

    matches = critical_patterns.findall(chunk.text)
    if matches:
        return min(2.0, len(matches) * 0.4)
    return 0.0


# ---------------------------------------------------------------------------
# Reviewed detector registry (content_prior-style) -- ONE canonical entry per
# detector MECHANISM, addressable by name and independently composable, instead of
# only reachable inside a fixed domain bundle.  Each entry's description carries an
# HONEST validation-status note -- mirroring geometry/content_prior.py's discipline
# of stating what is actually proven versus merely shipped.  The domain RULE_SETS
# below are convenience presets built FROM this registry (via dataclasses.replace
# for a domain's weight/name override), not a second, drifting copy of the rules.
# ---------------------------------------------------------------------------
from dataclasses import replace as _replace

LIGATURE_DETECTORS: dict[str, LigatureRule] = {
    "shared_identifier_reference": LigatureRule(
        name="shared_identifier_reference",
        ligature_type=LigatureType.REFERENCE,
        detector=_detect_data_reference,
        weight=1.0,
        description=(
            "Connect chunks that share an exact identifier (ticket/document/section/"
            "table/figure number, ...).  VALIDATED: synthetic (exp_ligature.py, "
            "exp_ligature_structural.py) AND REAL CORPUS (exp_ligature_real_corpus.py, "
            "RFC 9110, genuine 'Section N.M' cross-references) -- the best-in-class "
            "relational-preservation detector when it fires.  Precise/pairwise, not "
            "positional -- this is WHY it works (see 'detector precision' finding, "
            "MEMO_ligature_20260714.md).  Known caveat: the identifier must be chunked "
            "TOGETHER with the content it should protect (e.g. inline a citation label "
            "into its own opening sentence, not a separate header line) or protection "
            "silently fails on exactly the content it is meant to save -- discovered "
            "and documented during the real-corpus run."
        ),
    ),
    "causal_chain": LigatureRule(
        name="causal_chain",
        ligature_type=LigatureType.DEPENDENCY,
        detector=_detect_causal_chain,
        weight=1.0,
        description=(
            "Selective premise->conclusion link: fires only when chunk_j is chunk_i's "
            "IMMEDIATE successor and chunk_i (the premise) carries a forward causal "
            "connector (therefore/thus/hence/...).  Directional and premise-anchored by "
            "design -- the promiscuous predecessor (link any pair within +-2 positions "
            "where EITHER chunk had causal words) actively HURT retention (0.20, worse "
            "than plain) before this redesign.  VALIDATED: synthetic only "
            "(exp_ligature_causal.py) as of 2026-07-14 -- a real-corpus confirmation is "
            "pending/in progress.  Honest boundary: causal links via surface connectors "
            "are MIMICABLE (a distractor opening 'As a result ...' fooled the first, "
            "promiscuous cut) in a way exact shared-reference links are not -- this "
            "detector is inherently softer/more fragile than shared_identifier_reference."
        ),
    ),
    "blocker_dependency": LigatureRule(
        name="blocker_dependency",
        ligature_type=LigatureType.DEPENDENCY,
        detector=_detect_blocker_dependency,
        weight=1.5,
        description=(
            "Connect blocker items to downstream action items.  NOT INDIVIDUALLY "
            "VALIDATED: only ever tested bundled inside the project_management rule set "
            "alongside 3 other rules (exp_ligature.py's 'ligature_pm' condition); no "
            "isolated ablation exists, so the bundle's measured gain cannot be "
            "attributed to this detector specifically."
        ),
    ),
    "risk_stakeholder": LigatureRule(
        name="risk_stakeholder",
        ligature_type=LigatureType.DEPENDENCY,
        detector=_detect_risk_stakeholder,
        weight=1.2,
        description=(
            "Connect risk items to stakeholder-facing content.  NOT INDIVIDUALLY "
            "VALIDATED -- same caveat as blocker_dependency: only tested as part of the "
            "project_management bundle, never in isolation."
        ),
    ),
    "protection_critical": LigatureRule(
        name="protection_critical",
        ligature_type=LigatureType.PROTECTION,
        detector=_detect_protection_critical,
        weight=2.0,
        description=(
            "Protect critical/urgent content from removal.  Architecturally different "
            "from the others (PROTECTION type: scores individual chunks, not pairs).  "
            "NOT INDIVIDUALLY VALIDATED -- same caveat as blocker_dependency/"
            "risk_stakeholder: only ever exercised as part of the project_management "
            "bundle, never in isolation."
        ),
    ),
}

# Injected ligature edges are scaled RELATIVE to the graph's connectivity so a single
# structural edge is competitive with a dense distractor cluster (a raw fixed weight is
# out-voted by a clique).  "degree_max" scales to the best-connected node's degree.
LIGATURE_EDGE_SCALE = "degree_max"   # "degree_max" | "degree_mean" | "raw"
LIGATURE_EDGE_GAIN = 2.0             # multiplier: linked node degree ~2x best cluster member


def _edge_scale(adjacency):
    import numpy as _np
    pos = adjacency[adjacency > 0]
    if pos.size == 0 or LIGATURE_EDGE_SCALE == "raw":
        return 1.0
    row = adjacency.sum(axis=1)
    if LIGATURE_EDGE_SCALE == "degree_mean":
        nz = row[row > 0]
        return float(nz.mean()) if nz.size else 1.0
    return float(row.max()) * LIGATURE_EDGE_GAIN   # degree_max (default)


# ---------------------------------------------------------------------------
# Domain rule-set presets -- built FROM the reviewed registry above (single source
# of truth for the detector functions); a domain may override name/weight (e.g.
# "legal" historically used a higher section_reference weight) via dataclasses.replace,
# but the underlying detector, type, and honest description travel with it unchanged.
# ---------------------------------------------------------------------------

PROJECT_MANAGEMENT_RULES = [
    LIGATURE_DETECTORS["blocker_dependency"],
    LIGATURE_DETECTORS["risk_stakeholder"],
    _replace(LIGATURE_DETECTORS["shared_identifier_reference"], name="data_reference"),
    LIGATURE_DETECTORS["protection_critical"],
]

LEGAL_DOCUMENT_RULES = [
    _replace(LIGATURE_DETECTORS["shared_identifier_reference"],
             name="section_reference", weight=1.5),
    LIGATURE_DETECTORS["causal_chain"],
]

RAG_CONTEXT_RULES = [
    _replace(LIGATURE_DETECTORS["shared_identifier_reference"], name="document_reference"),
]

RULE_SETS: dict[str, list[LigatureRule]] = {
    "project_management": PROJECT_MANAGEMENT_RULES,
    "legal": LEGAL_DOCUMENT_RULES,
    "rag": RAG_CONTEXT_RULES,
}


# ---------------------------------------------------------------------------
# Public API — apply ligatures to a similarity graph
# ---------------------------------------------------------------------------

def apply_ligatures(
    adjacency: np.ndarray,
    chunks: Sequence[Chunk],
    rules: list[LigatureRule],
) -> tuple[np.ndarray, LigatureResult]:
    """
    Apply context-ligature rules to enhance a similarity graph.

    For each pair of chunks, each rule's detector is called. If it returns
    a nonzero weight, a ligature edge is added (or strengthened) in the
    adjacency matrix. For PROTECTION rules, the detector is called on
    individual chunks and increases all edges from that chunk.

    Parameters
    ----------
    adjacency : np.ndarray
        The original similarity adjacency matrix (n x n).
    chunks : Sequence[Chunk]
        The text chunks corresponding to graph nodes.
    rules : list[LigatureRule]
        Ligature rules to apply.

    Returns
    -------
    enhanced : np.ndarray
        The ligature-enhanced adjacency matrix.
    result : LigatureResult
        Metadata about what ligatures were applied.
    """
    from fiedler_optimizer.graph import compute_fiedler_vector

    n = len(chunks)
    enhanced = adjacency.copy()
    edges_added = 0
    rules_triggered: dict[str, int] = {}
    scale = _edge_scale(adjacency)   # graph-relative edge magnitude

    # Compute original λ₂ for comparison
    _, original_lambda2 = compute_fiedler_vector(adjacency)

    for rule in rules:
        triggered = 0

        if rule.ligature_type == LigatureType.PROTECTION:
            # Protection rules operate on individual chunks
            for i, chunk in enumerate(chunks):
                protection_weight = rule.detector(chunk)
                if protection_weight > 0:
                    # Boost all edges from this chunk
                    boost = protection_weight * rule.weight * scale
                    for j in range(n):
                        if i != j and enhanced[i, j] > 0:
                            enhanced[i, j] += boost
                            enhanced[j, i] += boost
                            edges_added += 1
                    triggered += 1
        else:
            # Pairwise rules operate on chunk pairs
            for i in range(n):
                for j in range(i + 1, n):
                    ligature_weight = rule.detector(chunks[i], chunks[j])
                    if ligature_weight > 0:
                        edge_boost = ligature_weight * rule.weight * scale
                        enhanced[i, j] += edge_boost
                        enhanced[j, i] += edge_boost
                        edges_added += 1
                        triggered += 1

        if triggered > 0:
            rules_triggered[rule.name] = triggered

    # Compute enhanced λ₂
    _, enhanced_lambda2 = compute_fiedler_vector(enhanced)

    return enhanced, LigatureResult(
        original_lambda2=original_lambda2,
        enhanced_lambda2=enhanced_lambda2,
        edges_added=edges_added,
        rules_triggered=rules_triggered,
    )
