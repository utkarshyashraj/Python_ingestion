"""Central, fully-configurable settings for the discovery engine.

Every weight and threshold that influences an algorithmic decision lives here
so that confidence calculations remain *transparent and configurable*. Nothing
in this file references business categories -- only generic structural and
semantic signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


@dataclass
class RelationshipWeights:
    """Weights for fusing pairwise (block-to-block) cohesion signals.

    Semantic coherence is first-class: spatial/formatting alone must not dominate
    when topical relatedness is weak (the classic over-grouping failure mode).
    """

    spatial_proximity: float = 0.22
    alignment_consistency: float = 0.14
    formatting_relationship: float = 0.16
    spacing_pattern: float = 0.10
    reading_order_coherence: float = 0.10
    semantic_coherence: float = 0.24
    visual_containment: float = 0.04

    def normalized(self) -> Dict[str, float]:
        return _normalize(asdict(self))


@dataclass
class GroupSimilarityWeights:
    """Weights for fusing cross-document logical-block similarity signals.

    Cross-document grouping is fundamentally about *topic*, so semantic evidence
    dominates; structural and contextual evidence act as secondary tie-breakers
    (documents with the same layout should not be grouped on layout alone).
    """

    semantic_similarity: float = 0.70
    structural_similarity: float = 0.18
    contextual_similarity: float = 0.12

    def normalized(self) -> Dict[str, float]:
        return _normalize(asdict(self))


@dataclass
class Thresholds:
    # A new content unit starts when consecutive-block cohesion drops below this.
    content_unit_cohesion: float = 0.45
    # Fused boundary score at/above which consecutive blocks are split.
    boundary_score_threshold: float = 0.48
    # Pairwise semantic coherence below this is treated as a topic transition
    # (especially when spatial/formatting would otherwise keep blocks together).
    semantic_boundary_gate: float = 0.18
    # Visual containment above this can explain co-location despite weak semantics
    # (e.g. cells inside a clearly detected container / table region).
    container_override: float = 0.55
    # When mean unit semantic coherence is below this, grouping confidence is
    # penalised unless strong container evidence is present.
    semantic_confidence_floor: float = 0.22
    # Similarity threshold for grouping content units into a discovered pattern.
    pattern_similarity: float = 0.62
    # Similarity threshold for grouping logical blocks across documents.
    cross_document_similarity: float = 0.6
    # Minimum semantic similarity for two blocks to be *eligible* to group at
    # all -- prevents same-layout blocks from grouping on structure alone.
    cross_document_semantic_gate: float = 0.16
    # Confidence below which any decision is additionally logged as low_confidence.
    low_confidence_flag: float = 0.5
    # A logical block is split when an internal cohesion valley is this far
    # below the block's mean internal cohesion.
    split_valley_delta: float = 0.22
    # Two adjacent units are merged when their boundary cohesion exceeds this.
    # Kept high so merge never undoes semantic boundary splits.
    merge_cohesion: float = 0.92
    # Confidence assigned to a single-block unit: it has no internal
    # relationships to fuse, so it is inherently less certain than a
    # multi-block unit whose grouping was justified by evidence.
    standalone_unit_confidence: float = 0.45


@dataclass
class EngineConfig:
    """Top-level configuration object passed through the whole pipeline."""

    relationship_weights: RelationshipWeights = field(default_factory=RelationshipWeights)
    group_weights: GroupSimilarityWeights = field(default_factory=GroupSimilarityWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)

    # Semantic embedding backend: "hashing" (dependency-free default) or
    # "sentence-transformers" (used only if the package is installed).
    embedding_backend: str = "hashing"
    embedding_dim: int = 512
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # Spatial proximity is scored relative to the document's typical line gap.
    # This multiplier controls how quickly the proximity score decays with gap.
    proximity_decay: float = 2.5

    # Raw cosine from the dependency-free backend is naturally compressed into a
    # small range; this reference value maps a "strong topical match" to ~1.0 so
    # semantic evidence has real dynamic range in the fused score. Set to 1.0
    # when using a transformer backend (which already produces well-spread
    # cosines).
    semantic_scale: float = 0.33

    # Optional, user-supplied lexicon for AFTER-THE-FACT naming only.
    # Default is empty: discovery never depends on business words such as
    # Feature / Fix / Bug. If you later want human names, pass your own seeds —
    # the engine will still discover structure without them.
    optional_label_lexicon: Dict[str, list] = field(default_factory=dict)

    # When set, only the first N pages of each PDF are extracted.
    max_pages: Optional[int] = None

    # Ingestion backend:
    #   "native"     — original PyMuPDF block clustering pipeline
    #   "structured" — pymupdf4llm markdown + table-row units (recommended)
    #   "docling"    — IBM Docling when installed; falls back to structured
    ingestion_backend: str = "native"

    # Logging
    readable_log: bool = True
    verbose_relationships: bool = False  # log every candidate pair (noisy)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["relationship_weights_normalized"] = self.relationship_weights.normalized()
        d["group_weights_normalized"] = self.group_weights.normalized()
        return d



DEFAULT_CONFIG = EngineConfig()
