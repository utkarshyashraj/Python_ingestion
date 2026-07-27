"""Relationship discovery.

Computes transparent, per-signal evidence for whether two raw blocks *belong
together*. Signals are combined (never a single fixed rule) using configurable,
normalised weights. All signals are relative to document statistics, so the
engine adapts to each document's own layout and typography.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import EngineConfig
from .models import Evidence, TextBlock
from .semantics import cosine
from .utils import clip01


class RelationshipEvaluator:
    def __init__(self, config: EngineConfig, embeddings: Dict[str, np.ndarray]) -> None:
        self.config = config
        self.embeddings = embeddings

    # ----- individual signals ------------------------------------------- #
    def _spatial_proximity(self, a: TextBlock, b: TextBlock, stats: Dict[str, float]) -> float:
        # a precedes b in reading order.
        gap = b.bounding_box.y0 - a.bounding_box.y1
        gap = max(0.0, gap)
        decay = self.config.proximity_decay * max(1e-3, stats["gap_median"])
        return clip01(math.exp(-gap / decay))

    def _alignment_consistency(self, a: TextBlock, b: TextBlock, stats: Dict[str, float]) -> float:
        pw = max(1.0, stats["page_width"])
        tol = 0.08 * pw
        left = 1.0 - clip01(abs(a.bounding_box.x0 - b.bounding_box.x0) / tol)
        center = 1.0 - clip01(abs(a.bounding_box.cx - b.bounding_box.cx) / tol)
        return clip01(max(left, center))

    def _formatting_relationship(self, a: TextBlock, b: TextBlock, stats: Dict[str, float]) -> float:
        sa = a.features.get("font_size", stats["size_median"])
        sb = b.features.get("font_size", stats["size_median"])
        size_sim = 1.0 - clip01(abs(sa - sb) / (3.0 * stats["size_std"]))
        prom_a = a.features.get("prominence", 0.0)
        prom_b = b.features.get("prominence", 0.0)
        # If the *following* block is notably more prominent it likely starts a
        # new section -> weak "belong together" relationship.
        if prom_b > prom_a + 0.4:
            return clip01(0.35 * size_sim)
        # Title-then-body (a more prominent) or continuation (similar) -> strong.
        return clip01(0.55 + 0.45 * size_sim)

    def _spacing_pattern(self, a: TextBlock, b: TextBlock, stats: Dict[str, float]) -> float:
        gap = max(0.0, b.bounding_box.y0 - a.bounding_box.y1)
        # Consistency with the document's typical intra-content spacing.
        deviation = abs(gap - stats["gap_median"]) / max(1e-3, stats["gap_std"] + stats["gap_median"])
        return clip01(math.exp(-deviation))

    def _reading_order_coherence(self, a: TextBlock, b: TextBlock) -> float:
        if a.page_number != b.page_number:
            return 0.0
        consecutive = 1.0 if (b.reading_order - a.reading_order) == 1 else 0.5
        same_col = 1.0 if a.features.get("column_bucket") == b.features.get("column_bucket") else 0.4
        return clip01(0.5 * consecutive + 0.5 * same_col)

    def _semantic_coherence(self, a: TextBlock, b: TextBlock) -> float:
        va = self.embeddings.get(a.id)
        vb = self.embeddings.get(b.id)
        if va is None or vb is None:
            return 0.0
        raw = cosine(va, vb)
        # Hashing embeddings compress cosines; scale so topical matches approach 1.0.
        scale = max(1e-6, self.config.semantic_scale)
        return clip01(raw / scale)

    def _visual_containment(self, a: TextBlock, b: TextBlock) -> float:
        if a.bounding_box.contains(b.bounding_box) or b.bounding_box.contains(a.bounding_box):
            return 1.0
        return clip01(a.bounding_box.iou(b.bounding_box))

    # ----- fusion -------------------------------------------------------- #
    def evaluate(self, a: TextBlock, b: TextBlock, stats: Dict[str, float]) -> Evidence:
        signals = {
            "spatial_proximity": self._spatial_proximity(a, b, stats),
            "alignment_consistency": self._alignment_consistency(a, b, stats),
            "formatting_relationship": self._formatting_relationship(a, b, stats),
            "spacing_pattern": self._spacing_pattern(a, b, stats),
            "reading_order_coherence": self._reading_order_coherence(a, b),
            "semantic_coherence": self._semantic_coherence(a, b),
            "visual_containment": self._visual_containment(a, b),
        }
        weights = self.config.relationship_weights.normalized()
        confidence = sum(signals[k] * weights[k] for k in weights)
        notes = []
        if b.features.get("prominence", 0.0) > a.features.get("prominence", 0.0) + 0.4:
            notes.append("following block more prominent -> possible section boundary")
        if signals["spatial_proximity"] < 0.3:
            notes.append("large vertical whitespace between blocks")
        return Evidence(signals=signals, weights=weights, confidence=clip01(confidence), notes=notes)
