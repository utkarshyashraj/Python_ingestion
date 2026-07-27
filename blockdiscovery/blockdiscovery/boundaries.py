"""Explicit boundary detection between consecutive raw PDF blocks.

Relationship evidence answers "how strongly do A and B belong together?".
Boundary evidence answers the complementary question: "how strongly should we
*stop* grouping here?". A high boundary score prevents over-grouping even when
blocks are spatially close and similarly formatted.

All signals are document-relative. No hardcoded section names, categories,
coordinates, or fixed font sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import EngineConfig
from .models import TextBlock
from .utils import clip01


def is_head(block: TextBlock) -> bool:
    """Locally prominent short line — relative to the document distribution."""
    f = block.features
    bold = f.get("bold_ratio", 0.0) > 0.5
    larger = f.get("size_ratio_median", 1.0) > 1.08
    elevated = f.get("size_zscore", 0.0) > 0.6
    short = f.get("is_short", 0.0) >= 1.0 or f.get("line_count", 1.0) <= 2.0
    return (bold or larger or elevated) and short


def is_title_like(block: TextBlock) -> bool:
    """Short phrase-like block that often starts a section or item row."""
    f = block.features
    text = (block.text or "").strip()
    if not text:
        return False
    short = f.get("is_short", 0.0) >= 1.0 and block.char_count <= 100
    few_lines = f.get("line_count", 1.0) <= 3.0
    not_sentence = not text.endswith((".", "?", ";", ","))
    return bool(short and few_lines and not_sentence)


def is_body_like(block: TextBlock) -> bool:
    """Longer descriptive block — typical continuation / description text."""
    f = block.features
    return (
        block.char_count >= 70
        or f.get("line_count", 1.0) >= 3.0
        or (f.get("is_short", 0.0) < 1.0 and f.get("prominence", 0.0) < 0.6)
    )


@dataclass
class BoundaryDecision:
    """Outcome of evaluating the boundary between two consecutive blocks."""

    signals: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    should_split: bool = False
    decision: str = "group"  # "group" | "split"
    reason_text: str = ""
    confidence: float = 0.0


class BoundaryDetector:
    """Compute explicit boundary evidence and a fused boundary score."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def evaluate(
        self,
        prev: TextBlock,
        cur: TextBlock,
        relationship_signals: Dict[str, float],
        relationship_score: float,
    ) -> BoundaryDecision:
        thr = self.config.thresholds
        signals: Dict[str, float] = {}
        reasons: List[str] = []

        # --- formatting / role transition ---------------------------------
        prom_jump = cur.features.get("prominence", 0.0) - prev.features.get("prominence", 0.0)
        formatting_transition = clip01(max(0.0, prom_jump) / 1.2)
        if is_head(cur) and prom_jump > 0.35:
            formatting_transition = max(formatting_transition, 0.85)
            reasons.append("formatting_transition")
        signals["formatting_transition"] = formatting_transition

        # --- spacing boundary ---------------------------------------------
        gap_ratio = cur.features.get("gap_above_ratio", 1.0)
        spacing_boundary = clip01((gap_ratio - 1.0) / 2.5)
        if gap_ratio >= 1.8:
            spacing_boundary = max(spacing_boundary, 0.75)
            reasons.append("spacing_boundary")
        signals["spacing_boundary"] = spacing_boundary

        # --- semantic topic transition ------------------------------------
        # Missing semantic evidence is treated as neutral (unknown), not as a
        # hard topic break — callers that omit embeddings must not over-split.
        has_semantic = "semantic_coherence" in relationship_signals
        semantic = relationship_signals.get("semantic_coherence", 0.5)
        if has_semantic:
            semantic_transition = clip01(1.0 - semantic)
        else:
            semantic_transition = 0.35
        # Amplify when spatial/formatting would otherwise dominate.
        spatial = relationship_signals.get("spatial_proximity", 0.0)
        formatting = relationship_signals.get(
            "formatting_relationship", relationship_signals.get("spatial_proximity", 0.0)
        )
        if has_semantic and semantic < thr.semantic_boundary_gate and spatial >= 0.75 and formatting >= 0.7:
            semantic_transition = max(semantic_transition, 0.82)
            reasons.append("semantic_transition")
        elif has_semantic and semantic < thr.semantic_boundary_gate * 0.7:
            semantic_transition = max(semantic_transition, 0.65)
            if "semantic_transition" not in reasons:
                reasons.append("semantic_transition")
        signals["semantic_transition"] = semantic_transition

        # --- structural role / repeated-item restart ----------------------
        structural_transition = 0.0
        title_here = is_title_like(cur)
        body_prev = is_body_like(prev)
        # Classic item restart: description ends, new short title begins —
        # even when vertical gap is tight (common in packed tables).
        if body_prev and title_here:
            structural_transition = max(structural_transition, 0.88)
            reasons.append("structural_transition")
        # Title-like after any whitespace bump.
        if title_here and gap_ratio >= 1.3:
            structural_transition = max(structural_transition, 0.72)
            if "structural_transition" not in reasons:
                reasons.append("title_like_after_whitespace")
        # Tight title-like restart with weak semantics (table/list rows).
        if (
            has_semantic
            and title_here
            and semantic < thr.semantic_boundary_gate
            and prev.char_count >= 40
        ):
            structural_transition = max(structural_transition, 0.78)
            if "structural_transition" not in reasons:
                reasons.append("structural_transition")
        # Title then bold multi-line column header.
        header_lines = [ln for ln in (cur.text or "").splitlines() if ln.strip()]
        looks_like_column_header = (
            cur.features.get("bold_ratio", 0.0) >= 0.4
            and (cur.text.count("/") >= 2 or len(header_lines) >= 3)
            and cur.char_count <= 200
        )
        if is_title_like(prev) and prev.char_count <= 90 and looks_like_column_header:
            structural_transition = max(structural_transition, 0.9)
            reasons.append("title_then_column_header")
        signals["structural_transition"] = clip01(structural_transition)

        # --- alignment pattern change -------------------------------------
        alignment = relationship_signals.get("alignment_consistency", 1.0)
        alignment_change = clip01(1.0 - alignment)
        # Weak alignment alone is not enough; combine with other weak signals.
        if alignment < 0.25 and (semantic < 0.25 or spacing_boundary > 0.4):
            alignment_change = max(alignment_change, 0.55)
            reasons.append("alignment_pattern_change")
        signals["alignment_pattern_change"] = alignment_change

        # --- reading-order discontinuity ----------------------------------
        order_gap = abs(cur.reading_order - prev.reading_order)
        same_col = prev.features.get("column_bucket") == cur.features.get("column_bucket")
        reading_discontinuity = 0.0
        if order_gap > 1:
            reading_discontinuity = clip01(0.4 + 0.2 * (order_gap - 1))
            reasons.append("reading_order_discontinuity")
        if not same_col and semantic < 0.3:
            reading_discontinuity = max(reading_discontinuity, 0.45)
        signals["reading_order_discontinuity"] = reading_discontinuity

        # --- density / typography jump ------------------------------------
        dens_a = prev.features.get("visual_density", 0.0)
        dens_b = cur.features.get("visual_density", 0.0)
        dens_ratio = abs(dens_a - dens_b) / max(dens_a, dens_b, 1e-6)
        density_transition = clip01(dens_ratio / 2.0)
        size_jump = abs(
            prev.features.get("size_ratio_median", 1.0) - cur.features.get("size_ratio_median", 1.0)
        )
        if size_jump > 0.35 and title_here:
            density_transition = max(density_transition, 0.55)
        signals["density_transition"] = density_transition

        # --- fused boundary score -----------------------------------------
        weights = {
            "formatting_transition": 0.18,
            "spacing_boundary": 0.16,
            "semantic_transition": 0.28,
            "structural_transition": 0.22,
            "alignment_pattern_change": 0.08,
            "reading_order_discontinuity": 0.05,
            "density_transition": 0.03,
        }
        score = sum(signals[k] * weights[k] for k in weights)
        score = clip01(score)

        # Hard veto paths (still evidence-driven, not document-specific).
        container = relationship_signals.get("visual_containment", 0.0)
        strong_container = container >= thr.container_override

        should_split = False
        if score >= thr.boundary_score_threshold:
            should_split = True
        if relationship_score < thr.content_unit_cohesion:
            should_split = True
            if "cohesion_below_threshold" not in reasons:
                reasons.append("cohesion_below_threshold")
        # Over-grouping gate: strong spatial + weak semantic without container.
        if (
            has_semantic
            and not strong_container
            and spatial >= 0.8
            and formatting >= 0.85
            and semantic < thr.semantic_boundary_gate
            and (structural_transition >= 0.5 or alignment < 0.35 or body_prev and title_here)
        ):
            should_split = True
            if "over_grouping_semantic_gate" not in reasons:
                reasons.append("over_grouping_semantic_gate")
        if strong_container and has_semantic and semantic < thr.semantic_boundary_gate:
            # Container explains co-location; do not force split on semantics alone.
            should_split = should_split and (
                formatting_transition >= 0.7 or structural_transition >= 0.85 or gap_ratio >= 2.2
            )

        decision = "split" if should_split else "group"
        if decision == "split":
            reason_text = (
                "Strong boundary evidence outweighs spatial proximity"
                if score >= thr.boundary_score_threshold and relationship_score >= thr.content_unit_cohesion
                else (
                    "Strong spatial relationship but weak semantic coherence and strong boundary evidence."
                    if semantic < thr.semantic_boundary_gate and spatial >= 0.75
                    else "Boundary signals exceeded grouping threshold."
                )
            )
        else:
            reason_text = "Relationship evidence outweighs boundary evidence."

        # Decision confidence: how sure we are about the split/group call.
        if decision == "split":
            conf = clip01(0.55 * score + 0.45 * (1.0 - relationship_score))
        else:
            conf = clip01(0.55 * relationship_score + 0.45 * (1.0 - score))

        # Deduplicate reasons while preserving order.
        seen = set()
        uniq_reasons = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                uniq_reasons.append(r)

        return BoundaryDecision(
            signals=signals,
            score=score,
            reasons=uniq_reasons,
            should_split=should_split,
            decision=decision,
            reason_text=reason_text,
            confidence=conf,
        )
