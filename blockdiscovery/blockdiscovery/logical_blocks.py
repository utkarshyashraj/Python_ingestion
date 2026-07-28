"""Logical block creation (the primary output) with merge/split refinement.

Each content unit becomes a :class:`LogicalBlock` carrying full traceability
and the explainable evidence that answers *"why were these blocks grouped
together?"*. A refinement pass may:

* SPLIT a block when it contains a deep internal cohesion / semantic valley
  (it was probably over-merged), or
* MERGE two adjacent blocks when their boundary cohesion is extremely high
  *and* semantic coherence is also strong (they were probably over-segmented).

Both operations emit dedicated structured events with their evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .boundaries import BoundaryDetector
from .config import EngineConfig
from .fingerprints import build_structural_fingerprint
from .logging_utils import DiscoveryLogger
from .models import ContentUnit, Document, Evidence, LogicalBlock, TextBlock
from .relationships import RelationshipEvaluator
from .utils import assign_roles, role_signature


def logical_block_from_unit(
    document: Document,
    index: int,
    unit: ContentUnit,
    pattern_id: Optional[str],
    block_index: Dict[str, TextBlock],
    total_blocks: int,
) -> LogicalBlock:
    """Promote a content unit to a LogicalBlock with full provenance."""
    head = block_index.get(unit.head_block_id or "")
    doc_pos = (head.reading_order / total_blocks) if head else (index / max(1, total_blocks))
    page_end = unit.page_end if unit.page_end is not None else unit.page_number
    return LogicalBlock(
        id=f"{document.id}_logical_block_{index:03d}",
        content_unit_id=unit.id,
        document_id=document.id,
        source_document=document.id,
        source_page=unit.page_number,
        page_end=page_end,
        source_block_ids=list(unit.block_ids),
        text=unit.text,
        structural_features=dict(unit.features),
        structural_fingerprint=dict(unit.structural_fingerprint),
        semantic_vector=unit.semantic_vector,
        discovered_pattern=pattern_id,
        role_sequence=list(unit.role_sequence),
        structural_signature=unit.structural_signature,
        confidence=unit.evidence.confidence,
        evidence=unit.evidence,
        doc_position=doc_pos,
    )


class LogicalBlockBuilder:
    def __init__(
        self,
        config: EngineConfig,
        evaluator: RelationshipEvaluator,
        logger: Optional[DiscoveryLogger] = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.logger = logger
        self.boundary_detector = BoundaryDetector(config)

    def build(
        self,
        document: Document,
        units: List[ContentUnit],
        unit_to_pattern: Dict[str, str],
    ) -> List[LogicalBlock]:
        if self.logger:
            self.logger.section("LOGICAL BLOCK")

        block_index = {b.id: b for b in document.blocks}
        total = max(1, len(document.blocks))

        logical_blocks: List[LogicalBlock] = []
        counter = 0
        for unit in units:
            counter += 1
            lb = self._from_unit(document, counter, unit, unit_to_pattern.get(unit.id), block_index, total)
            logical_blocks.append(lb)

        logical_blocks = self._split_pass(document, logical_blocks, block_index)
        logical_blocks = self._merge_pass(document, logical_blocks, block_index)
        return logical_blocks

    # ------------------------------------------------------------------ #
    def _from_unit(
        self,
        document: Document,
        index: int,
        unit: ContentUnit,
        pattern_id: Optional[str],
        block_index: Dict[str, TextBlock],
        total_blocks: int,
    ) -> LogicalBlock:
        lb = logical_block_from_unit(document, index, unit, pattern_id, block_index, total_blocks)
        self._log_created(lb)
        return lb

    def _log_created(self, lb: LogicalBlock) -> None:
        if not self.logger:
            return
        self.logger.event(
            "logical_block_created",
            document_id=lb.document_id,
            page_number=lb.source_page,
            logical_block_id=lb.id,
            content_unit_id=lb.content_unit_id,
            source_block_ids=lb.source_block_ids,
            pattern_id=lb.discovered_pattern,
            structural_fingerprint={
                k: round(v, 4) for k, v in lb.structural_fingerprint.items()
            },
            confidence=round(lb.confidence, 4),
            evidence=lb.evidence.to_dict(),
        )
        self.logger.push()
        self.logger.line(f"Created LogicalBlock: {lb.id}")
        self.logger.push()
        self.logger.line("Source:")
        self.logger.push()
        self.logger.kv("Document id", lb.document_id)
        self.logger.kv("Page", lb.source_page)
        self.logger.kv("Blocks", ", ".join(lb.source_block_ids))
        self.logger.pop()
        self.logger.kv("Discovered pattern", lb.discovered_pattern)
        self.logger.kv("Confidence", f"{lb.confidence:.2f}")
        self.logger.pop()
        self.logger.pop()

    # ------------------------------------------------------------------ #
    def _pair_scores(
        self, lb: LogicalBlock, block_index: Dict[str, TextBlock], stats: Dict[str, float]
    ) -> List[Dict[str, float]]:
        blocks = [block_index[bid] for bid in lb.source_block_ids if bid in block_index]
        rows = []
        for i in range(1, len(blocks)):
            ev = self.evaluator.evaluate(blocks[i - 1], blocks[i], stats)
            boundary = self.boundary_detector.evaluate(
                blocks[i - 1], blocks[i], ev.signals, ev.confidence
            )
            rows.append(
                {
                    "cohesion": ev.confidence,
                    "semantic": ev.signals.get("semantic_coherence", 0.0),
                    "boundary": boundary.score,
                    "should_split": 1.0 if boundary.should_split else 0.0,
                }
            )
        return rows

    def _split_pass(
        self, document: Document, blocks: List[LogicalBlock], block_index: Dict[str, TextBlock]
    ) -> List[LogicalBlock]:
        thr = self.config.thresholds
        stats = document.stats
        out: List[LogicalBlock] = []
        suffix = 0
        for lb in blocks:
            if len(lb.source_block_ids) < 3:
                out.append(lb)
                continue
            rows = self._pair_scores(lb, block_index, stats)
            if not rows:
                out.append(lb)
                continue

            cohesions = [r["cohesion"] for r in rows]
            semantics = [r["semantic"] for r in rows]
            boundaries = [r["boundary"] for r in rows]
            mean_c = float(np.mean(cohesions))
            mean_s = float(np.mean(semantics))

            # Prefer the weakest semantic / strongest boundary valley.
            valley = int(np.argmin([c - 0.6 * b + 0.4 * s for c, b, s in zip(cohesions, boundaries, semantics)]))
            valley_val = cohesions[valley]
            valley_sem = semantics[valley]
            valley_bound = boundaries[valley]
            force_split = rows[valley]["should_split"] >= 1.0
            deep_valley = valley_val < mean_c - thr.split_valley_delta
            semantic_valley = (
                valley_sem < thr.semantic_boundary_gate
                and mean_s < thr.semantic_confidence_floor
                and valley_bound >= thr.boundary_score_threshold * 0.85
            )

            if force_split or deep_valley or semantic_valley:
                left_ids = lb.source_block_ids[: valley + 1]
                right_ids = lb.source_block_ids[valley + 1 :]
                if not left_ids or not right_ids:
                    out.append(lb)
                    continue
                suffix += 1
                left = self._respawn(document, lb, left_ids, block_index, f"s{suffix}a")
                right = self._respawn(document, lb, right_ids, block_index, f"s{suffix}b")
                reason = (
                    "explicit_boundary_in_unit"
                    if force_split
                    else (
                        "semantic_coherence_valley"
                        if semantic_valley
                        else "internal_cohesion_valley"
                    )
                )
                if self.logger:
                    self.logger.event(
                        "logical_block_split",
                        document_id=document.id,
                        logical_block_id=lb.id,
                        into=[left.id, right.id],
                        valley_cohesion=round(valley_val, 4),
                        valley_semantic=round(valley_sem, 4),
                        valley_boundary=round(valley_bound, 4),
                        mean_cohesion=round(mean_c, 4),
                        reason=reason,
                        confidence=round(max(valley_bound, 1.0 - valley_val), 4),
                    )
                    self.logger.event(
                        "content_unit_split",
                        document_id=document.id,
                        content_unit_id=lb.content_unit_id,
                        into_block_ids=[left.id, right.id],
                        reason=reason,
                        confidence=round(max(valley_bound, 1.0 - valley_val), 4),
                    )
                out.extend([left, right])
            else:
                out.append(lb)
        return out

    def _merge_pass(
        self,
        document: Document,
        blocks: List[LogicalBlock],
        block_index: Dict[str, TextBlock],
    ) -> List[LogicalBlock]:
        thr = self.config.thresholds
        stats = document.stats
        out: List[LogicalBlock] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            if i + 1 < len(blocks):
                nxt = blocks[i + 1]
                if cur.source_page == nxt.source_page and cur.source_block_ids and nxt.source_block_ids:
                    a = block_index.get(cur.source_block_ids[-1])
                    b = block_index.get(nxt.source_block_ids[0])
                    if a and b:
                        ev = self.evaluator.evaluate(a, b, stats)
                        boundary = self.boundary_detector.evaluate(
                            a, b, ev.signals, ev.confidence
                        )
                        semantic = ev.signals.get("semantic_coherence")
                        semantic_ok = (
                            semantic is None or semantic >= thr.semantic_confidence_floor
                        )
                        # Merge only when relationship is very strong AND boundary
                        # is weak AND semantics support continuity (when available).
                        if (
                            ev.confidence >= thr.merge_cohesion
                            and not boundary.should_split
                            and boundary.score < thr.boundary_score_threshold * 0.7
                            and semantic_ok
                        ):
                            merged = self._merge(document, cur, nxt, ev, block_index)
                            if self.logger:
                                self.logger.event(
                                    "logical_block_merged",
                                    document_id=document.id,
                                    merged=[cur.id, nxt.id],
                                    into=merged.id,
                                    boundary_cohesion=round(ev.confidence, 4),
                                    boundary_score=round(boundary.score, 4),
                                    confidence=round(ev.confidence, 4),
                                    evidence=ev.to_dict(),
                                )
                            out.append(merged)
                            i += 2
                            continue
            out.append(cur)
            i += 1
        return out

    def _respawn(
        self,
        document: Document,
        parent: LogicalBlock,
        block_ids: List[str],
        block_index: Dict[str, TextBlock],
        tag: str,
    ) -> LogicalBlock:
        blocks = [block_index[bid] for bid in block_ids if bid in block_index]
        text = "\n".join(b.text for b in blocks)
        roles = assign_roles(blocks)
        signature = role_signature(roles)
        fingerprint = build_structural_fingerprint(blocks, roles)

        # Recompute confidence from internal relationships of the new side.
        stats = document.stats
        cohesions = []
        semantics = []
        for i in range(1, len(blocks)):
            ev = self.evaluator.evaluate(blocks[i - 1], blocks[i], stats)
            cohesions.append(ev.confidence)
            semantics.append(ev.signals.get("semantic_coherence", 0.0))
        if cohesions:
            conf = float(np.mean(cohesions))
            mean_sem = float(np.mean(semantics))
            if mean_sem < self.config.thresholds.semantic_confidence_floor:
                conf = max(0.15, conf * 0.7)
        else:
            conf = self.config.thresholds.standalone_unit_confidence

        evidence = Evidence(
            signals={
                "internal_cohesion": conf,
                "mean_semantic_coherence": float(np.mean(semantics)) if semantics else conf,
            },
            weights={"internal_cohesion": 0.6, "mean_semantic_coherence": 0.4},
            confidence=conf,
            notes=[f"split from {parent.id}"],
        )
        return LogicalBlock(
            id=f"{parent.id}_{tag}",
            content_unit_id=parent.content_unit_id,
            document_id=parent.document_id,
            source_document=parent.source_document,
            source_page=blocks[0].page_number if blocks else parent.source_page,
            page_end=blocks[-1].page_number if blocks else parent.source_page,
            source_block_ids=block_ids,
            text=text,
            structural_features={
                **parent.structural_features,
                "block_count": float(len(blocks)),
                "mean_semantic_coherence": evidence.signals.get("mean_semantic_coherence", 0.0),
            },
            structural_fingerprint=fingerprint,
            semantic_vector=parent.semantic_vector,
            discovered_pattern=parent.discovered_pattern,
            role_sequence=roles,
            structural_signature=signature,
            confidence=conf,
            evidence=evidence,
            doc_position=parent.doc_position,
        )

    def _merge(
        self,
        document: Document,
        a: LogicalBlock,
        b: LogicalBlock,
        ev: Evidence,
        block_index: Dict[str, TextBlock],
    ) -> LogicalBlock:
        block_ids = a.source_block_ids + b.source_block_ids
        blocks = [block_index[bid] for bid in block_ids if bid in block_index]
        roles = assign_roles(blocks)
        fingerprint = build_structural_fingerprint(blocks, roles, mean_signals=ev.signals)
        return LogicalBlock(
            id=f"{a.id}_merged",
            content_unit_id=a.content_unit_id,
            document_id=a.document_id,
            source_document=a.source_document,
            source_page=a.source_page,
            page_end=b.page_end or b.source_page,
            source_block_ids=block_ids,
            text=a.text + "\n" + b.text,
            structural_features={
                **a.structural_features,
                "block_count": float(len(block_ids)),
            },
            structural_fingerprint=fingerprint,
            semantic_vector=a.semantic_vector,
            discovered_pattern=a.discovered_pattern,
            role_sequence=roles,
            structural_signature=role_signature(roles),
            confidence=min(a.confidence, b.confidence, ev.confidence),
            evidence=ev,
            doc_position=a.doc_position,
        )
