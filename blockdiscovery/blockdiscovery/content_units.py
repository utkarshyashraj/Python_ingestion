"""Content unit discovery.

A raw PDF block is not necessarily a complete logical piece of information.
This module discovers coherent *content units* by evaluating both:

* ``relationship_score(A, B)`` — evidence that A and B belong together
* ``boundary_score(A, B)`` — evidence that a logical boundary sits between them

Spatial proximity alone is insufficient. Strong boundary evidence (semantic
topic transition, structural role change, formatting jump, spacing, etc.)
prevents over-grouping even when blocks are close and similarly formatted.

Every unit carries explainable evidence, a rich structural fingerprint, and
full source-block traceability.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .boundaries import BoundaryDetector
from .config import EngineConfig
from .fingerprints import build_structural_fingerprint
from .logging_utils import DiscoveryLogger
from .models import BoundingBox, ContentUnit, Document, Evidence, TextBlock
from .relationships import RelationshipEvaluator
from .semantics import EmbeddingBackend
from .utils import assign_roles, role_signature


def _union_bbox(blocks: List[TextBlock]) -> BoundingBox:
    x0 = min(b.bounding_box.x0 for b in blocks)
    y0 = min(b.bounding_box.y0 for b in blocks)
    x1 = max(b.bounding_box.x1 for b in blocks)
    y1 = max(b.bounding_box.y1 for b in blocks)
    return BoundingBox(x0, y0, x1, y1)


def _fmt_signal_block(log: DiscoveryLogger, title: str, signals: Dict[str, float]) -> None:
    log.line(f"{title}")
    log.push()
    width = max((len(k) for k in signals), default=10)
    for k, v in signals.items():
        log.line(f"{k:<{width}}  : {v:.2f}")
    log.pop()


class ContentUnitDiscovery:
    def __init__(
        self,
        config: EngineConfig,
        evaluator: RelationshipEvaluator,
        backend: EmbeddingBackend,
        logger: Optional[DiscoveryLogger] = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.backend = backend
        self.logger = logger
        self.boundary_detector = BoundaryDetector(config)
        # Collected for validation / human-readable discovery logs.
        self.transitions: List[Dict] = []
        self.stats: Dict[str, int] = {
            "boundaries_evaluated": 0,
            "boundaries_detected": 0,
            "blocks_merged": 0,
            "low_confidence_decisions": 0,
            "over_grouping_warnings": 0,
        }

    def discover(self, document: Document) -> List[ContentUnit]:
        log = self.logger
        stats = document.stats
        thr = self.config.thresholds
        if log:
            log.section("CONTENT UNIT DISCOVERY")
            log.line("Evaluating block relationships and boundaries...")

        text_blocks = [b for b in document.blocks if b.block_type == "text"]
        by_page: Dict[int, List[TextBlock]] = {}
        for b in text_blocks:
            by_page.setdefault(b.page_number, []).append(b)

        units: List[ContentUnit] = []
        unit_counter = 0

        for page_number in sorted(by_page):
            blocks = sorted(by_page[page_number], key=lambda b: b.reading_order)
            if not blocks:
                continue

            current: List[TextBlock] = [blocks[0]]
            current_cohesions: List[float] = []
            current_signal_rows: List[Dict[str, float]] = []
            current_boundary_scores: List[float] = []

            for i in range(1, len(blocks)):
                prev = blocks[i - 1]
                cur = blocks[i]
                evidence = self.evaluator.evaluate(prev, cur, stats)
                relationship_score = evidence.confidence
                boundary = self.boundary_detector.evaluate(
                    prev, cur, evidence.signals, relationship_score
                )

                spatial = evidence.signals.get("spatial_proximity", 0.0)
                formatting = evidence.signals.get("formatting_relationship", 0.0)
                semantic = evidence.signals.get("semantic_coherence", 0.0)
                container = evidence.signals.get("visual_containment", 0.0)
                over_group = (
                    spatial >= thr.over_grouping_spatial
                    and formatting >= thr.over_grouping_formatting
                    and semantic < thr.semantic_boundary_gate
                    and container < thr.container_override
                )

                # Soft size cap: growing multi-block units with weak semantics
                # must not keep absorbing neighbours.
                mean_sem_so_far = (
                    float(np.mean([r.get("semantic_coherence", 0.0) for r in current_signal_rows]))
                    if current_signal_rows
                    else semantic
                )
                size_cap_hit = (
                    len(current) >= thr.max_content_unit_blocks
                    and mean_sem_so_far < thr.semantic_confidence_floor
                    and container < thr.container_override
                )
                if size_cap_hit and not boundary.should_split:
                    boundary.should_split = True
                    boundary.decision = "split"
                    boundary.reasons = list(boundary.reasons) + ["max_unit_size_weak_semantics"]
                    boundary.reason_text = (
                        "Unit size cap reached while semantic coherence remains weak."
                    )
                    if "max_unit_size_weak_semantics" not in boundary.reasons:
                        pass

                self.stats["boundaries_evaluated"] += 1
                if boundary.should_split:
                    self.stats["boundaries_detected"] += 1
                else:
                    self.stats["blocks_merged"] += 1
                if boundary.confidence < thr.low_confidence_flag:
                    self.stats["low_confidence_decisions"] += 1

                transition = {
                    "document_id": document.id,
                    "page_number": page_number,
                    "block_a": prev.id,
                    "block_b": cur.id,
                    "decision": boundary.decision,
                    "relationship_score": round(relationship_score, 4),
                    "boundary_score": round(boundary.score, 4),
                    "relationship_evidence": {
                        k: round(v, 3) for k, v in evidence.signals.items()
                    },
                    "boundary_evidence": {
                        k: round(v, 3) for k, v in boundary.signals.items()
                    },
                    "confidence": round(boundary.confidence, 4),
                    "reason": boundary.reason_text,
                    "reasons": list(boundary.reasons),
                    "over_grouping": over_group,
                }
                self.transitions.append(transition)

                if over_group:
                    self.stats["over_grouping_warnings"] += 1
                if log:
                    log.event(
                        "candidate_relationship_evaluated",
                        document_id=document.id,
                        page_number=page_number,
                        block_a=prev.id,
                        block_b=cur.id,
                        cohesion=round(relationship_score, 4),
                        relationship_score=round(relationship_score, 4),
                        boundary=boundary.should_split,
                        signals={k: round(v, 3) for k, v in evidence.signals.items()},
                    )
                    log.event(
                        "boundary_evaluated",
                        document_id=document.id,
                        page_number=page_number,
                        block_a=prev.id,
                        block_b=cur.id,
                        boundary_score=round(boundary.score, 4),
                        decision=boundary.decision,
                        reasons=boundary.reasons,
                        signals={k: round(v, 3) for k, v in boundary.signals.items()},
                        confidence=round(boundary.confidence, 4),
                    )
                    if over_group:
                        log.event(
                            "over_grouping_warning",
                            document_id=document.id,
                            page_number=page_number,
                            block_a=prev.id,
                            block_b=cur.id,
                            spatial_proximity=round(spatial, 4),
                            formatting_relationship=round(formatting, 4),
                            semantic_coherence=round(semantic, 4),
                            decision="SPLIT_OR_REVIEW" if boundary.should_split else "REVIEW",
                            reason=(
                                "Spatial and formatting similarity are high, "
                                "but semantic coherence is weak."
                            ),
                            confidence=round(boundary.confidence, 4),
                        )
                    # Always narrate splits + over-grouping; groups when verbose.
                    if (
                        boundary.should_split
                        or over_group
                        or self.config.verbose_relationships
                    ):
                        self._log_evaluation(
                            document_id=document.id,
                            page_number=page_number,
                            prev=prev,
                            cur=cur,
                            relationship_score=relationship_score,
                            relationship_signals=evidence.signals,
                            boundary=boundary,
                            over_grouping=over_group,
                        )

                if not boundary.should_split:
                    current.append(cur)
                    current_cohesions.append(relationship_score)
                    current_signal_rows.append(evidence.signals)
                    current_boundary_scores.append(boundary.score)
                else:
                    if log:
                        log.event(
                            "content_unit_rejected",
                            document_id=document.id,
                            page_number=page_number,
                            block_a=prev.id,
                            block_b=cur.id,
                            reason="+".join(boundary.reasons) or boundary.reason_text,
                            cohesion=round(relationship_score, 4),
                            relationship_score=round(relationship_score, 4),
                            boundary_score=round(boundary.score, 4),
                            threshold=self.config.thresholds.content_unit_cohesion,
                            relationship_evidence={
                                k: round(v, 3) for k, v in evidence.signals.items()
                            },
                            boundary_evidence={
                                k: round(v, 3) for k, v in boundary.signals.items()
                            },
                            decision="split",
                            confidence=round(boundary.confidence, 4),
                            notes=evidence.notes,
                        )
                    unit_counter += 1
                    units.append(
                        self._finalize_unit(
                            document,
                            unit_counter,
                            current,
                            current_cohesions,
                            current_signal_rows,
                            current_boundary_scores,
                        )
                    )
                    current = [cur]
                    current_cohesions = []
                    current_signal_rows = []
                    current_boundary_scores = []

            unit_counter += 1
            units.append(
                self._finalize_unit(
                    document,
                    unit_counter,
                    current,
                    current_cohesions,
                    current_signal_rows,
                    current_boundary_scores,
                )
            )

        self._embed_units(document, units)
        return units

    def _log_evaluation(
        self,
        document_id: str,
        page_number: int,
        prev: TextBlock,
        cur: TextBlock,
        relationship_score: float,
        relationship_signals: Dict[str, float],
        boundary,
        over_grouping: bool = False,
    ) -> None:
        log = self.logger
        if not log:
            return
        if over_grouping:
            log.section("OVER-GROUPING WARNING")
            log.line("Candidate:")
            log.push()
            log.line(f"{prev.id} → {cur.id}")
            log.kv("Page", page_number)
            log.pop()
            log.line("")
            _fmt_signal_block(log, "Evidence:", relationship_signals)
            log.line("")
            log.line("Decision:")
            log.push()
            log.line("SPLIT_OR_REVIEW" if boundary.should_split else "REVIEW")
            log.kv(
                "Reason",
                "Spatial and formatting similarity are high, but semantic coherence is weak.",
            )
            log.kv("Confidence", f"{boundary.confidence:.2f}")
            log.pop()
        else:
            log.section("BOUNDARY EVALUATION")
            log.line("Previous Block:")
            log.push()
            log.line(prev.id)
            log.pop()
            log.line("Next Block:")
            log.push()
            log.line(cur.id)
            log.pop()
            log.line("")
            _fmt_signal_block(log, "Relationship Evidence:", relationship_signals)
            log.line("")
            _fmt_signal_block(log, "Boundary Evidence:", boundary.signals)
            log.line("")
            log.line("Decision:")
            log.push()
            decision_label = (
                "START_NEW_LOGICAL_BLOCK"
                if boundary.should_split
                else "GROUP_INTO_CONTENT_UNIT"
            )
            log.line(decision_label)
            if boundary.reasons:
                log.kv("Reasons", ", ".join(boundary.reasons))
            log.kv("Reason", boundary.reason_text)
            log.kv("Relationship score", f"{relationship_score:.2f}")
            log.kv("Boundary score", f"{boundary.score:.2f}")
            log.kv("Confidence", f"{boundary.confidence:.2f}")
            log.pop()

        log.event(
            "content_unit_decision",
            document_id=document_id,
            page_number=page_number,
            candidate_block_ids=[prev.id, cur.id],
            decision=boundary.decision,
            relationship_score=round(relationship_score, 4),
            relationship_evidence={k: round(v, 3) for k, v in relationship_signals.items()},
            boundary_evidence={k: round(v, 3) for k, v in boundary.signals.items()},
            boundary_score=round(boundary.score, 4),
            confidence=round(boundary.confidence, 4),
            reason=boundary.reason_text,
            over_grouping=over_grouping,
        )
        # Reset indent so subsequent ContentUnit creation logs are not nested
        # under this evaluation block.
        log._indent = 0

    def _finalize_unit(
        self,
        document: Document,
        index: int,
        blocks: List[TextBlock],
        cohesions: List[float],
        signal_rows: Optional[List[Dict[str, float]]] = None,
        boundary_scores: Optional[List[float]] = None,
    ) -> ContentUnit:
        head = max(blocks, key=lambda b: b.features.get("prominence", 0.0))
        roles = assign_roles(blocks, head.id)
        signature = role_signature(roles)
        text = "\n".join(b.text for b in blocks).strip()
        thr = self.config.thresholds

        if cohesions:
            internal = float(np.mean(cohesions))
            min_internal = float(np.min(cohesions))
        else:
            internal = thr.standalone_unit_confidence
            min_internal = internal

        signal_rows = signal_rows or []
        if signal_rows:
            keys = signal_rows[0].keys()
            avg_signals = {k: float(np.mean([row[k] for row in signal_rows])) for k in keys}
        else:
            avg_signals = {
                "spatial_proximity": internal,
                "reading_order_coherence": internal,
            }

        mean_semantic = float(avg_signals.get("semantic_coherence", internal))
        mean_containment = float(avg_signals.get("visual_containment", 0.0))

        # Explainable confidence: relationship cohesion, penalised when semantic
        # coherence is weak unless a container explains the grouping.
        confidence = internal
        notes: List[str] = []
        if len(blocks) == 1:
            notes.append("single-block unit (no internal relationships to fuse)")
        if len(blocks) > 1 and mean_semantic < thr.semantic_confidence_floor:
            if mean_containment >= thr.container_override:
                notes.append(
                    "low semantic coherence tolerated due to strong visual containment"
                )
            else:
                # Soft penalty — keep evidence visible, reduce over-confident claims.
                penalty = (thr.semantic_confidence_floor - mean_semantic) / max(
                    thr.semantic_confidence_floor, 1e-6
                )
                confidence = max(0.15, internal * (1.0 - 0.55 * penalty))
                notes.append(
                    "confidence reduced: weak semantic coherence relative to spatial/formatting"
                )
        if cohesions and min_internal < thr.content_unit_cohesion:
            notes.append("contains weak internal boundary (min cohesion below threshold)")
            confidence = min(confidence, 0.5 * confidence + 0.5 * min_internal)

        weights = self.config.relationship_weights.normalized()
        weights = {k: weights[k] for k in avg_signals if k in weights} or {
            k: 1.0 / len(avg_signals) for k in avg_signals
        }
        # Surface semantic weight explicitly in evidence.
        evidence = Evidence(
            signals=avg_signals,
            weights=weights,
            confidence=float(confidence),
            notes=notes,
        )

        fingerprint = build_structural_fingerprint(
            blocks, roles, boundary_scores=boundary_scores, mean_signals=avg_signals
        )

        prom_values = [b.features.get("prominence", 0.0) for b in blocks]
        unit_features = {
            "block_count": float(len(blocks)),
            "char_count": float(sum(b.char_count for b in blocks)),
            "line_count": float(sum(b.line_count for b in blocks)),
            "head_prominence": head.features.get("prominence", 0.0),
            "mean_prominence": float(np.mean(prom_values)) if prom_values else 0.0,
            "head_size_ratio": head.features.get("size_ratio_median", 1.0),
            "role_prominent": float(roles.count("PROMINENT")),
            "role_body": float(roles.count("BODY")),
            "role_meta": float(roles.count("META")),
            "page_fraction": head.features.get("page_fraction", 0.0),
            "internal_cohesion": internal,
            "min_internal_cohesion": min_internal,
            "mean_semantic_coherence": mean_semantic,
            "alignment_signature": fingerprint.get("alignment_signature", 0.0),
            "spacing_signature": fingerprint.get("spacing_signature", 0.0),
            "typography_signature": fingerprint.get("typography_signature", 0.0),
            "content_density": fingerprint.get("content_density", 0.0),
        }

        unit = ContentUnit(
            id=f"{document.id}_unit_{index:04d}",
            document_id=document.id,
            page_number=blocks[0].page_number,
            page_end=blocks[-1].page_number,
            block_ids=[b.id for b in blocks],
            text=text,
            head_block_id=head.id,
            role_sequence=roles,
            structural_signature=signature,
            structural_fingerprint=fingerprint,
            bounding_box=_union_bbox(blocks),
            features=unit_features,
            evidence=evidence,
        )

        if self.logger:
            self.logger.event(
                "content_unit_created",
                document_id=document.id,
                page_number=unit.page_number,
                content_unit_id=unit.id,
                source_block_ids=unit.block_ids,
                head_block_id=head.id,
                role_sequence=roles,
                structural_signature=signature,
                structural_fingerprint={k: round(v, 4) for k, v in fingerprint.items()},
                confidence=round(confidence, 4),
                evidence=evidence.to_dict(),
            )
            self.logger.event(
                "structural_fingerprint_created",
                document_id=document.id,
                content_unit_id=unit.id,
                fingerprint={k: round(v, 4) for k, v in fingerprint.items()},
                role_sequence=roles,
                confidence=round(confidence, 4),
            )
            self.logger.push()
            self.logger.line("Decision:")
            self.logger.push()
            self.logger.line(f"ContentUnit: {unit.id}")
            self.logger.kv("Page", unit.page_number)
            self.logger.kv("Blocks", ", ".join(unit.block_ids))
            self.logger.kv("Head block", head.id)
            self.logger.kv("Role sequence", ", ".join(roles))
            self.logger.kv("Structural signature", signature)
            self.logger.kv(
                "Fingerprint",
                (
                    f"align={fingerprint.get('alignment_signature', 0):.2f} "
                    f"space={fingerprint.get('spacing_signature', 0):.2f} "
                    f"type={fingerprint.get('typography_signature', 0):.2f} "
                    f"density={fingerprint.get('content_density', 0):.4f}"
                ),
            )
            _fmt_signal_block(self.logger, "Evidence:", avg_signals)
            if notes:
                self.logger.kv("Notes", "; ".join(notes))
            self.logger.kv("Confidence", f"{confidence:.2f}")
            self.logger.pop()
            self.logger.pop()
        return unit

    def _embed_units(self, document: Document, units: List[ContentUnit]) -> None:
        if not units:
            return
        vectors = self.backend.embed([u.text for u in units])
        for u, v in zip(units, vectors):
            u.semantic_vector = v.astype(float).tolist()
            if self.logger:
                self.logger.event(
                    "semantic_representation_created",
                    document_id=document.id,
                    content_unit_id=u.id,
                    dim=int(v.shape[0]),
                    backend=self.config.embedding_backend,
                )
