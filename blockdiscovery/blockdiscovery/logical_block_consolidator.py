"""Post-discovery logical block consolidation.

Runs *after* logical block discovery and *before* final section discovery:

    Logical Block Discovery
            ↓
    Continuation Detection
            ↓
    Logical Block Consolidation
            ↓
    Section Discovery

Adjacent blocks are merged only when continuation evidence outweighs new-
boundary evidence. Decisions are generic: they use punctuation, capitalization,
geometry, roles and typography — never document vocabulary or category names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import DiscoveryLogger
from .models import BoundingBox, Document, Evidence, LogicalBlock, TextBlock
from .utils import clip01, role_signature

# Closed-class grammatical particles used only as *weak* lexical evidence of
# sentence continuation. They are never sufficient alone to force a merge, and
# they are language-structure signals — not document categories.
_WEAK_CONTINUATION_STARTERS = frozenset(
    {
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "for",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "while",
        "when",
        "where",
        "because",
        "although",
        "though",
        "unless",
        "until",
        "whereas",
        "whether",
        "if",
        "as",
        "than",
        "then",
        "also",
        "however",
        "therefore",
        "thus",
        "hence",
        "moreover",
        "furthermore",
        "nevertheless",
        "nonetheless",
        "regardless",
        "including",
        "especially",
        "particularly",
        "such",
        "these",
        "those",
        "this",
        "their",
        "its",
        "his",
        "her",
        "our",
        "your",
        "with",
        "without",
        "from",
        "into",
        "onto",
        "upon",
        "about",
        "across",
        "after",
        "before",
        "during",
        "through",
        "under",
        "over",
        "between",
        "among",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
    }
)

_WEAK_INCOMPLETE_ENDINGS = frozenset(
    {
        "and",
        "or",
        "but",
        "nor",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "as",
        "that",
        "which",
        "who",
        "when",
        "where",
        "while",
        "because",
        "although",
        "if",
        "unless",
        "until",
        "including",
        "such",
        "than",
        "then",
        "also",
        "available",
        "used",
        "able",
        "conditions",
        "systems",
        "combinations",
    }
)

_CONTINUATION_WEIGHTS = {
    "sentence_continuation_score": 0.22,
    "lexical_continuation_score": 0.10,
    "capitalization_score": 0.08,
    "punctuation_score": 0.08,
    "spatial_proximity_score": 0.14,
    "alignment_score": 0.08,
    "typography_score": 0.08,
    "reading_order_score": 0.07,
    "page_continuation_score": 0.05,
    "role_compatibility_score": 0.10,
}

_BOUNDARY_WEIGHTS = {
    "heading_change": 0.24,
    "structural_change": 0.14,
    "spacing_break": 0.18,
    "reading_region_change": 0.14,
    "table_boundary": 0.18,
    "role_change": 0.12,
}


@dataclass
class ContinuationEvidence:
    """Per-signal evidence that *current* continues *previous*."""

    sentence_continuation_score: float = 0.0
    lexical_continuation_score: float = 0.0
    capitalization_score: float = 0.0
    punctuation_score: float = 0.0
    spatial_proximity_score: float = 0.0
    alignment_score: float = 0.0
    typography_score: float = 0.0
    reading_order_score: float = 0.0
    page_continuation_score: float = 0.0
    role_compatibility_score: float = 0.0
    new_boundary_penalty: float = 0.0
    total_score: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "sentence_continuation_score": round(self.sentence_continuation_score, 4),
            "lexical_continuation_score": round(self.lexical_continuation_score, 4),
            "capitalization_score": round(self.capitalization_score, 4),
            "punctuation_score": round(self.punctuation_score, 4),
            "spatial_proximity_score": round(self.spatial_proximity_score, 4),
            "alignment_score": round(self.alignment_score, 4),
            "typography_score": round(self.typography_score, 4),
            "reading_order_score": round(self.reading_order_score, 4),
            "page_continuation_score": round(self.page_continuation_score, 4),
            "role_compatibility_score": round(self.role_compatibility_score, 4),
            "new_boundary_penalty": round(self.new_boundary_penalty, 4),
            "total_score": round(self.total_score, 4),
        }


@dataclass
class NewBoundaryEvidence:
    """Per-signal evidence that *current* opens a new logical unit."""

    heading_change: float = 0.0
    structural_change: float = 0.0
    spacing_break: float = 0.0
    reading_region_change: float = 0.0
    table_boundary: float = 0.0
    role_change: float = 0.0
    total_score: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "heading_change": round(self.heading_change, 4),
            "structural_change": round(self.structural_change, 4),
            "spacing_break": round(self.spacing_break, 4),
            "reading_region_change": round(self.reading_region_change, 4),
            "table_boundary": round(self.table_boundary, 4),
            "role_change": round(self.role_change, 4),
            "total_score": round(self.total_score, 4),
        }


@dataclass
class ConsolidationDecision:
    previous_block_id: str
    current_block_id: str
    previous_page: int
    current_page: int
    previous_preview: str
    current_preview: str
    continuation: ContinuationEvidence
    new_boundary: NewBoundaryEvidence
    decision: str  # MERGE | KEEP_SEPARATE
    confidence: float
    reasons: List[str] = field(default_factory=list)
    veto: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_block_id": self.previous_block_id,
            "current_block_id": self.current_block_id,
            "previous_page": self.previous_page,
            "current_page": self.current_page,
            "previous_preview": self.previous_preview,
            "current_preview": self.current_preview,
            "continuation_score": round(self.continuation.total_score, 4),
            "new_boundary_score": round(self.new_boundary.total_score, 4),
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "evidence": {
                "continuation": self.continuation.as_dict(),
                "new_boundary": self.new_boundary.as_dict(),
            },
            "reasons": list(self.reasons),
            "veto": self.veto,
        }


@dataclass
class ConsolidationResult:
    blocks: List[LogicalBlock]
    decisions: List[ConsolidationDecision] = field(default_factory=list)
    merge_chains: List[List[str]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def _leading_opener_class(text: str) -> str:
    """Character-class of the first non-space glyph (enumeration evidence)."""
    for ch in text or "":
        if ch.isspace():
            continue
        if ch.isdigit():
            return "D"
        if ch.isalpha():
            return "A"
        return "P"  # punctuation / bullet / symbol opener
    return ""


def _field_text_at(block: LogicalBlock, position: int = 0) -> str:
    fields = block.structured_fields or []
    for f in fields:
        if int(f.get("field_position", -1)) == position:
            return str(f.get("field_text") or "")
    if fields and position == 0:
        return str(fields[0].get("field_text") or "")
    return ""


def _structured_field_count(block: LogicalBlock) -> int:
    fields = block.structured_fields or []
    if not fields:
        return 0
    return 1 + max(int(f.get("field_position", 0)) for f in fields)


def _leading_field_empty(block: LogicalBlock) -> bool:
    if block.block_type != "structured_record":
        return False
    fields = block.structured_fields or []
    if not fields:
        return False
    return not _field_text_at(block, 0).strip()


def _merge_structured_fields(chain: Sequence[LogicalBlock]) -> Optional[List[Dict[str, Any]]]:
    """Merge cell texts by column position across wrapped table-row fragments."""
    if not any(b.structured_fields for b in chain):
        return None
    by_pos: Dict[int, Dict[str, Any]] = {}
    for block in chain:
        for field in block.structured_fields or []:
            pos = int(field.get("field_position", 0))
            text = str(field.get("field_text") or "").strip()
            if pos not in by_pos:
                entry = dict(field)
                entry["field_text"] = text
                entry.pop("field_part", None)
                by_pos[pos] = entry
                continue
            prev = str(by_pos[pos].get("field_text") or "").strip()
            if not text:
                continue
            if not prev:
                by_pos[pos]["field_text"] = text
            elif text == prev or prev.endswith(text):
                continue
            elif text.startswith(prev):
                by_pos[pos]["field_text"] = text
            else:
                by_pos[pos]["field_text"] = f"{prev} {text}".strip()
    return [by_pos[p] for p in sorted(by_pos)]


def _full_text(text: str) -> str:
    return " ".join((text or "").split())


def _first_alpha(text: str) -> str:
    for ch in text or "":
        if ch.isalpha():
            return ch
    return ""


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in text or "":
        if ch.isalnum() or ch in {"'", "-"}:
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def _ends_with_terminal(text: str) -> bool:
    s = (text or "").rstrip()
    while s and s[-1] in {")", "]", '"', "'", "”", "’"}:
        s = s[:-1].rstrip()
    return bool(s) and s[-1] in {".", "!", "?", ":", ";"}


def _ends_open(text: str) -> bool:
    s = (text or "").rstrip()
    if not s:
        return False
    return s[-1] in {"(", "[", "{", '"', "'", "“", "‘", ",", "-"}


def _dominant_role(block: LogicalBlock) -> str:
    roles = block.role_sequence or []
    if not roles:
        prom = block.structural_features.get(
            "prominence", block.structural_features.get("head_prominence", 0.0)
        )
        if prom >= 0.55 or block.structural_features.get("marker_depth", 0.0) > 0:
            return "PROMINENT"
        return "BODY"
    if roles[0] == "PROMINENT":
        return "PROMINENT"
    if "PROMINENT" in roles and roles.count("BODY") == 0:
        return "PROMINENT"
    return roles[-1] if roles[-1] in {"PROMINENT", "BODY", "META"} else "BODY"


def _bbox_for(
    block: LogicalBlock, document: Optional[Document]
) -> Optional[BoundingBox]:
    if document is None:
        return None
    boxes: List[BoundingBox] = []
    for bid in block.source_block_ids:
        tb = document.block_by_id(bid)
        if tb is not None and tb.bounding_box is not None:
            boxes.append(tb.bounding_box)
    if not boxes:
        return None
    return BoundingBox(
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
    )


def _union_bbox(boxes: Sequence[Optional[BoundingBox]]) -> Optional[BoundingBox]:
    present = [b for b in boxes if b is not None]
    if not present:
        return None
    return BoundingBox(
        min(b.x0 for b in present),
        min(b.y0 for b in present),
        max(b.x1 for b in present),
        max(b.y1 for b in present),
    )


class LogicalBlockConsolidator:
    """Merge adjacent logical blocks that are continuations of one unit."""

    def __init__(
        self,
        logger: Optional[DiscoveryLogger] = None,
        merge_margin: float = 0.22,
        min_continuation: float = 0.58,
    ) -> None:
        self.logger = logger
        self.merge_margin = merge_margin
        self.min_continuation = min_continuation

    def consolidate(
        self,
        document: Document,
        blocks: List[LogicalBlock],
    ) -> ConsolidationResult:
        log = self.logger
        if log:
            log.section("BLOCK CONSOLIDATION")
            log.line(
                f"Evaluating {max(0, len(blocks) - 1)} adjacent pair(s) "
                "for continuation vs new-boundary evidence..."
            )

        if len(blocks) <= 1:
            return ConsolidationResult(
                blocks=list(blocks),
                stats={
                    "initial_logical_blocks": len(blocks),
                    "consolidated_logical_blocks": len(blocks),
                    "blocks_merged": 0,
                    "merge_chains": 0,
                    "average_consolidation_confidence": 0.0,
                    "pairs_evaluated": 0,
                    "merge_decisions": 0,
                    "keep_separate_decisions": 0,
                },
            )

        ordered = sorted(
            blocks,
            key=lambda b: (
                b.source_page,
                b.doc_position,
                b.id,
            ),
        )
        decisions: List[ConsolidationDecision] = []
        for i in range(1, len(ordered)):
            decision = self.evaluate_pair(document, ordered[i - 1], ordered[i])
            decisions.append(decision)
            self._log_decision(decision)

        # Build merge chains from consecutive MERGE decisions.
        chains: List[List[LogicalBlock]] = [[ordered[0]]]
        for i, decision in enumerate(decisions):
            if decision.decision == "MERGE":
                chains[-1].append(ordered[i + 1])
            else:
                chains.append([ordered[i + 1]])

        consolidated: List[LogicalBlock] = []
        merge_chains: List[List[str]] = []
        counter = 0
        confidences: List[float] = []
        for chain in chains:
            if len(chain) == 1:
                consolidated.append(chain[0])
                continue
            counter += 1
            chain_decisions = [
                d
                for d in decisions
                if d.previous_block_id in {b.id for b in chain}
                and d.current_block_id in {b.id for b in chain}
                and d.decision == "MERGE"
            ]
            merged = self._merge_chain(document, chain, chain_decisions, counter)
            consolidated.append(merged)
            merge_chains.append([b.id for b in chain])
            if chain_decisions:
                confidences.append(
                    float(np.mean([d.confidence for d in chain_decisions]))
                )

        stats = {
            "initial_logical_blocks": len(blocks),
            "consolidated_logical_blocks": len(consolidated),
            "blocks_merged": sum(len(c) for c in merge_chains),
            "merge_chains": len(merge_chains),
            "average_consolidation_confidence": round(
                float(np.mean(confidences)) if confidences else 0.0, 4
            ),
            "pairs_evaluated": len(decisions),
            "merge_decisions": sum(1 for d in decisions if d.decision == "MERGE"),
            "keep_separate_decisions": sum(
                1 for d in decisions if d.decision == "KEEP_SEPARATE"
            ),
        }
        if log:
            log.kv("Initial logical blocks", stats["initial_logical_blocks"])
            log.kv("Consolidated logical blocks", stats["consolidated_logical_blocks"])
            log.kv("Merge chains", stats["merge_chains"])
            log.kv("Blocks merged", stats["blocks_merged"])
            log.kv(
                "Average consolidation confidence",
                f"{stats['average_consolidation_confidence']:.2f}",
            )
            for chain in merge_chains[:20]:
                log.event(
                    "logical_block_merged",
                    document_id=document.id,
                    source_ids=chain,
                    decision="MERGE_CHAIN",
                    confidence=stats["average_consolidation_confidence"],
                    reason="Continuation of same logical content across adjacent blocks.",
                    member_count=len(chain),
                )

        return ConsolidationResult(
            blocks=consolidated,
            decisions=decisions,
            merge_chains=merge_chains,
            stats=stats,
        )

    # ------------------------------------------------------------------ #
    def evaluate_pair(
        self,
        document: Document,
        previous: LogicalBlock,
        current: LogicalBlock,
    ) -> ConsolidationDecision:
        veto = self._hard_veto(previous, current)
        continuation = self._continuation_evidence(document, previous, current)
        boundary = self._boundary_evidence(document, previous, current)
        continuation.new_boundary_penalty = boundary.total_score
        continuation.total_score = self._weighted(
            continuation.as_dict(), _CONTINUATION_WEIGHTS
        )
        boundary.total_score = self._weighted(boundary.as_dict(), _BOUNDARY_WEIGHTS)

        reasons: List[str] = []
        if veto:
            decision = "KEEP_SEPARATE"
            confidence = 0.95
            reasons.append(veto)
        else:
            margin = continuation.total_score - boundary.total_score
            prev_text = previous.text or ""
            cur_text = current.text or ""
            unfinished = (not _ends_with_terminal(prev_text)) or _ends_open(prev_text)
            # True line/cell wrap: unfinished previous *and* a real continuation
            # cue (lowercase start or closed-class starter). Incomplete text alone
            # is not enough — that over-merged list bullets and title lines.
            wrap_like = (
                unfinished
                and continuation.sentence_continuation_score >= 0.78
                and continuation.total_score >= self.min_continuation
                and margin >= self.merge_margin
            )
            cur_tokens = _tokens(cur_text)
            first_token = cur_tokens[0].casefold() if cur_tokens else ""
            # Same-unit prose that continues with a referring sentence after a
            # finished clause (common for feature descriptions spanning boxes).
            anaphoric_continue = (
                first_token in {"this", "these", "that", "those", "such", "it"}
                and len(cur_text) >= 40
                and _dominant_role(current) != "PROMINENT"
                and continuation.sentence_continuation_score >= 0.60
                and margin >= self.merge_margin
            )
            # Structured records that already cleared the hard veto for wrap /
            # anaphoric continuation across cells or page splits.
            empty_lead = _leading_field_empty(current)
            prev_lead = _field_text_at(previous, 0).strip()
            label_wrap = (
                previous.block_type == "structured_record"
                and current.block_type == "structured_record"
                and prev_lead.endswith(":")
                and bool(_field_text_at(current, 0).strip())
            )
            structured_continue = (
                previous.block_type == "structured_record"
                and current.block_type == "structured_record"
                and (
                    empty_lead
                    or label_wrap
                    or continuation.sentence_continuation_score >= 0.30
                )
                and (
                    empty_lead
                    or label_wrap
                    or margin >= self.merge_margin
                )
            )
            if wrap_like or anaphoric_continue or structured_continue:
                decision = "MERGE"
                confidence = clip01(0.5 + 0.5 * margin + 0.2 * continuation.total_score)
                reasons.extend(self._merge_reasons(continuation, boundary))
            else:
                decision = "KEEP_SEPARATE"
                confidence = clip01(0.5 + 0.5 * max(0.0, -margin) + 0.2 * boundary.total_score)
                reasons.extend(self._separate_reasons(continuation, boundary))
                if continuation.total_score >= self.min_continuation and not unfinished:
                    reasons.append(
                        "Adjacent units look related but previous text is finished; "
                        "kept as separate items to avoid over-grouping."
                    )

        return ConsolidationDecision(
            previous_block_id=previous.id,
            current_block_id=current.id,
            previous_page=previous.source_page,
            current_page=current.source_page,
            previous_preview=_full_text(previous.text),
            current_preview=_full_text(current.text),
            continuation=continuation,
            new_boundary=boundary,
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            veto=veto,
        )

    @staticmethod
    def _weighted(signals: Dict[str, float], weights: Dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for key, weight in weights.items():
            if key not in signals:
                continue
            total += float(signals[key]) * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        return clip01(total / weight_sum)

    def _hard_veto(
        self, previous: LogicalBlock, current: LogicalBlock
    ) -> Optional[str]:
        # Preserve structured table records by default, but allow a wrap
        # continuation when the previous cell text is clearly unfinished and
        # the next fragment continues the sentence. Page-split tables often
        # receive different source_table_id values, so table-id mismatch alone
        # is never a hard veto when wrap evidence is strong.
        prev_structured = previous.block_type == "structured_record"
        cur_structured = current.block_type == "structured_record"
        if prev_structured != cur_structured:
            return "Structured record adjacent to non-record content."
        if prev_structured and cur_structured:
            prev_text = previous.text or ""
            cur_text = current.text or ""
            first = _first_alpha(cur_text)
            tokens = _tokens(cur_text)
            first_token = tokens[0].casefold() if tokens else ""
            incomplete = not _ends_with_terminal(prev_text) or _ends_open(prev_text)
            empty_lead = _leading_field_empty(current)
            prev_lead = _field_text_at(previous, 0).strip()
            label_wrap = prev_lead.endswith(":") and bool(_field_text_at(current, 0).strip())
            wrap_continuation = (
                empty_lead
                or label_wrap
                or (
                    incomplete
                    and (
                        (bool(first) and first.islower())
                        or first_token in _WEAK_CONTINUATION_STARTERS
                    )
                )
            )
            # Same logical record often continues with a new sentence that
            # anaphorically refers to the previous description ("This feature…").
            prev_prom = previous.structural_features.get(
                "prominence", previous.structural_features.get("head_prominence", 0.0)
            )
            cur_prom = current.structural_features.get(
                "prominence", current.structural_features.get("head_prominence", 0.0)
            )
            anaphoric_continuation = (
                first_token in {"this", "these", "that", "those", "such", "it"}
                and len(cur_text) >= 40
                and current.structural_features.get("marker_depth", 0.0) <= 0
                and cur_prom <= prev_prom + 0.25
            )
            # Split header label rows: later row mostly empty leading cells.
            same_width = _structured_field_count(previous) == _structured_field_count(
                current
            ) and _structured_field_count(previous) >= 2
            header_fragment = same_width and empty_lead
            if not (wrap_continuation or anaphoric_continuation or header_fragment):
                return "Both blocks are structured records; table rows stay independent."

        prev_role = _dominant_role(previous)
        cur_role = _dominant_role(current)
        # Parallel enumerated items (same bullet/number opener class, short
        # lines) are independent items — never wrap-merge them.
        prev_opener = _leading_opener_class(previous.text or "")
        cur_opener = _leading_opener_class(current.text or "")
        if (
            prev_opener
            and prev_opener == cur_opener
            and prev_opener in {"P", "D"}
            and len((previous.text or "").split()) <= 24
            and len((current.text or "").split()) <= 24
        ):
            return "Parallel enumerated items share an opener class; keep separate."

        # Extractor section/page headers are atomic openers — never absorb the
        # following body unless the next fragment is a true lowercase wrap.
        prev_is_layout_head = (
            previous.block_type != "structured_record"
            and (
                previous.structural_features.get("layout_section_header", 0.0) >= 1.0
                or previous.structural_features.get("layout_page_header", 0.0) >= 1.0
            )
        )
        if prev_is_layout_head:
            first = _first_alpha(current.text or "")
            if not (first and first.islower()):
                return "Layout heading unit stays separate from following content."

        # Compact title-like PROMINENT units are group openers, not wrap stems.
        # Structured table rows are excluded — their "prominence" is geometric.
        prev_text = previous.text or ""
        prev_words = len(prev_text.split())
        if (
            previous.block_type != "structured_record"
            and prev_role == "PROMINENT"
            and prev_words <= 14
            and len(prev_text) <= 120
            and not _ends_open(prev_text)
            and not (bool(_first_alpha(current.text or "")) and _first_alpha(current.text).islower())
        ):
            return "Short prominent heading stays separate from following content."

        # A new prominent head after body content is a classic new-block cue,
        # unless the previous text is clearly unfinished (handled without veto).
        if (
            prev_role == "BODY"
            and cur_role == "PROMINENT"
            and _ends_with_terminal(previous.text)
            and _first_alpha(current.text).isupper()
        ):
            return "Body followed by a new prominent unit with terminal punctuation."
        return None

    def _continuation_evidence(
        self,
        document: Document,
        previous: LogicalBlock,
        current: LogicalBlock,
    ) -> ContinuationEvidence:
        prev_text = previous.text or ""
        cur_text = current.text or ""
        prev_tokens = _tokens(prev_text)
        cur_tokens = _tokens(cur_text)
        first_token = (cur_tokens[0].casefold() if cur_tokens else "")
        last_token = (prev_tokens[-1].casefold() if prev_tokens else "")
        first_alpha = _first_alpha(cur_text)
        incomplete = (not _ends_with_terminal(prev_text)) or _ends_open(prev_text)
        lowercase_start = bool(first_alpha) and first_alpha.islower()
        # Closed-class starters only count as wrap when the next unit begins in
        # lowercase. Capitalised "In/To/For…" opens a new sentence, not a wrap —
        # otherwise short headings merge into the following paragraph.
        starter_hit = (
            first_token in _WEAK_CONTINUATION_STARTERS and lowercase_start
        )
        anaphoric_start = first_token in {
            "this",
            "these",
            "that",
            "those",
            "such",
            "it",
        }
        ending_hit = last_token in _WEAK_INCOMPLETE_ENDINGS

        complete_to_complete = (
            _ends_with_terminal(prev_text)
            and bool(first_alpha)
            and first_alpha.isupper()
            and not starter_hit
            and not anaphoric_start
        )

        sentence = 0.0
        if incomplete and lowercase_start:
            sentence = 0.92
        elif incomplete and starter_hit:
            sentence = 0.78
        elif anaphoric_start and len(cur_text) >= 40:
            # Referring sentence that continues the same logical unit.
            sentence = 0.72
        elif incomplete:
            sentence = 0.55
        elif lowercase_start:
            sentence = 0.45
        elif starter_hit:
            sentence = 0.35
        if complete_to_complete:
            # Two finished, title-cased units are independent by default.
            sentence = min(sentence, 0.12)

        lexical = 0.0
        if starter_hit:
            lexical += 0.55
        if ending_hit and incomplete:
            lexical += 0.35
        if incomplete and (starter_hit or ending_hit):
            lexical += 0.15
        lexical = clip01(lexical)
        if complete_to_complete:
            lexical = min(lexical, 0.10)

        capitalization = 0.0
        if lowercase_start:
            capitalization = 0.95
        elif first_alpha and first_alpha.isupper() and incomplete:
            capitalization = 0.25
        elif first_alpha and first_alpha.isupper():
            capitalization = 0.05
        if complete_to_complete:
            capitalization = 0.05

        punctuation = 0.0
        if _ends_open(prev_text):
            punctuation = 0.95
        elif not _ends_with_terminal(prev_text):
            punctuation = 0.70
        elif prev_text.rstrip().endswith((":", ";")):
            punctuation = 0.55
        else:
            punctuation = 0.10
        if complete_to_complete:
            punctuation = 0.08

        prev_box = _bbox_for(previous, document)
        cur_box = _bbox_for(current, document)
        spatial, alignment, region_break = self._geometry_scores(
            document, previous, current, prev_box, cur_box
        )

        typography = self._typography_score(previous, current)
        reading_order = 1.0 if previous.doc_position <= current.doc_position else 0.0
        if previous.source_page == current.source_page:
            page_cont = 0.85
        else:
            # Cross-page continuation is allowed when text looks unfinished.
            page_cont = 0.75 if incomplete else 0.35

        role_compat = self._role_compatibility(previous, current)
        # Finished sentence → finished sentence is only a strong anti-merge
        # signal when the layout also looks like a break. A tight BODY→BODY
        # run may still be one multi-sentence logical unit.
        if complete_to_complete and spatial < 0.80:
            spatial = min(spatial, 0.28)
            alignment = min(alignment, 0.40)
            role_compat = min(role_compat, 0.28)
            typography = min(typography, 0.30)
            page_cont = min(page_cont, 0.30)

        return ContinuationEvidence(
            sentence_continuation_score=sentence,
            lexical_continuation_score=lexical,
            capitalization_score=capitalization,
            punctuation_score=punctuation,
            spatial_proximity_score=spatial,
            alignment_score=alignment,
            typography_score=typography,
            reading_order_score=reading_order,
            page_continuation_score=page_cont,
            role_compatibility_score=role_compat,
        )

    def _boundary_evidence(
        self,
        document: Document,
        previous: LogicalBlock,
        current: LogicalBlock,
    ) -> NewBoundaryEvidence:
        prev_role = _dominant_role(previous)
        cur_role = _dominant_role(current)
        prev_prom = previous.structural_features.get(
            "prominence", previous.structural_features.get("head_prominence", 0.0)
        )
        cur_prom = current.structural_features.get(
            "prominence", current.structural_features.get("head_prominence", 0.0)
        )
        prev_depth = previous.structural_features.get("marker_depth", 0.0)
        cur_depth = current.structural_features.get("marker_depth", 0.0)

        heading_change = 0.0
        if cur_role == "PROMINENT" and prev_role != "PROMINENT":
            heading_change = clip01(0.55 + 0.45 * cur_prom)
        elif prev_role == "PROMINENT" and cur_role == "PROMINENT":
            # Adjacent prominent units are almost always distinct logical heads.
            heading_change = clip01(0.70 + 0.25 * max(cur_prom, prev_prom))
        elif cur_depth > 0 and cur_depth <= max(prev_depth, 1.0) and prev_depth == 0:
            heading_change = 0.85
        elif cur_prom > prev_prom + 0.25 and _first_alpha(current.text).isupper():
            heading_change = clip01((cur_prom - prev_prom) * 1.5)

        structural_change = 0.0
        if previous.structural_signature and current.structural_signature:
            if previous.structural_signature != current.structural_signature:
                structural_change = 0.45
        if previous.discovered_pattern and current.discovered_pattern:
            if previous.discovered_pattern != current.discovered_pattern:
                structural_change = max(structural_change, 0.35)
        if previous.block_type != current.block_type:
            structural_change = max(structural_change, 0.80)
        # Parallel bullets / numbered lines are distinct items even when close.
        prev_opener = _leading_opener_class(previous.text or "")
        cur_opener = _leading_opener_class(current.text or "")
        if (
            prev_opener
            and prev_opener == cur_opener
            and prev_opener in {"P", "D"}
            and len((previous.text or "").split()) <= 24
            and len((current.text or "").split()) <= 24
        ):
            structural_change = max(structural_change, 0.95)

        complete_to_complete = (
            _ends_with_terminal(previous.text)
            and bool(_first_alpha(current.text))
            and _first_alpha(current.text).isupper()
        )
        prev_box = _bbox_for(previous, document)
        cur_box = _bbox_for(current, document)
        spatial, _, region_break = self._geometry_scores(
            document, previous, current, prev_box, cur_box
        )
        spacing_break = clip01(1.0 - spatial)
        if previous.source_page != current.source_page and _ends_with_terminal(previous.text):
            spacing_break = max(spacing_break, 0.55)
        if complete_to_complete:
            # Finished title-cased units with a non-trivial gap are independent.
            structural_change = max(structural_change, 0.55)
            if spatial < 0.80:
                structural_change = max(structural_change, 0.78)
                spacing_break = max(spacing_break, 0.68)

        table_boundary = 0.0
        if previous.block_type == "structured_record" and current.block_type == "structured_record":
            # Wrap / anaphoric fragments of one record should not be punished as
            # hard table boundaries; independent rows still get the veto.
            incomplete = not _ends_with_terminal(previous.text) or _ends_open(previous.text)
            first = _first_alpha(current.text)
            tokens = _tokens(current.text)
            first_token = tokens[0].casefold() if tokens else ""
            lowercase = bool(first) and first.islower()
            anaphoric = first_token in {"this", "these", "that", "those", "such", "it"}
            empty_lead = _leading_field_empty(current)
            label_wrap = _field_text_at(previous, 0).strip().endswith(":") and bool(
                _field_text_at(current, 0).strip()
            )
            table_boundary = (
                0.15
                if (empty_lead or label_wrap or (incomplete and lowercase) or anaphoric)
                else 1.0
            )
        elif previous.block_type == "structured_record" or current.block_type == "structured_record":
            table_boundary = 1.0
        elif previous.source_table_id or current.source_table_id:
            table_boundary = 0.85

        role_change = 0.0
        if prev_role != cur_role:
            if prev_role == "BODY" and cur_role == "PROMINENT":
                role_change = 0.90
            elif prev_role == "PROMINENT" and cur_role == "BODY":
                role_change = 0.20  # title → body is often one unit
            else:
                role_change = 0.55
        elif prev_role == "PROMINENT" and cur_role == "PROMINENT":
            role_change = 0.80

        return NewBoundaryEvidence(
            heading_change=heading_change,
            structural_change=structural_change,
            spacing_break=spacing_break,
            reading_region_change=region_break,
            table_boundary=table_boundary,
            role_change=role_change,
        )

    def _geometry_scores(
        self,
        document: Document,
        previous: LogicalBlock,
        current: LogicalBlock,
        prev_box: Optional[BoundingBox],
        cur_box: Optional[BoundingBox],
    ) -> Tuple[float, float, float]:
        """Return (spatial_proximity, alignment, reading_region_break)."""
        # Feature fallbacks when absolute boxes are unavailable.
        fa = previous.structural_features
        fb = current.structural_features
        if prev_box is None or cur_box is None:
            same_page = previous.source_page == current.source_page
            dx = abs(fa.get("rel_x0", 0.0) - fb.get("rel_x0", 0.0))
            alignment = clip01(1.0 - dx * 4.0)
            gap_ratio = fb.get("gap_above_ratio", 1.0)
            spatial = clip01(1.0 / (1.0 + max(0.0, gap_ratio - 1.0))) if same_page else 0.45
            region = clip01(dx * 3.0)
            return spatial, alignment, region

        page_h = 792.0
        page_w = 612.0
        if document.pages:
            # Prefer the current page metrics when available.
            page = next(
                (p for p in document.pages if p.page_number == current.source_page),
                document.pages[0],
            )
            page_h = page.height or page_h
            page_w = page.width or page_w

        same_page = previous.source_page == current.source_page
        if same_page:
            gap = max(0.0, cur_box.y0 - prev_box.y1)
            # Typical body line spacing is a fraction of page height.
            spatial = clip01(1.0 - gap / max(18.0, page_h * 0.08))
        else:
            # End-of-page → start-of-next-page is expected for continuations.
            near_bottom = prev_box.y1 / page_h
            near_top = cur_box.y0 / page_h
            spatial = clip01(0.35 + 0.45 * near_bottom + 0.20 * (1.0 - near_top))

        dx0 = abs(prev_box.x0 - cur_box.x0) / max(1.0, page_w)
        width_ratio = min(prev_box.width, cur_box.width) / max(
            1.0, max(prev_box.width, cur_box.width)
        )
        alignment = clip01(0.7 * (1.0 - dx0 * 5.0) + 0.3 * width_ratio)

        # Distinct columns: large horizontal separation with overlapping y.
        col_sep = abs(prev_box.cx - cur_box.cx) / max(1.0, page_w)
        y_overlap = prev_box.intersection_area(
            BoundingBox(prev_box.x0, cur_box.y0, prev_box.x1, cur_box.y1)
        ) > 0 or (
            same_page and abs(prev_box.cy - cur_box.cy) < page_h * 0.15 and col_sep > 0.28
        )
        region_break = clip01(col_sep * 2.2) if (same_page and col_sep > 0.22) else 0.0
        if y_overlap and col_sep > 0.28:
            region_break = max(region_break, 0.85)
            spatial = min(spatial, 0.25)
        return spatial, alignment, region_break

    @staticmethod
    def _typography_score(previous: LogicalBlock, current: LogicalBlock) -> float:
        fa = previous.structural_features
        fb = current.structural_features
        size_a = fa.get("mean_line_height_ratio", fa.get("head_size_ratio", fa.get("line_height_ratio", 1.0)))
        size_b = fb.get("mean_line_height_ratio", fb.get("head_size_ratio", fb.get("line_height_ratio", 1.0)))
        size_sim = clip01(1.0 - abs(size_a - size_b) / max(0.4, max(size_a, size_b)))
        # Compatible body→body or title→body.
        prev_role = _dominant_role(previous)
        cur_role = _dominant_role(current)
        if prev_role == "BODY" and cur_role == "BODY":
            return clip01(0.55 + 0.45 * size_sim)
        if prev_role == "PROMINENT" and cur_role == "BODY":
            return clip01(0.40 + 0.35 * size_sim)
        if prev_role == "BODY" and cur_role == "PROMINENT":
            return clip01(0.15 * size_sim)
        return clip01(0.35 + 0.40 * size_sim)

    @staticmethod
    def _role_compatibility(previous: LogicalBlock, current: LogicalBlock) -> float:
        prev_role = _dominant_role(previous)
        cur_role = _dominant_role(current)
        if prev_role == "BODY" and cur_role == "BODY":
            return 0.95
        if prev_role == "PROMINENT" and cur_role == "BODY":
            return 0.70
        if prev_role == "META" and cur_role == "BODY":
            return 0.55
        if prev_role == "BODY" and cur_role == "META":
            return 0.50
        if prev_role == "BODY" and cur_role == "PROMINENT":
            return 0.15
        if prev_role == cur_role:
            return 0.60
        return 0.30

    @staticmethod
    def _merge_reasons(
        continuation: ContinuationEvidence, boundary: NewBoundaryEvidence
    ) -> List[str]:
        reasons = []
        if continuation.sentence_continuation_score >= 0.6:
            reasons.append("Previous block appears incomplete; current continues the sentence.")
        if continuation.capitalization_score >= 0.7:
            reasons.append("Current block begins with lowercase text.")
        if continuation.lexical_continuation_score >= 0.5:
            reasons.append("Lexical continuity markers support the same reading flow.")
        if continuation.spatial_proximity_score >= 0.6:
            reasons.append("Blocks are spatially close in the same reading region.")
        if continuation.alignment_score >= 0.7:
            reasons.append("Compatible horizontal alignment.")
        if continuation.typography_score >= 0.6:
            reasons.append("Compatible typography.")
        if continuation.page_continuation_score >= 0.6 and boundary.spacing_break < 0.5:
            reasons.append("Compatible across page boundary.")
        if boundary.heading_change < 0.25 and boundary.table_boundary < 0.5:
            reasons.append("No strong new heading or table boundary.")
        if not reasons:
            reasons.append("Continuation evidence outweighs new-boundary evidence.")
        return reasons

    @staticmethod
    def _separate_reasons(
        continuation: ContinuationEvidence, boundary: NewBoundaryEvidence
    ) -> List[str]:
        reasons = []
        if boundary.heading_change >= 0.5:
            reasons.append("Strong new prominent structure detected.")
        if boundary.table_boundary >= 0.5:
            reasons.append("Table / structured-record boundary.")
        if boundary.spacing_break >= 0.55:
            reasons.append("Significant spatial boundary.")
        if boundary.reading_region_change >= 0.5:
            reasons.append("Different reading column / region.")
        if boundary.role_change >= 0.7:
            reasons.append("Role change indicates a new logical unit.")
        if boundary.structural_change >= 0.5:
            reasons.append("Structural pattern changed.")
        if continuation.sentence_continuation_score < 0.3:
            reasons.append("Previous block does not look incomplete.")
        if not reasons:
            reasons.append("New-boundary evidence dominates continuation evidence.")
        return reasons

    def _merge_chain(
        self,
        document: Document,
        chain: List[LogicalBlock],
        decisions: List[ConsolidationDecision],
        index: int,
    ) -> LogicalBlock:
        head = chain[0]
        texts = [b.text for b in chain if b.text]
        # Join with space when the next piece continues mid-sentence; newline otherwise.
        merged_parts: List[str] = [texts[0]] if texts else []
        for i in range(1, len(texts)):
            prev = merged_parts[-1]
            nxt = texts[i]
            if (not _ends_with_terminal(prev)) or _first_alpha(nxt).islower():
                merged_parts[-1] = prev.rstrip() + " " + nxt.lstrip()
            else:
                merged_parts.append(nxt)
        text = "\n".join(merged_parts)

        source_block_ids: List[str] = []
        for b in chain:
            for bid in b.source_block_ids:
                if bid not in source_block_ids:
                    source_block_ids.append(bid)

        roles: List[str] = []
        for b in chain:
            roles.extend(b.role_sequence or [])
        if not roles:
            roles = [_dominant_role(b) for b in chain]

        conf = float(np.mean([b.confidence for b in chain]))
        if decisions:
            conf = clip01(
                0.5 * conf + 0.5 * float(np.mean([d.confidence for d in decisions]))
            )

        cont_means = {
            k: float(np.mean([getattr(d.continuation, k) for d in decisions]))
            for k in ContinuationEvidence().__dict__.keys()
            if k not in {"total_score", "new_boundary_penalty"}
        } if decisions else {}
        notes = []
        for d in decisions:
            notes.extend(d.reasons)
        notes.append(f"merged_from:{','.join(b.id for b in chain)}")

        consolidated = LogicalBlock(
            id=f"{document.id}_consolidated_block_{index:04d}",
            content_unit_id=head.content_unit_id,
            document_id=document.id,
            source_document=head.source_document,
            source_page=head.source_page,
            page_end=chain[-1].page_end or chain[-1].source_page,
            source_block_ids=source_block_ids,
            text=text,
            structural_features={
                **dict(head.structural_features),
                "block_count": float(len(source_block_ids)),
                "merged_logical_block_count": float(len(chain)),
            },
            semantic_vector=head.semantic_vector,
            discovered_pattern=head.discovered_pattern,
            role_sequence=roles,
            structural_signature=role_signature(roles),
            structural_fingerprint={
                **dict(head.structural_fingerprint),
                "block_count": float(len(chain)),
            },
            confidence=conf,
            evidence=Evidence(
                signals={k: round(v, 4) for k, v in cont_means.items()},
                weights=dict(_CONTINUATION_WEIGHTS),
                confidence=conf,
                notes=notes[:20],
            ),
            group_id=None,
            section_group_id=None,
            doc_position=head.doc_position,
            block_type=head.block_type if head.block_type != "heading" else "content",
            structured_fields=_merge_structured_fields(chain) or head.structured_fields,
            source_table_id=head.source_table_id,
            source_logical_block_ids=[b.id for b in chain],
            source_content_unit_ids=[b.content_unit_id for b in chain],
            consolidation_evidence=Evidence(
                signals={
                    "continuation_score": round(
                        float(np.mean([d.continuation.total_score for d in decisions])), 4
                    )
                    if decisions
                    else 0.0,
                    "new_boundary_score": round(
                        float(np.mean([d.new_boundary.total_score for d in decisions])), 4
                    )
                    if decisions
                    else 0.0,
                    "chain_length": float(len(chain)),
                },
                weights={"continuation_score": 0.7, "new_boundary_score": 0.3},
                confidence=conf,
                notes=[f"chain:{' → '.join(b.id for b in chain)}"],
            ),
        )
        return consolidated

    def _log_decision(self, decision: ConsolidationDecision) -> None:
        log = self.logger
        if not log:
            return
        log.push()
        log.line("Continuation Decision:")
        log.kv("Previous block", f"{decision.previous_block_id} (page {decision.previous_page})")
        log.line(f"  text: {decision.previous_preview}")
        log.kv("Current block", f"{decision.current_block_id} (page {decision.current_page})")
        log.line(f"  text: {decision.current_preview}")
        log.line("Continuation evidence:")
        for key, value in decision.continuation.as_dict().items():
            if key in {"total_score", "new_boundary_penalty"}:
                continue
            log.line(f"  {key}: {value:.2f}")
        log.line("New boundary evidence:")
        for key, value in decision.new_boundary.as_dict().items():
            if key == "total_score":
                continue
            log.line(f"  {key}: {value:.2f}")
        log.kv("Continuation score", f"{decision.continuation.total_score:.2f}")
        log.kv("New boundary score", f"{decision.new_boundary.total_score:.2f}")
        log.kv("Decision", decision.decision)
        log.kv("Confidence", f"{decision.confidence:.2f}")
        for reason in decision.reasons:
            log.line(f"  reason: {reason}")
        log.pop()
        log.event(
            "logical_block_continuation_evaluated",
            document_id=decision.previous_block_id.rsplit("_logical_block_", 1)[0]
            if "_logical_block_" in decision.previous_block_id
            else "",
            page=decision.current_page,
            source_ids=[decision.previous_block_id, decision.current_block_id],
            decision=decision.decision,
            confidence=round(decision.confidence, 4),
            evidence=decision.to_dict()["evidence"],
            reason="; ".join(decision.reasons),
            continuation_score=round(decision.continuation.total_score, 4),
            new_boundary_score=round(decision.new_boundary.total_score, 4),
            veto=decision.veto,
        )


def write_consolidation_log(path: str, result: ConsolidationResult, document_id: str) -> str:
    """Write a focused human-readable consolidation report."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: List[str] = []
    A = lines.append
    sep = "=" * 60
    A(sep)
    A("BLOCK CONSOLIDATION")
    A(sep)
    A("")
    A(f"Document: {document_id}")
    A(f"Initial logical blocks: {result.stats.get('initial_logical_blocks', 0)}")
    A(f"Consolidated logical blocks: {result.stats.get('consolidated_logical_blocks', 0)}")
    A(f"Blocks merged: {result.stats.get('blocks_merged', 0)}")
    A(f"Number of merge chains: {result.stats.get('merge_chains', 0)}")
    A(
        "Average consolidation confidence: "
        f"{result.stats.get('average_consolidation_confidence', 0.0):.2f}"
    )
    A("")

    if result.merge_chains:
        A("Merged chains:")
        A("")
        for chain in result.merge_chains:
            A(f"  {' → '.join(chain)}")
            A("  Reason: Continuation of same logical content.")
            A("")
    else:
        A("No merge chains.")
        A("")

    A(sep)
    A("PAIR DECISIONS")
    A(sep)
    A("")
    for d in result.decisions:
        A("Continuation Decision:")
        A(f"  previous_block_id: {d.previous_block_id}")
        A(f"  current_block_id: {d.current_block_id}")
        A(f"  previous page: {d.previous_page}")
        A(f"  current page: {d.current_page}")
        A(f"  previous text: {d.previous_preview}")
        A(f"  current text: {d.current_preview}")
        A(f"  continuation_score: {d.continuation.total_score:.2f}")
        A(f"  new_boundary_score: {d.new_boundary.total_score:.2f}")
        A("  Continuation evidence:")
        for k, v in d.continuation.as_dict().items():
            if k in {"total_score", "new_boundary_penalty"}:
                continue
            A(f"    {k}: {v:.2f}")
        A("  New boundary evidence:")
        for k, v in d.new_boundary.as_dict().items():
            if k == "total_score":
                continue
            A(f"    {k}: {v:.2f}")
        A(f"  Decision: {d.decision}")
        A(f"  Confidence: {d.confidence:.2f}")
        for reason in d.reasons:
            A(f"  Reason: {reason}")
        A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
