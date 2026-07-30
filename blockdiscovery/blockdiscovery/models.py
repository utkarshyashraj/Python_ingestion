"""Core data models for the PDF logical block discovery engine.

Every object here is designed to preserve *complete source traceability*:

    LogicalGroup -> LogicalBlock -> ContentUnit -> TextBlock -> Page -> Document

No information is discarded during transformation. Higher-level objects only
ever reference lower-level objects by id (and keep the ids around), so any
logical result can always be traced back to the exact bytes on the page.

These models are intentionally free of business logic. They describe *observed
evidence* and *discovered structure* -- never predefined categories such as
"Feature", "Fix" or "Bug".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in PDF point coordinates (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def intersection_area(self, other: "BoundingBox") -> float:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: "BoundingBox", tol: float = 1.0) -> bool:
        return (
            self.x0 - tol <= other.x0
            and self.y0 - tol <= other.y0
            and self.x1 + tol >= other.x1
            and self.y1 + tol >= other.y1
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "x0": round(self.x0, 3),
            "y0": round(self.y0, 3),
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
        }


# --------------------------------------------------------------------------- #
# Raw extraction evidence
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    """A run of text sharing identical font characteristics."""

    text: str
    font: str
    size: float
    flags: int
    color: int
    bbox: BoundingBox

    # Derived font-weight/style evidence (PyMuPDF flag bits).
    @property
    def is_bold(self) -> bool:
        # bit 4 (16) = bold in PyMuPDF span flags; also detect by font name.
        return bool(self.flags & 2 ** 4) or "bold" in self.font.lower() or "black" in self.font.lower()

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2 ** 1) or "italic" in self.font.lower() or "oblique" in self.font.lower()

    @property
    def is_monospace(self) -> bool:
        return bool(self.flags & 2 ** 3) or "mono" in self.font.lower() or "courier" in self.font.lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "font": self.font,
            "size": round(self.size, 3),
            "flags": self.flags,
            "color": self.color,
            "bbox": self.bbox.to_dict(),
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "is_monospace": self.is_monospace,
        }


@dataclass
class Line:
    """A single visual line of text (group of spans)."""

    text: str
    bbox: BoundingBox
    spans: List[Span] = field(default_factory=list)
    wmode: int = 0
    direction: Tuple[float, float] = (1.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class Formatting:
    """Summary of the typographic evidence for a block (relative reasoning is
    done later against document-level statistics -- nothing absolute here)."""

    dominant_size: float
    max_size: float
    min_size: float
    dominant_font: str
    bold_ratio: float
    italic_ratio: float
    monospace_ratio: float
    color: int
    size_variety: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_size": round(self.dominant_size, 3),
            "max_size": round(self.max_size, 3),
            "min_size": round(self.min_size, 3),
            "dominant_font": self.dominant_font,
            "bold_ratio": round(self.bold_ratio, 3),
            "italic_ratio": round(self.italic_ratio, 3),
            "monospace_ratio": round(self.monospace_ratio, 3),
            "color": self.color,
            "size_variety": self.size_variety,
        }


@dataclass
class TextBlock:
    """A raw block extracted from the PDF, with full source traceability.

    This is the atomic unit of *observed evidence*. It carries no interpretation
    beyond what PyMuPDF reported plus lightweight typographic summaries.
    """

    id: str
    document_id: str
    page_number: int
    text: str
    bounding_box: BoundingBox
    spans: List[Span] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    formatting: Optional[Formatting] = None
    reading_order: int = 0
    block_type: str = "text"  # "text" | "image"
    raw_block_no: int = 0

    # Populated by the feature-generation layer.
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self, include_spans: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
            "formatting": self.formatting.to_dict() if self.formatting else None,
            "reading_order": self.reading_order,
            "block_type": self.block_type,
            "line_count": self.line_count,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }
        if include_spans:
            d["spans"] = [s.to_dict() for s in self.spans]
            d["lines"] = [ln.to_dict() for ln in self.lines]
        return d


@dataclass
class PageInfo:
    document_id: str
    page_number: int
    width: float
    height: float
    rotation: int = 0
    block_ids: List[str] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "rotation": self.rotation,
            "block_count": len(self.block_ids),
            "link_count": len(self.links),
            "image_count": len(self.images),
        }


@dataclass
class Document:
    id: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[PageInfo] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)

    # Document-level statistics used for *relative* reasoning (computed later).
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def block_by_id(self, block_id: str) -> Optional[TextBlock]:
        for b in self.blocks:
            if b.id == block_id:
                return b
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metadata": self.metadata,
            "page_count": self.page_count,
            "block_count": len(self.blocks),
            "stats": self.stats,
        }


# --------------------------------------------------------------------------- #
# Discovered structure
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """A transparent, per-signal evidence bundle plus the fused confidence.

    Confidence is NEVER stored without the evidence that produced it.
    """

    signals: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "confidence": round(self.confidence, 4),
            "notes": self.notes,
        }


@dataclass
class ContentUnit:
    """A discovered, coherent unit of information composed of >=1 raw blocks.

    A content unit is discovered from evidence -- it is not assumed to exist.
    """

    id: str
    document_id: str
    page_number: int
    block_ids: List[str]
    text: str
    head_block_id: Optional[str] = None  # most visually prominent block
    role_sequence: List[str] = field(default_factory=list)  # relative roles
    structural_signature: str = ""
    structural_fingerprint: Dict[str, float] = field(default_factory=dict)
    bounding_box: Optional[BoundingBox] = None
    features: Dict[str, float] = field(default_factory=dict)
    semantic_vector: Optional[List[float]] = None
    evidence: Evidence = field(default_factory=Evidence)
    page_end: Optional[int] = None  # inclusive; defaults to page_number

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "page_end": self.page_end if self.page_end is not None else self.page_number,
            "block_ids": list(self.block_ids),
            "head_block_id": self.head_block_id,
            "text": self.text,
            "role_sequence": self.role_sequence,
            "structural_signature": self.structural_signature,
            "structural_fingerprint": {
                k: round(v, 4) for k, v in self.structural_fingerprint.items()
            },
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "evidence": self.evidence.to_dict(),
        }
        if include_vector and self.semantic_vector is not None:
            d["semantic_vector"] = self.semantic_vector
        return d


@dataclass
class DiscoveredPattern:
    """A recurring structural/semantic pattern discovered across content units.

    Deliberately named generically (e.g. ``pattern_003``). Any human-friendly
    label is optional and applied later -- never required for discovery.
    """

    id: str
    member_unit_ids: List[str] = field(default_factory=list)
    representative_signature: str = ""
    centroid: Optional[List[float]] = None
    role_template: List[str] = field(default_factory=list)
    inferred_label: Optional[str] = None  # optional, may stay None forever
    label_confidence: float = 0.0
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def size(self) -> int:
        return len(self.member_unit_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "size": self.size,
            "member_unit_ids": list(self.member_unit_ids),
            "representative_signature": self.representative_signature,
            "role_template": self.role_template,
            "inferred_label": self.inferred_label,
            "label_confidence": round(self.label_confidence, 4),
            "evidence": self.evidence.to_dict(),
        }


@dataclass
class LogicalBlock:
    """The primary output unit: a meaningful, explainable block of information.

    Always carries the evidence that justifies *why* its source blocks were
    grouped together, and full traceability to the original document/page.
    """

    id: str
    content_unit_id: str
    document_id: str
    source_document: str
    source_page: int
    source_block_ids: List[str]
    text: str
    structural_features: Dict[str, float] = field(default_factory=dict)
    semantic_vector: Optional[List[float]] = None
    discovered_pattern: Optional[str] = None
    role_sequence: List[str] = field(default_factory=list)
    structural_signature: str = ""
    structural_fingerprint: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: Evidence = field(default_factory=Evidence)
    group_id: Optional[str] = None
    section_group_id: Optional[str] = None
    doc_position: float = 0.0  # 0..1 relative position within document
    page_end: Optional[int] = None
    block_type: str = "content"  # "content" | "structured_record" | "heading"
    structured_fields: Optional[List[Dict[str, str]]] = None
    source_table_id: Optional[str] = None
    # Populated by the post-discovery consolidation layer when several discovered
    # logical blocks are joined into one coherent unit.
    source_logical_block_ids: Optional[List[str]] = None
    source_content_unit_ids: Optional[List[str]] = None
    consolidation_evidence: Optional[Evidence] = None

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "content_unit_id": self.content_unit_id,
            "document_id": self.document_id,
            "source_document": self.source_document,
            "source_page": self.source_page,
            "page_end": self.page_end if self.page_end is not None else self.source_page,
            "source_block_ids": list(self.source_block_ids),
            "text": self.text,
            "block_type": self.block_type,
            "structural_features": {k: round(v, 4) for k, v in self.structural_features.items()},
            "structural_fingerprint": {
                k: round(v, 4) for k, v in self.structural_fingerprint.items()
            },
            "discovered_pattern": self.discovered_pattern,
            "role_sequence": self.role_sequence,
            "structural_signature": self.structural_signature,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence.to_dict(),
            "group_id": self.group_id,
            "section_group_id": self.section_group_id,
            "doc_position": round(self.doc_position, 4),
        }
        if self.structured_fields is not None:
            d["structured_fields"] = self.structured_fields
        if self.source_table_id is not None:
            d["source_table_id"] = self.source_table_id
        if self.source_logical_block_ids is not None:
            d["source_logical_block_ids"] = list(self.source_logical_block_ids)
        if self.source_content_unit_ids is not None:
            d["source_content_unit_ids"] = list(self.source_content_unit_ids)
        if self.consolidation_evidence is not None:
            d["consolidation_evidence"] = self.consolidation_evidence.to_dict()
        if include_vector and self.semantic_vector is not None:
            d["semantic_vector"] = self.semantic_vector
        return d


@dataclass
class LogicalGroup:
    """A cross-document association of similar logical blocks.

    Members are grouped by fused semantic + structural + contextual similarity,
    never by exact text equality.
    """

    id: str
    member_block_ids: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    dominant_pattern: Optional[str] = None
    inferred_label: Optional[str] = None
    centroid: Optional[List[float]] = None
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def size(self) -> int:
        return len(self.member_block_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "size": self.size,
            "member_block_ids": list(self.member_block_ids),
            "document_ids": list(self.document_ids),
            "dominant_pattern": self.dominant_pattern,
            "inferred_label": self.inferred_label,
            "evidence": self.evidence.to_dict(),
        }


@dataclass
class SectionGroup:
    """A within-document section discovered from layout evidence.

    A section starts at a visually prominent heading-like logical block and owns
    the following content items until the next comparable heading. Sections are
    identified generically (e.g. ``…_section_003``). Optional human labels may be
    attached later and are never required for discovery.

    Nesting is evidence-driven: ``depth`` / ``parent_section_id`` /
    ``child_section_ids`` form a tree (e.g. umbrella → monthly band →
    features/fixes leaf) without vocabulary hardcoding.
    """

    id: str
    document_id: str
    heading_block_id: str
    heading_text: str
    page_start: int
    page_end: int
    member_logical_block_ids: List[str] = field(default_factory=list)
    member_source_block_ids: List[str] = field(default_factory=list)
    depth: int = 0
    parent_section_id: Optional[str] = None
    child_section_ids: List[str] = field(default_factory=list)
    inferred_label: Optional[str] = None  # optional; unused by default
    label_confidence: float = 0.0
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def item_count(self) -> int:
        return len(self.member_logical_block_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "heading_block_id": self.heading_block_id,
            "heading_text": self.heading_text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "item_count": self.item_count,
            "member_logical_block_ids": list(self.member_logical_block_ids),
            "member_source_block_ids": list(self.member_source_block_ids),
            "depth": self.depth,
            "parent_section_id": self.parent_section_id,
            "child_section_ids": list(self.child_section_ids),
            "inferred_label": self.inferred_label,
            "label_confidence": round(self.label_confidence, 4),
            "evidence": self.evidence.to_dict(),
        }
