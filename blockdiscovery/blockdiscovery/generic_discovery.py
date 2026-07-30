"""Generic, adaptive logical block discovery.

The pipeline implemented here is:

    Evidence -> Features -> Relationships -> Boundaries -> Structure -> Blocks

Nothing in this module knows any document vocabulary, section name, column name
or content category, and no regular expression participates in any structural
decision. Every threshold that separates "same block" from "new block" is
derived at run time from the document's own distribution of evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .extraction_units import RawExtractionUnit
from .logging_utils import DiscoveryLogger
from .models import (
    BoundingBox,
    ContentUnit,
    Document,
    Evidence,
    Formatting,
    LogicalBlock,
    SectionGroup,
    TextBlock,
)
from .semantics import EmbeddingBackend, cosine
from .utils import clip01

# Relationship / boundary signal weights. These describe how much each *kind*
# of evidence counts, not what any document contains.
_REL_WEIGHTS = {
    "spatial_proximity": 0.14,
    "alignment_relationship": 0.12,
    "typography_relationship": 0.14,
    "spacing_relationship": 0.10,
    "reading_order_relationship": 0.08,
    "structural_similarity": 0.12,
    "container_relationship": 0.10,
    "semantic_similarity": 0.20,
}

# Similarity of *form* between two units is only evidence that they belong to
# one block while it is surprising. Once a recurring template predicts that
# similarity, it says nothing about block membership, so these signals are
# discounted by how completely the template explains them. Signals about
# *content* continuity are never discounted — that is what stops high visual
# similarity from hiding weak semantic coherence.
_FORM_SIGNALS = (
    "spatial_proximity",
    "alignment_relationship",
    "typography_relationship",
    "spacing_relationship",
    "structural_similarity",
    "container_relationship",
)

_BOUND_WEIGHTS = {
    "semantic_transition": 0.12,
    "spatial_boundary": 0.09,
    "formatting_transition": 0.08,
    "alignment_transition": 0.05,
    "structural_transition": 0.10,
    "repetition_boundary": 0.14,
    "container_boundary": 0.12,
    "reading_order_discontinuity": 0.05,
    "prominence_onset": 0.07,
    "hierarchy_onset": 0.10,
    "layout_transition": 0.08,
}


# --------------------------------------------------------------------------- #
# Adaptive thresholding
# --------------------------------------------------------------------------- #
def otsu_threshold(values: Sequence[float], bins: int = 64) -> Optional[float]:
    """Pick the value that best separates a 1-D distribution into two classes.

    Adaptive by construction: the cut point comes from the data, so no fixed
    numeric threshold decides whether two units belong together.
    """
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 4:
        return None
    # A handful of extreme edges (page breaks, for instance) would otherwise be
    # separated as their own class and drag the cut away from the real split, so
    # the search range is winsorized to the central mass first.
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    if hi - lo < 1e-9:
        lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return None
    vals = [min(hi, max(lo, v)) for v in vals]
    hist = [0] * bins
    for v in vals:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        hist[idx] += 1
    total = len(vals)
    sum_all = sum((i + 0.5) * hist[i] for i in range(bins))
    sum_bg = 0.0
    w_bg = 0
    best_var = -1.0
    best_idx = 0
    for i in range(bins):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += (i + 0.5) * hist[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_idx = i
    return lo + (best_idx + 1) / bins * (hi - lo)


def _ratio(a: float, b: float) -> float:
    """Symmetric 0..1 similarity between two non-negative magnitudes."""
    hi = max(abs(a), abs(b))
    if hi < 1e-9:
        return 1.0
    return clip01(min(abs(a), abs(b)) / hi)


def _char_profile(text: str) -> Dict[str, float]:
    if not text:
        return {"digit_ratio": 0.0, "upper_ratio": 0.0, "punct_ratio": 0.0, "space_ratio": 0.0}
    n = len(text)
    digits = sum(1 for c in text if c.isdigit())
    uppers = sum(1 for c in text if c.isupper())
    spaces = sum(1 for c in text if c.isspace())
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return {
        "digit_ratio": digits / n,
        "upper_ratio": uppers / n,
        "punct_ratio": punct / n,
        "space_ratio": spaces / n,
    }


def _first_char_class(text: str) -> str:
    s = (text or "").lstrip()
    if not s:
        return "E"
    c = s[0]
    if c.isdigit():
        return "D"
    if c.isupper():
        return "U"
    if c.islower():
        return "L"
    return "P"


# Opening quote glyphs (straight + typographic). Used only to look past a
# leading quote when judging whether a unit starts a capitalised entry —
# no vocabulary and no regex.
_QUOTE_OPENERS = frozenset({'"', "'", "“", "‘", "„", "‹", "«"})


def _first_alpha(text: str) -> str:
    for ch in text or "":
        if ch.isalpha():
            return ch
    return ""


def _starts_capitalised_unit(text: str) -> bool:
    """True when the first letter (ignoring leading quotes/space) is uppercase."""
    fa = _first_alpha(text)
    return bool(fa) and fa.isupper()


def _opens_quoted_capital(text: str) -> bool:
    """Quoted capitalised entry opener (definition / glossary style lines)."""
    s = (text or "").lstrip()
    if not s or s[0] not in _QUOTE_OPENERS:
        return False
    return _starts_capitalised_unit(s)


def _split_bullet_parts(text: str, bullet_chars: frozenset) -> List[str]:
    """Split a cell on bullet glyphs into nested item parts (no regex)."""
    if not text or not any(ch in bullet_chars for ch in text):
        return [text] if text is not None else []
    parts: List[str] = []
    buf: List[str] = []
    for ch in text:
        if ch in bullet_chars:
            piece = " ".join("".join(buf).split())
            if piece:
                parts.append(piece)
            buf = []
            continue
        buf.append(ch)
    tail = " ".join("".join(buf).split())
    if tail:
        parts.append(tail)
    return parts or [text]


# --------------------------------------------------------------------------- #
# Candidate units
# --------------------------------------------------------------------------- #
@dataclass
class CandidateUnit:
    """A normalized candidate. Not yet a logical block."""

    id: str
    document_id: str
    page_number: int
    order: int
    text: str
    raw_unit_ids: List[str]
    block_ids: List[str] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    layout_class: str = "unknown"
    cells: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    grid_id: Optional[str] = None
    grid_row_index: Optional[int] = None
    marker_depth: int = 0
    features: Dict[str, float] = field(default_factory=dict)
    shape_signature: str = ""
    repetition_count: int = 1
    semantic_vector: Optional[np.ndarray] = None
    refinement: str = "PRESERVE"
    refinement_reason: str = ""


@dataclass
class BoundaryDecision:
    unit_a: str
    unit_b: str
    page_a: int
    page_b: int
    relationship_signals: Dict[str, float]
    boundary_signals: Dict[str, float]
    relationship_score: float
    boundary_score: float
    net: float
    decision: str
    confidence: float
    reason: str


@dataclass
class GridSchema:
    grid_id: str
    page_number: int
    row_count: int
    column_count: int
    structure_type: str
    column_signatures: List[Dict[str, float]]
    row_signatures: List[str]
    geometry: Dict[str, float]
    repetition_evidence: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "page_number": self.page_number,
            "structure_type": self.structure_type,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_signatures": self.column_signatures,
            "row_signatures": self.row_signatures,
            "geometry": self.geometry,
            "repetition_evidence": self.repetition_evidence,
        }


@dataclass
class DiscoveryTrace:
    """Everything the human-readable log needs, captured as evidence."""

    candidates: List[CandidateUnit] = field(default_factory=list)
    boundaries: List[BoundaryDecision] = field(default_factory=list)
    grids: List[GridSchema] = field(default_factory=list)
    refinements: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    graph_nodes: int = 0
    graph_edges: int = 0
    adaptive_thresholds: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    heading_contexts: List[Dict[str, Any]] = field(default_factory=list)
    over_grouping: List[Dict[str, Any]] = field(default_factory=list)


class GenericDiscoveryEngine:
    """Evidence-driven discovery over backend-neutral raw extraction units."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        logger: Optional[DiscoveryLogger] = None,
    ) -> None:
        self.backend = backend
        self.logger = logger

    # ------------------------------------------------------------------ #
    def run(
        self,
        document: Document,
        raw_units: List[RawExtractionUnit],
    ) -> Tuple[List[ContentUnit], List[LogicalBlock], List[SectionGroup], DiscoveryTrace]:
        trace = DiscoveryTrace()
        blocks = self._materialize_blocks(document, raw_units)
        candidates = self._build_candidates(document, raw_units, blocks)
        if not candidates:
            return [], [], [], trace

        self._extract_features(document, candidates)
        self._discover_repetition(candidates)
        self._embed(candidates)
        trace.grids = self._discover_grid_schemas(candidates)
        self._announce_candidates(candidates)
        self._reject_page_furniture(candidates, document, trace)

        active = [c for c in candidates if c.refinement != "REJECT"]
        boundaries, thresholds = self._analyze(active, trace)
        trace.candidates = candidates
        trace.boundaries = boundaries
        trace.adaptive_thresholds = thresholds
        trace.graph_nodes = len(active)
        trace.graph_edges = len(boundaries)

        groups = self._partition(active, boundaries)
        units, logical_blocks = self._refine_and_build(document, groups, trace)
        sections = self._discover_contexts(document, logical_blocks, candidates, trace)

        trace.stats = {
            "raw_units": len(raw_units),
            "candidates": len(candidates),
            "rejected": len(trace.rejected),
            "boundaries_evaluated": len(boundaries),
            "boundaries_detected": sum(
                1 for b in boundaries if b.decision == "START_NEW_LOGICAL_BLOCK"
            ),
            "continued": sum(1 for b in boundaries if b.decision == "CONTINUE_LOGICAL_BLOCK"),
            "content_units": len(units),
            "logical_blocks": len(logical_blocks),
            "grids": len(trace.grids),
            "merges": sum(1 for r in trace.refinements if r["decision"] == "MERGE"),
            "splits": sum(1 for r in trace.refinements if r["decision"] == "SPLIT"),
            "preserves": sum(1 for r in trace.refinements if r["decision"] == "PRESERVE"),
            "low_confidence": sum(1 for b in logical_blocks if b.confidence < 0.5),
        }
        return units, logical_blocks, sections, trace

    # ------------------------------------------------------------------ #
    def _materialize_blocks(
        self, document: Document, raw_units: List[RawExtractionUnit]
    ) -> Dict[str, TextBlock]:
        """Give every raw unit a TextBlock so provenance reaches the PDF page."""
        blocks: Dict[str, TextBlock] = {}
        page_index = {p.page_number: p for p in document.pages}
        for i, ru in enumerate(raw_units, start=1):
            x0, y0, x1, y1 = ru.bbox
            height = max(1.0, y1 - y0)
            approx_size = height / max(1, ru.line_count)
            tb = TextBlock(
                id=f"{document.id}_block_{i:04d}",
                document_id=document.id,
                page_number=ru.page_number,
                text=ru.text,
                bounding_box=BoundingBox(x0, y0, x1, y1),
                formatting=Formatting(
                    dominant_size=approx_size,
                    max_size=approx_size,
                    min_size=approx_size,
                    dominant_font=ru.backend,
                    bold_ratio=0.0,
                    italic_ratio=0.0,
                    monospace_ratio=0.0,
                    color=0,
                    size_variety=1,
                ),
                reading_order=i,
                block_type="text",
                raw_block_no=i,
            )
            blocks[ru.id] = tb
            document.blocks.append(tb)
            page = page_index.get(ru.page_number)
            if page is not None:
                page.block_ids.append(tb.id)
        return blocks

    def _build_candidates(
        self,
        document: Document,
        raw_units: List[RawExtractionUnit],
        blocks: Dict[str, TextBlock],
    ) -> List[CandidateUnit]:
        """Normalize raw units into candidates, splitting only on evidence."""
        candidates: List[CandidateUnit] = []
        counter = 0
        for ru in raw_units:
            tb = blocks[ru.id]
            parts = self._internal_split_parts(ru)
            if len(parts) <= 1:
                counter += 1
                candidates.append(
                    CandidateUnit(
                        id=f"{document.id}_candidate_{counter:04d}",
                        document_id=document.id,
                        page_number=ru.page_number,
                        order=counter,
                        text=ru.text,
                        raw_unit_ids=[ru.id],
                        block_ids=[tb.id],
                        bbox=ru.bbox,
                        layout_class=ru.layout_class,
                        cells=list(ru.cells),
                        lines=list(ru.lines) or [ru.text],
                        grid_id=ru.grid_id,
                        grid_row_index=ru.grid_row_index,
                        marker_depth=ru.marker_depth,
                    )
                )
                continue
            # Evidence said this extraction unit holds several enumerated items.
            x0, y0, x1, y1 = ru.bbox
            slice_h = (y1 - y0) / len(parts)
            for pi, part in enumerate(parts):
                counter += 1
                candidates.append(
                    CandidateUnit(
                        id=f"{document.id}_candidate_{counter:04d}",
                        document_id=document.id,
                        page_number=ru.page_number,
                        order=counter,
                        text=part,
                        raw_unit_ids=[ru.id],
                        block_ids=[tb.id],
                        bbox=(x0, y0 + pi * slice_h, x1, y0 + (pi + 1) * slice_h),
                        layout_class=ru.layout_class,
                        cells=[],
                        lines=[part],
                        grid_id=ru.grid_id,
                        grid_row_index=ru.grid_row_index,
                        marker_depth=ru.marker_depth,
                    )
                )
        return candidates

    @staticmethod
    def _internal_split_parts(ru: RawExtractionUnit) -> List[str]:
        """Detect repeated enumerated items inside one extraction unit.

        Purely character-class evidence: a run of lines that each open with the
        same non-alphabetic opener class is a repeated item series, whatever the
        words are.
        """
        lines = [ln for ln in ru.lines if ln.strip()]
        if ru.grid_id is not None or len(lines) < 2:
            return []
        openers = [_first_char_class(ln) for ln in lines]
        enumerated = sum(1 for o in openers if o == "D")
        if enumerated < 2 or enumerated < len(lines) * 0.6:
            return []
        # Uniform-ish item lengths reinforce "series of items" over "wrapped prose".
        lengths = [len(ln) for ln in lines]
        mean_len = sum(lengths) / len(lengths)
        if mean_len < 1:
            return []
        spread = (max(lengths) - min(lengths)) / mean_len
        if spread > 2.5:
            return []
        return lines

    # ------------------------------------------------------------------ #
    def _extract_features(self, document: Document, candidates: List[CandidateUnit]) -> None:
        page_w = float(document.stats.get("page_width", 612.0) or 612.0)
        page_h = float(document.stats.get("page_height", 792.0) or 792.0)

        heights = []
        for c in candidates:
            x0, y0, x1, y1 = c.bbox
            heights.append(max(1.0, (y1 - y0) / max(1, len(c.lines))))
        median_line_h = float(np.median(heights)) if heights else 12.0

        # Vertical gaps to the previous candidate on the same page.
        gaps: List[float] = []
        for i, c in enumerate(candidates):
            if i == 0 or candidates[i - 1].page_number != c.page_number:
                gaps.append(0.0)
                continue
            gaps.append(max(0.0, c.bbox[1] - candidates[i - 1].bbox[3]))
        positive = [g for g in gaps if g > 0]
        median_gap = float(np.median(positive)) if positive else 1.0

        for i, c in enumerate(candidates):
            x0, y0, x1, y1 = c.bbox
            width = max(0.0, x1 - x0)
            height = max(0.0, y1 - y0)
            line_h = height / max(1, len(c.lines))
            prof = _char_profile(c.text)
            tokens = c.text.split()
            c.features = {
                "rel_x0": clip01(x0 / page_w),
                "rel_x1": clip01(x1 / page_w),
                "rel_width": clip01(width / page_w),
                "rel_y0": clip01(y0 / page_h),
                "rel_height": clip01(height / page_h),
                "line_height_ratio": line_h / max(1e-6, median_line_h),
                "gap_above_ratio": gaps[i] / max(1e-6, median_gap),
                "char_count": float(len(c.text)),
                "token_count": float(len(tokens)),
                "line_count": float(len(c.lines)),
                "cell_count": float(len(c.cells)),
                "marker_depth": float(c.marker_depth),
                "in_grid": 1.0 if c.grid_id else 0.0,
                "page_fraction": c.page_number / max(1, document.page_count),
                "terminal_punctuation": 1.0 if c.text.rstrip().endswith((".", "?", "!")) else 0.0,
                "layout_section_header": 1.0 if c.layout_class == "section-header" else 0.0,
                "layout_page_header": 1.0 if c.layout_class == "page-header" else 0.0,
                "layout_list_item": 1.0 if c.layout_class == "list-item" else 0.0,
                **prof,
            }
            c.features["prominence"] = clip01(
                0.5 * clip01((c.features["line_height_ratio"] - 0.9) / 0.8)
                + 0.3 * clip01(c.features["marker_depth"] / 3.0)
                + 0.2 * (1.0 - clip01(c.features["char_count"] / 400.0))
            )
            c.shape_signature = self._shape_signature(c)

    @staticmethod
    def _shape_signature(c: CandidateUnit) -> str:
        """Structure-only signature: geometry and topology, never content."""
        f = c.features
        x_bucket = int(f.get("rel_x0", 0.0) * 8)
        w_bucket = int(f.get("rel_width", 0.0) * 8)
        h_bucket = min(5, int(f.get("line_height_ratio", 1.0) * 2))
        cells = int(f.get("cell_count", 0))
        depth = int(f.get("marker_depth", 0))
        line_bucket = min(4, int(f.get("line_count", 1.0)))
        return (
            f"L:{c.layout_class}|C:{cells}|X:{x_bucket}|W:{w_bucket}"
            f"|H:{h_bucket}|D:{depth}|R:{line_bucket}"
        )

    def _discover_repetition(self, candidates: List[CandidateUnit]) -> None:
        counts: Dict[str, int] = {}
        for c in candidates:
            counts[c.shape_signature] = counts.get(c.shape_signature, 0) + 1
        for c in candidates:
            c.repetition_count = counts[c.shape_signature]
            c.features["repetition_count"] = float(c.repetition_count)
            c.features["is_repeated_shape"] = 1.0 if c.repetition_count >= 3 else 0.0

    def _embed(self, candidates: List[CandidateUnit]) -> None:
        texts = [c.text for c in candidates]
        if not texts:
            return
        vectors = self.backend.embed(texts)
        for c, v in zip(candidates, vectors):
            c.semantic_vector = v

    # ------------------------------------------------------------------ #
    def _discover_grid_schemas(self, candidates: List[CandidateUnit]) -> List[GridSchema]:
        """Describe every grid purely from geometry, cell counts and repetition."""
        by_grid: Dict[str, List[CandidateUnit]] = {}
        for c in candidates:
            if c.grid_id:
                by_grid.setdefault(c.grid_id, []).append(c)

        schemas: List[GridSchema] = []
        for grid_id, rows in sorted(by_grid.items()):
            rows.sort(key=lambda r: r.order)
            counts = [len(r.cells) for r in rows]
            column_count = max(set(counts), key=counts.count) if counts else 0
            consistent = sum(1 for n in counts if n == column_count) / max(1, len(counts))

            column_signatures: List[Dict[str, float]] = []
            for col in range(column_count):
                col_cells = [r.cells[col] for r in rows if col < len(r.cells)]
                if not col_cells:
                    continue
                lengths = [len(v) for v in col_cells]
                profiles = [_char_profile(v) for v in col_cells]
                column_signatures.append(
                    {
                        "column_position": float(col),
                        "mean_char_count": round(float(np.mean(lengths)), 2),
                        "char_count_stability": round(
                            1.0 - clip01(float(np.std(lengths)) / max(1.0, float(np.mean(lengths)))),
                            3,
                        ),
                        "mean_digit_ratio": round(
                            float(np.mean([p["digit_ratio"] for p in profiles])), 3
                        ),
                        "mean_upper_ratio": round(
                            float(np.mean([p["upper_ratio"] for p in profiles])), 3
                        ),
                        "fill_ratio": round(
                            sum(1 for v in col_cells if v.strip()) / len(col_cells), 3
                        ),
                    }
                )

            row_signatures = sorted({r.shape_signature for r in rows})
            xs = [r.bbox[0] for r in rows]
            widths = [r.bbox[2] - r.bbox[0] for r in rows]
            row_heights = [r.bbox[3] - r.bbox[1] for r in rows]
            geometry = {
                "left_edge_stability": round(
                    1.0 - clip01(float(np.std(xs)) / max(1.0, abs(float(np.mean(xs))))), 3
                ),
                "width_stability": round(
                    1.0 - clip01(float(np.std(widths)) / max(1.0, float(np.mean(widths)))), 3
                ),
                "row_height_stability": round(
                    1.0 - clip01(float(np.std(row_heights)) / max(1.0, float(np.mean(row_heights)))),
                    3,
                ),
                "mean_row_height": round(float(np.mean(row_heights)), 2),
            }
            repetition = {
                "row_count": float(len(rows)),
                "column_consistency": round(consistent, 3),
                "distinct_row_shapes": float(len(row_signatures)),
                "shape_repetition": round(
                    float(np.mean([r.repetition_count for r in rows])), 2
                ),
            }
            header_rows = self._header_like_rows(rows)
            structure_type = (
                "repeated_grid"
                if len(rows) >= 2 and consistent >= 0.6 and len(row_signatures) <= max(2, len(rows) // 2)
                else ("grid" if len(rows) >= 2 else "single_row_structure")
            )
            repetition["header_like_rows"] = float(len(header_rows))
            schema = GridSchema(
                grid_id=grid_id,
                page_number=rows[0].page_number,
                row_count=len(rows),
                column_count=column_count,
                structure_type=structure_type,
                column_signatures=column_signatures,
                row_signatures=row_signatures,
                geometry=geometry,
                repetition_evidence=repetition,
            )
            schemas.append(schema)
            for r in rows:
                r.features["grid_structure_repeated"] = (
                    1.0 if structure_type == "repeated_grid" else 0.0
                )
                r.features["grid_column_consistency"] = consistent
            if self.logger:
                self.logger.event(
                    "table_structure_discovered",
                    document_id=rows[0].document_id,
                    grid_id=grid_id,
                    page=rows[0].page_number,
                    structure_type=structure_type,
                    column_count=column_count,
                    row_count=len(rows),
                    geometry=geometry,
                    repetition_evidence=repetition,
                    evidence="discovered from cell counts, row geometry and shape repetition",
                )
        return schemas

    def _announce_candidates(self, candidates: List[CandidateUnit]) -> None:
        """Emit one machine-readable record per candidate, before any grouping."""
        if not self.logger:
            return
        for c in candidates:
            self.logger.event(
                "candidate_unit_created",
                document_id=c.document_id,
                page=c.page_number,
                source_ids=c.raw_unit_ids,
                candidate_id=c.id,
                block_ids=c.block_ids,
                layout_class=c.layout_class,
                shape_signature=c.shape_signature,
                repetition_count=c.repetition_count,
                decision="CANDIDATE",
                confidence=round(
                    clip01(
                        0.6
                        + 0.2 * clip01(c.features.get("char_count", 0.0) / 60.0)
                        + 0.2 * clip01(c.features.get("cell_count", 0.0) / 3.0)
                    ),
                    4,
                ),
                evidence={k: round(v, 4) for k, v in c.features.items()},
                reason="Normalized from one extraction unit; not yet a logical block.",
            )
            if c.grid_id:
                self.logger.event(
                    "structured_record_created",
                    document_id=c.document_id,
                    page=c.page_number,
                    source_ids=c.raw_unit_ids,
                    candidate_id=c.id,
                    grid_id=c.grid_id,
                    row_index=c.grid_row_index,
                    field_count=len([x for x in c.cells if x.strip()]),
                    header_like=bool(c.features.get("grid_header_like", 0.0)),
                    decision="STRUCTURED_RECORD_CANDIDATE",
                    confidence=round(
                        clip01(0.5 + 0.5 * c.features.get("grid_column_consistency", 0.0)), 4
                    ),
                    evidence={
                        "cell_count": c.features.get("cell_count", 0.0),
                        "grid_structure_repeated": c.features.get(
                            "grid_structure_repeated", 0.0
                        ),
                    },
                    reason="Row position within a structure discovered from geometry.",
                )

    @staticmethod
    def _header_like_rows(rows: List[CandidateUnit]) -> List[CandidateUnit]:
        """Find label rows as content-shape outliers against the rest of the grid.

        A header row is recognised because its cells look unlike the cells below
        them — shorter, more title-cased, fewer digits — not because of what any
        cell says. Rows are only considered at the top of the structure, which is
        where a label row can be.
        """
        if len(rows) < 3:
            return []

        def profile(row: CandidateUnit) -> np.ndarray:
            cells = [c for c in row.cells if c.strip()] or [row.text]
            prof = [_char_profile(c) for c in cells]
            return np.array(
                [
                    float(np.mean([len(c) for c in cells])) / 40.0,
                    float(np.mean([p["digit_ratio"] for p in prof])),
                    float(np.mean([p["upper_ratio"] for p in prof])),
                    float(np.mean([p["punct_ratio"] for p in prof])),
                ],
                dtype=np.float32,
            )

        vectors = [profile(r) for r in rows]
        headers: List[CandidateUnit] = []
        for idx in (0,):
            others = vectors[idx + 1 :]
            if len(others) < 2:
                break
            baseline = np.median(np.vstack(others), axis=0)
            spread = np.median(np.abs(np.vstack(others) - baseline), axis=0)
            scale = np.maximum(spread, 0.05)
            deviation = float(np.mean(np.abs(vectors[idx] - baseline) / scale))
            if deviation >= 3.0:
                rows[idx].features["grid_header_like"] = 1.0
                headers.append(rows[idx])
        return headers

    # ------------------------------------------------------------------ #
    def _reject_page_furniture(
        self,
        candidates: List[CandidateUnit],
        document: Document,
        trace: DiscoveryTrace,
    ) -> None:
        """Drop repeated running elements discovered by cross-page repetition.

        Navigation crumbs, running headers and footers are found because the
        *same short text* recurs at the *same relative height* on several pages
        — never because of a list of known strings.
        """
        if document.page_count < 3:
            return
        buckets: Dict[Tuple[str, int], List[CandidateUnit]] = {}
        for c in candidates:
            key = (c.text.strip().casefold(), int(c.features.get("rel_y0", 0.0) * 12))
            buckets.setdefault(key, []).append(c)

        page_threshold = max(3, int(document.page_count * 0.3))
        for (_, _), members in buckets.items():
            pages = {m.page_number for m in members}
            if len(pages) < page_threshold:
                continue
            if not members or members[0].features.get("char_count", 0.0) > 80:
                continue
            for m in members:
                m.refinement = "REJECT"
                m.refinement_reason = (
                    f"Identical short content recurs on {len(pages)} pages at the same "
                    "relative page height — running page furniture, not content."
                )
                trace.rejected.append(
                    {
                        "candidate_id": m.id,
                        "page": m.page_number,
                        "text": m.text[:80],
                        "decision": "REJECT",
                        "evidence": {
                            "pages_with_same_content": len(pages),
                            "page_threshold": page_threshold,
                            "char_count": m.features.get("char_count", 0.0),
                            "relative_height_band": round(m.features.get("rel_y0", 0.0), 3),
                        },
                        "reason": m.refinement_reason,
                    }
                )
                if self.logger:
                    self.logger.event(
                        "content_unit_rejected",
                        document_id=m.document_id,
                        page=m.page_number,
                        source_ids=[m.id],
                        decision="REJECT",
                        confidence=round(clip01(len(pages) / max(1, document.page_count)), 4),
                        evidence={"repeat_pages": len(pages), "threshold": page_threshold},
                        reason=m.refinement_reason,
                    )

    # ------------------------------------------------------------------ #
    def _relationship(self, a: CandidateUnit, b: CandidateUnit) -> Dict[str, float]:
        fa, fb = a.features, b.features
        gap_ratio = fb.get("gap_above_ratio", 1.0)
        same_page = a.page_number == b.page_number

        spatial = clip01(1.0 / (1.0 + max(0.0, gap_ratio - 1.0))) if same_page else 0.0
        alignment = clip01(1.0 - abs(fa.get("rel_x0", 0.0) - fb.get("rel_x0", 0.0)) * 4.0)
        width_sim = _ratio(fa.get("rel_width", 0.0), fb.get("rel_width", 0.0))
        typography = clip01(
            0.6 * _ratio(fa.get("line_height_ratio", 1.0), fb.get("line_height_ratio", 1.0))
            + 0.2 * (1.0 if a.layout_class == b.layout_class else 0.0)
            + 0.2 * (1.0 if a.marker_depth == b.marker_depth else 0.0)
        )
        spacing = clip01(1.0 - clip01((gap_ratio - 1.0) / 3.0)) if same_page else 0.0
        order_rel = 1.0 if (same_page and b.order == a.order + 1) else (0.5 if same_page else 0.0)
        structural = clip01(
            0.5 * (1.0 if a.shape_signature == b.shape_signature else 0.0)
            + 0.3 * _ratio(fa.get("cell_count", 0.0), fb.get("cell_count", 0.0))
            + 0.2 * width_sim
        )
        container = 1.0 if (a.grid_id and a.grid_id == b.grid_id) else 0.0
        if a.semantic_vector is not None and b.semantic_vector is not None:
            semantic = clip01(max(0.0, cosine(a.semantic_vector, b.semantic_vector)))
        else:
            semantic = 0.0
        return {
            "spatial_proximity": spatial,
            "alignment_relationship": alignment,
            "typography_relationship": typography,
            "spacing_relationship": spacing,
            "reading_order_relationship": order_rel,
            "structural_similarity": structural,
            "container_relationship": container,
            "semantic_similarity": semantic,
        }

    @staticmethod
    def _form_discount(a: CandidateUnit, b: CandidateUnit) -> Dict[str, float]:
        """How much of this pair's *form* similarity carries no membership news.

        Two situations make matching geometry and typography uninformative:

        * a recurring template already predicts the match, so both units are
          simply instances of the same shape, and
        * the units sit in different structural containers, where a shared left
          edge or line height is a page-layout artefact rather than cohesion.

        Both are discovered from shape repetition and container topology; the
        text is never consulted.
        """
        template = 0.0
        if (
            a.shape_signature == b.shape_signature
            and a.repetition_count >= 3
            and b.repetition_count >= 3
        ):
            template = clip01(
                0.55 + 0.45 * clip01((min(a.repetition_count, b.repetition_count) - 3) / 8.0)
            )
        if a.grid_id and a.grid_id == b.grid_id and a.grid_row_index != b.grid_row_index:
            template = max(
                template,
                clip01(0.45 + 0.55 * a.features.get("grid_structure_repeated", 0.0)),
            )
        # A label row and a data row share a grid's geometry by construction, so
        # that shared geometry says nothing about them being one record.
        header_split = 0.0
        if a.features.get("grid_header_like", 0.0) != b.features.get("grid_header_like", 0.0):
            header_split = 0.8
        container_change = 0.0 if a.grid_id == b.grid_id else 0.7
        return {
            "template": max(template, header_split),
            "container_change": container_change,
            "total": max(template, container_change, header_split),
        }

    def _boundary(
        self,
        a: CandidateUnit,
        b: CandidateUnit,
        rel: Dict[str, float],
        discount: Dict[str, float],
    ) -> Dict[str, float]:
        same_page = a.page_number == b.page_number
        gap_ratio = b.features.get("gap_above_ratio", 1.0)

        # A rise in prominence is asymmetric evidence: the second unit opens
        # something, whereas a fall merely continues under what came before.
        # Kept separate from the extractor's measured heading level, because a
        # font-size bump inside a stanza and an actual level change are
        # different observations and must be allowed to earn different weights.
        onset = clip01(
            max(0.0, b.features.get("prominence", 0.0) - a.features.get("prominence", 0.0)) * 2.5
        )
        hierarchy_onset = (
            1.0
            if b.marker_depth > 0
            and (a.marker_depth == 0 or b.marker_depth <= a.marker_depth)
            else 0.0
        )

        # Extractor layout class is geometry/typography evidence. A change of
        # class (heading ↔ body ↔ list ↔ table) is a real structural boundary.
        # Distinct list items are also separate content units by construction.
        layout_transition = 0.0
        if a.layout_class != b.layout_class:
            layout_transition = 0.85
        # Each list item is its own unit — including consecutive list items.
        if a.layout_class == "list-item" or b.layout_class == "list-item":
            layout_transition = max(layout_transition, 0.95)
        # Consecutive section headers are distinct heads, not one blob.
        if a.layout_class == "section-header" and b.layout_class == "section-header":
            layout_transition = max(layout_transition, 0.95)
        if (a.grid_id is None) != (b.grid_id is None):
            layout_transition = max(layout_transition, 0.90)

        return {
            "semantic_transition": clip01(1.0 - rel["semantic_similarity"]),
            "spatial_boundary": clip01((gap_ratio - 1.0) / 2.5) if same_page else 1.0,
            "formatting_transition": clip01(1.0 - rel["typography_relationship"]),
            "alignment_transition": clip01(1.0 - rel["alignment_relationship"]),
            "structural_transition": clip01(1.0 - rel["structural_similarity"]),
            "repetition_boundary": discount["template"],
            "container_boundary": 0.0
            if (a.grid_id and a.grid_id == b.grid_id)
            else (1.0 if (a.grid_id or b.grid_id) else 0.3),
            "reading_order_discontinuity": 0.0 if same_page else 1.0,
            "prominence_onset": onset,
            "hierarchy_onset": hierarchy_onset,
            "layout_transition": layout_transition,
        }

    @staticmethod
    def _relationship_score(
        rel: Dict[str, float],
        discount: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Weighted relationship score with uninformative form similarity removed."""
        weights = weights or _REL_WEIGHTS
        total = 0.0
        for key, weight in weights.items():
            value = rel[key]
            if key in _FORM_SIGNALS:
                value *= 1.0 - discount
            total += value * weight
        return total

    @staticmethod
    def _informative_weights(
        base: Dict[str, float],
        observations: List[Dict[str, float]],
        discounts: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Rescale weights by each signal's spread across this document.

        Evidence only discriminates to the extent that it differs between edges.
        A signal that reads the same everywhere is uninformative here, whatever
        it measures, so its share of the budget is handed to signals that vary.
        The total weight budget is preserved so scores stay comparable.
        """
        if len(observations) < 3:
            return dict(base)

        spreads: Dict[str, float] = {}
        for key in base:
            series = []
            for i, obs in enumerate(observations):
                value = obs[key]
                if discounts is not None and key in _FORM_SIGNALS:
                    value *= 1.0 - discounts[i]
                series.append(value)
            spreads[key] = float(np.std(series))

        widest = max(spreads.values()) if spreads else 0.0
        if widest < 1e-6:
            return dict(base)

        # Square root keeps a moderately varying signal relevant rather than
        # letting the single most variable one take the whole budget.
        scaled = {k: base[k] * math.sqrt(spreads[k] / widest) for k in base}
        budget = sum(base.values())
        total = sum(scaled.values())
        if total < 1e-9:
            return dict(base)
        return {k: v * budget / total for k, v in scaled.items()}

    def _analyze(
        self, candidates: List[CandidateUnit], trace: Optional[DiscoveryTrace] = None
    ) -> Tuple[List[BoundaryDecision], Dict[str, float]]:
        """Build the weighted relationship graph and cut it adaptively."""
        pairs: List[
            Tuple[CandidateUnit, CandidateUnit, Dict[str, float], Dict[str, float], float]
        ] = []
        for i in range(1, len(candidates)):
            a, b = candidates[i - 1], candidates[i]
            rel = self._relationship(a, b)
            discount = self._form_discount(a, b)
            bnd = self._boundary(a, b, rel, discount)
            pairs.append((a, b, rel, bnd, discount))

        # A signal that never varies in this document cannot separate anything,
        # so its weight is scaled by how much it actually moves here. This is
        # what stops a uniformly left-aligned layout from voting on every edge.
        rel_weights = self._informative_weights(
            _REL_WEIGHTS, [p[2] for p in pairs], discounts=[p[4]["total"] for p in pairs]
        )
        bnd_weights = self._informative_weights(_BOUND_WEIGHTS, [p[3] for p in pairs])

        nets: List[float] = []
        for _, _, rel, bnd, discount in pairs:
            r = self._relationship_score(rel, discount["total"], rel_weights)
            s = sum(bnd[k] * w for k, w in bnd_weights.items())
            nets.append(r - s)

        cut = otsu_threshold(nets)
        if cut is None:
            cut = float(np.median(nets)) if nets else 0.0
        thresholds = {
            "edge_cut_net_score": round(float(cut), 4),
            "net_score_mean": round(float(np.mean(nets)), 4) if nets else 0.0,
            "net_score_std": round(float(np.std(nets)), 4) if nets else 0.0,
            "derivation": "1-D maximum-variance separation over this document's own edge scores",
        }
        thresholds.update(
            {f"weight_relationship.{k}": round(v, 4) for k, v in rel_weights.items()}
        )
        thresholds.update({f"weight_boundary.{k}": round(v, 4) for k, v in bnd_weights.items()})

        decisions: List[BoundaryDecision] = []
        for (a, b, rel, bnd, discount), net in zip(pairs, nets):
            explained = discount["total"]
            r = self._relationship_score(rel, explained, rel_weights)
            s = sum(bnd[k] * w for k, w in bnd_weights.items())
            rel = {
                **rel,
                "form_discount_total": round(explained, 3),
                "form_discount_template": round(discount["template"], 3),
                "form_discount_container_change": round(discount["container_change"], 3),
            }
            # Decisive structural evidence forces a new block even when the
            # adaptive cut would otherwise continue. This prevents over-grouping
            # of headings, list items and table rows into one prose blob.
            forced_split = False
            force_reason = ""
            if bnd.get("hierarchy_onset", 0.0) >= 0.95:
                forced_split = True
                force_reason = "Heading-level onset from formatting evidence."
            elif bnd.get("layout_transition", 0.0) >= 0.90:
                forced_split = True
                force_reason = (
                    f"Layout class transition {a.layout_class!r} → {b.layout_class!r}."
                    if a.layout_class != b.layout_class
                    else "List-item atomic boundary."
                )
            elif bnd.get("repetition_boundary", 0.0) >= 0.70 and (
                a.grid_id and a.grid_id == b.grid_id
            ):
                forced_split = True
                force_reason = "Repeated grid-row template boundary."
            elif (
                a.layout_class == "list-item"
                or b.layout_class == "list-item"
                or a.layout_class == "section-header"
                or b.layout_class == "section-header"
                or a.layout_class == "page-header"
                or b.layout_class == "page-header"
            ) and a.layout_class != b.layout_class:
                forced_split = True
                force_reason = "Atomic layout role change (list/heading)."
            elif (
                (not a.grid_id)
                and (not b.grid_id)
                and _opens_quoted_capital(a.text)
                and _opens_quoted_capital(b.text)
            ):
                # Parallel quoted capitalised entries (definition / glossary
                # style) share form but are independent items.
                forced_split = True
                force_reason = "Parallel quoted term entries stay atomic."
            elif (
                (not a.grid_id)
                and a.text.rstrip().endswith((".", "?", "!"))
                and bool(b.text.strip())
                and _starts_capitalised_unit(b.text)
            ):
                # Finished prose unit followed by a new capitalised unit —
                # keep as separate items (anti-over-grouping). Look past a
                # leading quote so “ Term ”… lines are not glued together.
                forced_split = True
                force_reason = "Finished sentence followed by a new capitalised unit."
            elif (
                (not a.grid_id)
                and (not b.grid_id)
                and a.layout_class in {"text", "section-header", "page-header", None, ""}
                and b.layout_class in {"text", "section-header", "page-header", None, ""}
            ):
                # Adjacent compact title-like fragments stay atomic so nested
                # section hierarchy can open each one (release/band/leaf titles).
                at = (a.text or "").strip()
                bt = (b.text or "").strip()
                aw = len(at.split())
                bw = len(bt.split())
                a_title = (
                    0 < aw <= 14
                    and len(at) <= 120
                    and not at.endswith((".", "?", "!", ";", ":"))
                    and _starts_capitalised_unit(at)
                )
                b_title = (
                    0 < bw <= 14
                    and len(bt) <= 120
                    and not bt.endswith((".", "?", "!", ";", ":"))
                    and _starts_capitalised_unit(bt)
                )
                if a_title and (b_title or (bw <= 20 and _starts_capitalised_unit(bt))):
                    forced_split = True
                    force_reason = "Adjacent title-like units stay atomic."
            elif (
                (not a.grid_id)
                and a.text.rstrip().endswith(":")
                and bool(b.text.strip())
                and (
                    _starts_capitalised_unit(b.text)
                    or b.text.lstrip().startswith("```")
                    or b.layout_class != a.layout_class
                )
            ):
                forced_split = True
                force_reason = "Intro/label line followed by a distinct unit."
            elif (not a.grid_id) and (
                a.text.lstrip().startswith("```") or b.text.lstrip().startswith("```")
            ):
                forced_split = True
                force_reason = "Code/example fence boundary."

            split = forced_split or (net < cut)
            spread = abs(net - cut)
            confidence = clip01(0.5 + spread * 2.0)
            if forced_split:
                confidence = max(confidence, 0.85)
            top = max(bnd.items(), key=lambda kv: kv[1] * bnd_weights.get(kv[0], 0.0))
            top_rel = max(
                ((k, v) for k, v in rel.items() if k in rel_weights),
                key=lambda kv: kv[1] * rel_weights[kv[0]],
            )
            if split:
                reason = (
                    force_reason
                    if forced_split
                    else (
                        f"Boundary evidence dominates; strongest signal '{top[0]}'={top[1]:.2f} "
                        f"against relationship score {r:.2f}."
                    )
                )
                if (not forced_split) and explained >= 0.4:
                    cause = (
                        "a recurring template already predicts it"
                        if discount["template"] >= discount["container_change"]
                        else "the two units sit in different structural containers"
                    )
                    reason += (
                        f" Form similarity was discounted by {explained:.2f} because "
                        f"{cause}."
                    )
            else:
                reason = (
                    f"Relationship evidence dominates; strongest signal "
                    f"'{top_rel[0]}'={top_rel[1]:.2f} against boundary score {s:.2f}."
                )
            decision = BoundaryDecision(
                unit_a=a.id,
                unit_b=b.id,
                page_a=a.page_number,
                page_b=b.page_number,
                relationship_signals=rel,
                boundary_signals=bnd,
                relationship_score=r,
                boundary_score=s,
                net=net,
                decision="START_NEW_LOGICAL_BLOCK" if split else "CONTINUE_LOGICAL_BLOCK",
                confidence=confidence,
                reason=reason,
            )
            decisions.append(decision)

            raw_form = float(np.mean([rel[k] for k in _FORM_SIGNALS]))
            if trace is not None and raw_form >= 0.75 and rel["semantic_similarity"] < 0.25:
                trace.over_grouping.append(
                    {
                        "unit_a": a.id,
                        "unit_b": b.id,
                        "page": b.page_number,
                        "mean_form_similarity": round(raw_form, 3),
                        "semantic_similarity": round(rel["semantic_similarity"], 3),
                        "form_discount_applied": round(explained, 3),
                        "decision": decision.decision,
                        "resolved": decision.decision == "START_NEW_LOGICAL_BLOCK",
                        "text_b": b.text[:80],
                    }
                )

            if not self.logger:
                continue

            evidence_payload = {
                "relationship": {k: round(v, 3) for k, v in rel.items()},
                "boundary": {k: round(v, 3) for k, v in bnd.items()},
                "relationship_score": round(r, 4),
                "boundary_score": round(s, 4),
                "net": round(net, 4),
                "adaptive_cut": round(float(cut), 4),
            }
            self.logger.event(
                "boundary_evaluated",
                document_id=a.document_id,
                page=b.page_number,
                source_ids=[a.id, b.id],
                decision=decision.decision,
                confidence=round(confidence, 4),
                evidence=evidence_payload,
                reason=reason,
            )

            # Transitions inside one discovered structure are record boundaries.
            if a.grid_id and a.grid_id == b.grid_id:
                record_decision = (
                    "START_NEW_RECORD" if split else "CONTINUE_CURRENT_RECORD"
                )
                self.logger.event(
                    "record_boundary_evaluated",
                    document_id=a.document_id,
                    page=b.page_number,
                    source_ids=[a.id, b.id],
                    grid_id=a.grid_id,
                    decision=record_decision,
                    confidence=round(confidence, 4),
                    evidence=evidence_payload,
                    reason=reason,
                )
                if split:
                    self.logger.event(
                        "record_boundary_detected",
                        document_id=a.document_id,
                        page=b.page_number,
                        source_ids=[a.id, b.id],
                        grid_id=a.grid_id,
                        decision="START_NEW_RECORD",
                        confidence=round(confidence, 4),
                        evidence={
                            "repetition_boundary": round(bnd["repetition_boundary"], 3),
                            "container_boundary": round(bnd["container_boundary"], 3),
                            "semantic_transition": round(bnd["semantic_transition"], 3),
                        },
                        reason=reason,
                    )

            # High form similarity with weak semantics is exactly the state that
            # produces over-grouping, so it is always surfaced.
            if raw_form >= 0.75 and rel["semantic_similarity"] < 0.25:
                self.logger.event(
                    "over_grouping_warning",
                    document_id=a.document_id,
                    page=b.page_number,
                    source_ids=[a.id, b.id],
                    decision=decision.decision,
                    confidence=round(confidence, 4),
                    evidence={
                        "mean_form_similarity": round(raw_form, 3),
                        "semantic_similarity": round(rel["semantic_similarity"], 3),
                        "form_discount_applied": round(explained, 3),
                        "net": round(net, 4),
                    },
                    reason=(
                        "Form similarity is high while semantic coherence is weak; "
                        "form evidence was discounted before deciding."
                    ),
                )
        return decisions, thresholds

    @staticmethod
    def _partition(
        candidates: List[CandidateUnit], boundaries: List[BoundaryDecision]
    ) -> List[List[CandidateUnit]]:
        """Connected components after removing edges the boundary analysis cut."""
        if not candidates:
            return []
        by_pair = {(b.unit_a, b.unit_b): b for b in boundaries}
        groups: List[List[CandidateUnit]] = [[candidates[0]]]
        for i in range(1, len(candidates)):
            edge = by_pair.get((candidates[i - 1].id, candidates[i].id))
            if edge is None or edge.decision == "START_NEW_LOGICAL_BLOCK":
                groups.append([candidates[i]])
            else:
                groups[-1].append(candidates[i])
        return groups

    # ------------------------------------------------------------------ #
    def _refine_and_build(
        self,
        document: Document,
        groups: List[List[CandidateUnit]],
        trace: DiscoveryTrace,
    ) -> Tuple[List[ContentUnit], List[LogicalBlock]]:
        units: List[ContentUnit] = []
        logical_blocks: List[LogicalBlock] = []
        total_blocks = max(1, len(document.blocks))

        for gi, group in enumerate(groups, start=1):
            if not group:
                continue
            decision = "PRESERVE" if len(group) == 1 else "MERGE"
            multi_raw = len({rid for c in group for rid in c.raw_unit_ids})
            if len(group) > 1 and multi_raw == 1:
                decision = "SPLIT"

            texts = [c.text for c in group]
            text = "\n".join(texts)
            block_ids: List[str] = []
            for c in group:
                for bid in c.block_ids:
                    if bid not in block_ids:
                        block_ids.append(bid)

            roles = self._roles(group)
            fingerprint = self._fingerprint(group)
            confidence, ev_signals = self._group_confidence(group)
            reason = {
                "PRESERVE": "Boundaries on both sides; the candidate stands alone as one unit.",
                "MERGE": "Relationship evidence exceeded boundary evidence across the run.",
                "SPLIT": "One extraction unit produced several independent candidates.",
            }[decision]

            trace.refinements.append(
                {
                    "group_index": gi,
                    "candidate_ids": [c.id for c in group],
                    "decision": decision,
                    "evidence": {k: round(v, 3) for k, v in ev_signals.items()},
                    "reason": reason,
                    "confidence": round(confidence, 4),
                }
            )
            for c in group:
                c.refinement = decision
                c.refinement_reason = reason

            unit = ContentUnit(
                id=f"{document.id}_unit_{gi:04d}",
                document_id=document.id,
                page_number=group[0].page_number,
                page_end=group[-1].page_number,
                block_ids=block_ids,
                text=text,
                head_block_id=block_ids[0] if block_ids else None,
                role_sequence=roles,
                structural_signature=self._role_signature(roles),
                structural_fingerprint=fingerprint,
                bounding_box=BoundingBox(
                    min(c.bbox[0] for c in group),
                    min(c.bbox[1] for c in group),
                    max(c.bbox[2] for c in group),
                    max(c.bbox[3] for c in group),
                ),
                features={
                    **{k: v for k, v in group[0].features.items()},
                    "block_count": float(len(block_ids)),
                    "candidate_count": float(len(group)),
                },
                evidence=Evidence(
                    signals=ev_signals,
                    weights={k: 1.0 / max(1, len(ev_signals)) for k in ev_signals},
                    confidence=confidence,
                    notes=[f"refinement:{decision}", reason],
                ),
            )
            units.append(unit)

            if self.logger:
                refine_payload = dict(
                    document_id=document.id,
                    page=unit.page_number,
                    source_ids=[c.id for c in group],
                    content_unit_id=unit.id,
                    decision=decision,
                    confidence=round(confidence, 4),
                    evidence={k: round(v, 3) for k, v in ev_signals.items()},
                    reason=reason,
                )
                self.logger.event("content_unit_refined", **refine_payload)
                if decision == "MERGE":
                    self.logger.event("content_unit_merged", **refine_payload)
                elif decision == "SPLIT":
                    self.logger.event("content_unit_split", **refine_payload)
                    self.logger.event(
                        "logical_block_split",
                        **{**refine_payload, "logical_block_id": f"{document.id}_logical_block_{gi:04d}"},
                    )

            head_block = next(
                (b for b in document.blocks if b.id == unit.head_block_id), None
            )
            doc_pos = (
                head_block.reading_order / total_blocks if head_block else gi / max(1, len(groups))
            )
            grid_ids = sorted({c.grid_id for c in group if c.grid_id})
            structured_fields = self._structured_fields(group)
            lb = LogicalBlock(
                id=f"{document.id}_logical_block_{gi:04d}",
                content_unit_id=unit.id,
                document_id=document.id,
                source_document=document.id,
                source_page=unit.page_number,
                page_end=unit.page_end,
                source_block_ids=list(block_ids),
                text=text,
                structural_features=dict(unit.features),
                structural_fingerprint=fingerprint,
                role_sequence=roles,
                structural_signature=unit.structural_signature,
                confidence=confidence,
                evidence=unit.evidence,
                doc_position=doc_pos,
                block_type="structured_record" if grid_ids else "content",
                structured_fields=structured_fields or None,
                source_table_id=grid_ids[0] if grid_ids else None,
            )
            logical_blocks.append(lb)
            if self.logger:
                self.logger.event(
                    "logical_block_created",
                    document_id=document.id,
                    page=lb.source_page,
                    source_ids=[c.id for c in group],
                    logical_block_id=lb.id,
                    content_unit_id=unit.id,
                    source_block_ids=lb.source_block_ids,
                    block_type=lb.block_type,
                    decision="CREATE",
                    confidence=round(confidence, 4),
                    evidence={k: round(v, 3) for k, v in ev_signals.items()},
                    reason=reason,
                )
        return units, logical_blocks

    @staticmethod
    def _structured_fields(group: List[CandidateUnit]) -> List[Dict[str, Any]]:
        fields_out: List[Dict[str, Any]] = []
        bullet_chars = frozenset("•·∙●◦▪▸")
        for c in group:
            for pos, cell in enumerate(c.cells):
                # Keep empty cells so column indices stay aligned with the grid.
                parts = _split_bullet_parts(cell, bullet_chars)
                if len(parts) <= 1:
                    parts = [cell]
                for pi, part in enumerate(parts):
                    prof = _char_profile(part)
                    entry: Dict[str, Any] = {
                        "field_position": pos,
                        "field_text": part,
                        "column_signature": (
                            f"pos{pos}"
                            f"|len{min(6, int(math.log1p(len(part)) / 1.2))}"
                            f"|d{int(prof['digit_ratio'] * 4)}"
                            f"|u{int(prof['upper_ratio'] * 4)}"
                        ),
                    }
                    if len(parts) > 1:
                        entry["field_part"] = pi
                    fields_out.append(entry)
        return fields_out

    @staticmethod
    def _roles(group: List[CandidateUnit]) -> List[str]:
        """Relative roles from prominence only — never from words."""
        proms = [c.features.get("prominence", 0.0) for c in group]
        hi = max(proms) if proms else 0.0
        roles = []
        for c, p in zip(group, proms):
            if p >= max(0.45, hi * 0.85):
                roles.append("PROMINENT")
            elif c.features.get("char_count", 0.0) <= 40:
                roles.append("META")
            else:
                roles.append("BODY")
        return roles

    @staticmethod
    def _role_signature(roles: List[str]) -> str:
        return (
            f"P{roles.count('PROMINENT')}B{roles.count('BODY')}M{roles.count('META')}"
        )

    @staticmethod
    def _fingerprint(group: List[CandidateUnit]) -> Dict[str, float]:
        f = [c.features for c in group]
        roles_prom = sum(1 for c in group if c.features.get("prominence", 0.0) >= 0.45)
        return {
            "block_count": float(len(group)),
            "role_prominent": float(roles_prom),
            "role_body": float(len(group) - roles_prom),
            "cell_count": float(max((c.features.get("cell_count", 0.0) for c in group), default=0.0)),
            "mean_rel_x0": round(float(np.mean([x.get("rel_x0", 0.0) for x in f])), 4),
            "mean_rel_width": round(float(np.mean([x.get("rel_width", 0.0) for x in f])), 4),
            "mean_line_height_ratio": round(
                float(np.mean([x.get("line_height_ratio", 1.0) for x in f])), 4
            ),
            "mean_digit_ratio": round(float(np.mean([x.get("digit_ratio", 0.0) for x in f])), 4),
            "mean_upper_ratio": round(float(np.mean([x.get("upper_ratio", 0.0) for x in f])), 4),
            "content_density": round(
                float(np.mean([x.get("char_count", 0.0) for x in f])) / 500.0, 4
            ),
            "marker_depth": float(max((x.get("marker_depth", 0.0) for x in f), default=0.0)),
            "in_grid": float(max((x.get("in_grid", 0.0) for x in f), default=0.0)),
            "repetition": round(float(np.mean([x.get("repetition_count", 1.0) for x in f])), 3),
            "local_position": round(float(np.mean([x.get("page_fraction", 0.0) for x in f])), 4),
        }

    def _group_confidence(self, group: List[CandidateUnit]) -> Tuple[float, Dict[str, float]]:
        if len(group) == 1:
            c = group[0]
            structure_support = clip01(
                0.4
                + 0.3 * clip01(c.features.get("cell_count", 0.0) / 4.0)
                + 0.3 * clip01(c.repetition_count / 8.0)
            )
            signals = {
                "structural_self_support": round(structure_support, 3),
                "boundary_isolation": 1.0,
                "repetition_support": round(clip01(c.repetition_count / 8.0), 3),
                "content_density": round(clip01(c.features.get("char_count", 0.0) / 400.0), 3),
            }
            conf = clip01(0.45 + 0.35 * structure_support + 0.2 * signals["content_density"])
            return conf, signals

        rels = []
        discounts = []
        for i in range(1, len(group)):
            rels.append(self._relationship(group[i - 1], group[i]))
            discounts.append(self._form_discount(group[i - 1], group[i])["total"])
        agg = {k: float(np.mean([r[k] for r in rels])) for k in rels[0]}
        mean_discount = float(np.mean(discounts))
        conf = clip01(
            self._relationship_score(agg, mean_discount) / sum(_REL_WEIGHTS.values())
        )
        signals = {k: round(v, 3) for k, v in agg.items()}
        signals["form_discount_total"] = round(mean_discount, 3)
        return clip01(0.35 + 0.65 * conf), signals

    # ------------------------------------------------------------------ #
    def _discover_contexts(
        self,
        document: Document,
        logical_blocks: List[LogicalBlock],
        candidates: List[CandidateUnit],
        trace: DiscoveryTrace,
    ) -> List[SectionGroup]:
        """Contexts need heading-like evidence *and* meaningful members."""
        if not logical_blocks:
            return []

        proms = [b.structural_features.get("prominence", 0.0) for b in logical_blocks]
        depth_present = any(
            b.structural_features.get("marker_depth", 0.0) > 0 for b in logical_blocks
        )
        prom_cut = float(np.percentile(proms, 75)) if proms else 0.6

        def heading_like(b: LogicalBlock) -> bool:
            f = b.structural_features
            if f.get("in_grid", 0.0) >= 1.0:
                return False
            if b.block_type == "structured_record":
                return False
            text = (b.text or "").strip()
            words = len(text.split())
            # Sentence-length units (even bold warnings) stay as items.
            if text.endswith((".", "?", "!")) and words >= 8:
                return False
            depth = f.get("marker_depth", 0.0)
            if depth_present:
                return depth > 0
            return (
                f.get("prominence", 0.0) >= prom_cut
                and f.get("char_count", 0.0) <= 120
                and words <= 14
                and f.get("terminal_punctuation", 0.0) < 1.0
            )

        sections: List[SectionGroup] = []
        counter = 0
        open_head: Optional[LogicalBlock] = None
        members: List[LogicalBlock] = []

        def close() -> None:
            nonlocal counter, open_head, members
            if open_head is None:
                return
            meaningful = [
                m
                for m in members
                if m.structural_features.get("char_count", 0.0) >= 12
            ]
            if meaningful:
                counter += 1
                sg = SectionGroup(
                    id=f"{document.id}_context_{counter:03d}",
                    document_id=document.id,
                    heading_block_id=open_head.id,
                    heading_text=open_head.text,
                    page_start=open_head.source_page,
                    page_end=meaningful[-1].page_end or meaningful[-1].source_page,
                    member_logical_block_ids=[m.id for m in meaningful],
                    member_source_block_ids=[
                        bid for m in meaningful for bid in m.source_block_ids
                    ],
                    evidence=Evidence(
                        signals={
                            "heading_prominence": round(
                                open_head.structural_features.get("prominence", 0.0), 3
                            ),
                            "heading_marker_depth": open_head.structural_features.get(
                                "marker_depth", 0.0
                            ),
                            "member_count": float(len(meaningful)),
                            "structural_continuity": round(
                                clip01(len(meaningful) / 8.0), 3
                            ),
                        },
                        weights={"heading_prominence": 0.4, "member_count": 0.6},
                        confidence=clip01(0.4 + 0.6 * clip01(len(meaningful) / 6.0)),
                        notes=["context discovered from heading evidence plus members"],
                    ),
                )
                sections.append(sg)
                for m in meaningful:
                    m.section_group_id = sg.id
                if self.logger:
                    self.logger.event(
                        "section_context_discovered",
                        document_id=document.id,
                        page=sg.page_start,
                        source_ids=[open_head.id],
                        context_id=sg.id,
                        decision="CREATE_CONTEXT",
                        confidence=round(sg.evidence.confidence, 4),
                        evidence=sg.evidence.signals,
                        reason="Heading evidence is supported by meaningful descendants.",
                    )
            else:
                trace.heading_contexts.append(
                    {
                        "logical_block_id": open_head.id,
                        "page": open_head.source_page,
                        "text": open_head.text,
                        "reason": "Heading-like evidence with no meaningful descendants.",
                    }
                )
                if self.logger:
                    self.logger.event(
                        "heading_context_detected",
                        document_id=document.id,
                        page=open_head.source_page,
                        source_ids=[open_head.id],
                        decision="NO_CONTEXT",
                        confidence=0.5,
                        evidence={"member_count": 0.0},
                        reason="Heading-like evidence with no meaningful descendants.",
                    )
            open_head = None
            members = []

        for b in logical_blocks:
            if heading_like(b):
                close()
                open_head = b
                members = []
            elif open_head is not None:
                members.append(b)
        close()
        return sections


def detect_pattern_consolidation(
    patterns: List[Any],
    similarity_floor: float = 0.92,
) -> List[Dict[str, Any]]:
    """Report structurally near-identical patterns without merging them."""
    findings: List[Dict[str, Any]] = []
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            a, b = patterns[i], patterns[j]
            va = np.asarray(a.centroid or [], dtype=np.float32)
            vb = np.asarray(b.centroid or [], dtype=np.float32)
            if va.size == 0 or vb.size == 0 or va.size != vb.size:
                continue
            sim = float(cosine(va, vb))
            if sim >= similarity_floor:
                findings.append(
                    {
                        "pattern_a": a.id,
                        "pattern_b": b.id,
                        "similarity": round(sim, 4),
                        "signature_a": a.representative_signature,
                        "signature_b": b.representative_signature,
                        "finding": "possible_pattern_consolidation",
                    }
                )
    return findings
