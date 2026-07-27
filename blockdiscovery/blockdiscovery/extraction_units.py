"""Backend-neutral extraction primitives.

Extraction backends answer only one question: *what is physically on the page?*
They emit :class:`RawExtractionUnit` objects carrying geometry, extractor layout
role and text. They never decide what a logical block is — that is the job of
:mod:`blockdiscovery.generic_discovery`.

No document vocabulary is consulted anywhere in this module, and no regular
expression is used to decide structure. Markdown/pipe-grid handling below is
pure machine-format parsing of the extractor's own output syntax.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .logging_utils import DiscoveryLogger
from .models import Document, PageInfo
from .utils import document_slug

BBox = Tuple[float, float, float, float]

# Characters the extractor uses to draw a grid separator row. Recognising them
# is machine-format parsing, not semantic interpretation.
_RULE_CHARS = frozenset("-:| ")
_EMPHASIS_TOKENS = ("**", "__")


def collapse_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def strip_markup(text: str) -> str:
    """Remove extractor markup artefacts (emphasis runs and angle-bracket tags).

    Regex-free on purpose: this is a character scan over a machine-generated
    format, and it never inspects words.
    """
    t = (text or "").replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    for token in _EMPHASIS_TOKENS:
        t = t.replace(token, "")
    out: List[str] = []
    depth = 0
    for ch in t:
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth:
                depth -= 1
        elif depth == 0:
            out.append(ch)
    return collapse_whitespace("".join(out))


def is_rule_row(cells: Sequence[str]) -> bool:
    """True when a grid line is a drawn separator rather than data."""
    filled = [c for c in cells if c.strip()]
    if not filled:
        return False
    for c in filled:
        stripped = set(c.strip())
        if not stripped or not stripped.issubset(_RULE_CHARS) or "-" not in stripped:
            return False
    return True


def split_grid_line(line: str) -> List[str]:
    """Split one pipe-delimited grid line into raw cell strings."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [strip_markup(c) for c in s.split("|")]


def leading_marker_depth(line: str) -> int:
    """Count the extractor's leading heading markers (typography evidence).

    The marker count reflects the relative font size the extractor measured; it
    carries no semantic meaning and is treated as one weak signal only.
    """
    s = line.lstrip()
    depth = 0
    for ch in s:
        if ch == "#":
            depth += 1
        else:
            break
    if depth == 0 or depth > 6:
        return 0
    rest = s[depth:]
    return depth if rest[:1] in (" ", "\t") else 0


@dataclass
class RawExtractionUnit:
    """One physical thing the extractor found on a page."""

    id: str
    document_id: str
    page_number: int
    order: int
    text: str
    bbox: BBox
    layout_class: str
    backend: str
    page_width: float = 612.0
    page_height: float = 792.0
    marker_depth: int = 0
    grid_id: Optional[str] = None
    grid_row_index: Optional[int] = None
    grid_row_count: Optional[int] = None
    cells: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    line_count: int = 1

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "page_number": self.page_number,
            "order": self.order,
            "layout_class": self.layout_class,
            "bbox": [round(v, 2) for v in self.bbox],
            "marker_depth": self.marker_depth,
            "grid_id": self.grid_id,
            "grid_row_index": self.grid_row_index,
            "cell_count": len(self.cells),
            "line_count": self.line_count,
            "text": self.text,
        }


@dataclass
class ExtractionResult:
    document: Document
    raw_units: List[RawExtractionUnit] = field(default_factory=list)
    page_stats: Dict[int, Dict[str, int]] = field(default_factory=dict)


def _build_document(path: str, doc_id: str, page_limit: int, backend: str) -> Document:
    import pymupdf

    pdf = pymupdf.open(path)
    document = Document(
        id=doc_id,
        source_path=os.path.abspath(path),
        metadata={
            "title": (pdf.metadata or {}).get("title") or "",
            "page_count_total": pdf.page_count,
            "page_count_processed": page_limit,
            "extraction_backend": backend,
        },
    )
    for i in range(min(page_limit, pdf.page_count)):
        page = pdf[i]
        document.pages.append(
            PageInfo(
                document_id=doc_id,
                page_number=i + 1,
                width=float(page.rect.width),
                height=float(page.rect.height),
            )
        )
    pdf.close()
    first = document.pages[0] if document.pages else None
    document.stats = {
        "page_width": first.width if first else 612.0,
        "page_height": first.height if first else 792.0,
    }
    return document


def _grid_rows_from_segment(segment: str) -> List[List[str]]:
    """Parse the extractor's pipe-grid syntax into raw rows (data rows only)."""
    rows: List[List[str]] = []
    for line in segment.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = split_grid_line(line)
        if is_rule_row(cells):
            continue
        if not any(c.strip() for c in cells):
            continue
        rows.append(cells)
    return rows


def _interpolate_row_bbox(table_bbox: BBox, row_index: int, row_count: int) -> BBox:
    x0, y0, x1, y1 = table_bbox
    if row_count <= 0:
        return table_bbox
    height = (y1 - y0) / row_count
    return (x0, y0 + row_index * height, x1, y0 + (row_index + 1) * height)


def extract_raw_units(
    path: str,
    max_pages: Optional[int] = None,
    logger: Optional[DiscoveryLogger] = None,
    document_id: Optional[str] = None,
    backend: str = "structured",
) -> ExtractionResult:
    """Extract geometry-bearing raw units using pymupdf4llm layout boxes."""
    import pymupdf
    import pymupdf4llm

    doc_id = document_id or document_slug(path)
    pdf = pymupdf.open(path)
    total_pages = pdf.page_count
    pdf.close()
    page_limit = min(total_pages, max_pages) if max_pages else total_pages

    document = _build_document(path, doc_id, page_limit, backend)
    page_size = {p.page_number: (p.width, p.height) for p in document.pages}

    if logger:
        logger.section("EXTRACTION")
        logger.kv("Backend", backend)
        logger.kv("Document", os.path.basename(path))
        logger.kv("Pages", page_limit)

    chunks = pymupdf4llm.to_markdown(
        path, pages=list(range(page_limit)), page_chunks=True
    )

    raw_units: List[RawExtractionUnit] = []
    page_stats: Dict[int, Dict[str, int]] = {
        pn: {"raw_units": 0, "grids": 0, "heading_like": 0, "layout_boxes": 0}
        for pn in range(1, page_limit + 1)
    }
    counter = 0
    grid_counter = 0

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        page_number = int(meta.get("page_number", 0) or 0)
        if page_number < 1 or page_number > page_limit:
            continue
        text = chunk.get("text") or ""
        boxes = chunk.get("page_boxes") or []
        pw, ph = page_size.get(page_number, (612.0, 792.0))
        page_stats[page_number]["layout_boxes"] = len(boxes)

        for box in boxes:
            pos = box.get("pos")
            if not pos:
                continue
            segment = text[pos[0] : pos[1]]
            if not segment.strip():
                continue
            bbox = tuple(float(v) for v in box.get("bbox", (0, 0, pw, 0)))  # type: ignore
            layout_class = str(box.get("class") or "unknown")

            grid_rows = _grid_rows_from_segment(segment)
            if grid_rows:
                grid_counter += 1
                grid_id = f"{doc_id}_grid_{grid_counter:03d}"
                page_stats[page_number]["grids"] += 1
                for ri, cells in enumerate(grid_rows):
                    counter += 1
                    row_text = collapse_whitespace(" ".join(c for c in cells if c.strip()))
                    if not row_text:
                        continue
                    raw_units.append(
                        RawExtractionUnit(
                            id=f"{doc_id}_raw_{counter:04d}",
                            document_id=doc_id,
                            page_number=page_number,
                            order=counter,
                            text=row_text,
                            bbox=_interpolate_row_bbox(bbox, ri, len(grid_rows)),
                            layout_class=layout_class,
                            backend=backend,
                            page_width=pw,
                            page_height=ph,
                            grid_id=grid_id,
                            grid_row_index=ri,
                            grid_row_count=len(grid_rows),
                            cells=[c for c in cells],
                            lines=[row_text],
                            line_count=1,
                        )
                    )
                    page_stats[page_number]["raw_units"] += 1
                continue

            # Non-grid segment: keep extractor's block, record marker depth.
            lines = [ln for ln in segment.splitlines() if ln.strip()]
            depth = max((leading_marker_depth(ln) for ln in lines), default=0)
            cleaned_lines = []
            for ln in lines:
                d = leading_marker_depth(ln)
                cleaned_lines.append(strip_markup(ln[ln.find("#") + d :] if d else ln))
            body = collapse_whitespace(" ".join(cleaned_lines))
            if not body:
                continue
            counter += 1
            if depth > 0:
                page_stats[page_number]["heading_like"] += 1
            raw_units.append(
                RawExtractionUnit(
                    id=f"{doc_id}_raw_{counter:04d}",
                    document_id=doc_id,
                    page_number=page_number,
                    order=counter,
                    text=body,
                    bbox=bbox,
                    layout_class=layout_class,
                    backend=backend,
                    page_width=pw,
                    page_height=ph,
                    marker_depth=depth,
                    lines=[ln for ln in cleaned_lines if ln],
                    line_count=len(lines),
                )
            )
            page_stats[page_number]["raw_units"] += 1

    if logger:
        logger.event(
            "raw_extraction_completed",
            document_id=doc_id,
            backend=backend,
            pages=page_limit,
            raw_units=len(raw_units),
            grids=grid_counter,
            page_stats=page_stats,
        )
        logger.kv("Raw extraction units", len(raw_units))
        logger.kv("Grids found by extractor geometry", grid_counter)

    return ExtractionResult(document=document, raw_units=raw_units, page_stats=page_stats)
