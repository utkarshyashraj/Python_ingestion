"""Structure-aware PDF ingestion adapters.

Two backends share one output contract so downstream pattern discovery and
cross-document grouping stay unchanged:

* ``structured`` — PyMuPDF layout + pymupdf4llm markdown (works out of the box)
* ``docling``    — IBM Docling (optional; used when installed)

Both produce:
  Document (with synthetic TextBlocks for traceability)
  ContentUnits (one per table row / heading / paragraph)
  SectionGroups (from markdown headings / Docling titles)

No Feature / Fix / Bug lexicon is used. Column headers become *structural*
field slots only, never business category labels.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .logging_utils import DiscoveryLogger
from .logical_blocks import logical_block_from_unit
from .models import (
    BoundingBox,
    ContentUnit,
    Document,
    Evidence,
    Formatting,
    LogicalBlock,
    PageInfo,
    SectionGroup,
    TextBlock,
)
from .utils import document_slug


def _clean_cell(text: str) -> str:
    t = (text or "").replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    t = re.sub(r"\*\*|__", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _split_md_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_clean_cell(c) for c in line.split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_md_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) is not None for c in cells if c)


def _looks_like_column_labels(cells: List[str]) -> bool:
    """Heuristic: short label-like cells, not prose and not long numeric ids."""
    filled = [c for c in cells if c]
    if len(filled) < 2:
        return False
    return all(len(c) <= 48 and not re.search(r"\d{6,}", c) for c in filled)


@dataclass
class StructuredIngestResult:
    document: Document
    content_units: List[ContentUnit] = field(default_factory=list)
    section_groups: List[SectionGroup] = field(default_factory=list)


def _empty_document(path: str, doc_id: str, page_count: int, backend: str) -> Document:
    import fitz

    pdf = fitz.open(path)
    document = Document(
        id=doc_id,
        source_path=os.path.abspath(path),
        metadata={
            "title": (pdf.metadata or {}).get("title") or "",
            "page_count_total": pdf.page_count,
            "page_count_processed": page_count,
            "ingestion_backend": backend,
        },
    )
    for i in range(min(page_count, pdf.page_count)):
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
    document.stats = {
        "size_median": 11.0,
        "size_mean": 11.0,
        "size_std": 1.0,
        "gap_median": 6.0,
        "gap_std": 4.0,
        "page_width": document.pages[0].width if document.pages else 612.0,
        "page_height": document.pages[0].height if document.pages else 792.0,
        "ingestion_backend": 1.0,
    }
    return document


def parse_markdown_chunks(
    doc_id: str,
    document: Document,
    chunks: List[dict],
    page_limit: int,
    backend_name: str,
    logger: Optional[DiscoveryLogger] = None,
) -> StructuredIngestResult:
    """Parse page-chunk markdown into content units + section groups."""
    log = logger
    units: List[ContentUnit] = []
    sections: List[SectionGroup] = []
    blocks: List[TextBlock] = []
    block_counter = 0
    unit_counter = 0
    section_counter = 0
    current_section: Optional[SectionGroup] = None
    pending_section_members: List[str] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    def flush_section() -> None:
        nonlocal current_section, pending_section_members
        if current_section is None:
            return
        current_section.member_logical_block_ids = list(pending_section_members)
        sections.append(current_section)
        if log:
            log.event(
                "section_group_created",
                document_id=doc_id,
                section_id=current_section.id,
                heading=current_section.heading_text[:120],
                page_start=current_section.page_start,
                item_count=len(pending_section_members),
                confidence=round(current_section.evidence.confidence, 4),
            )
        pending_section_members = []

    def new_block(page_number: int, text: str, role: str) -> TextBlock:
        nonlocal block_counter
        block_counter += 1
        tb = TextBlock(
            id=f"{doc_id}_block_{block_counter:04d}",
            document_id=doc_id,
            page_number=page_number,
            text=text,
            bounding_box=BoundingBox(0, 0, 1, 1),
            formatting=Formatting(
                dominant_size=14.0 if role == "PROMINENT" else 11.0,
                max_size=14.0 if role == "PROMINENT" else 11.0,
                min_size=11.0,
                dominant_font=backend_name,
                bold_ratio=1.0 if role == "PROMINENT" else 0.0,
                italic_ratio=0.0,
                monospace_ratio=0.0,
                color=0,
                size_variety=1,
            ),
            reading_order=block_counter,
            block_type="text",
            raw_block_no=block_counter,
            features={
                "prominence": 1.0 if role == "PROMINENT" else 0.2,
                "from_structured_ingest": 1.0,
            },
        )
        blocks.append(tb)
        page = next((p for p in document.pages if p.page_number == page_number), None)
        if page is not None:
            page.block_ids.append(tb.id)
        return tb

    def new_unit(
        page_number: int,
        text: str,
        role_sequence: List[str],
        source_blocks: List[TextBlock],
        kind: str,
        fingerprint_extra: Optional[Dict[str, float]] = None,
        structural_fields: Optional[Dict[str, str]] = None,
    ) -> ContentUnit:
        nonlocal unit_counter
        unit_counter += 1
        head = source_blocks[0] if source_blocks else None
        sig = (
            f"P{role_sequence.count('PROMINENT')}"
            f"B{role_sequence.count('BODY')}"
            f"M{role_sequence.count('META')}"
        )
        fp: Dict[str, float] = {
            "block_count": float(len(source_blocks)),
            "role_prominent": float(role_sequence.count("PROMINENT")),
            "role_body": float(role_sequence.count("BODY")),
            "role_meta": float(role_sequence.count("META")),
            "from_table_row": 1.0 if kind == "table_row" else 0.0,
            "from_heading": 1.0 if kind == "heading" else 0.0,
            "from_paragraph": 1.0 if kind == "paragraph" else 0.0,
            "field_slot_count": float(len(structural_fields or {})),
            "char_count_log": float(len(text)),
            "local_position": page_number / max(1, page_limit),
        }
        if fingerprint_extra:
            fp.update(fingerprint_extra)
        conf = 0.9 if kind == "table_row" else (0.75 if kind == "heading" else 0.55)
        evidence = Evidence(
            signals={
                "structure_source": 1.0,
                "table_row_evidence": 1.0 if kind == "table_row" else 0.0,
                "heading_evidence": 1.0 if kind == "heading" else 0.0,
            },
            weights={"structure_source": 1.0},
            confidence=conf,
            notes=[f"{backend_name}_ingest:{kind}"],
        )
        if structural_fields:
            for i, (k, v) in enumerate(structural_fields.items()):
                evidence.notes.append(f"field_{i}:{k}={v[:120]}")
        unit = ContentUnit(
            id=f"{doc_id}_unit_{unit_counter:04d}",
            document_id=doc_id,
            page_number=page_number,
            page_end=page_number,
            block_ids=[b.id for b in source_blocks],
            text=text,
            head_block_id=head.id if head else None,
            role_sequence=role_sequence,
            structural_signature=sig,
            structural_fingerprint=fp,
            bounding_box=BoundingBox(0, float(page_number), 1, float(page_number) + 0.5),
            features={
                "block_count": float(len(source_blocks)),
                "ingestion_kind_table_row": 1.0 if kind == "table_row" else 0.0,
                "ingestion_kind_heading": 1.0 if kind == "heading" else 0.0,
                "field_slot_count": float(len(structural_fields or {})),
                "head_prominence": 1.0 if "PROMINENT" in role_sequence else 0.2,
                "head_size_ratio": 1.2 if kind == "heading" else 1.0,
                "role_prominent": float(role_sequence.count("PROMINENT")),
                "role_body": float(role_sequence.count("BODY")),
                "role_meta": float(role_sequence.count("META")),
                "page_fraction": page_number / max(1, page_limit),
                "char_count": float(len(text)),
            },
            evidence=evidence,
        )
        units.append(unit)
        if log:
            log.event(
                "content_unit_created",
                document_id=doc_id,
                page_number=page_number,
                content_unit_id=unit.id,
                source_block_ids=unit.block_ids,
                kind=kind,
                role_sequence=role_sequence,
                structural_signature=sig,
                structural_fingerprint={k: round(v, 4) for k, v in fp.items()},
                confidence=round(conf, 4),
                evidence=evidence.to_dict(),
            )
            log.event(
                "structural_fingerprint_created",
                document_id=doc_id,
                content_unit_id=unit.id,
                fingerprint={k: round(v, 4) for k, v in fp.items()},
                confidence=round(conf, 4),
            )
            preview = text[:100].replace("\n", " ")
            log.push()
            log.line(f"ContentUnit {unit.id} [{kind}] p{page_number}: {preview}")
            log.pop()
        return unit

    # Persist markdown table headers across page-chunk boundaries so continued
    # table fragments still become row units.
    carried_header: List[str] = []

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        if "page_number" in meta:
            page_number = int(meta["page_number"])
        elif "page" in meta:
            # Some exporters use 0-based page.
            page_number = int(meta["page"]) + 1
        else:
            page_number = 1
        if page_number < 1:
            page_number = 1
        if page_number > page_limit:
            page_number = page_limit
        text = chunk.get("text") or ""
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line.strip():
                i += 1
                continue

            hm = heading_re.match(line.strip())
            if hm:
                level = len(hm.group(1))
                title = _clean_cell(hm.group(2))
                if not title:
                    i += 1
                    continue
                carried_header = []
                flush_section()
                section_counter += 1
                head_block = new_block(page_number, title, "PROMINENT")
                unit = new_unit(
                    page_number,
                    title,
                    ["PROMINENT"],
                    [head_block],
                    "heading",
                    {"heading_level": float(level)},
                )
                current_section = SectionGroup(
                    id=f"{doc_id}_section_{section_counter:03d}",
                    document_id=doc_id,
                    heading_block_id=unit.id,
                    heading_text=title,
                    page_start=page_number,
                    page_end=page_number,
                    evidence=Evidence(
                        signals={"heading_level": float(level), "structured_heading": 1.0},
                        weights={"structured_heading": 1.0},
                        confidence=0.85,
                        notes=[f"{backend_name}_heading"],
                    ),
                )
                pending_section_members = []
                i += 1
                continue

            if line.strip().startswith("|"):
                table_lines: List[str] = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                header: List[str] = list(carried_header)
                # If this fragment includes a markdown separator, the row before
                # it is a fresh header only when it looks like column labels.
                # Page-continued tables often get a false separator after a
                # wrapped cell — do not overwrite a good carried header.
                sep_idx = next((ti for ti, tl in enumerate(table_lines) if _is_separator(tl)), -1)
                if sep_idx > 0:
                    candidate = _split_md_row(table_lines[sep_idx - 1])
                    if not header or _looks_like_column_labels(candidate):
                        header = [c or f"field_{j + 1}" for j, c in enumerate(candidate)]
                        carried_header = list(header)
                for ti, tline in enumerate(table_lines):
                    if _is_separator(tline):
                        continue
                    # Skip the header row that preceded a separator.
                    if sep_idx > 0 and ti == sep_idx - 1:
                        continue
                    cells = _split_md_row(tline)
                    if not any(cells):
                        continue
                    if not header:
                        # Continuation without separator: treat first row as header
                        # only when cells look short/label-like.
                        if _looks_like_column_labels(cells):
                            header = [c or f"field_{j + 1}" for j, c in enumerate(cells)]
                            carried_header = list(header)
                            continue
                        # Otherwise invent generic field slots so the row is kept.
                        header = [f"field_{j + 1}" for j in range(len(cells))]
                        carried_header = list(header)
                    if len(cells) < len(header):
                        cells = cells + [""] * (len(header) - len(cells))
                    # Grow header if this row has more columns (layout glitch).
                    if len(cells) > len(header):
                        header = header + [f"field_{j + 1}" for j in range(len(header), len(cells))]
                        carried_header = list(header)
                    cells = cells[: len(header)]
                    fields = {header[j]: cells[j] for j in range(len(header)) if cells[j]}
                    if not fields:
                        continue
                    # Skip tiny continuation crumbs.
                    if len(fields) == 1 and len(list(fields.values())[0]) <= 12:
                        continue
                    row_text = " | ".join(f"{k}: {v}" for k, v in fields.items())
                    tb = new_block(page_number, row_text, "BODY")
                    new_unit(
                        page_number,
                        row_text,
                        ["PROMINENT", "BODY"] if len(fields) >= 2 else ["BODY"],
                        [tb],
                        "table_row",
                        {
                            "column_count": float(len(header)),
                            "filled_fields": float(len(fields)),
                        },
                        fields,
                    )
                    pending_section_members.append(units[-1].id)
                    if current_section is not None:
                        current_section.page_end = max(current_section.page_end, page_number)
                        current_section.member_source_block_ids.append(tb.id)
                if header:
                    carried_header = list(header)
                continue

            # Non-table text clears carried table header only when substantial.
            carried_header = []
            para_lines = [line.strip()]
            i += 1
            while i < len(lines):
                nxt = lines[i].rstrip()
                if not nxt.strip() or nxt.strip().startswith("|") or heading_re.match(nxt.strip()):
                    break
                para_lines.append(nxt.strip())
                i += 1
            para = _clean_cell(" ".join(para_lines))
            if len(para) < 3 or para.lower() in {"back to top", "---", "***"}:
                continue
            role = "PROMINENT" if len(para) <= 80 and not para.endswith(".") else "BODY"
            tb = new_block(page_number, para, role)
            new_unit(page_number, para, [role], [tb], "paragraph")
            pending_section_members.append(units[-1].id)
            if current_section is not None:
                current_section.page_end = max(current_section.page_end, page_number)
                current_section.member_source_block_ids.append(tb.id)

    flush_section()
    document.blocks = blocks
    if log:
        table_rows = sum(1 for u in units if u.features.get("ingestion_kind_table_row"))
        log.event(
            "document_extraction_completed",
            document_id=doc_id,
            pages=page_limit,
            raw_blocks=len(blocks),
            content_units=len(units),
            sections=len(sections),
            table_row_units=table_rows,
            backend=backend_name,
        )
        log.kv("Pages", page_limit)
        log.kv("Content units", len(units))
        log.kv("Table-row units", table_rows)
        log.kv("Section groups", len(sections))

    return StructuredIngestResult(
        document=document,
        content_units=units,
        section_groups=sections,
    )


class StructuredMarkdownIngestor:
    """Ingest via pymupdf4llm markdown — table rows become content units."""

    def __init__(self, logger: Optional[DiscoveryLogger] = None, max_pages: Optional[int] = None) -> None:
        self.logger = logger
        self.max_pages = max_pages

    def ingest(self, path: str, document_id: Optional[str] = None) -> StructuredIngestResult:
        try:
            import pymupdf4llm
            import fitz
        except ImportError as e:
            raise ImportError(
                "structured ingestion requires pymupdf4llm. Install with: pip install pymupdf4llm"
            ) from e

        doc_id = document_id or document_slug(path)
        log = self.logger
        if log:
            log.section("STRUCTURED INGESTION")
            log.kv("Backend", "pymupdf4llm")
            log.kv("Document", os.path.basename(path))

        pdf = fitz.open(path)
        total_pages = pdf.page_count
        page_limit = min(total_pages, self.max_pages) if self.max_pages else total_pages
        pdf.close()

        document = _empty_document(path, doc_id, page_limit, "structured")
        chunks = pymupdf4llm.to_markdown(path, pages=list(range(page_limit)), page_chunks=True)
        return parse_markdown_chunks(doc_id, document, chunks, page_limit, "structured", log)


class DoclingIngestor:
    """Optional Docling backend. Raises ImportError if Docling is unavailable."""

    def __init__(self, logger: Optional[DiscoveryLogger] = None, max_pages: Optional[int] = None) -> None:
        self.logger = logger
        self.max_pages = max_pages

    def ingest(self, path: str, document_id: Optional[str] = None) -> StructuredIngestResult:
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as e:
            raise ImportError(
                "Docling is not installed (or failed to build on this platform). "
                "Use ingestion_backend='structured' or: pip install docling"
            ) from e

        log = self.logger
        if log:
            log.section("STRUCTURED INGESTION")
            log.kv("Backend", "docling")
            log.kv("Document", os.path.basename(path))

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(path)
        md = result.document.export_to_markdown()

        doc_id = document_id or document_slug(path)
        import fitz

        pdf = fitz.open(path)
        total = pdf.page_count
        pdf.close()
        page_limit = min(total, self.max_pages) if self.max_pages else total
        document = _empty_document(path, doc_id, page_limit, "docling")
        # Docling export is document-global; page provenance may be coarse.
        chunks = [{"metadata": {"page_number": 1}, "text": md}]
        return parse_markdown_chunks(doc_id, document, chunks, page_limit, "docling", log)


def units_to_logical_blocks(
    document: Document,
    units: List[ContentUnit],
    unit_to_pattern: Dict[str, str],
    logger: Optional[DiscoveryLogger] = None,
) -> List[LogicalBlock]:
    """Promote structured content units to logical blocks (no heuristic split/merge)."""
    total = max(1, len(document.blocks))
    block_index = {b.id: b for b in document.blocks}
    out: List[LogicalBlock] = []
    for i, unit in enumerate(units, start=1):
        lb = logical_block_from_unit(
            document, i, unit, unit_to_pattern.get(unit.id), block_index, total
        )
        # Prefer basename for display consistency with structured path consumers.
        lb.source_document = os.path.basename(document.source_path)
        out.append(lb)
        if logger:
            logger.event(
                "logical_block_created",
                document_id=lb.document_id,
                page_number=lb.source_page,
                logical_block_id=lb.id,
                content_unit_id=lb.content_unit_id,
                source_block_ids=lb.source_block_ids,
                pattern_id=lb.discovered_pattern,
                confidence=round(lb.confidence, 4),
            )
    return out


def remap_section_members(
    sections: List[SectionGroup],
    logical_blocks: List[LogicalBlock],
) -> None:
    """Map provisional unit-id membership / headings to logical block ids."""
    unit_to_lb = {lb.content_unit_id: lb.id for lb in logical_blocks}
    for section in sections:
        section.member_logical_block_ids = [
            unit_to_lb[uid] for uid in section.member_logical_block_ids if uid in unit_to_lb
        ]
        if section.heading_block_id in unit_to_lb:
            section.heading_block_id = unit_to_lb[section.heading_block_id]


def ingest_pdf(
    path: str,
    backend: str = "structured",
    max_pages: Optional[int] = None,
    logger: Optional[DiscoveryLogger] = None,
    document_id: Optional[str] = None,
) -> StructuredIngestResult:
    """Factory: choose structured (default) or docling ingestion."""
    if backend == "docling":
        try:
            return DoclingIngestor(logger=logger, max_pages=max_pages).ingest(path, document_id)
        except ImportError:
            if logger:
                logger.line("Docling unavailable — falling back to structured (pymupdf4llm).")
            return StructuredMarkdownIngestor(logger=logger, max_pages=max_pages).ingest(path, document_id)
    if backend == "structured":
        return StructuredMarkdownIngestor(logger=logger, max_pages=max_pages).ingest(path, document_id)
    raise ValueError(f"Unknown ingestion backend: {backend!r} (use 'structured' or 'docling')")
