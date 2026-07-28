"""Generic PDF extraction layer (PyMuPDF).

This layer extracts *observable evidence only*. It contains no business logic,
no keyword checks, no section-name assumptions and no fixed coordinates. It
simply reports what PyMuPDF sees: metadata, pages, blocks, lines, spans, fonts,
bounding boxes, links and images, preserving reading order where available.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from .logging_utils import DiscoveryLogger
from .models import (
    BoundingBox,
    Document,
    Formatting,
    Line,
    PageInfo,
    Span,
    TextBlock,
)
from .utils import document_slug


def _summarize_formatting(spans: List[Span]) -> Optional[Formatting]:
    if not spans:
        return None
    # Weight font characteristics by the amount of text they cover.
    size_weight: Dict[float, int] = {}
    font_weight: Dict[str, int] = {}
    total_chars = 0
    bold_chars = italic_chars = mono_chars = 0
    for s in spans:
        n = max(1, len(s.text.strip()))
        total_chars += n
        rsize = round(s.size, 1)
        size_weight[rsize] = size_weight.get(rsize, 0) + n
        font_weight[s.font] = font_weight.get(s.font, 0) + n
        if s.is_bold:
            bold_chars += n
        if s.is_italic:
            italic_chars += n
        if s.is_monospace:
            mono_chars += n
    dominant_size = max(size_weight.items(), key=lambda kv: kv[1])[0]
    dominant_font = max(font_weight.items(), key=lambda kv: kv[1])[0]
    sizes = [s.size for s in spans]
    return Formatting(
        dominant_size=dominant_size,
        max_size=max(sizes),
        min_size=min(sizes),
        dominant_font=dominant_font,
        bold_ratio=bold_chars / total_chars if total_chars else 0.0,
        italic_ratio=italic_chars / total_chars if total_chars else 0.0,
        monospace_ratio=mono_chars / total_chars if total_chars else 0.0,
        color=spans[0].color,
        size_variety=len(size_weight),
    )


class PDFExtractor:
    """Turns a PDF file into a :class:`Document` of raw evidence."""

    def __init__(self, logger: Optional[DiscoveryLogger] = None) -> None:
        self.logger = logger

    def extract(
        self,
        path: str,
        document_id: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> Document:
        log = self.logger
        doc_id = document_id or document_slug(path)
        if log:
            page_note = f" (first {max_pages} pages)" if max_pages else ""
            log.section("DOCUMENT", f"Processing document id: {doc_id}{page_note}")
            log.event(
                "document_started",
                document_id=doc_id,
                source_path=path,
                max_pages=max_pages,
                readable=None,
            )

        fitz_doc = fitz.open(path)
        metadata = dict(fitz_doc.metadata or {})
        document = Document(id=doc_id, source_path=path, metadata=metadata)

        block_counter = 0
        reading_order = 0
        page_limit = fitz_doc.page_count
        if max_pages is not None:
            page_limit = max(0, min(fitz_doc.page_count, int(max_pages)))

        if log:
            log.section("EXTRACTION")
            if max_pages is not None:
                log.line(f"Page limit applied: processing first {page_limit} of {fitz_doc.page_count} pages")

        for page_index in range(page_limit):
            page = fitz_doc[page_index]
            page_number = page_index + 1
            rect = page.rect
            page_info = PageInfo(
                document_id=doc_id,
                page_number=page_number,
                width=float(rect.width),
                height=float(rect.height),
                rotation=int(page.rotation),
            )

            # Links & images are extra evidence (used opportunistically later).
            try:
                page_info.links = [
                    {
                        "kind": lk.get("kind"),
                        "uri": lk.get("uri"),
                        "from": list(lk.get("from")) if lk.get("from") else None,
                    }
                    for lk in page.get_links()
                ]
            except Exception:
                page_info.links = []

            try:
                for img in page.get_image_info(xrefs=True):
                    bb = img.get("bbox")
                    page_info.images.append(
                        {
                            "bbox": list(bb) if bb else None,
                            "width": img.get("width"),
                            "height": img.get("height"),
                            "xref": img.get("xref"),
                        }
                    )
            except Exception:
                page_info.images = []

            raw = page.get_text("dict")
            page_blocks = 0

            for raw_block in raw.get("blocks", []):
                btype = raw_block.get("type", 0)
                if btype == 1:
                    # Image block -> keep as evidence with an image TextBlock.
                    bb = raw_block.get("bbox", [0, 0, 0, 0])
                    block_counter += 1
                    tb = TextBlock(
                        id=f"{doc_id}_block_{block_counter:04d}",
                        document_id=doc_id,
                        page_number=page_number,
                        text="",
                        bounding_box=BoundingBox(*bb),
                        block_type="image",
                        reading_order=reading_order,
                        raw_block_no=raw_block.get("number", 0),
                    )
                    reading_order += 1
                    document.blocks.append(tb)
                    page_info.block_ids.append(tb.id)
                    page_blocks += 1
                    continue

                lines: List[Line] = []
                all_spans: List[Span] = []
                for raw_line in raw_block.get("lines", []):
                    line_spans: List[Span] = []
                    for raw_span in raw_line.get("spans", []):
                        sbb = raw_span.get("bbox", [0, 0, 0, 0])
                        span = Span(
                            text=raw_span.get("text", ""),
                            font=raw_span.get("font", ""),
                            size=float(raw_span.get("size", 0.0)),
                            flags=int(raw_span.get("flags", 0)),
                            color=int(raw_span.get("color", 0)),
                            bbox=BoundingBox(*sbb),
                        )
                        line_spans.append(span)
                        all_spans.append(span)
                    line_text = "".join(s.text for s in line_spans)
                    lbb = raw_line.get("bbox", [0, 0, 0, 0])
                    lines.append(
                        Line(
                            text=line_text,
                            bbox=BoundingBox(*lbb),
                            spans=line_spans,
                            wmode=raw_line.get("wmode", 0),
                            direction=tuple(raw_line.get("dir", (1.0, 0.0))),
                        )
                    )

                text = "\n".join(ln.text for ln in lines).strip()
                if not text:
                    continue

                bb = raw_block.get("bbox", [0, 0, 0, 0])
                block_counter += 1
                tb = TextBlock(
                    id=f"{doc_id}_block_{block_counter:04d}",
                    document_id=doc_id,
                    page_number=page_number,
                    text=text,
                    bounding_box=BoundingBox(*bb),
                    spans=all_spans,
                    lines=lines,
                    formatting=_summarize_formatting(all_spans),
                    reading_order=reading_order,
                    raw_block_no=raw_block.get("number", 0),
                )
                reading_order += 1
                document.blocks.append(tb)
                page_info.block_ids.append(tb.id)
                page_blocks += 1

            document.pages.append(page_info)
            if log:
                log.event(
                    "page_processed",
                    document_id=doc_id,
                    page_number=page_number,
                    block_count=page_blocks,
                    width=round(page_info.width, 1),
                    height=round(page_info.height, 1),
                )

        fitz_doc.close()

        if log:
            log.line(f"Pages detected: {document.page_count}")
            log.line(f"Text blocks extracted: {len(document.blocks)}")
            log.event(
                "raw_blocks_created",
                document_id=doc_id,
                block_count=len(document.blocks),
            )
            log.event(
                "document_extraction_completed",
                document_id=doc_id,
                page_count=document.page_count,
                block_count=len(document.blocks),
                metadata_keys=list(metadata.keys()),
            )
        return document
