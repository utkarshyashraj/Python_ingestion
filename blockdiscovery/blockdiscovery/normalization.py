"""Raw block normalization.

Cleans and orders raw blocks without discarding evidence:
* normalises whitespace in the derived ``text`` (spans keep their raw text),
* establishes a stable reading order per page using PyMuPDF's order as the
  primary signal and (page, y, x) geometry as a tie-breaker,
* drops fully-empty blocks.

No business logic, no keyword logic.
"""

from __future__ import annotations

import re
from typing import Optional

from .logging_utils import DiscoveryLogger
from .models import Document


_WS_RE = re.compile(r"[ \t\u00a0]+")


def _clean(text: str) -> str:
    lines = [ _WS_RE.sub(" ", ln).strip() for ln in text.splitlines() ]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines)


def normalize_document(document: Document, logger: Optional[DiscoveryLogger] = None) -> Document:
    for b in document.blocks:
        b.text = _clean(b.text)

    document.blocks = [b for b in document.blocks if b.text or b.block_type == "image"]

    # Stable reading order: page first, then original order, then geometry.
    document.blocks.sort(
        key=lambda b: (
            b.page_number,
            b.reading_order,
            round(b.bounding_box.y0, 1),
            round(b.bounding_box.x0, 1),
        )
    )
    for i, b in enumerate(document.blocks):
        b.reading_order = i

    # Rebuild per-page block id ordering to match.
    page_map = {p.page_number: p for p in document.pages}
    for p in document.pages:
        p.block_ids = []
    for b in document.blocks:
        if b.page_number in page_map:
            page_map[b.page_number].block_ids.append(b.id)

    if logger:
        logger.event(
            "block_normalization_completed",
            document_id=document.id,
            block_count=len(document.blocks),
        )
    return document
