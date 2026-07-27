"""Within-document section group discovery.

Discovers section-like containers from *layout evidence only*:

  1. Rank logical blocks by relative prominence (font size / bold / shortness).
  2. Treat high-prominence or title-like short blocks as section starts.
  3. Assign subsequent content items to the open section until the next
     comparable heading appears.

Sections are identified as ``DiscoveredSection_001``, ``DiscoveredSection_002``,
… — never as hardcoded business types (Feature / Fix / Bug / …).

Optional naming via ``EngineConfig.optional_label_lexicon`` is disabled by
default and is not part of discovery.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import EngineConfig
from .logging_utils import DiscoveryLogger
from .models import Document, Evidence, LogicalBlock, SectionGroup
from .semantics import EmbeddingBackend, cosine


def _preview(text: str, n: int = 90) -> str:
    return " ".join(text.split())[:n]


def _is_section_heading(block: LogicalBlock, prom_threshold: float) -> bool:
    """Relative heading test -- never absolute font size or fixed keywords."""
    f = block.structural_features
    prom = f.get("head_prominence", f.get("mean_prominence", 0.0))
    size_ratio = f.get("head_size_ratio", 1.0)
    chars = f.get("char_count", float(len(block.text)))
    blocks = f.get("block_count", float(len(block.source_block_ids)))
    roles = block.role_sequence or []
    text = (block.text or "").strip()

    shortish = chars <= 120
    mostly_head = blocks <= 2 or (roles.count("PROMINENT") >= 1 and roles.count("BODY") <= 1)
    prominent = prom >= prom_threshold or size_ratio >= 1.15
    # Title-like: short, few source blocks, does not read as a full sentence.
    title_like = (
        shortish
        and blocks <= 2
        and not text.endswith((".", "?", ";"))
        and len(text.split()) <= 16
    )
    looks_like_column_header = (
        chars <= 200
        and size_ratio < 1.12
        and prom < prom_threshold + 0.15
        and (text.count("/") >= 2 or text.count("\n") >= 2)
        and f.get("block_count", blocks) <= 2
    )
    # Navigational crumbs ("Back to top") — very short, no version digits.
    looks_like_nav = (
        chars <= 20
        and len(text.split()) <= 4
        and size_ratio < 1.1
        and prom < prom_threshold
        and not any(ch.isdigit() for ch in text)
    )
    # Body sentence leftover (ends with '.') should not open a section.
    looks_like_sentence = text.endswith(".") and size_ratio < 1.12 and prom <= prom_threshold + 0.05
    if looks_like_column_header or looks_like_nav or looks_like_sentence:
        return False
    # Accept either elevated prominence OR a compact title-like unit (common
    # when section titles share the body font size).
    return bool((prominent and shortish and mostly_head) or title_like)


class SectionGroupDiscovery:
    def __init__(
        self,
        config: EngineConfig,
        backend: EmbeddingBackend,
        logger: Optional[DiscoveryLogger] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.logger = logger

    def discover(
        self,
        document: Document,
        logical_blocks: List[LogicalBlock],
    ) -> List[SectionGroup]:
        log = self.logger
        if not logical_blocks:
            return []

        if log:
            log.section("SECTION GROUP DISCOVERY")
            log.line("Discovering section headings from relative layout evidence...")

        # Prominence threshold = 70th percentile of head prominence (relative).
        proms = [
            b.structural_features.get("head_prominence", b.structural_features.get("mean_prominence", 0.0))
            for b in logical_blocks
        ]
        prom_threshold = float(np.percentile(proms, 70)) if proms else 0.8

        ordered = sorted(logical_blocks, key=lambda b: (b.source_page, b.doc_position, b.id))
        heading_flags = [_is_section_heading(b, prom_threshold) for b in ordered]

        # Prefer the stronger of consecutive heading candidates (umbrella title
        # followed by a more specific section title): keep both if both pass,
        # membership of intervening empty sections is fine.
        sections: List[SectionGroup] = []
        open_heading: Optional[LogicalBlock] = None
        open_members: List[LogicalBlock] = []
        section_idx = 0

        def _flush() -> None:
            nonlocal section_idx, open_heading, open_members
            if open_heading is None:
                return
            section_idx += 1
            members = [m for m in open_members if m.id != open_heading.id]
            source_ids: List[str] = []
            for m in members:
                source_ids.extend(m.source_block_ids)
            pages = [open_heading.source_page] + [m.source_page for m in members]
            ev = Evidence(
                signals={
                    "heading_prominence": open_heading.structural_features.get(
                        "head_prominence", open_heading.structural_features.get("mean_prominence", 0.0)
                    ),
                    "heading_size_ratio": open_heading.structural_features.get("head_size_ratio", 1.0),
                    "member_count": float(len(members)),
                    "page_span": float(max(pages) - min(pages) + 1),
                },
                weights={
                    "heading_prominence": 0.4,
                    "heading_size_ratio": 0.2,
                    "member_count": 0.3,
                    "page_span": 0.1,
                },
                confidence=min(
                    1.0,
                    0.45
                    + 0.15 * min(1.0, open_heading.structural_features.get("head_size_ratio", 1.0) - 1.0)
                    + 0.05 * min(10, len(members)),
                ),
                notes=["section opened by relative prominence / size elevation"],
            )
            sg = SectionGroup(
                id=f"{document.id}_section_{section_idx:03d}",
                document_id=document.id,
                heading_block_id=open_heading.id,
                heading_text=_preview(open_heading.text, 200),
                page_start=min(pages),
                page_end=max(pages),
                member_logical_block_ids=[m.id for m in members],
                member_source_block_ids=source_ids,
                evidence=ev,
            )
            for m in members:
                m.section_group_id = sg.id
            open_heading.section_group_id = sg.id
            sections.append(sg)
            open_heading = None
            open_members = []

        preamble: List[LogicalBlock] = []
        for block, is_head in zip(ordered, heading_flags):
            if is_head:
                if open_heading is None and preamble:
                    # Materialise a preamble section for leading non-heading content.
                    open_heading = LogicalBlock(
                        id=f"{document.id}_preamble_heading",
                        content_unit_id="",
                        document_id=document.id,
                        source_document=document.source_path.split("/")[-1],
                        source_page=preamble[0].source_page,
                        source_block_ids=[],
                        text="(document preamble — no section heading yet)",
                        structural_features={"head_prominence": 0.0, "head_size_ratio": 1.0, "char_count": 0, "block_count": 0},
                        confidence=0.4,
                        evidence=Evidence(confidence=0.4, notes=["synthetic preamble heading"]),
                    )
                    open_members = list(preamble)
                    preamble = []
                    _flush()
                elif open_heading is not None:
                    _flush()
                open_heading = block
                open_members = [block]
            else:
                if open_heading is None:
                    preamble.append(block)
                else:
                    open_members.append(block)

        if open_heading is not None:
            _flush()
        elif preamble:
            # Entire slice had no strong headings — one catch-all section.
            open_heading = LogicalBlock(
                id=f"{document.id}_preamble_heading",
                content_unit_id="",
                document_id=document.id,
                source_document=document.source_path.split("/")[-1],
                source_page=preamble[0].source_page,
                source_block_ids=[],
                text="(document content — no section heading detected)",
                structural_features={"head_prominence": 0.0, "head_size_ratio": 1.0, "char_count": 0, "block_count": 0},
                confidence=0.35,
                evidence=Evidence(confidence=0.35),
            )
            open_members = list(preamble)
            _flush()

        self._maybe_label(sections)
        sections = self._collapse_empty_umbrellas(sections)

        if log:
            for sg in sections:
                generic = sg.id.split("_section_")[-1] if "_section_" in sg.id else sg.id
                display = f"DiscoveredSection_{generic}"
                log.event(
                    "section_group_created",
                    document_id=document.id,
                    section_group_id=sg.id,
                    heading_text=sg.heading_text,
                    page_start=sg.page_start,
                    page_end=sg.page_end,
                    item_count=sg.item_count,
                    member_logical_block_ids=sg.member_logical_block_ids,
                    confidence=round(sg.evidence.confidence, 4),
                    evidence=sg.evidence.to_dict(),
                )
                log.push()
                log.line(f"SectionGroup: {display}")
                log.push()
                log.kv("Heading", sg.heading_text)
                log.kv("Pages", f"{sg.page_start}-{sg.page_end}")
                log.kv("Items", sg.item_count)
                log.kv("Source blocks", len(sg.member_source_block_ids))
                log.evidence_block(sg.evidence.signals)
                log.pop()
                log.pop()

        return sections

    def _collapse_empty_umbrellas(self, sections: List[SectionGroup]) -> List[SectionGroup]:
        """Fold heading-only sections into the next content section when useful.

        Example: umbrella title "New Features and Bug Fixes…" (0 items) followed by
        "New Features in Siebel CRM 26.3" (N items) → keep the content section and
        note the umbrella in evidence.
        """
        if len(sections) < 2:
            return sections
        out: List[SectionGroup] = []
        i = 0
        while i < len(sections):
            cur = sections[i]
            if (
                cur.item_count == 0
                and i + 1 < len(sections)
                and sections[i + 1].item_count > 0
                and sections[i + 1].page_start >= cur.page_start
                and cur.inferred_label
                and sections[i + 1].inferred_label == cur.inferred_label
            ):
                nxt = sections[i + 1]
                nxt.evidence.notes.append(f"preceded by umbrella heading: {cur.heading_text}")
                out.append(nxt)
                i += 2
                continue
            out.append(cur)
            i += 1
        return out

    def _maybe_label(self, sections: List[SectionGroup]) -> None:
        """OPTIONAL labelling from a user lexicon — never used for grouping."""
        lexicon = self.config.optional_label_lexicon
        if not lexicon or not sections:
            return
        labels = list(lexicon.keys())
        seed_texts = [" ".join(lexicon[l]) for l in labels]
        seed_vecs = self.backend.embed(seed_texts)
        heading_vecs = self.backend.embed([s.heading_text for s in sections])
        for s, hv in zip(sections, heading_vecs):
            sims = [cosine(hv, seed_vecs[i]) for i in range(len(labels))]
            best = int(np.argmax(sims)) if sims else -1
            if best >= 0 and sims[best] >= 0.12:
                s.inferred_label = labels[best]
                s.label_confidence = float(sims[best])
                s.evidence.notes.append(
                    f"optional label '{labels[best]}' from heading similarity {sims[best]:.2f}"
                )


def write_human_section_log(
    path: str,
    document: Document,
    logical_blocks: List[LogicalBlock],
    sections: List[SectionGroup],
    page_limit: Optional[int] = None,
) -> None:
    """Write a concise, human-readable section→items→blocks log.

    Uses only discovered generic ids (no business category names).
    """
    by_id = {b.id: b for b in logical_blocks}
    lines: List[str] = []
    A = lines.append
    page_note = f" (pages 1-{page_limit})" if page_limit else ""
    A("=" * 78)
    A(f"HUMAN-READABLE DISCOVERY LOG — {document.source_path.split('/')[-1]}{page_note}")
    A("Groups are discovered from layout evidence (generic). No hardcoded")
    A("category names (Feature / Fix / Bug / …) are used.")
    A("=" * 78)
    A("")
    A("[DOCUMENT]")
    A(f"  File: {document.source_path}")
    A(f"  Document id: {document.id}")
    A(f"  Pages processed: {document.page_count}")
    A(f"  Raw text blocks: {len(document.blocks)}")
    A(f"  Logical blocks: {len(logical_blocks)}")
    A(f"  Discovered section groups: {len(sections)}")
    A("")

    # Document order only — never sort by business label.
    sections_sorted = sorted(sections, key=lambda s: (s.page_start, s.id))

    A("[DISCOVERED SECTION SUMMARY]")
    for sg in sections_sorted:
        # Generic display name derived from the discovered id, e.g. section_007 → DiscoveredSection_007
        generic = sg.id.split("_section_")[-1] if "_section_" in sg.id else sg.id
        display = f"DiscoveredSection_{generic}"
        A(
            f"  • {display:28}  {sg.item_count:3d} items  "
            f"pages {sg.page_start}-{sg.page_end}  | heading: {sg.heading_text[:60]}"
        )
    A("")

    for sg in sections_sorted:
        generic = sg.id.split("_section_")[-1] if "_section_" in sg.id else sg.id
        display = f"DiscoveredSection_{generic}"
        A("-" * 78)
        A(f"[SECTION GROUP] {display}")
        A(f"  section_id : {sg.id}")
        A(f"  heading    : {sg.heading_text}")
        A(f"  pages      : {sg.page_start}-{sg.page_end}")
        A(f"  items      : {sg.item_count}")
        A(f"  source blocks in group: {len(sg.member_source_block_ids)}")
        A("  discovery evidence:")
        for k, v in sg.evidence.signals.items():
            A(f"    - {k}: {v:.2f}" if isinstance(v, float) else f"    - {k}: {v}")
        A("")
        if not sg.member_logical_block_ids:
            A("  (no member items — heading only)")
            A("")
            continue

        for i, mid in enumerate(sg.member_logical_block_ids, start=1):
            b = by_id.get(mid)
            if not b:
                continue
            A(f"  [ITEM {i}] {b.id}")
            A(f"    page            : {b.source_page}")
            A(f"    confidence      : {b.confidence:.2f}")
            A(f"    pattern         : {b.discovered_pattern}")
            A(f"    role sequence   : {', '.join(b.role_sequence) or '-'}")
            A(f"    source blocks   : {', '.join(b.source_block_ids)}")
            if b.evidence and b.evidence.signals:
                A("    item evidence   :")
                for k, v in b.evidence.signals.items():
                    A(f"      - {k}: {v:.2f}" if isinstance(v, float) else f"      - {k}: {v}")
            preview = _preview(b.text, 160)
            A(f"    text            : {preview}")
            A("")

    A("=" * 78)
    A("[END OF LOG]")
    A("=" * 78)
    A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
