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

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .config import EngineConfig
from .logging_utils import DiscoveryLogger
from .models import Document, Evidence, LogicalBlock, SectionGroup
from .semantics import EmbeddingBackend, cosine


def _full_text(text: str) -> str:
    """Preserve every word for human-readable logs (collapse runs of whitespace only)."""
    return " ".join((text or "").split())


def _is_section_heading(block: LogicalBlock, prom_threshold: float) -> bool:
    """Relative heading test -- never absolute font size or fixed keywords."""
    f = block.structural_features
    prom = f.get("head_prominence", f.get("mean_prominence", f.get("prominence", 0.0)))
    size_ratio = f.get("head_size_ratio", 1.0)
    chars = f.get("char_count", float(len(block.text)))
    blocks = f.get("block_count", float(len(block.source_block_ids)))
    roles = block.role_sequence or []
    text = (block.text or "").strip()
    words = len(text.split())
    marker_depth = f.get("marker_depth", 0.0)
    layout_section = f.get("layout_section_header", 0.0) >= 1.0
    layout_page = f.get("layout_page_header", 0.0) >= 1.0
    layout_header = layout_section or layout_page
    # Repeated running chrome (true page headers) never opens a section.
    # Unique page-header-styled titles still may (extractor often marks release
    # band titles this way).
    if layout_page and (
        f.get("repetition_count", 1.0) >= 2.0 or f.get("is_repeated_shape", 0.0) >= 1.0
    ):
        return False

    # Structured table rows / list items are never section openers.
    if block.block_type == "structured_record":
        return False
    if f.get("layout_list_item", 0.0) >= 1.0:
        return False
    # Enumerated / bullet lines are items, never section openers.
    lead = ""
    for ch in text:
        if not ch.isspace():
            lead = ch
            break
    if lead and not lead.isalnum() and lead not in {'"', "'", "(", "[", "{"}:
        return False
    # Lowercase-start fragments are body continuations, not headings.
    if lead and lead.islower():
        return False
    # Numbered list openers (digits then '.' or ')') are items, not headings.
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    if (
        i > 0
        and i < len(text)
        and text[i] in ".)"
        and (i + 1 >= len(text) or text[i + 1].isspace())
    ):
        return False

    shortish = chars <= 120 and words <= 14
    mostly_head = blocks <= 2 or (roles.count("PROMINENT") >= 1 and roles.count("BODY") <= 1)
    prominent = (
        prom >= prom_threshold
        or size_ratio >= 1.15
        or marker_depth >= 1.0
        or layout_header
    )
    # Title-like: short, few source blocks, does not read as a full sentence.
    title_like = (
        shortish
        and blocks <= 2
        and not text.endswith((".", "?", ";", "!"))
        and words <= 14
    )
    looks_like_column_header = (
        chars <= 200
        and size_ratio < 1.12
        and prom < prom_threshold + 0.15
        and not layout_header
        and (text.count("/") >= 2 or text.count("\n") >= 2)
        and f.get("block_count", blocks) <= 2
    )
    # Navigational crumbs: very short labels without identifiers. Real section
    # titles that are short still pass when unique (handled below via layout /
    # title_like); repeated chrome is demoted later in discover().
    looks_like_nav = (
        chars <= 12
        and words <= 2
        and size_ratio < 1.1
        and prom < prom_threshold
        and not layout_header
        and not any(ch.isdigit() for ch in text)
    )
    # Prose / warning sentences (even if bold / section-header styled) must not
    # open a section — they remain items under the previous heading.
    looks_like_sentence = text.endswith((".", "?", "!")) and words >= 4
    # Intro / instruction lines that end with ':' introduce following items;
    # they are not section openers themselves.
    looks_like_intro = text.endswith(":") and words >= 4
    if looks_like_column_header or looks_like_nav or looks_like_sentence or looks_like_intro:
        return False
    # Extractor section-header layout + compact title shape is strong evidence.
    if layout_header and title_like:
        return True
    # Accept either elevated prominence OR a compact title-like unit (common
    # when section titles share the body font size, e.g. archive-file headings).
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

        # Identical micro-labels that recur (e.g. page chrome) are not section
        # openers — demote by repetition evidence only.
        micro = [
            (b.text or "").strip()
            for b, flag in zip(ordered, heading_flags)
            if flag and len((b.text or "").split()) <= 4 and len((b.text or "").strip()) <= 24
        ]
        micro_counts = Counter(micro)
        for i, (block, flag) in enumerate(zip(ordered, heading_flags)):
            if not flag:
                continue
            key = (block.text or "").strip()
            if micro_counts.get(key, 0) >= 2 and len(key.split()) <= 4:
                heading_flags[i] = False

        # Stacked same-page titles stay as headings when both qualify — the
        # hierarchy stack nests the later (usually narrower) title under the
        # earlier container instead of flattening it to an item.

        # Hierarchy from extractor layout roles (generic, not vocabulary):
        #   page-header → never a section opener
        #   mid-stream interrupters in a continuing multi-column table → items
        # Nested titles / leaf table headings stay as headings so a score-based
        # stack can nest them (umbrella → band → features/fixes leaf).
        def _field_width(block: LogicalBlock) -> int:
            fields = block.structured_fields or []
            if not fields:
                return 0
            return 1 + max(int(f.get("field_position", 0)) for f in fields)

        def _nearby_structured(
            idx: int, direction: int, limit: int = 8, respect_heads: bool = True
        ) -> Optional[LogicalBlock]:
            step = 1 if direction > 0 else -1
            j = idx + step
            seen = 0
            while 0 <= j < len(ordered) and seen < limit:
                b = ordered[j]
                if respect_heads and heading_flags[j]:
                    break
                if b.block_type == "structured_record" and _field_width(b) >= 2:
                    return b
                j += step
                seen += 1
            return None

        for i, (block, flag) in enumerate(zip(ordered, heading_flags)):
            if not flag:
                continue
            f = block.structural_features
            # Repeated page chrome only — unique page-header-styled titles stay.
            if f.get("layout_page_header", 0.0) >= 1.0 and (
                f.get("repetition_count", 1.0) >= 2.0
                or f.get("is_repeated_shape", 0.0) >= 1.0
            ):
                heading_flags[i] = False
                continue
            # Heading sitting inside a continuing multi-column record stream
            # (same column width before and after). Extractors often assign a
            # new grid id across pages/fragments — keep as an item so a single
            # leaf section owns the whole table.
            prev_s = _nearby_structured(i, -1)
            next_s = _nearby_structured(i, 1)
            if (
                prev_s is not None
                and next_s is not None
                and _field_width(prev_s) == _field_width(next_s)
                and _field_width(prev_s) >= 3
            ):
                heading_flags[i] = False

        # Relative hierarchy score among remaining headings. Higher = shallower
        # (closer to the document root). Signals are layout/shape/adjacency only.
        # Major section-header peers share one score tier so they stay siblings
        # (not nested under the document title).
        def _heading_score(idx: int) -> float:
            block = ordered[idx]
            f = block.structural_features
            text = (block.text or "").strip()
            words = len(text.split())
            layout_sec = f.get("layout_section_header", 0.0) >= 1.0
            layout_page = f.get("layout_page_header", 0.0) >= 1.0
            next_head = idx + 1 < len(ordered) and heading_flags[idx + 1]
            imm_table = _nearby_structured(idx, 1, limit=2) is not None
            # Peer major sections: extractor section-header not opening onto a grid.
            if layout_sec and not imm_table:
                return 100.0
            # Container band: next unit is another heading, or a unique
            # page-header-styled title that wraps nested leaves.
            if next_head or (layout_page and not imm_table and words >= 8):
                return 55.0 + (10.0 if words >= 8 else 0.0)
            # Leaf subsection: immediately above a table, or a compact title
            # that does not wrap further headings (so empty-feature peers still
            # sit beside Fixes rather than parenting them).
            if imm_table or (not next_head and words <= 8):
                return 25.0
            # Soft subsection without an immediate table.
            return 40.0 + (10.0 if words >= 8 else 0.0)

        heading_scores: Dict[int, float] = {
            i: _heading_score(i) for i, flag in enumerate(heading_flags) if flag
        }

        # Nested section stack: open sections until a same-or-higher score peer
        # arrives; children attach via parent_section_id / child_section_ids.
        sections: List[SectionGroup] = []
        # stack entries: (score, SectionGroup being filled, member blocks)
        stack: List[tuple] = []
        section_idx = 0

        def _close_top() -> None:
            nonlocal section_idx
            if not stack:
                return
            _score, sg, members = stack.pop()
            body = [m for m in members if m.id != sg.heading_block_id]
            source_ids: List[str] = []
            for m in body:
                source_ids.extend(m.source_block_ids)
            pages = [sg.page_start] + [m.source_page for m in body]
            # Extend page_end for descendants already closed under this node.
            page_end = max([sg.page_end] + pages)
            for cid in sg.child_section_ids:
                child = next((s for s in sections if s.id == cid), None)
                if child is not None:
                    page_end = max(page_end, child.page_end)
            sg.page_end = page_end
            sg.member_logical_block_ids = [m.id for m in body]
            sg.member_source_block_ids = source_ids
            sg.evidence.signals["member_count"] = float(len(body))
            sg.evidence.signals["page_span"] = float(page_end - sg.page_start + 1)
            sg.evidence.signals["depth"] = float(sg.depth)
            sg.evidence.signals["hierarchy_score"] = float(_score)
            for m in body:
                m.section_group_id = sg.id
            # Heading block is the opener — tag it too when present in ordered.
            for b in ordered:
                if b.id == sg.heading_block_id:
                    b.section_group_id = sg.id
                    break
            sections.append(sg)

        def _open_section(block: LogicalBlock, score: float) -> None:
            nonlocal section_idx
            while stack and stack[-1][0] <= score:
                _close_top()
            parent_id = stack[-1][1].id if stack else None
            depth = (stack[-1][1].depth + 1) if stack else 0
            section_idx += 1
            ev = Evidence(
                signals={
                    "heading_prominence": block.structural_features.get(
                        "head_prominence",
                        block.structural_features.get("mean_prominence", 0.0),
                    ),
                    "heading_size_ratio": block.structural_features.get(
                        "head_size_ratio", 1.0
                    ),
                    "member_count": 0.0,
                    "page_span": 1.0,
                    "depth": float(depth),
                    "hierarchy_score": float(score),
                },
                weights={
                    "heading_prominence": 0.35,
                    "heading_size_ratio": 0.2,
                    "member_count": 0.25,
                    "page_span": 0.1,
                    "depth": 0.1,
                },
                confidence=min(
                    1.0,
                    0.45
                    + 0.15
                    * min(
                        1.0,
                        block.structural_features.get("head_size_ratio", 1.0) - 1.0,
                    )
                    + 0.05 * max(0, 3 - depth),
                ),
                notes=[
                    "section opened by relative prominence / hierarchy score",
                    f"depth={depth}",
                ],
            )
            sg = SectionGroup(
                id=f"{document.id}_section_{section_idx:03d}",
                document_id=document.id,
                heading_block_id=block.id,
                heading_text=_full_text(block.text),
                page_start=block.source_page,
                page_end=block.source_page,
                depth=depth,
                parent_section_id=parent_id,
                evidence=ev,
            )
            if parent_id:
                for _sc, parent_sg, _mem in stack:
                    if parent_sg.id == parent_id:
                        parent_sg.child_section_ids.append(sg.id)
                        break
            stack.append((score, sg, [block]))

        preamble: List[LogicalBlock] = []
        for i, (block, is_head) in enumerate(zip(ordered, heading_flags)):
            if is_head:
                if not stack and preamble:
                    # Materialise a preamble section for leading non-heading content.
                    synth = LogicalBlock(
                        id=f"{document.id}_preamble_heading",
                        content_unit_id="",
                        document_id=document.id,
                        source_document=document.id,
                        source_page=preamble[0].source_page,
                        source_block_ids=[],
                        text="(document preamble — no section heading yet)",
                        structural_features={
                            "head_prominence": 0.0,
                            "head_size_ratio": 1.0,
                            "char_count": 0,
                            "block_count": 0,
                        },
                        confidence=0.4,
                        evidence=Evidence(
                            confidence=0.4, notes=["synthetic preamble heading"]
                        ),
                    )
                    _open_section(synth, score=1000.0)
                    stack[-1][2].extend(preamble)
                    preamble = []
                    _close_top()
                _open_section(block, heading_scores.get(i, 0.0))
            else:
                if not stack:
                    preamble.append(block)
                else:
                    stack[-1][2].append(block)

        while stack:
            _close_top()
        if preamble:
            # Entire slice had no strong headings — one catch-all section.
            synth = LogicalBlock(
                id=f"{document.id}_preamble_heading",
                content_unit_id="",
                document_id=document.id,
                source_document=document.id,
                source_page=preamble[0].source_page,
                source_block_ids=[],
                text="(document content — no section heading detected)",
                structural_features={
                    "head_prominence": 0.0,
                    "head_size_ratio": 1.0,
                    "char_count": 0,
                    "block_count": 0,
                },
                confidence=0.35,
                evidence=Evidence(confidence=0.35),
            )
            _open_section(synth, score=1000.0)
            stack[-1][2].extend(preamble)
            while stack:
                _close_top()

        # sections were appended in close-order (children before parents).
        # Reorder to document / nesting order: parents before children.
        by_id = {s.id: s for s in sections}
        roots = [s for s in sections if not s.parent_section_id]
        # Preserve discovery order among siblings via section id sort key.
        roots.sort(key=lambda s: s.id)

        ordered_sections: List[SectionGroup] = []

        def _walk(node: SectionGroup) -> None:
            ordered_sections.append(node)
            for cid in node.child_section_ids:
                child = by_id.get(cid)
                if child is not None:
                    _walk(child)

        for root in roots:
            _walk(root)
        # Any orphans (should be none) append at end.
        seen = {s.id for s in ordered_sections}
        for s in sections:
            if s.id not in seen:
                ordered_sections.append(s)
        sections = ordered_sections

        self._maybe_label(sections)
        # Empty umbrellas with children are valid nested containers — do not collapse.

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
                log.kv("Depth", sg.depth)
                log.kv("Parent", sg.parent_section_id or "-")
                log.kv("Children", len(sg.child_section_ids))
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
    A(f"HUMAN-READABLE DISCOVERY LOG{page_note}")
    A("Groups are discovered from layout evidence (generic). No hardcoded")
    A("category names (Feature / Fix / Bug / …) are used.")
    A("=" * 78)
    A("")
    A("[DOCUMENT]")
    A(f"  Document id: {document.id}")
    A(f"  Pages processed: {document.page_count}")
    A(f"  Raw text blocks: {len(document.blocks)}")
    A(f"  Logical blocks: {len(logical_blocks)}")
    A(f"  Discovered section groups: {len(sections)}")
    A("")

    # Document / nesting order — parents before children.
    sections_sorted = list(sections)
    by_sid = {s.id: s for s in sections_sorted}

    A("[DISCOVERED SECTION SUMMARY]")
    for sg in sections_sorted:
        generic = sg.id.split("_section_")[-1] if "_section_" in sg.id else sg.id
        display = f"DiscoveredSection_{generic}"
        indent = "  " * (sg.depth + 1)
        marker = "•" if sg.depth == 0 else ("◦" if sg.depth == 1 else "▪")
        A(
            f"{indent}{marker} {display:28}  {sg.item_count:3d} items  "
            f"depth={sg.depth}  pages {sg.page_start}-{sg.page_end}  "
            f"| heading: {_full_text(sg.heading_text)}"
        )
    A("")

    for sg in sections_sorted:
        generic = sg.id.split("_section_")[-1] if "_section_" in sg.id else sg.id
        display = f"DiscoveredSection_{generic}"
        A("-" * 78)
        A(f"[SECTION GROUP] {display}")
        A(f"  section_id : {sg.id}")
        A(f"  heading    : {sg.heading_text}")
        A(f"  depth      : {sg.depth}")
        A(f"  parent     : {sg.parent_section_id or '-'}")
        if sg.child_section_ids:
            child_heads = [
                f"{cid} ({by_sid[cid].heading_text})" if cid in by_sid else cid
                for cid in sg.child_section_ids
            ]
            A(f"  children   : {', '.join(child_heads)}")
        else:
            A("  children   : -")
        A(f"  pages      : {sg.page_start}-{sg.page_end}")
        A(f"  items      : {sg.item_count}")
        A(f"  source blocks in group: {len(sg.member_source_block_ids)}")
        A("  discovery evidence:")
        for k, v in sg.evidence.signals.items():
            A(f"    - {k}: {v:.2f}" if isinstance(v, float) else f"    - {k}: {v}")
        A("")
        if not sg.member_logical_block_ids:
            A("  (no direct member items — nested children hold content)"
              if sg.child_section_ids
              else "  (no member items — heading only)")
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
            A(f"    text            : {_full_text(b.text)}")
            if b.structured_fields:
                A("    fields          :")
                for sf in b.structured_fields:
                    part = sf.get("field_part")
                    prefix = f"      [{sf.get('field_position', '?')}"
                    if part is not None:
                        prefix += f".{part}"
                    prefix += "]"
                    A(f"{prefix} {_full_text(str(sf.get('field_text', '')))}")
            A("")

    A("=" * 78)
    A("[END OF LOG]")
    A("=" * 78)
    A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
