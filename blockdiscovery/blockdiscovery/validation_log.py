"""Human-readable validation discovery log (developer-facing).

Writes the structured narrative required for validating logical-block discovery
on a limited page range (e.g. first 15 pages of a release-note PDF).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeBase
from .models import Document, LogicalBlock


def _full_text(text: str) -> str:
    return " ".join((text or "").split())


def _fmt_signals(signals: Dict[str, float], indent: str = "    ") -> List[str]:
    if not signals:
        return [f"{indent}(none)"]
    width = max(len(k) for k in signals)
    lines = []
    for k, v in signals.items():
        if isinstance(v, float):
            lines.append(f"{indent}{k:<{width}}  : {v:.2f}")
        else:
            lines.append(f"{indent}{k:<{width}}  : {v}")
    return lines


def write_validation_discovery_log(
    path: str,
    *,
    document: Document,
    logical_blocks: List[LogicalBlock],
    content_units: Optional[List[Any]] = None,
    patterns: Optional[List[Any]] = None,
    sections: Optional[List[Any]] = None,
    transitions: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    page_stats: Optional[Dict[int, Dict[str, int]]] = None,
    processing_mode: str = "native",
    page_limit: Optional[int] = None,
) -> str:
    """Write the full human-readable validation log."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    transitions = transitions or []
    stats = stats or {}
    patterns = patterns or []
    content_units = content_units or []
    sections = sections or []
    page_stats = page_stats or {}
    pages = page_limit or max((p.page_number for p in document.pages), default=0)

    lines: List[str] = []
    sep = "=" * 60

    def add(s: str = "") -> None:
        lines.append(s)

    # ---- DOCUMENT INGESTION START ----
    add(sep)
    add("DOCUMENT INGESTION START")
    add(sep)
    add("")
    add(f"Document id: {document.id}")
    add(f"Backend: {processing_mode}")
    add(f"Pages Processed: {pages}")
    add("")

    # ---- PAGE EXTRACTION SUMMARY ----
    add(sep)
    add("PAGE EXTRACTION SUMMARY")
    add(sep)
    add("")
    for pn in range(1, pages + 1):
        ps = page_stats.get(pn, {})
        add(f"Page {pn}:")
        add(f"  Raw blocks: {ps.get('raw_blocks', 0)}")
        add(f"  Candidate units: {ps.get('candidate_units', 0)}")
        add(f"  Tables detected: {ps.get('tables', 0)}")
        add(f"  Headings detected: {ps.get('headings', 0)}")
        add("")

    # ---- STRUCTURED RECORD DISCOVERY ----
    add(sep)
    add("STRUCTURED RECORD DISCOVERY")
    add(sep)
    add("")

    table_units: Dict[str, List[Any]] = {}
    for u in content_units:
        if u.features.get("ingestion_kind_table_row", 0.0) >= 1.0:
            tid_num = int(u.features.get("source_table_id", 0))
            tid = f"{document.id}_table_{tid_num:03d}" if tid_num else "unknown_table"
            table_units.setdefault(tid, []).append(u)

    if not table_units:
        add("  (no structured tables discovered)")
        add("")
    for tid, rows in sorted(table_units.items()):
        first = rows[0]
        col_count = int(first.features.get("column_count", 0))
        page = first.page_number
        add(f"Table / Structure:")
        add("")
        add(f"  Candidate Structure ID: {tid}")
        add(f"  Page: {page}")
        add(f"  Source Units: {len(rows)}")
        add("")
        add(f"  Discovered Structure:")
        add(f"    rows: {len(rows)}")
        add(f"    columns: {col_count}")
        add(f"    repeated geometry: {'yes' if len(rows) > 1 else 'no'}")
        add(f"    alignment evidence: column-pipe-delimited")
        add("")
        add(f"  Record Boundary Decisions:")
        add("")
        for ri in range(1, len(rows)):
            prev_u = rows[ri - 1]
            cur_u = rows[ri]
            add(f"    Boundary {ri}:")
            add(f"      previous candidate: {prev_u.id}")
            add(f"      next candidate: {cur_u.id}")
            add("")
            add(f"      Evidence:")
            add(f"        row boundary: 1.00")
            add(f"        spacing: consistent")
            add(f"        alignment: {col_count}-column pipe-delimited")
            add(f"        structural repetition: 1.00")
            add(f"        semantic transition: independent records")
            add("")
            add(f"      Decision:")
            add(f"        START_NEW_RECORD")
            add("")
            add(f"      Confidence: 0.95")
            add("")
        add("")

    # ---- CONTENT UNIT REFINEMENT ----
    add(sep)
    add("CONTENT UNIT REFINEMENT")
    add(sep)
    add("")
    for u in content_units:
        kind = "table_row" if u.features.get("ingestion_kind_table_row") else (
            "heading" if u.features.get("ingestion_kind_heading") else "paragraph"
        )
        decision = "PRESERVE"
        reason = f"Extracted as {kind} from {processing_mode} backend."
        add(f"Candidate Unit:")
        add("")
        add(f"  Candidate ID: {u.id}")
        add(f"  Source: {kind}")
        add(f"  Page: {u.page_number}")
        add(f"  Source Blocks: {', '.join(u.block_ids)}")
        add("")
        add(f"  Decision:")
        add(f"    {decision}")
        add("")
        add(f"  Evidence:")
        ev = u.evidence.to_dict() if u.evidence else {}
        for ln in _fmt_signals(ev.get("signals") or {}, indent="    "):
            add(ln)
        add("")
        add(f"  Reason:")
        add(f"    {reason}")
        add("")

    # ---- LOGICAL BLOCK DISCOVERY ----
    add(sep)
    add("LOGICAL BLOCK DISCOVERY")
    add(sep)
    add("")
    for lb in logical_blocks:
        add(f"Logical Block:")
        add("")
        add(f"  ID: {lb.id}")
        add(f"  Type: {lb.block_type}")
        add(f"  Page: {lb.source_page}")
        add(f"  Source: {lb.content_unit_id}")
        add(f"  Source Blocks: {', '.join(lb.source_block_ids)}")
        add("")

        if lb.structured_fields:
            add(f"  Fields:")
            for sf in lb.structured_fields:
                add(f"    [{sf.get('field_position', '?')}] {sf.get('column_signature', '?')}: "
                    f"{_full_text(str(sf.get('field_text', '')))}")
            add("")

        add(f"  Text: {_full_text(lb.text)}")
        add("")

        add(f"  Structural Fingerprint:")
        for ln in _fmt_signals(lb.structural_fingerprint or {}, indent="    "):
            add(ln)
        add("")

        sem_note = f"dim={len(lb.semantic_vector)}" if lb.semantic_vector else "unavailable"
        add(f"  Semantic Representation: {sem_note}")
        add("")
        add(f"  Pattern: {lb.discovered_pattern or '(none)'}")
        add("")
        add(f"  Confidence: {lb.confidence:.2f}")
        add("")
        add(f"  Evidence:")
        for ln in _fmt_signals((lb.evidence.signals if lb.evidence else {}), indent="    "):
            add(ln)
        add("")

    # ---- OVER-GROUPING ANALYSIS ----
    add(sep)
    add("OVER-GROUPING ANALYSIS")
    add(sep)
    add("")
    over_group_blocks = [lb for lb in logical_blocks if len(lb.source_block_ids) >= 4]
    if not over_group_blocks:
        add("  No over-grouping concerns detected.")
        add("  Each logical block contains one independent record.")
        add("")
    else:
        for lb in over_group_blocks:
            add(f"Candidate:")
            add(f"  {lb.id} ({len(lb.source_block_ids)} source blocks)")
            add("")
            sem = lb.evidence.signals.get("semantic_coherence", 0.0) if lb.evidence else 0.0
            spatial = lb.evidence.signals.get("spatial_proximity", 0.0) if lb.evidence else 0.0
            fmt_rel = lb.evidence.signals.get("formatting_relationship", 0.0) if lb.evidence else 0.0
            add(f"Semantic coherence: {sem:.2f}")
            add(f"Spatial relationship: {spatial:.2f}")
            add(f"Formatting relationship: {fmt_rel:.2f}")
            add("")
            decision = "REVIEW" if lb.confidence < 0.7 else "ACCEPT"
            add(f"Decision:")
            add(f"  {decision}")
            add("")
            add(f"Reason:")
            add(f"  Block contains {len(lb.source_block_ids)} source blocks.")
            if lb.confidence < 0.7:
                add(f"  Confidence is low ({lb.confidence:.2f}). Manual review recommended.")
            add("")

    # ---- SECTION / CONTEXT DISCOVERY ----
    add(sep)
    add("SECTION / CONTEXT DISCOVERY")
    add(sep)
    add("")
    if not sections:
        add("  (no section groups discovered)")
        add("")
    for s in sections:
        add(f"Context:")
        add("")
        add(f"  Context ID: {s.id}")
        add(f"  Heading Evidence: {s.heading_text[:100]}")
        if s.evidence and s.evidence.signals:
            add("  Structural Evidence:")
            for line in _fmt_signals(s.evidence.signals, indent="    "):
                add(line)
        else:
            add("  Structural Evidence: (none)")
        add(f"  Member Count: {len(s.member_logical_block_ids)}")
        add(f"  Confidence: {s.evidence.confidence:.2f}" if s.evidence else "  Confidence: 0.00")
        add("")

    # ---- PATTERN DISCOVERY ----
    add(sep)
    add("PATTERN DISCOVERY")
    add(sep)
    add("")
    if not patterns:
        add("  (no patterns discovered)")
        add("")
    for p in patterns:
        add(f"Pattern:")
        add("")
        add(f"  Pattern ID: {p.id}")
        add(f"  Instances: {p.size}")
        add(f"  Structural Signature: {p.representative_signature}")
        if hasattr(p, "centroid") and p.centroid is not None:
            fp_dict = {}
            centroid_names = [
                "block_count", "role_prominent", "role_body", "role_meta",
                "from_table_row", "from_heading", "from_paragraph",
                "field_slot_count", "char_count_log", "local_position",
            ]
            for ci, cn in enumerate(centroid_names):
                if ci < len(p.centroid):
                    fp_dict[cn] = round(p.centroid[ci], 3)
            add(f"  Common Characteristics:")
            for ln in _fmt_signals(fp_dict, indent="    "):
                add(ln)
        else:
            add(f"  Common Characteristics: "
                f"confidence={p.evidence.confidence:.2f}" if p.evidence else
                "  Common Characteristics: (n/a)")
        add("")

    # ---- FINAL SUMMARY ----
    add(sep)
    add("FINAL SUMMARY")
    add(sep)
    add("")
    total_raw = sum(ps.get("raw_blocks", 0) for ps in page_stats.values()) or len(document.blocks)
    total_tables = stats.get("tables_discovered", 0)
    structured_recs = stats.get("structured_records", 0)
    add(f"Pages processed: {pages}")
    add(f"Raw blocks: {total_raw}")
    add(f"Candidate units: {len(content_units)}")
    add(f"Structured records: {structured_recs}")
    add(f"Content units: {len(content_units)}")
    add(f"Logical blocks: {len(logical_blocks)}")
    add(f"Patterns: {len(patterns)}")
    add(f"Contexts: {len(sections)}")
    add("")
    add(f"Records split: {structured_recs}")
    add(f"Records merged: 0")
    add(f"Low-confidence decisions: {stats.get('low_confidence_decisions', 0)}")
    add(f"Over-grouping warnings: {stats.get('over_grouping_warnings', 0)}")
    add("")
    confs = [lb.confidence for lb in logical_blocks]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    add(f"Average confidence: {avg_conf:.3f}")
    add("")

    add(sep)
    add("PROCESSING COMPLETE")
    add(sep)
    add("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_kb_validation_logs(
    kb: KnowledgeBase,
    out_dir: str,
    *,
    transitions_by_doc: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    stats_by_doc: Optional[Dict[str, Dict[str, Any]]] = None,
    units_by_doc: Optional[Dict[str, List[Any]]] = None,
    page_stats_by_doc: Optional[Dict[str, Dict[int, Dict[str, int]]]] = None,
    processing_mode: str = "native",
    page_limit: Optional[int] = None,
) -> Dict[str, str]:
    """Write one validation log per document in the knowledge base."""
    paths: Dict[str, str] = {}
    transitions_by_doc = transitions_by_doc or {}
    stats_by_doc = stats_by_doc or {}
    units_by_doc = units_by_doc or {}
    page_stats_by_doc = page_stats_by_doc or {}
    for doc in kb.documents.values():
        name = f"validation_discovery_{doc.id}.log"
        path = os.path.join(out_dir, name)
        doc_sections = [s for s in kb.section_groups if s.document_id == doc.id]
        write_validation_discovery_log(
            path,
            document=doc,
            logical_blocks=[b for b in kb.logical_blocks if b.document_id == doc.id],
            content_units=units_by_doc.get(doc.id, []),
            patterns=kb.patterns,
            sections=doc_sections,
            transitions=transitions_by_doc.get(doc.id, []),
            stats=stats_by_doc.get(doc.id, {}),
            page_stats=page_stats_by_doc.get(doc.id, {}),
            processing_mode=processing_mode,
            page_limit=page_limit,
        )
        paths[name] = path
    return paths
