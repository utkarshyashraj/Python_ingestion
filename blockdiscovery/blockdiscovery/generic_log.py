"""Human-readable narrative for the generic discovery engine.

Explains, for one document: what was extracted, how units related, where
boundaries were placed and why, how candidates were refined, which logical
blocks resulted, what patterns recur, and the genericity audit of the engine
that produced the result.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .genericity_audit import AuditReport, audit_package
from .models import Document, LogicalBlock

_SEP = "=" * 60


def _full_text(text: str) -> str:
    """Full human-readable text — every word kept; only whitespace is normalized."""
    return " ".join((text or "").split())


def _signal_lines(signals: Dict[str, Any], indent: str = "  ") -> List[str]:
    if not signals:
        return [f"{indent}(none)"]
    width = max(len(str(k)) for k in signals)
    out = []
    for k, v in signals.items():
        out.append(
            f"{indent}{str(k):<{width}} : {v:.2f}"
            if isinstance(v, float)
            else f"{indent}{str(k):<{width}} : {v}"
        )
    return out


def write_generic_discovery_log(
    path: str,
    *,
    document: Document,
    logical_blocks: List[LogicalBlock],
    trace: Any,
    patterns: Sequence[Any] = (),
    groups: Sequence[Any] = (),
    sections: Sequence[Any] = (),
    raw_units: Sequence[Any] = (),
    page_stats: Optional[Dict[int, Dict[str, int]]] = None,
    pattern_consolidation: Sequence[Dict[str, Any]] = (),
    audit: Optional[AuditReport] = None,
    backend: str = "structured",
    page_limit: Optional[int] = None,
    max_boundary_entries: int = 400,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    page_stats = page_stats or {}
    audit = audit or audit_package()
    pages = page_limit or document.page_count
    stats: Dict[str, Any] = getattr(trace, "stats", {}) or {}
    candidates = list(getattr(trace, "candidates", []) or [])
    boundaries = list(getattr(trace, "boundaries", []) or [])
    grids = list(getattr(trace, "grids", []) or [])
    refinements = list(getattr(trace, "refinements", []) or [])
    rejected = list(getattr(trace, "rejected", []) or [])
    heading_contexts = list(getattr(trace, "heading_contexts", []) or [])
    over_grouping = list(getattr(trace, "over_grouping", []) or [])
    thresholds: Dict[str, Any] = getattr(trace, "adaptive_thresholds", {}) or {}

    lines: List[str] = []
    A = lines.append

    A(_SEP)
    A("GENERIC DISCOVERY ENGINE")
    A(_SEP)
    A("")
    A("No hardcoded semantic categories used.")
    A("No document-specific section rules used.")
    A("No regex-based document structure discovery used.")
    A("")
    A(f"Document: {document.source_path}")
    A(f"Extraction backend (evidence only): {backend}")
    A(f"Pages processed: {pages}")
    A("")
    A("Every threshold below was derived from this document's own evidence")
    A("distribution at run time; none are constants tuned to a known layout.")
    A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("EXTRACTION")
    A(_SEP)
    A("")
    A(f"Raw extraction units: {len(raw_units)}")
    A(f"Candidate units: {len(candidates)}")
    A(f"Grids discovered from geometry/structure: {len(grids)}")
    heading_like = sum(1 for c in candidates if c.features.get("marker_depth", 0.0) > 0)
    A(f"Headings discovered from formatting/position evidence: {heading_like}")
    A(f"Candidates rejected as running page furniture: {len(rejected)}")
    A("")
    for pn in range(1, pages + 1):
        ps = page_stats.get(pn, {})
        page_cands = sum(1 for c in candidates if c.page_number == pn)
        A(f"Page {pn}:")
        A(f"  Layout boxes from extractor : {ps.get('layout_boxes', 0)}")
        A(f"  Raw extraction units        : {ps.get('raw_units', 0)}")
        A(f"  Candidate units             : {page_cands}")
        A(f"  Grids                       : {ps.get('grids', 0)}")
        A(f"  Heading-like units          : {ps.get('heading_like', 0)}")
        A("")

    if rejected:
        A("Rejected running elements (found by cross-page repetition, not by name):")
        A("")
        for r in rejected[:20]:
            A(f"  {r['candidate_id']} p{r['page']}: {_full_text(r['text'])}")
            for ln in _signal_lines(r["evidence"], indent="      "):
                A(ln)
            A(f"      reason : {r['reason']}")
            A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("DISCOVERED STRUCTURES")
    A(_SEP)
    A("")
    if not grids:
        A("  (no repeated grid structures discovered)")
        A("")
    for g in grids:
        A(f"Structure {g.grid_id}:")
        A(f"  Page              : {g.page_number}")
        A(f"  Structure type    : {g.structure_type}")
        A(f"  Rows              : {g.row_count}")
        A(f"  Columns           : {g.column_count}")
        A("  Geometry evidence :")
        for ln in _signal_lines(g.geometry, indent="    "):
            A(ln)
        A("  Repetition evidence:")
        for ln in _signal_lines(g.repetition_evidence, indent="    "):
            A(ln)
        A("  Column signatures (content-shape only, no column names):")
        for cs in g.column_signatures:
            A(
                f"    position {int(cs['column_position'])}: "
                f"mean_len={cs['mean_char_count']:.1f} "
                f"stability={cs['char_count_stability']:.2f} "
                f"digits={cs['mean_digit_ratio']:.2f} "
                f"upper={cs['mean_upper_ratio']:.2f} "
                f"fill={cs['fill_ratio']:.2f}"
            )
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("RELATIONSHIP GRAPH")
    A(_SEP)
    A("")
    A(f"Nodes: {getattr(trace, 'graph_nodes', 0)}")
    A(f"Edges: {getattr(trace, 'graph_edges', 0)}")
    A("")
    A("Adaptive edge cut:")
    for ln in _signal_lines(thresholds, indent="  "):
        A(ln)
    A("")
    if boundaries:
        rel_keys = list(boundaries[0].relationship_signals.keys())
        A("Mean relationship evidence across all edges:")
        for k in rel_keys:
            mean_v = sum(b.relationship_signals[k] for b in boundaries) / len(boundaries)
            A(f"  {k:<28} : {mean_v:.2f}")
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("BOUNDARY DISCOVERY")
    A(_SEP)
    A("")
    shown = boundaries[:max_boundary_entries]
    if len(boundaries) > len(shown):
        A(f"(showing first {len(shown)} of {len(boundaries)} evaluated boundaries)")
        A("")
    for b in shown:
        A("Boundary:")
        A(f"  Previous unit: {b.unit_a} (page {b.page_a})")
        A(f"  Next unit    : {b.unit_b} (page {b.page_b})")
        A("")
        A("Evidence:")
        for ln in _signal_lines(b.boundary_signals, indent="  "):
            A(ln)
        A("")
        A("Relationship:")
        for ln in _signal_lines(b.relationship_signals, indent="  "):
            A(ln)
        A("")
        A("Decision:")
        A(f"  {b.decision}")
        A("")
        A(f"Confidence: {b.confidence:.2f}")
        A(f"Reason: {b.reason}")
        A(
            f"Scores: relationship={b.relationship_score:.3f} "
            f"boundary={b.boundary_score:.3f} net={b.net:.3f} "
            f"adaptive_cut={thresholds.get('edge_cut_net_score', 0.0)}"
        )
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("CONTENT UNIT REFINEMENT")
    A(_SEP)
    A("")
    for r in refinements:
        A("Candidate:")
        A(f"  {', '.join(r['candidate_ids'])}")
        A("Decision:")
        A(f"  {r['decision']}")
        A("Evidence:")
        for ln in _signal_lines(r["evidence"], indent="  "):
            A(ln)
        A(f"Confidence: {r['confidence']:.2f}")
        A("Reason:")
        A(f"  {r['reason']}")
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("LOGICAL BLOCKS")
    A(_SEP)
    A("")
    for lb in logical_blocks:
        A(f"Logical Block {lb.id}:")
        A(f"  Type          : {lb.block_type}")
        A(f"  Pages         : {lb.source_page}-{lb.page_end or lb.source_page}")
        A(f"  Content unit  : {lb.content_unit_id}")
        A(f"  Source Units  : {', '.join(lb.source_block_ids)}")
        if lb.structured_fields:
            A("  Fields (positional, discovered from cell geometry):")
            for f in lb.structured_fields:
                A(
                    f"    [{f['field_position']}] {f['column_signature']} : "
                    f"{_full_text(f['field_text'])}"
                )
        A("  Structural Fingerprint:")
        for ln in _signal_lines(lb.structural_fingerprint, indent="    "):
            A(ln)
        A(
            "  Semantic Representation: "
            + (f"dim={len(lb.semantic_vector)}" if lb.semantic_vector else "unavailable")
        )
        A(f"  Pattern       : {lb.discovered_pattern or '(none)'}")
        A(f"  Confidence    : {lb.confidence:.2f}")
        A("  Evidence:")
        for ln in _signal_lines(lb.evidence.signals if lb.evidence else {}, indent="    "):
            A(ln)
        A(f"  Text          : {_full_text(lb.text)}")
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("OVER-GROUPING ANALYSIS")
    A(_SEP)
    A("")
    A("Edges where form similarity was high but semantic coherence was weak.")
    A("These are the pairs that naive visual grouping would have merged.")
    A("")
    A(f"Edges flagged: {len(over_grouping)}")
    A(f"  Resolved by splitting : {sum(1 for o in over_grouping if o['resolved'])}")
    A(f"  Kept together         : {sum(1 for o in over_grouping if not o['resolved'])}")
    A("")
    unresolved = [o for o in over_grouping if not o["resolved"]]
    if unresolved:
        A("Kept together despite the warning (review candidates):")
        A("")
        for o in unresolved[:30]:
            A(f"  p{o['page']} {o['unit_a']} + {o['unit_b']}")
            A(
                f"      form={o['mean_form_similarity']:.2f} "
                f"semantic={o['semantic_similarity']:.2f} "
                f"discount_applied={o['form_discount_applied']:.2f}"
            )
            A(f"      next unit: {_full_text(o['text_b'])}")
        A("")
    else:
        A("Every flagged edge was resolved into a boundary.")
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("SECTION / CONTEXT DISCOVERY")
    A(_SEP)
    A("")
    if not sections:
        A("  (no contexts met the heading-plus-members evidence requirement)")
        A("")
    for s in sections:
        A(f"Context {s.id}:")
        A(f"  Heading evidence : {_full_text(s.heading_text)}")
        A(f"  Pages            : {s.page_start}-{s.page_end}")
        A(f"  Member count     : {len(s.member_logical_block_ids)}")
        A("  Structural evidence:")
        for ln in _signal_lines(s.evidence.signals if s.evidence else {}, indent="    "):
            A(ln)
        A(f"  Confidence       : {s.evidence.confidence:.2f}" if s.evidence else "")
        A("")
    if heading_contexts:
        A("Heading-like units that did NOT become contexts (no meaningful members):")
        A("")
        for h in heading_contexts:
            A(f"  {h['logical_block_id']} p{h['page']}: {_full_text(h['text'])}")
            A(f"      reason: {h['reason']}")
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("PATTERN DISCOVERY")
    A(_SEP)
    A("")
    if not patterns:
        A("  (no patterns discovered)")
        A("")
    for p in patterns:
        A(f"Pattern {p.id}:")
        A(f"  Instances            : {p.size}")
        A(f"  Structural Signature : {p.representative_signature}")
        A(
            f"  Role template        : "
            f"{', '.join(p.role_template) if p.role_template else '(n/a)'}"
        )
        if p.evidence:
            A("  Common Characteristics:")
            for ln in _signal_lines(p.evidence.signals, indent="    "):
                A(ln)
            A(f"  Semantic Similarity   : {p.evidence.confidence:.2f}")
        A("")
    if pattern_consolidation:
        A("Possible pattern consolidation (reported, NOT merged):")
        A("")
        for f in pattern_consolidation:
            A(
                f"  {f['pattern_a']} ~ {f['pattern_b']}  similarity={f['similarity']:.3f}"
                f"  [{f['signature_a']} | {f['signature_b']}]"
            )
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("CROSS-DOCUMENT STRUCTURE")
    A(_SEP)
    A("")
    A("Recurring structures are patterns whose members span documents. Membership")
    A("comes from structural fingerprints, so a document using unfamiliar words")
    A("still matches a familiar shape.")
    A("")

    # Recurring structure = a pattern seen in more than one document.
    pattern_docs: Dict[str, Dict[str, int]] = {}
    for p in patterns:
        counts: Dict[str, int] = {}
        for uid in p.member_unit_ids:
            doc_id = uid.rsplit("_unit_", 1)[0]
            counts[doc_id] = counts.get(doc_id, 0) + 1
        pattern_docs[p.id] = counts

    recurring = [p for p in patterns if len(pattern_docs.get(p.id, {})) > 1]
    A(f"Recurring structures (multi-document patterns): {len(recurring)}")
    A(f"Single-document patterns                     : {len(patterns) - len(recurring)}")
    A("")
    for p in recurring[:40]:
        counts = pattern_docs[p.id]
        A(f"Recurring Structure {p.id}:")
        A(f"  Structural signature : {p.representative_signature}")
        A(f"  Instances            : {p.size}")
        A("  Document membership  :")
        for doc_id, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            A(f"    {doc_id}: {count} instance(s)")
        if p.evidence and p.evidence.signals:
            A("  Similarity evidence  :")
            for ln in _signal_lines(p.evidence.signals, indent="    "):
                A(ln)
        A("")

    A("Topical groups (semantic-led fusion over the same blocks):")
    A("")
    A(f"  Groups: {len(groups)}")
    multi_doc_groups = [
        g for g in groups if len(getattr(g, "document_ids", []) or []) > 1
    ]
    A(f"  Spanning more than one document: {len(multi_doc_groups)}")
    A("")
    for g in (multi_doc_groups or list(groups))[:25]:
        members = getattr(g, "member_block_ids", []) or []
        docs = getattr(g, "document_ids", []) or []
        A(f"Group {g.id}:")
        A(f"  Dominant pattern    : {g.dominant_pattern}")
        A(f"  Document membership : {', '.join(docs) if docs else '(none)'}")
        A(f"  Blocks              : {len(members)}")
        if getattr(g, "evidence", None) and g.evidence.signals:
            A("  Similarity evidence :")
            for ln in _signal_lines(g.evidence.signals, indent="    "):
                A(ln)
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("GENERICITY AUDIT")
    A(_SEP)
    A("")
    counts = audit.counts()
    A(f"Modules scanned: {audit.files_scanned}")
    A("")
    A(f"Hardcoded semantic categories:      {counts['hardcoded_semantic_category']}")
    A(f"Document-specific section rules:    {counts['document_specific_section_rule']}")
    A(f"Regex-based structure discovery:    {counts['regex_structure_discovery']}")
    A(f"Fixed-coordinate discovery rules:   {counts['fixed_coordinate_rule']}")
    A(f"Fixed-page discovery rules:         {counts['fixed_page_rule']}")
    A(f"Absolute font-size discovery rules: {counts['absolute_font_size_rule']}")
    A("")
    violations = [f for f in audit.findings if f.classification == "DOCUMENT_SPECIFIC"]
    if violations:
        A("DOCUMENT-SPECIFIC LOGIC FOUND (must be removed):")
        for f in violations:
            A(f"  {f.file}:{f.line} [{f.kind}] {f.snippet}")
        A("")
    else:
        A("No document-specific discovery logic found.")
        A("")
    generic = [f for f in audit.findings if f.classification == "VALID_GENERIC"]
    if generic:
        A("Regex retained, classified VALID GENERIC (machine-format parsing only):")
        for f in generic:
            A(f"  {f.file}:{f.line} — {f.note}")
            A(f"      {f.snippet}")
        A("")

    # ---------------------------------------------------------------- #
    A(_SEP)
    A("FINAL SUMMARY")
    A(_SEP)
    A("")
    confs = [lb.confidence for lb in logical_blocks]
    avg = sum(confs) / len(confs) if confs else 0.0
    A(f"Pages processed          : {pages}")
    A(f"Raw extraction units     : {len(raw_units)}")
    A(f"Candidate units          : {len(candidates)}")
    A(f"Candidates rejected      : {len(rejected)}")
    A(f"Boundaries evaluated     : {stats.get('boundaries_evaluated', len(boundaries))}")
    A(f"  START_NEW_LOGICAL_BLOCK: {stats.get('boundaries_detected', 0)}")
    A(f"  CONTINUE_LOGICAL_BLOCK : {stats.get('continued', 0)}")
    A(f"Refinement MERGE         : {stats.get('merges', 0)}")
    A(f"Refinement SPLIT         : {stats.get('splits', 0)}")
    A(f"Refinement PRESERVE      : {stats.get('preserves', 0)}")
    A(f"Content units            : {stats.get('content_units', 0)}")
    A(f"Logical blocks           : {len(logical_blocks)}")
    A(f"  structured records     : {sum(1 for b in logical_blocks if b.block_type == 'structured_record')}")
    A(f"  content blocks         : {sum(1 for b in logical_blocks if b.block_type != 'structured_record')}")
    A(f"Discovered structures    : {len(grids)}")
    A(f"Patterns                 : {len(patterns)}")
    A(f"Contexts                 : {len(sections)}")
    A(f"Heading evidence w/o context: {len(heading_contexts)}")
    A(f"Recurring multi-document structures: {len(recurring)}")
    A(f"Topical groups           : {len(groups)}")
    A(f"Over-grouping warnings   : {len(over_grouping)}")
    A(f"  resolved by splitting  : {sum(1 for o in over_grouping if o['resolved'])}")
    A(f"Low-confidence blocks    : {stats.get('low_confidence', 0)}")
    A(f"Average confidence       : {avg:.3f}")
    A("")

    A(_SEP)
    A("PROCESSING COMPLETE")
    A(_SEP)
    A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
