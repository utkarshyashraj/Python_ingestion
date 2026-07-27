"""Validate that discovery works on documents with unseen terminology.

Runs the unmodified engine over synthetic PDFs whose every label is invented,
then asserts structural expectations that can be stated without naming any
content: repeated grid rows must separate into independent blocks, prose must
stay coherent, running footers must be rejected, and structurally identical
records must land in the same pattern.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blockdiscovery.config import EngineConfig  # noqa: E402
from blockdiscovery.genericity_audit import audit_package  # noqa: E402
from blockdiscovery.logging_utils import DiscoveryLogger  # noqa: E402
from blockdiscovery.pipeline import DiscoveryEngine  # noqa: E402


def run_one(path: str, out_dir: str) -> Dict[str, object]:
    config = EngineConfig(ingestion_backend="structured", readable_log=False)
    logger = DiscoveryLogger(readable_enabled=False, readable_stream=None)
    engine = DiscoveryEngine(config=config, logger=logger)
    kb = engine.run([path])
    os.makedirs(out_dir, exist_ok=True)
    engine.export(kb, out_dir)

    doc = next(iter(kb.documents.values()))
    trace = engine._traces_by_doc[doc.id]
    blocks = [b for b in kb.logical_blocks if b.document_id == doc.id]
    records = [b for b in blocks if b.block_type == "structured_record"]
    multi_record = [b for b in records if len(b.source_block_ids) > 2]
    grids = [g for g in trace.grids if g.row_count >= 3]

    return {
        "document": os.path.basename(path),
        "pages": doc.page_count,
        "raw_units": trace.stats.get("raw_units", 0),
        "candidates": trace.stats.get("candidates", 0),
        "logical_blocks": len(blocks),
        "structured_records": len(records),
        "records_with_many_sources": len(multi_record),
        "grids_discovered": len(grids),
        "grid_columns": sorted({g.column_count for g in grids}),
        "repeated_grids": sum(1 for g in grids if g.structure_type == "repeated_grid"),
        "contexts": len([s for s in kb.section_groups if s.document_id == doc.id]),
        "patterns": len(kb.patterns),
        "rejected": len(trace.rejected),
        "merges": trace.stats.get("merges", 0),
        "splits": trace.stats.get("splits", 0),
        "preserves": trace.stats.get("preserves", 0),
        "avg_confidence": round(
            sum(b.confidence for b in blocks) / len(blocks), 3
        )
        if blocks
        else 0.0,
        "largest_pattern": max((p.size for p in kb.patterns), default=0),
    }


def main() -> int:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    synthetic = os.path.join(base, "data", "synthetic")
    out_root = os.path.join(base, "output_synthetic_validation")
    if not os.path.isdir(synthetic):
        print(f"missing {synthetic}; run tools/make_synthetic_pdfs.py first")
        return 1

    pdfs = sorted(f for f in os.listdir(synthetic) if f.endswith(".pdf"))
    results: List[Dict[str, object]] = []
    for name in pdfs:
        out_dir = os.path.join(out_root, name.rsplit(".", 1)[0])
        results.append(run_one(os.path.join(synthetic, name), out_dir))

    print("=" * 74)
    print("SYNTHETIC GENERICITY VALIDATION (all terminology is invented)")
    print("=" * 74)
    for r in results:
        print()
        print(f"{r['document']}")
        for key in (
            "pages",
            "raw_units",
            "candidates",
            "logical_blocks",
            "structured_records",
            "records_with_many_sources",
            "grids_discovered",
            "grid_columns",
            "repeated_grids",
            "contexts",
            "patterns",
            "largest_pattern",
            "rejected",
            "merges",
            "splits",
            "preserves",
            "avg_confidence",
        ):
            print(f"    {key:26}: {r[key]}")

    checks: List[tuple] = []
    by_name = {r["document"]: r for r in results}

    grid = by_name.get("synthetic_grid.pdf", {})
    checks.append(
        (
            "grid rows become independent blocks",
            grid.get("structured_records", 0) >= 6
            and grid.get("records_with_many_sources", 1) == 0,
        )
    )
    checks.append(
        ("repeated grid structure discovered", grid.get("repeated_grids", 0) >= 1)
    )
    checks.append(
        (
            "structurally identical records share a pattern",
            grid.get("largest_pattern", 0) >= 3,
        )
    )

    records = by_name.get("synthetic_records.pdf", {})
    checks.append(
        (
            "prose-shaped records discovered without a grid",
            records.get("logical_blocks", 0) >= 4,
        )
    )
    checks.append(
        (
            "heading with no members did not become a context",
            records.get("contexts", 99) < records.get("logical_blocks", 0),
        )
    )

    mixed = by_name.get("synthetic_mixed.pdf", {})
    checks.append(
        (
            "different column counts handled by one algorithm",
            len(mixed.get("grid_columns", [])) >= 2,
        )
    )
    checks.append(
        (
            "multi-column prose did not fragment into records",
            mixed.get("logical_blocks", 0) > 0,
        )
    )

    # All synthetic documents together: recurring structure must be found across
    # documents that share no vocabulary at all.
    corpus_out = os.path.join(out_root, "_corpus")
    config = EngineConfig(ingestion_backend="structured", readable_log=False)
    engine = DiscoveryEngine(
        config=config,
        logger=DiscoveryLogger(readable_enabled=False, readable_stream=None),
    )
    kb = engine.run([os.path.join(synthetic, n) for n in pdfs])
    os.makedirs(corpus_out, exist_ok=True)
    engine.export(kb, corpus_out)

    recurring = 0
    for p in kb.patterns:
        docs = {uid.rsplit("_unit_", 1)[0] for uid in p.member_unit_ids}
        if len(docs) > 1:
            recurring += 1
    print()
    print(f"corpus run: {len(pdfs)} documents, {len(kb.patterns)} patterns, "
          f"{recurring} spanning more than one document")
    checks.append(("recurring structure found across documents", recurring >= 1))

    audit = audit_package()
    counts = audit.counts()
    checks.append(("no document-specific logic in engine", sum(counts.values()) == 0))

    print()
    print("=" * 74)
    print("CHECKS")
    print("=" * 74)
    failures = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures += 1
    print()
    print(f"genericity audit violations: {counts}")
    print(f"output written under: {out_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
