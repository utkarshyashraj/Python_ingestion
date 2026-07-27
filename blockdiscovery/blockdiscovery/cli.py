"""Command-line interface.

Usage:
    python -m blockdiscovery.cli ingest DOC1.pdf DOC2.pdf ... [--out output]
    python -m blockdiscovery.cli search "authentication" --out output
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

from .config import EngineConfig
from .logging_utils import DiscoveryLogger
from .pipeline import DiscoveryEngine


def _expand(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        matches = sorted(glob.glob(p))
        out.extend(matches if matches else [p])
    return [p for p in out if p.lower().endswith(".pdf")]


def cmd_ingest(args: argparse.Namespace) -> int:
    pdfs = _expand(args.pdfs)
    if not pdfs:
        print("No PDF files matched.", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    config = EngineConfig(
        verbose_relationships=args.verbose,
        ingestion_backend=args.backend,
        max_pages=args.max_pages,
    )
    logger = DiscoveryLogger(
        structured_path=os.path.join(args.out, "events.jsonl"),
        readable_path=os.path.join(args.out, "discovery.log"),
        readable_enabled=not args.quiet,
        low_confidence_threshold=config.thresholds.low_confidence_flag,
    )
    engine = DiscoveryEngine(config=config, logger=logger)
    kb = engine.run(pdfs)
    paths = engine.export(kb, args.out)
    logger.close()

    print("\n=== Discovered collection tree ===")
    for g in kb.groups:
        label = f" [{g.inferred_label}]" if g.inferred_label else ""
        print(f"{g.id}{label}  (pattern={g.dominant_pattern}, docs={len(g.document_ids)}, blocks={g.size})")
        for b in kb.blocks_in_group(g.id):
            preview = b.text[:70].replace("\n", " ")
            print(f"    - {b.document_id} p{b.source_page} {b.id} conf={b.confidence:.2f}: {preview}")
    print("\nArtefacts written:")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    pdfs = _expand(args.pdfs)
    if not pdfs:
        print("No PDF files matched.", file=sys.stderr)
        return 2
    logger = DiscoveryLogger(readable_enabled=False)
    engine = DiscoveryEngine(logger=logger)
    kb = engine.run(pdfs)
    logger.close()
    results = kb.search(args.query, top_k=args.top_k, document_id=args.document)
    print(f"\nQuery: {args.query!r}")
    for b, score in results:
        preview = b.text[:90].replace("\n", " ")
        print(f"  [{score:.3f}] {b.document_id} p{b.source_page} {b.id} (pattern={b.discovered_pattern}): {preview}")
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="blockdiscovery", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest PDFs and discover logical blocks/groups.")
    p_ing.add_argument("pdfs", nargs="+", help="PDF paths or globs.")
    p_ing.add_argument("--out", default="output", help="Output directory.")
    p_ing.add_argument("--verbose", action="store_true", help="Log every candidate relationship.")
    p_ing.add_argument("--quiet", action="store_true", help="Suppress readable log to stdout.")
    p_ing.add_argument(
        "--backend",
        choices=["native", "structured", "docling"],
        default="native",
        help="Ingestion backend: native block clustering, structured (pymupdf4llm), or docling.",
    )
    p_ing.add_argument("--max-pages", type=int, default=None, help="Process only the first N pages.")
    p_ing.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Ingest PDFs then run a semantic query.")
    p_search.add_argument("query", help="Search text.")
    p_search.add_argument("pdfs", nargs="+", help="PDF paths or globs.")
    p_search.add_argument("--document", default=None, help="Restrict to a document id.")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
