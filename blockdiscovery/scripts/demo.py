"""End-to-end demonstration of the discovery engine.

Generates the sample PDFs (if needed), ingests them, and shows:
  * the discovered collection tree (cross-document logical groups),
  * OPTIONAL after-the-fact descriptive labels (via a user-supplied lexicon --
    the core algorithm never depends on these words),
  * full provenance tracing for one logical block,
  * a couple of semantic searches.

Run:  python scripts/demo.py
"""

from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from blockdiscovery.config import EngineConfig
from blockdiscovery.logging_utils import DiscoveryLogger
from blockdiscovery.pipeline import DiscoveryEngine


def ensure_samples() -> list:
    sample_dir = os.path.join(ROOT, "data", "synthetic")
    pdfs = sorted(glob.glob(os.path.join(sample_dir, "*.pdf")))
    if not pdfs:
        import scripts.generate_sample_pdfs as gen  # type: ignore
        gen.main()
        pdfs = sorted(glob.glob(os.path.join(sample_dir, "*.pdf")))
    return pdfs


def main() -> None:
    pdfs = ensure_samples()
    out_dir = os.path.join(ROOT, "output")

    # No business lexicon — discovery stays future-agnostic.
    logger = DiscoveryLogger(
        structured_path=os.path.join(out_dir, "events.jsonl"),
        readable_path=os.path.join(out_dir, "discovery.log"),
        readable_enabled=False,
        low_confidence_threshold=0.5,
    )
    config = EngineConfig(optional_label_lexicon={})
    engine = DiscoveryEngine(config=config, logger=logger)
    kb = engine.run(pdfs)
    engine.export(kb, out_dir)
    logger.close()

    print("=" * 78)
    print("DISCOVERED COLLECTION TREE  (cross-document logical groups)")
    print("=" * 78)
    multi = [g for g in kb.groups if g.size > 1]
    single = [g for g in kb.groups if g.size == 1]
    for g in multi:
        print(f"\n{g.id}  (pattern={g.dominant_pattern}, docs={len(g.document_ids)}, blocks={g.size})")
        for b in kb.blocks_in_group(g.id):
            preview = b.text[:64].replace("\n", " ")
            print(f"    - {b.document_id:16} p{b.source_page} conf={b.confidence:.2f}  {preview}")
    print(f"\n(+ {len(single)} singleton groups not shown)")

    print("\n" + "=" * 78)
    print("DISCOVERED PATTERNS  (generic, no business names)")
    print("=" * 78)
    for p in kb.patterns[:8]:
        print(f"  {p.id}: size={p.size:2d} signature={p.representative_signature} roles={p.role_template}")

    print("\n" + "=" * 78)
    print("PROVENANCE TRACE  (logical group -> block -> source blocks -> page -> PDF)")
    print("=" * 78)
    target = next((b for b in kb.logical_blocks if len(b.source_block_ids) >= 3), kb.logical_blocks[0])
    trace = kb.trace(target.id)
    print(f"  logical_block: {target.id}  (group={target.group_id}, pattern={target.discovered_pattern})")
    print(f"  source_document: {trace['source_document']['source_path']}")
    print(f"  source_page: {trace['source_page']}")
    print(f"  confidence: {target.confidence:.2f}  evidence signals: "
          + ", ".join(f"{k}={v:.2f}" for k, v in target.evidence.signals.items()))
    for sb in trace["source_blocks"]:
        print(f"      block {sb['id']}  size={sb['formatting']['dominant_size'] if sb['formatting'] else '-'}"
              f"  bold={sb['formatting']['bold_ratio'] if sb['formatting'] else '-'}  text={sb['text'][:48]!r}")

    print("\n" + "=" * 78)
    print("SEMANTIC SEARCH")
    print("=" * 78)
    for q in ["authentication and login", "billing calculation error", "crash when exporting"]:
        print(f"\n  query: {q!r}")
        for b, score in kb.search(q, top_k=3):
            print(f"    [{score:.3f}] {b.document_id} {b.id}: {b.text[:56].replace(chr(10), ' ')}")

    print("\nArtefacts written to:", out_dir)


if __name__ == "__main__":
    main()
