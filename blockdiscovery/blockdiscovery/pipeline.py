"""Pipeline orchestrator.

Wires the layers together in the evidence-first order mandated by the spec:

    PDF -> Evidence Extraction -> Feature Generation -> Relationship Discovery
        -> Content Unit Discovery -> Pattern Discovery -> Logical Block Discovery
        -> Section Group Discovery -> Cross-Document Similarity
        -> Logical Group Formation

Produces a :class:`KnowledgeBase` and writes readable + structured logs and JSON
artefacts. No business categories are introduced into the core discovery flow;
optional labels may be attached after structure is found.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .config import DEFAULT_CONFIG, EngineConfig
from .content_units import ContentUnitDiscovery
from .cross_document import CrossDocumentGrouping
from .extraction import PDFExtractor
from .extraction_units import extract_raw_units
from .features import FeatureGenerator
from .generic_discovery import (
    GenericDiscoveryEngine,
    detect_pattern_consolidation,
)
from .generic_log import write_generic_discovery_log
from .genericity_audit import audit_package
from .knowledge import KnowledgeBase
from .logging_utils import DiscoveryLogger
from .logical_block_consolidator import (
    ConsolidationResult,
    LogicalBlockConsolidator,
    write_consolidation_log,
)
from .logical_blocks import LogicalBlockBuilder
from .models import ContentUnit, Document, LogicalBlock, SectionGroup
from .normalization import normalize_document
from .patterns import PatternDiscovery
from .relationships import RelationshipEvaluator
from .section_groups import SectionGroupDiscovery, write_human_section_log
from .semantics import build_backend
from .validation_log import write_kb_validation_logs


class DiscoveryEngine:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        logger: Optional[DiscoveryLogger] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        thr = self.config.thresholds.low_confidence_flag
        self.logger = logger or DiscoveryLogger(
            readable_enabled=self.config.readable_log,
            low_confidence_threshold=thr,
        )
        self.backend = build_backend(
            self.config.embedding_backend,
            self.config.embedding_dim,
            self.config.sentence_transformer_model,
        )
        # Per-document discovery traces for validation logs.
        self._transitions_by_doc: Dict[str, List] = {}
        self._stats_by_doc: Dict[str, Dict] = {}
        self._units_by_doc: Dict[str, List[ContentUnit]] = {}
        self._page_stats_by_doc: Dict[str, Dict[int, Dict[str, int]]] = {}
        self._traces_by_doc: Dict[str, object] = {}
        self._raw_units_by_doc: Dict[str, List] = {}
        self._pattern_consolidation: List[Dict] = []
        self._consolidation_by_doc: Dict[str, ConsolidationResult] = {}


    # ------------------------------------------------------------------ #
    def extract_and_feature(self, path: str, document_id: Optional[str] = None) -> Document:
        """Steps 1-4a: extraction, normalization and feature generation."""
        extractor = PDFExtractor(self.logger)
        document = extractor.extract(path, document_id, max_pages=self.config.max_pages)
        normalize_document(document, self.logger)
        FeatureGenerator(self.logger).generate(document)
        return document

    def discover_units(self, document: Document):
        """Steps 4b-5: block embeddings, relationships and content units."""
        text_blocks = [b for b in document.blocks if b.block_type == "text"]
        if text_blocks:
            vecs = self.backend.embed([b.text for b in text_blocks])
            block_embeddings: Dict[str, np.ndarray] = {b.id: v for b, v in zip(text_blocks, vecs)}
        else:
            block_embeddings = {}
        evaluator = RelationshipEvaluator(self.config, block_embeddings)
        discoverer = ContentUnitDiscovery(self.config, evaluator, self.backend, self.logger)
        units = discoverer.discover(document)
        self._transitions_by_doc[document.id] = list(discoverer.transitions)
        self._stats_by_doc[document.id] = dict(discoverer.stats)
        self._units_by_doc[document.id] = list(units)
        return units, evaluator

    def run(self, paths: List[str]) -> KnowledgeBase:
        if self.config.ingestion_backend in ("structured", "docling"):
            return self._run_structured(paths)
        return self._run_native(paths)

    def _run_structured(self, paths: List[str]) -> KnowledgeBase:
        """Extraction-only backend feeding the generic evidence-first engine.

        The backend contributes geometry and layout evidence; every grouping,
        boundary and context decision is made by :class:`GenericDiscoveryEngine`
        from that evidence alone.
        """
        log = self.logger
        all_documents: Dict[str, Document] = {}
        all_units: List[ContentUnit] = []
        all_logical_blocks: List[LogicalBlock] = []
        all_section_groups: List[SectionGroup] = []
        extraction: Dict[str, object] = {}

        for path in paths:
            result = extract_raw_units(
                path,
                max_pages=self.config.max_pages,
                logger=log,
                backend=self.config.ingestion_backend,
            )
            extraction[result.document.id] = result
            all_documents[result.document.id] = result.document
            self._page_stats_by_doc[result.document.id] = dict(result.page_stats)

        # Fit the embedding space on raw evidence before any grouping happens.
        corpus = [ru.text for r in extraction.values() for ru in r.raw_units if ru.text]
        self.backend.fit(corpus)

        engine = GenericDiscoveryEngine(self.backend, log)
        consolidator = LogicalBlockConsolidator(log)
        section_discoverer = SectionGroupDiscovery(self.config, self.backend, log)
        per_doc_blocks: Dict[str, List[LogicalBlock]] = {}

        for doc_id, result in extraction.items():
            document = result.document
            # Section discovery from the engine is discarded and re-run after
            # consolidation so section membership reflects merged blocks.
            units, blocks, _sections, trace = engine.run(document, result.raw_units)
            consolidation = consolidator.consolidate(document, blocks)
            blocks = consolidation.blocks
            self._consolidation_by_doc[doc_id] = consolidation
            self._units_by_doc[doc_id] = list(units)
            self._traces_by_doc[doc_id] = trace
            self._raw_units_by_doc[doc_id] = list(result.raw_units)
            self._stats_by_doc[doc_id] = {
                **dict(trace.stats),
                **{f"consolidation_{k}": v for k, v in consolidation.stats.items()},
            }
            per_doc_blocks[doc_id] = blocks
            all_units.extend(units)
            all_logical_blocks.extend(blocks)

        if all_units:
            vectors = self.backend.embed([u.text for u in all_units])
            unit_vectors = {u.id: v.astype(float).tolist() for u, v in zip(all_units, vectors)}
            for u, v in zip(all_units, vectors):
                u.semantic_vector = v.astype(float).tolist()
                if log:
                    log.event(
                        "semantic_representation_created",
                        document_id=u.document_id,
                        content_unit_id=u.id,
                        dim=int(v.shape[0]),
                        backend=self.config.embedding_backend,
                    )
            for lb in all_logical_blocks:
                if lb.semantic_vector is None:
                    lb.semantic_vector = unit_vectors.get(lb.content_unit_id)

        patterns, unit_to_pattern = PatternDiscovery(self.config, log).discover(all_units)
        for lb in all_logical_blocks:
            lb.discovered_pattern = self._pattern_for_block(lb, unit_to_pattern)

        # Re-discover sections on consolidated blocks.
        all_section_groups = []
        for doc_id, blocks in per_doc_blocks.items():
            sections = section_discoverer.discover(all_documents[doc_id], blocks)
            all_section_groups.extend(sections)

        self._pattern_consolidation = detect_pattern_consolidation(patterns)
        if log:
            for finding in self._pattern_consolidation:
                log.event("possible_pattern_consolidation", **finding)

        groups = CrossDocumentGrouping(self.config, self.backend, log).group(all_logical_blocks)

        kb = KnowledgeBase(
            backend=self.backend,
            documents=all_documents,
            logical_blocks=all_logical_blocks,
            patterns=patterns,
            groups=groups,
            section_groups=all_section_groups,
        )
        kb.build_indexes()
        self._log_summary(kb)
        return kb

    def _run_native(self, paths: List[str]) -> KnowledgeBase:
        """Original evidence-first block clustering pipeline."""
        log = self.logger
        all_documents: Dict[str, Document] = {}
        all_units: List[ContentUnit] = []
        all_logical_blocks: List[LogicalBlock] = []
        all_section_groups: List[SectionGroup] = []
        per_doc_units: Dict[str, List[ContentUnit]] = {}
        per_doc_evaluator: Dict[str, RelationshipEvaluator] = {}
        per_doc_logical: Dict[str, List[LogicalBlock]] = {}

        for path in paths:
            document = self.extract_and_feature(path)
            all_documents[document.id] = document

        corpus = [
            b.text
            for d in all_documents.values()
            for b in d.blocks
            if b.block_type == "text" and b.text
        ]
        self.backend.fit(corpus)

        for document in all_documents.values():
            units, evaluator = self.discover_units(document)
            per_doc_units[document.id] = units
            per_doc_evaluator[document.id] = evaluator
            all_units.extend(units)

        patterns, unit_to_pattern = PatternDiscovery(self.config, log).discover(all_units)
        consolidator = LogicalBlockConsolidator(log)

        for doc_id, units in per_doc_units.items():
            document = all_documents[doc_id]
            builder = LogicalBlockBuilder(self.config, per_doc_evaluator[doc_id], log)
            blocks = builder.build(document, units, unit_to_pattern)
            consolidation = consolidator.consolidate(document, blocks)
            self._consolidation_by_doc[doc_id] = consolidation
            self._stats_by_doc[doc_id] = {
                **self._stats_by_doc.get(doc_id, {}),
                **{f"consolidation_{k}": v for k, v in consolidation.stats.items()},
            }
            for lb in consolidation.blocks:
                lb.discovered_pattern = self._pattern_for_block(lb, unit_to_pattern)
            per_doc_logical[doc_id] = consolidation.blocks
            all_logical_blocks.extend(consolidation.blocks)

        # Section discovery runs after consolidation so membership stays coherent.
        section_discoverer = SectionGroupDiscovery(self.config, self.backend, log)
        for doc_id, blocks in per_doc_logical.items():
            sections = section_discoverer.discover(all_documents[doc_id], blocks)
            all_section_groups.extend(sections)

        groups = CrossDocumentGrouping(self.config, self.backend, log).group(all_logical_blocks)

        kb = KnowledgeBase(
            backend=self.backend,
            documents=all_documents,
            logical_blocks=all_logical_blocks,
            patterns=patterns,
            groups=groups,
            section_groups=all_section_groups,
        )
        kb.build_indexes()
        self._log_summary(kb)
        return kb

    def _log_summary(self, kb: KnowledgeBase) -> None:
        log = self.logger
        if not log:
            return
        log.section("SUMMARY")
        log.kv("Ingestion backend", self.config.ingestion_backend)
        log.kv("Documents", len(kb.documents))
        log.kv("Logical blocks", len(kb.logical_blocks))
        log.kv("Section groups", len(kb.section_groups))
        log.kv("Discovered patterns", len(kb.patterns))
        log.kv("Logical groups", len(kb.groups))
        log.event(
            "processing_completed",
            ingestion_backend=self.config.ingestion_backend,
            documents=len(kb.documents),
            logical_blocks=len(kb.logical_blocks),
            section_groups=len(kb.section_groups),
            patterns=len(kb.patterns),
            groups=len(kb.groups),
            event_counts=log.summary(),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pattern_for_block(
        block: LogicalBlock, unit_to_pattern: Dict[str, str]
    ) -> Optional[str]:
        """Prefer the dominant pattern among consolidated source units."""
        unit_ids = block.source_content_unit_ids or [block.content_unit_id]
        counts: Dict[str, int] = {}
        for uid in unit_ids:
            pid = unit_to_pattern.get(uid)
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        if not counts:
            return unit_to_pattern.get(block.content_unit_id)
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def export(self, kb: KnowledgeBase, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        def _dump(name: str, obj) -> None:
            p = os.path.join(out_dir, name)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            paths[name] = p

        _dump("documents.json", {d.id: d.to_dict() for d in kb.documents.values()})
        _dump("logical_blocks.json", [b.to_dict() for b in kb.logical_blocks])
        _dump("patterns.json", [p.to_dict() for p in kb.patterns])
        _dump("logical_groups.json", [g.to_dict() for g in kb.groups])
        _dump("section_groups.json", [s.to_dict() for s in kb.section_groups])

        tree = self._collection_tree(kb)
        _dump("collection_tree.json", tree)

        # Human-readable section → items log (one file per document).
        for doc in kb.documents.values():
            doc_blocks = [b for b in kb.logical_blocks if b.document_id == doc.id]
            doc_sections = [s for s in kb.section_groups if s.document_id == doc.id]
            human_name = f"human_readable_{doc.id}.log"
            human_path = os.path.join(out_dir, human_name)
            write_human_section_log(
                human_path,
                document=doc,
                logical_blocks=doc_blocks,
                sections=doc_sections,
                page_limit=self.config.max_pages,
            )
            paths[human_name] = human_path

        # Post-discovery consolidation reports (one per document when available).
        consolidation_export: Dict[str, object] = {}
        for doc in kb.documents.values():
            result = self._consolidation_by_doc.get(doc.id)
            if result is None:
                continue
            cons_name = f"block_consolidation_{doc.id}.log"
            cons_path = os.path.join(out_dir, cons_name)
            write_consolidation_log(cons_path, result, doc.id)
            paths[cons_name] = cons_path
            consolidation_export[doc.id] = {
                "stats": result.stats,
                "merge_chains": result.merge_chains,
                "decisions": [d.to_dict() for d in result.decisions],
            }
        if consolidation_export:
            _dump("block_consolidation.json", consolidation_export)

        # Section-12 style validation discovery log (native path has full
        # boundary transitions; structured still gets extraction + LB summary).
        if self._traces_by_doc:
            audit = audit_package()
            for doc in kb.documents.values():
                trace = self._traces_by_doc.get(doc.id)
                if trace is None:
                    continue
                name = f"generic_discovery_{doc.id}.log"
                generic_path = os.path.join(out_dir, name)
                write_generic_discovery_log(
                    generic_path,
                    document=doc,
                    logical_blocks=[b for b in kb.logical_blocks if b.document_id == doc.id],
                    trace=trace,
                    patterns=kb.patterns,
                    groups=kb.groups,
                    sections=[s for s in kb.section_groups if s.document_id == doc.id],
                    raw_units=self._raw_units_by_doc.get(doc.id, []),
                    page_stats=self._page_stats_by_doc.get(doc.id, {}),
                    pattern_consolidation=self._pattern_consolidation,
                    audit=audit,
                    backend=self.config.ingestion_backend,
                    page_limit=self.config.max_pages,
                )
                paths[name] = generic_path
            _dump("genericity_audit.json", audit.to_dict())
            _dump("pattern_consolidation.json", self._pattern_consolidation)
        else:
            val_paths = write_kb_validation_logs(
                kb,
                out_dir,
                transitions_by_doc=self._transitions_by_doc,
                stats_by_doc=self._stats_by_doc,
                units_by_doc=self._units_by_doc,
                page_stats_by_doc=self._page_stats_by_doc,
                processing_mode=self.config.ingestion_backend,
                page_limit=self.config.max_pages,
            )
            paths.update(val_paths)
        return paths

    def _collection_tree(self, kb: KnowledgeBase) -> Dict:
        tree = {"section_groups": [], "cross_document_groups": []}
        lb_by_id = {b.id: b for b in kb.logical_blocks}
        for s in kb.section_groups:
            tree["section_groups"].append(
                {
                    "section_id": s.id,
                    "label": s.inferred_label,
                    "heading": s.heading_text,
                    "depth": s.depth,
                    "parent_section_id": s.parent_section_id,
                    "child_section_ids": list(s.child_section_ids),
                    "pages": f"{s.page_start}-{s.page_end}",
                    "items": [
                        {
                            "logical_block_id": bid,
                            "preview": (lb_by_id[bid].text[:80].replace("\n", " ") if bid in lb_by_id else ""),
                        }
                        for bid in s.member_logical_block_ids
                    ],
                }
            )
        for g in kb.groups:
            node = {
                "group_id": g.id,
                "dominant_pattern": g.dominant_pattern,
                "inferred_label": g.inferred_label,
                "documents": {},
            }
            for b in kb.blocks_in_group(g.id):
                node["documents"].setdefault(b.document_id, []).append(
                    {
                        "logical_block_id": b.id,
                        "page": b.source_page,
                        "pattern": b.discovered_pattern,
                        "confidence": round(b.confidence, 3),
                        "preview": b.text[:80].replace("\n", " "),
                    }
                )
            tree["cross_document_groups"].append(node)
        return tree
