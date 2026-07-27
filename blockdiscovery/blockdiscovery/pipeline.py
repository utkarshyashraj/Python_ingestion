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
from .features import FeatureGenerator
from .knowledge import KnowledgeBase
from .logging_utils import DiscoveryLogger
from .logical_blocks import LogicalBlockBuilder
from .models import ContentUnit, Document, LogicalBlock, SectionGroup
from .normalization import normalize_document
from .patterns import PatternDiscovery
from .relationships import RelationshipEvaluator
from .section_groups import SectionGroupDiscovery, write_human_section_log
from .semantics import build_backend
from .structured_ingest import (
    ingest_pdf,
    remap_section_members,
    units_to_logical_blocks,
)


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
        units = ContentUnitDiscovery(self.config, evaluator, self.backend, self.logger).discover(document)
        return units, evaluator

    def run(self, paths: List[str]) -> KnowledgeBase:
        if self.config.ingestion_backend in ("structured", "docling"):
            return self._run_structured(paths)
        return self._run_native(paths)

    def _run_structured(self, paths: List[str]) -> KnowledgeBase:
        """Structure-aware path: table rows / headings → units → patterns → groups."""
        log = self.logger
        all_documents: Dict[str, Document] = {}
        all_units: List[ContentUnit] = []
        all_logical_blocks: List[LogicalBlock] = []
        all_section_groups: List[SectionGroup] = []
        per_doc_units: Dict[str, List[ContentUnit]] = {}
        per_doc_sections: Dict[str, List[SectionGroup]] = {}

        for path in paths:
            result = ingest_pdf(
                path,
                backend=self.config.ingestion_backend,
                max_pages=self.config.max_pages,
                logger=log,
            )
            all_documents[result.document.id] = result.document
            per_doc_units[result.document.id] = result.content_units
            per_doc_sections[result.document.id] = result.section_groups
            all_units.extend(result.content_units)

        corpus = [u.text for u in all_units if u.text]
        self.backend.fit(corpus)
        if all_units:
            vectors = self.backend.embed([u.text for u in all_units])
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

        patterns, unit_to_pattern = PatternDiscovery(self.config, log).discover(all_units)

        for doc_id, units in per_doc_units.items():
            document = all_documents[doc_id]
            blocks = units_to_logical_blocks(document, units, unit_to_pattern, log)
            sections = per_doc_sections.get(doc_id, [])
            remap_section_members(sections, blocks)
            for lb in blocks:
                # Attach section id when membership matches.
                for s in sections:
                    if lb.id in s.member_logical_block_ids:
                        lb.section_group_id = s.id
                        break
            all_logical_blocks.extend(blocks)
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

        for doc_id, units in per_doc_units.items():
            document = all_documents[doc_id]
            builder = LogicalBlockBuilder(self.config, per_doc_evaluator[doc_id], log)
            blocks = builder.build(document, units, unit_to_pattern)
            per_doc_logical[doc_id] = blocks
            all_logical_blocks.extend(blocks)

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
