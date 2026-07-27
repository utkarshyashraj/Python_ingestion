"""Searchable knowledge structure.

Holds the discovered artefacts for a whole document collection and exposes
retrieval that does not depend on fixed layouts or regex rules:

* ``search(query)``            -> logical blocks ranked by semantic similarity
* ``blocks_in_group(group_id)`` -> all related content across documents/versions
* ``related(block_id)``         -> cross-document siblings of a logical block

Every returned item retains full traceability to its source blocks/page/PDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Document, DiscoveredPattern, LogicalBlock, LogicalGroup, SectionGroup
from .semantics import EmbeddingBackend, cosine


@dataclass
class KnowledgeBase:
    backend: EmbeddingBackend
    documents: Dict[str, Document] = field(default_factory=dict)
    logical_blocks: List[LogicalBlock] = field(default_factory=list)
    patterns: List[DiscoveredPattern] = field(default_factory=list)
    groups: List[LogicalGroup] = field(default_factory=list)
    section_groups: List[SectionGroup] = field(default_factory=list)

    _block_index: Dict[str, LogicalBlock] = field(default_factory=dict, init=False)
    _group_index: Dict[str, LogicalGroup] = field(default_factory=dict, init=False)
    _section_index: Dict[str, SectionGroup] = field(default_factory=dict, init=False)

    def build_indexes(self) -> None:
        self._block_index = {b.id: b for b in self.logical_blocks}
        self._group_index = {g.id: g for g in self.groups}
        self._section_index = {s.id: s for s in self.section_groups}

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        pattern_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[LogicalBlock, float]]:
        qv = self.backend.embed([query])[0]
        scored: List[Tuple[LogicalBlock, float]] = []
        for b in self.logical_blocks:
            if document_id and b.document_id != document_id:
                continue
            if pattern_id and b.discovered_pattern != pattern_id:
                continue
            if b.semantic_vector is None:
                continue
            score = cosine(qv, np.asarray(b.semantic_vector, dtype=np.float32))
            if score >= min_score:
                scored.append((b, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def blocks_in_group(self, group_id: str) -> List[LogicalBlock]:
        g = self._group_index.get(group_id)
        if not g:
            return []
        return [self._block_index[bid] for bid in g.member_block_ids if bid in self._block_index]

    def blocks_in_section(self, section_id: str) -> List[LogicalBlock]:
        s = self._section_index.get(section_id)
        if not s:
            return []
        return [self._block_index[bid] for bid in s.member_logical_block_ids if bid in self._block_index]

    def sections_by_label(self, label: str) -> List[SectionGroup]:
        return [s for s in self.section_groups if (s.inferred_label or "").lower() == label.lower()]

    def related(self, block_id: str, cross_document_only: bool = True) -> List[LogicalBlock]:
        b = self._block_index.get(block_id)
        if not b or not b.group_id:
            return []
        siblings = self.blocks_in_group(b.group_id)
        out = []
        for s in siblings:
            if s.id == block_id:
                continue
            if cross_document_only and s.document_id == b.document_id:
                continue
            out.append(s)
        return out

    def blocks_by_document(self, document_id: str) -> List[LogicalBlock]:
        return [b for b in self.logical_blocks if b.document_id == document_id]

    def blocks_by_pattern(self, pattern_id: str) -> List[LogicalBlock]:
        return [b for b in self.logical_blocks if b.discovered_pattern == pattern_id]

    def trace(self, block_id: str) -> Dict:
        """Full provenance chain for a logical block."""
        b = self._block_index.get(block_id)
        if not b:
            return {}
        doc = self.documents.get(b.document_id)
        source_blocks = []
        if doc:
            for bid in b.source_block_ids:
                tb = doc.block_by_id(bid)
                if tb:
                    source_blocks.append(tb.to_dict())
        return {
            "logical_block": b.to_dict(),
            "group": self._group_index.get(b.group_id).to_dict() if b.group_id and b.group_id in self._group_index else None,
            "source_document": doc.to_dict() if doc else None,
            "source_page": b.source_page,
            "source_blocks": source_blocks,
        }
