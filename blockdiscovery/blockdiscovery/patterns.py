"""Structural / semantic pattern discovery.

After content units exist, this module discovers *recurring patterns* by
combining structural similarity (role-count signatures, block counts) with
semantic similarity. Patterns are named generically (``pattern_001`` ...) and
are NOT business categories -- any human label is applied later and optionally.

Uses dependency-free online (leader) clustering with a configurable similarity
threshold, so it works without scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EngineConfig
from .fingerprints import fingerprint_vector
from .logging_utils import DiscoveryLogger
from .models import ContentUnit, DiscoveredPattern, Evidence
from .semantics import cosine


# Patterns describe recurring *shape*, so structural evidence leads and semantic
# evidence only modulates. Two records with different wording but the same layout
# must therefore land in one pattern.
_STRUCT_WEIGHT = 0.72
_SEM_WEIGHT = 0.28


def _struct_vector(unit: ContentUnit) -> np.ndarray:
    """Prefer rich structural fingerprints; fall back to role/count features."""
    if unit.structural_fingerprint:
        return fingerprint_vector(unit.structural_fingerprint)
    f = unit.features
    v = np.array(
        [
            f.get("role_prominent", 0.0),
            f.get("role_body", 0.0),
            f.get("role_meta", 0.0),
            np.log1p(f.get("block_count", 1.0)),
            min(3.0, f.get("head_size_ratio", 1.0)),
        ],
        dtype=np.float32,
    )
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class _Cluster:
    __slots__ = ("members", "sem_centroid", "struct_centroid", "signatures")

    def __init__(self, sem: np.ndarray, struct: np.ndarray, signature: str, unit_id: str) -> None:
        self.members: List[str] = [unit_id]
        self.sem_centroid = sem.copy()
        self.struct_centroid = struct.copy()
        self.signatures = Counter([signature])

    def similarity(self, sem: np.ndarray, struct: np.ndarray) -> Tuple[float, float, float]:
        sem_sim = max(0.0, cosine(self.sem_centroid, sem))
        struct_sim = max(0.0, cosine(self.struct_centroid, struct))
        fused = _SEM_WEIGHT * sem_sim + _STRUCT_WEIGHT * struct_sim
        return fused, sem_sim, struct_sim

    def add(self, sem: np.ndarray, struct: np.ndarray, signature: str, unit_id: str) -> None:
        n = len(self.members)
        self.sem_centroid = (self.sem_centroid * n + sem) / (n + 1)
        self.struct_centroid = (self.struct_centroid * n + struct) / (n + 1)
        self.signatures[signature] += 1
        self.members.append(unit_id)


class PatternDiscovery:
    def __init__(self, config: EngineConfig, logger: Optional[DiscoveryLogger] = None) -> None:
        self.config = config
        self.logger = logger

    def discover(self, units: List[ContentUnit]) -> Tuple[List[DiscoveredPattern], Dict[str, str]]:
        log = self.logger
        if log:
            log.section("PATTERN DISCOVERY")

        clusters: List[_Cluster] = []
        unit_index = {u.id: u for u in units}
        sem_cache: Dict[str, np.ndarray] = {}
        struct_cache: Dict[str, np.ndarray] = {}

        for u in units:
            sem = np.asarray(u.semantic_vector, dtype=np.float32) if u.semantic_vector is not None else np.zeros(1, dtype=np.float32)
            struct = _struct_vector(u)
            sem_cache[u.id] = sem
            struct_cache[u.id] = struct

            best_idx = -1
            best_sim = 0.0
            for idx, c in enumerate(clusters):
                fused, _, _ = c.similarity(sem, struct)
                if fused > best_sim:
                    best_sim = fused
                    best_idx = idx

            if best_idx >= 0 and best_sim >= self.config.thresholds.pattern_similarity:
                clusters[best_idx].add(sem, struct, u.structural_signature, u.id)
            else:
                clusters.append(_Cluster(sem, struct, u.structural_signature, u.id))

        # Order clusters by size (largest, most-recurring first) for stable ids.
        clusters.sort(key=lambda c: len(c.members), reverse=True)

        patterns: List[DiscoveredPattern] = []
        unit_to_pattern: Dict[str, str] = {}
        for i, c in enumerate(clusters, start=1):
            pid = f"pattern_{i:03d}"
            representative_signature = c.signatures.most_common(1)[0][0]
            # Role template taken from a representative member.
            rep_unit = max(
                (unit_index[m] for m in c.members),
                key=lambda uu: uu.features.get("head_prominence", 0.0),
            )
            evidence = Evidence(
                signals={
                    "member_count": float(len(c.members)),
                    "signature_purity": c.signatures.most_common(1)[0][1] / len(c.members),
                },
                weights={"member_count": 0.5, "signature_purity": 0.5},
                confidence=min(1.0, c.signatures.most_common(1)[0][1] / len(c.members)),
            )
            pattern = DiscoveredPattern(
                id=pid,
                member_unit_ids=list(c.members),
                representative_signature=representative_signature,
                centroid=c.sem_centroid.astype(float).tolist(),
                role_template=rep_unit.role_sequence,
                evidence=evidence,
            )
            patterns.append(pattern)
            for m in c.members:
                unit_to_pattern[m] = pid

            if log:
                fp = rep_unit.structural_fingerprint or {}
                log.event(
                    "pattern_discovered",
                    pattern_id=pid,
                    member_count=len(c.members),
                    representative_signature=representative_signature,
                    role_template=rep_unit.role_sequence,
                    structural_fingerprint={
                        k: round(v, 4) for k, v in fp.items()
                    },
                    common_characteristics={
                        "field_count": fp.get("field_slot_count", 0),
                        "column_count": fp.get("column_count", 0),
                        "from_table_row": fp.get("from_table_row", 0),
                        "from_heading": fp.get("from_heading", 0),
                        "from_paragraph": fp.get("from_paragraph", 0),
                    },
                    confidence=round(evidence.confidence, 4),
                )
                log.push()
                log.line(f"ContentUnit: {rep_unit.id}")
                log.line("")
                log.line("Structural signature:")
                log.push()
                for role in rep_unit.role_sequence:
                    label = {
                        "PROMINENT": "Prominent text",
                        "BODY": "Supporting paragraph",
                        "META": "Metadata-like content",
                    }.get(role, role)
                    log.line(f"- {label}")
                log.pop()
                log.line(f"Discovered pattern: {pid} ({len(c.members)} unit(s), signature {representative_signature})")
                log.line("Semantic representation generated.")
                log.pop()

        return patterns, unit_to_pattern
