"""Cross-document logical grouping.

Compares logical blocks across (and within) documents using a *fused* similarity
of semantic + structural + contextual evidence -- never exact text equality --
and forms Discovered Logical Groups via union-find connected components above a
configurable threshold.

Optionally, and only if the user supplies a label lexicon, an after-the-fact
descriptive label is attached to each group. The lexicon is empty by default:
the grouping algorithm itself never depends on any business terminology.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import numpy as np

from .config import EngineConfig
from .fingerprints import fingerprint_vector
from .logging_utils import DiscoveryLogger
from .models import Evidence, LogicalBlock, LogicalGroup
from .semantics import EmbeddingBackend, cosine


_SIG_RE = re.compile(r"P(\d+)B(\d+)M(\d+)")


def _sig_vector(signature: str) -> np.ndarray:
    m = _SIG_RE.match(signature or "")
    if not m:
        return np.zeros(3, dtype=np.float32)
    v = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class CrossDocumentGrouping:
    def __init__(
        self,
        config: EngineConfig,
        backend: EmbeddingBackend,
        logger: Optional[DiscoveryLogger] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.logger = logger

    def _fused_matrix(self, logical_blocks: List[LogicalBlock]) -> np.ndarray:
        """Vectorised fused similarity matrix -- scales to thousands of blocks.

        Avoids O(n^2) Python-level pair calls by computing each signal with
        numpy matrix operations, then fusing and applying the semantic gate.
        """
        n = len(logical_blocks)
        dim = len(logical_blocks[0].semantic_vector) if logical_blocks and logical_blocks[0].semantic_vector else 1
        fp_dim = len(fingerprint_vector({}))
        sem_vecs = np.zeros((n, dim), dtype=np.float32)
        sig_vecs = np.zeros((n, 3), dtype=np.float32)
        fp_vecs = np.zeros((n, fp_dim), dtype=np.float32)
        pos = np.zeros(n, dtype=np.float32)
        pat = np.empty(n, dtype=object)
        for i, b in enumerate(logical_blocks):
            if b.semantic_vector is not None:
                v = np.asarray(b.semantic_vector, dtype=np.float32)
                nv = np.linalg.norm(v)
                sem_vecs[i] = v / nv if nv > 0 else v
            sig_vecs[i] = _sig_vector(b.structural_signature)
            fp_vecs[i] = fingerprint_vector(b.structural_fingerprint or {})
            pos[i] = b.doc_position
            pat[i] = b.discovered_pattern

        sem = sem_vecs @ sem_vecs.T
        np.clip(sem, 0.0, 1.0, out=sem)

        struct = sig_vecs @ sig_vecs.T
        np.clip(struct, 0.0, 1.0, out=struct)
        # The full structural fingerprint carries relative layout, so it is the
        # signal that lets an unfamiliar document match a familiar shape.
        fp_sim = fp_vecs @ fp_vecs.T
        np.clip(fp_sim, 0.0, 1.0, out=fp_sim)
        struct = np.maximum(struct, fp_sim)
        # Same discovered pattern -> full structural agreement.
        pat_codes = {p: k for k, p in enumerate(set(pat))}
        codes = np.array([pat_codes[p] for p in pat])
        same_pat = (codes[:, None] == codes[None, :]) & np.array([p is not None for p in pat])[:, None]
        struct = np.maximum(struct, same_pat.astype(np.float32))

        ctx = 1.0 - np.abs(pos[:, None] - pos[None, :])

        scale = max(1e-6, self.config.semantic_scale)
        sem_scaled = np.minimum(1.0, sem / scale)

        w = self.config.group_weights.normalized()
        fused = (
            w["semantic_similarity"] * sem_scaled
            + w["structural_similarity"] * struct
            + w["contextual_similarity"] * ctx
        ).astype(np.float32)

        # Eligibility needs either semantic agreement or near-identical
        # structure; requiring both would make shared layout undiscoverable
        # between documents that use different words for the same thing.
        weak_semantics = sem < self.config.thresholds.cross_document_semantic_gate
        weak_structure = struct < self.config.thresholds.cross_document_structural_gate
        fused[weak_semantics & weak_structure] = 0.0
        np.fill_diagonal(fused, 0.0)
        return fused

    def group(self, logical_blocks: List[LogicalBlock]) -> List[LogicalGroup]:
        log = self.logger
        if log:
            log.section("CROSS-DOCUMENT GROUPING")
        n = len(logical_blocks)
        thr = self.config.thresholds.cross_document_similarity
        if n == 0:
            return []

        fused = self._fused_matrix(logical_blocks)

        # Average-linkage leader clustering using *sparse candidate neighbours*:
        # a block only considers clusters it shares an above-threshold edge with,
        # then joins the one with the highest *mean* similarity to its members
        # (>= threshold). Averaging avoids single-link chaining; sparsity keeps
        # it scalable to thousands of blocks. Stable order -> deterministic ids.
        order = sorted(range(n), key=lambda k: (logical_blocks[k].document_id, logical_blocks[k].doc_position))
        assign = np.full(n, -1, dtype=np.int64)
        members: List[List[int]] = []
        log_budget = 3000
        for idx in order:
            row = fused[idx]
            nb = np.where(row >= thr)[0]
            candidate_clusters = {}
            for j in nb:
                c = assign[j]
                if c >= 0:
                    candidate_clusters.setdefault(int(c), []).append(int(j))
            best_c, best_avg, best_j = -1, 0.0, -1
            for c, _js in candidate_clusters.items():
                mem = members[c]
                avg = float(row[mem].mean())
                if avg > best_avg:
                    best_avg = avg
                    best_c = c
                    best_j = int(mem[int(np.argmax(row[mem]))])
            if best_c >= 0 and best_avg >= thr:
                members[best_c].append(idx)
                assign[idx] = best_c
                if log and log_budget > 0 and best_j >= 0:
                    a, b = logical_blocks[idx], logical_blocks[best_j]
                    if a.document_id != b.document_id:
                        log_budget -= 1
                        log.event(
                            "cross_document_similarity_calculated",
                            block_a=a.id, block_b=b.id,
                            document_a=a.document_id, document_b=b.document_id,
                            similarity=round(float(row[best_j]), 4),
                            decisive_edge=True,
                        )
            else:
                members.append([idx])
                assign[idx] = len(members) - 1

        groups: List[LogicalGroup] = []
        gi = 0
        for idxs in sorted(members, key=lambda c: len(c), reverse=True):
            gi += 1
            gid = f"logical_group_{gi:03d}"
            block_members = [logical_blocks[k] for k in idxs]
            member_ids = [m.id for m in block_members]
            doc_ids = sorted({m.document_id for m in block_members})
            patterns = Counter(m.discovered_pattern for m in block_members if m.discovered_pattern)
            dominant = patterns.most_common(1)[0][0] if patterns else None

            vecs = [np.asarray(m.semantic_vector, dtype=np.float32) for m in block_members if m.semantic_vector is not None]
            centroid = np.mean(vecs, axis=0) if vecs else None

            # Average intra-group similarity as evidence (from the fused matrix).
            if len(idxs) > 1:
                sub = fused[np.ix_(idxs, idxs)]
                iu = np.triu_indices(len(idxs), k=1)
                avg_sim = float(sub[iu].mean())
            else:
                avg_sim = 1.0

            evidence = Evidence(
                signals={"avg_pairwise_similarity": avg_sim, "member_count": float(len(block_members)),
                         "document_spread": float(len(doc_ids))},
                weights={"avg_pairwise_similarity": 1.0},
                confidence=avg_sim,
            )
            group = LogicalGroup(
                id=gid,
                member_block_ids=member_ids,
                document_ids=doc_ids,
                dominant_pattern=dominant,
                centroid=centroid.astype(float).tolist() if centroid is not None else None,
                evidence=evidence,
            )
            for m in block_members:
                m.group_id = gid
            groups.append(group)

            if log:
                log.event(
                    "logical_group_created",
                    group_id=gid,
                    member_count=len(members),
                    document_ids=doc_ids,
                    dominant_pattern=dominant,
                    confidence=round(avg_sim, 4),
                    member_block_ids=member_ids,
                )
                log.push()
                log.line(f"{gid}: {len(members)} block(s) across {len(doc_ids)} document(s)")
                log.pop()

        self._maybe_label(groups)
        return groups

    # ------------------------------------------------------------------ #
    def _maybe_label(self, groups: List[LogicalGroup]) -> None:
        """OPTIONAL, off by default. Attach a descriptive label to each group by
        comparing its centroid to user-supplied seed phrases. This never affects
        grouping; it only names an already-discovered group after the fact."""
        lexicon = self.config.optional_label_lexicon
        if not lexicon:
            return
        labels = list(lexicon.keys())
        seed_texts = [" ".join(lexicon[l]) for l in labels]
        seed_vecs = self.backend.embed(seed_texts)
        for g in groups:
            if g.centroid is None:
                continue
            c = np.asarray(g.centroid, dtype=np.float32)
            sims = [cosine(c, seed_vecs[i]) for i in range(len(labels))]
            best = int(np.argmax(sims)) if sims else -1
            if best >= 0 and sims[best] > 0.15:
                g.inferred_label = labels[best]
                g.evidence.notes.append(f"optional label '{labels[best]}' (sim={sims[best]:.2f})")
                if self.logger:
                    self.logger.event(
                        "logical_group_updated",
                        group_id=g.id,
                        inferred_label=labels[best],
                        label_similarity=round(float(sims[best]), 4),
                    )
