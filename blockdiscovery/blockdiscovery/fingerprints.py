"""Structural fingerprint generation for content units / logical blocks.

A role sequence such as ``PROMINENT, BODY, BODY`` is only one weak feature.
Fingerprints combine role, geometry, alignment, spacing, typography, density,
local position, and boundary characteristics so similar-looking but different
content types are less likely to collapse into the same pattern.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .models import TextBlock
from .utils import clip01


def build_structural_fingerprint(
    blocks: List[TextBlock],
    roles: List[str],
    boundary_scores: Optional[List[float]] = None,
    mean_signals: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Build a rich, document-relative structural fingerprint."""
    if not blocks:
        return {
            "block_count": 0.0,
            "role_prominent": 0.0,
            "role_body": 0.0,
            "role_meta": 0.0,
        }

    xs0 = [b.bounding_box.x0 for b in blocks]
    rel_cx = [b.features.get("rel_cx", 0.5) for b in blocks]
    gaps = []
    for i in range(1, len(blocks)):
        gap = blocks[i].bounding_box.y0 - blocks[i - 1].bounding_box.y1
        gaps.append(max(0.0, gap))

    sizes = [b.features.get("size_ratio_median", 1.0) for b in blocks]
    bold = [b.features.get("bold_ratio", 0.0) for b in blocks]
    dens = [b.features.get("visual_density", 0.0) for b in blocks]
    cols = [b.features.get("column_bucket", 0.0) for b in blocks]

    left_spread = float(np.std(xs0)) if len(xs0) > 1 else 0.0
    cx_spread = float(np.std(rel_cx)) if len(rel_cx) > 1 else 0.0
    gap_mean = float(np.mean(gaps)) if gaps else 0.0
    gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    size_spread = float(np.std(sizes)) if len(sizes) > 1 else 0.0

    # Alignment signature: low spread => consistent left/center column.
    if cx_spread < 0.04:
        alignment_sig = 1.0  # tight column
    elif cx_spread < 0.12:
        alignment_sig = 0.6
    else:
        alignment_sig = 0.2  # mixed / multi-column

    # Spacing signature: regularity of internal gaps.
    if not gaps:
        spacing_sig = 0.5
    else:
        spacing_sig = clip01(1.0 - (gap_std / (gap_mean + 1e-3)) / 3.0)

    # Typography signature: head vs body contrast + boldness.
    head_size = max(sizes) if sizes else 1.0
    body_sizes = [s for s, r in zip(sizes, roles) if r != "PROMINENT"] or sizes
    typography_sig = clip01(
        0.4 * min(2.0, head_size) / 2.0
        + 0.3 * float(np.mean(bold))
        + 0.3 * (1.0 - min(1.0, size_spread))
    )

    content_density = float(np.mean(dens)) if dens else 0.0
    col_entropy = len(set(int(c) for c in cols)) / 3.0

    boundary_mean = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    boundary_max = float(np.max(boundary_scores)) if boundary_scores else 0.0

    mean_signals = mean_signals or {}
    fp = {
        "block_count": float(len(blocks)),
        "role_prominent": float(roles.count("PROMINENT")),
        "role_body": float(roles.count("BODY")),
        "role_meta": float(roles.count("META")),
        "alignment_signature": alignment_sig,
        "alignment_spread": clip01(cx_spread * 5.0),
        "left_spread_norm": clip01(left_spread / 80.0),
        "spacing_signature": spacing_sig,
        "spacing_mean": gap_mean,
        "spacing_std": gap_std,
        "typography_signature": typography_sig,
        "head_size_ratio": head_size,
        "mean_body_size_ratio": float(np.mean(body_sizes)) if body_sizes else 1.0,
        "content_density": content_density,
        "column_diversity": col_entropy,
        "local_position": blocks[0].features.get("page_fraction", 0.0),
        "page_number_norm": float(blocks[0].page_number) / 100.0,
        "boundary_mean": boundary_mean,
        "boundary_max": boundary_max,
        "mean_semantic": float(mean_signals.get("semantic_coherence", 0.0)),
        "mean_spatial": float(mean_signals.get("spatial_proximity", 0.0)),
        "char_count_log": float(np.log1p(sum(b.char_count for b in blocks))),
    }
    return fp


# Union of the keys emitted by the native builder and the generic engine. A
# fingerprint supplies whatever it measured; absent dimensions stay at zero, so
# one vectorizer serves both without either knowing about the other.
_VECTOR_KEYS = (
    "role_prominent",
    "role_body",
    "role_meta",
    "block_count",
    "alignment_signature",
    "spacing_signature",
    "typography_signature",
    "content_density",
    "column_diversity",
    "head_size_ratio",
    "mean_semantic",
    "boundary_mean",
    "char_count_log",
    "local_position",
    # Generic-engine dimensions.
    "cell_count",
    "mean_rel_x0",
    "mean_rel_width",
    "mean_line_height_ratio",
    "mean_digit_ratio",
    "mean_upper_ratio",
    "marker_depth",
    "in_grid",
    "repetition",
)
_BLOCK_COUNT_INDEX = _VECTOR_KEYS.index("block_count")
_REPETITION_INDEX = _VECTOR_KEYS.index("repetition")


def fingerprint_vector(fp: Dict[str, float]) -> np.ndarray:
    """Numeric vector used by pattern discovery (subset of fingerprint keys)."""
    v = np.array([fp.get(k, 0.0) for k in _VECTOR_KEYS], dtype=np.float32)
    # Soft-scale counts so magnitude does not dominate cosine direction.
    v[_BLOCK_COUNT_INDEX] = np.log1p(v[_BLOCK_COUNT_INDEX])
    v[_REPETITION_INDEX] = np.log1p(v[_REPETITION_INDEX])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
