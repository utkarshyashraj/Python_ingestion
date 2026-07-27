"""Feature generation.

Computes document-level statistics first, then per-block geometry, typography,
layout and structural features. Crucially, every "importance"-style feature is
*relative* to the current document (e.g. font-size z-score against the local
distribution) so the engine never depends on absolute values like
``font_size == 16``.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from .logging_utils import DiscoveryLogger
from .models import Document, TextBlock


def _median(values: List[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def _std(values: List[float], default: float = 0.0) -> float:
    return statistics.pstdev(values) if len(values) > 1 else default


def compute_document_stats(document: Document) -> Dict[str, float]:
    text_blocks = [b for b in document.blocks if b.block_type == "text" and b.formatting]

    sizes = [b.formatting.dominant_size for b in text_blocks]
    heights = [b.bounding_box.height for b in text_blocks]
    widths = [b.bounding_box.width for b in text_blocks]
    x0s = [b.bounding_box.x0 for b in text_blocks]

    # Vertical gaps between consecutive blocks on the same page (typical spacing).
    gaps: List[float] = []
    prev: Optional[TextBlock] = None
    for b in text_blocks:
        if prev is not None and prev.page_number == b.page_number:
            gap = b.bounding_box.y0 - prev.bounding_box.y1
            if gap >= 0:
                gaps.append(gap)
        prev = b

    page_w = _median([p.width for p in document.pages], 612.0)
    page_h = _median([p.height for p in document.pages], 792.0)

    stats = {
        "size_median": _median(sizes, 10.0),
        "size_mean": (sum(sizes) / len(sizes)) if sizes else 10.0,
        "size_std": _std(sizes, 1.0) or 1.0,
        "size_max": max(sizes) if sizes else 10.0,
        "size_min": min(sizes) if sizes else 10.0,
        "height_median": _median(heights, 12.0),
        "width_median": _median(widths, 200.0),
        "gap_median": _median(gaps, 6.0) or 6.0,
        "gap_std": _std(gaps, 4.0) or 4.0,
        "x0_median": _median(x0s, 72.0),
        "x0_std": _std(x0s, 1.0) or 1.0,
        "page_width": page_w,
        "page_height": page_h,
        "bold_block_ratio": (
            sum(1 for b in text_blocks if b.formatting.bold_ratio > 0.5) / len(text_blocks)
            if text_blocks else 0.0
        ),
    }
    document.stats = stats
    return stats


def _z(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (value - mean) / std


class FeatureGenerator:
    """Attaches a feature dict to every block, using document-relative signals."""

    def __init__(self, logger: Optional[DiscoveryLogger] = None) -> None:
        self.logger = logger

    def generate(self, document: Document) -> Document:
        stats = compute_document_stats(document)
        text_blocks = [b for b in document.blocks if b.block_type == "text"]

        # Group blocks by page for local layout features.
        by_page: Dict[int, List[TextBlock]] = {}
        for b in text_blocks:
            by_page.setdefault(b.page_number, []).append(b)

        for page_number, blocks in by_page.items():
            page = next((p for p in document.pages if p.page_number == page_number), None)
            pw = page.width if page else stats["page_width"]
            ph = page.height if page else stats["page_height"]
            blocks_sorted = sorted(blocks, key=lambda b: b.reading_order)

            for i, b in enumerate(blocks_sorted):
                bb = b.bounding_box
                fmt = b.formatting

                # ---- Geometry ------------------------------------------------
                f: Dict[str, float] = {
                    "x0": bb.x0,
                    "y0": bb.y0,
                    "width": bb.width,
                    "height": bb.height,
                    "area": bb.area,
                    "rel_x": bb.x0 / pw if pw else 0.0,
                    "rel_y": bb.y0 / ph if ph else 0.0,
                    "rel_width": bb.width / pw if pw else 0.0,
                    "rel_cx": bb.cx / pw if pw else 0.0,
                    "page_fraction": (b.reading_order + 1) / max(1, len(text_blocks)),
                }

                # Column position estimate: left / center / right thirds.
                f["column_bucket"] = min(2, int(f["rel_cx"] * 3))

                # ---- Typography (relative!) ---------------------------------
                if fmt:
                    f["font_size"] = fmt.dominant_size
                    f["size_zscore"] = _z(fmt.dominant_size, stats["size_mean"], stats["size_std"])
                    f["size_ratio_median"] = (
                        fmt.dominant_size / stats["size_median"] if stats["size_median"] else 1.0
                    )
                    f["bold_ratio"] = fmt.bold_ratio
                    f["italic_ratio"] = fmt.italic_ratio
                    f["monospace_ratio"] = fmt.monospace_ratio
                    f["size_variety"] = float(fmt.size_variety)
                else:
                    f["font_size"] = stats["size_median"]
                    f["size_zscore"] = 0.0
                    f["size_ratio_median"] = 1.0
                    f["bold_ratio"] = 0.0
                    f["italic_ratio"] = 0.0
                    f["monospace_ratio"] = 0.0
                    f["size_variety"] = 1.0

                # ---- Layout --------------------------------------------------
                f["line_count"] = float(b.line_count)
                f["char_count"] = float(b.char_count)
                f["is_short"] = 1.0 if b.char_count <= 60 else 0.0
                f["left_aligned_to_median"] = 1.0 if abs(bb.x0 - stats["x0_median"]) <= 3.0 else 0.0

                # Whitespace above / below relative to typical gap.
                gap_above = None
                gap_below = None
                if i > 0:
                    gap_above = bb.y0 - blocks_sorted[i - 1].bounding_box.y1
                if i < len(blocks_sorted) - 1:
                    gap_below = blocks_sorted[i + 1].bounding_box.y0 - bb.y1
                f["gap_above"] = gap_above if gap_above is not None else stats["gap_median"]
                f["gap_below"] = gap_below if gap_below is not None else stats["gap_median"]
                f["gap_above_ratio"] = (f["gap_above"] / stats["gap_median"]) if stats["gap_median"] else 1.0
                f["gap_below_ratio"] = (f["gap_below"] / stats["gap_median"]) if stats["gap_median"] else 1.0

                # Visual density: chars per unit area (prominent titles are sparse).
                f["visual_density"] = (b.char_count / bb.area) if bb.area > 0 else 0.0

                # ---- Prominence: fused, document-relative --------------------
                # A block is "prominent" when it is larger and/or bolder than the
                # local norm and short -- classic heading-like evidence, but
                # expressed relatively rather than via fixed thresholds.
                prominence = 0.0
                prominence += max(0.0, f["size_zscore"]) * 0.5
                prominence += f["bold_ratio"] * 0.9
                prominence += (0.4 if f["is_short"] else 0.0)
                prominence += max(0.0, f["size_ratio_median"] - 1.0) * 0.8
                f["prominence"] = prominence

                b.features = f

        if self.logger:
            self.logger.section("FEATURES")
            self.logger.kv("Font size (median)", round(stats["size_median"], 2))
            self.logger.kv("Font size (max/min)", f"{stats['size_max']:.1f}/{stats['size_min']:.1f}")
            self.logger.kv("Typical block gap", round(stats["gap_median"], 2))
            self.logger.event(
                "feature_generation_completed",
                document_id=document.id,
                stats={k: round(v, 3) for k, v in stats.items()},
            )
        return document
