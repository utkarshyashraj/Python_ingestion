"""Unit tests for post-discovery logical block consolidation."""

from __future__ import annotations

import unittest

from blockdiscovery.logical_block_consolidator import LogicalBlockConsolidator
from blockdiscovery.models import (
    BoundingBox,
    Document,
    Evidence,
    LogicalBlock,
    PageInfo,
    TextBlock,
)


def _block(
    doc_id: str,
    idx: int,
    text: str,
    *,
    page: int = 1,
    roles: list | None = None,
    block_type: str = "content",
    source_table_id: str | None = None,
    features: dict | None = None,
    fingerprint: dict | None = None,
    confidence: float = 0.8,
    doc_position: float | None = None,
) -> LogicalBlock:
    return LogicalBlock(
        id=f"{doc_id}_logical_block_{idx:04d}",
        content_unit_id=f"{doc_id}_unit_{idx:04d}",
        document_id=doc_id,
        source_document=f"{doc_id}.pdf",
        source_page=page,
        page_end=page,
        source_block_ids=[f"{doc_id}_block_{idx:04d}"],
        text=text,
        structural_features=features or {"prominence": 0.2, "rel_x0": 0.1, "gap_above_ratio": 1.0},
        role_sequence=roles or ["BODY"],
        structural_signature="P0B1M0",
        structural_fingerprint=fingerprint or {"block_count": 1.0},
        confidence=confidence,
        evidence=Evidence(confidence=confidence),
        doc_position=doc_position if doc_position is not None else idx / 100.0,
        block_type=block_type,
        source_table_id=source_table_id,
    )


def _doc_with_boxes(
    doc_id: str,
    blocks: list[LogicalBlock],
    boxes: dict[str, BoundingBox],
    page_width: float = 612.0,
    page_height: float = 792.0,
) -> Document:
    document = Document(
        id=doc_id,
        source_path=f"{doc_id}.pdf",
        pages=[
            PageInfo(
                document_id=doc_id,
                page_number=p,
                width=page_width,
                height=page_height,
            )
            for p in sorted({b.source_page for b in blocks})
        ],
    )
    for b in blocks:
        bid = b.source_block_ids[0]
        document.blocks.append(
            TextBlock(
                id=bid,
                document_id=doc_id,
                page_number=b.source_page,
                text=b.text,
                bounding_box=boxes[bid],
                reading_order=int(b.doc_position * 1000),
            )
        )
    return document


class LogicalBlockConsolidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.consolidator = LogicalBlockConsolidator(logger=None)

    def test_lowercase_continuation_merges(self) -> None:
        prev = _block("d", 1, 'utilities have used could introduce "table locks"')
        cur = _block(
            "d",
            2,
            "or other conditions that would block end users from being able to use the application.",
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(50, 100, 500, 130),
                cur.source_block_ids[0]: BoundingBox(50, 135, 500, 165),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "MERGE")
        self.assertGreater(decision.continuation.capitalization_score, 0.7)
        self.assertGreater(decision.continuation.sentence_continuation_score, 0.6)

    def test_sentence_continuation_across_blocks(self) -> None:
        prev = _block(
            "d",
            1,
            "Siebel UCM: Publish in Hybrid Mode\n"
            "Universal Customer Master (UCM) is enhanced to enable publishing",
            roles=["PROMINENT", "BODY"],
            features={"prominence": 0.4, "rel_x0": 0.1, "gap_above_ratio": 1.0},
        )
        cur = _block(
            "d",
            2,
            "regardless of their integration protocol (SOAP or REST).",
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 200, 520, 260),
                cur.source_block_ids[0]: BoundingBox(40, 265, 520, 290),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "MERGE")

    def test_same_paragraph_across_page_boundary(self) -> None:
        prev = _block(
            "d",
            1,
            "Universal Customer Master is enhanced to enable publishing",
            page=4,
        )
        cur = _block(
            "d",
            2,
            "to multiple middleware systems or combinations of middleware systems.",
            page=5,
            features={"prominence": 0.2, "rel_x0": 0.1, "gap_above_ratio": 0.5},
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                # Near bottom of page 4 → near top of page 5.
                prev.source_block_ids[0]: BoundingBox(48, 720, 540, 760),
                cur.source_block_ids[0]: BoundingBox(48, 60, 540, 100),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "MERGE")
        self.assertGreater(decision.continuation.page_continuation_score, 0.5)

    def test_multi_block_continuation_chain(self) -> None:
        a = _block(
            "d",
            1,
            "Siebel UCM: Publish in Hybrid Mode\nUniversal Customer Master (UCM) is enhanced",
            roles=["PROMINENT", "BODY"],
        )
        b = _block("d", 2, "regardless of their integration protocol (SOAP or REST).")
        c = _block(
            "d",
            3,
            "This feature removes the current single-middleware restriction and "
            "supports mixed environments.",
        )
        document = _doc_with_boxes(
            "d",
            [a, b, c],
            {
                a.source_block_ids[0]: BoundingBox(40, 100, 520, 160),
                b.source_block_ids[0]: BoundingBox(40, 165, 520, 190),
                c.source_block_ids[0]: BoundingBox(40, 195, 520, 250),
            },
        )
        result = self.consolidator.consolidate(document, [a, b, c])
        self.assertEqual(result.stats["consolidated_logical_blocks"], 1)
        self.assertEqual(result.stats["merge_chains"], 1)
        self.assertEqual(len(result.blocks[0].source_logical_block_ids or []), 3)
        self.assertIn("regardless", result.blocks[0].text)
        self.assertIn("single-middleware", result.blocks[0].text)

    def test_strong_new_heading_keeps_separate(self) -> None:
        prev = _block(
            "d",
            1,
            "Schema Changes without Downtime for Updates and Migration.",
            roles=["PROMINENT", "BODY"],
            features={"prominence": 0.6, "marker_depth": 2.0, "rel_x0": 0.1},
        )
        cur = _block(
            "d",
            2,
            "Replacement of Obsolete Data Types from Oracle and MSSQL Databases.",
            roles=["PROMINENT"],
            features={"prominence": 0.85, "marker_depth": 2.0, "rel_x0": 0.1, "gap_above_ratio": 2.5},
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 100, 520, 180),
                cur.source_block_ids[0]: BoundingBox(40, 240, 520, 270),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")
        self.assertGreater(decision.new_boundary.heading_change, 0.4)

    def test_typography_change_supports_separation(self) -> None:
        prev = _block(
            "d",
            1,
            "This paragraph explains a completed idea in normal body text.",
            roles=["BODY"],
            features={"prominence": 0.2, "mean_line_height_ratio": 1.0, "rel_x0": 0.1},
        )
        cur = _block(
            "d",
            2,
            "Brand New Topic Header",
            roles=["PROMINENT"],
            features={
                "prominence": 0.9,
                "marker_depth": 2.0,
                "mean_line_height_ratio": 1.8,
                "rel_x0": 0.1,
                "gap_above_ratio": 3.0,
            },
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 100, 520, 160),
                cur.source_block_ids[0]: BoundingBox(40, 220, 300, 245),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")

    def test_large_spatial_gap_keeps_separate(self) -> None:
        prev = _block(
            "d",
            1,
            "First completed thought ends here.",
            features={"prominence": 0.2, "rel_x0": 0.1, "gap_above_ratio": 1.0},
        )
        cur = _block(
            "d",
            2,
            "Second completed thought begins independently.",
            features={"prominence": 0.2, "rel_x0": 0.1, "gap_above_ratio": 8.0},
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 100, 520, 140),
                cur.source_block_ids[0]: BoundingBox(40, 420, 520, 460),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")
        self.assertGreater(decision.new_boundary.spacing_break, 0.5)

    def test_different_reading_columns_keep_separate(self) -> None:
        prev = _block("d", 1, "Left column narrative continues in its own flow.")
        cur = _block("d", 2, "Right column narrative occupies a different band.")
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 120, 260, 220),
                cur.source_block_ids[0]: BoundingBox(320, 120, 540, 220),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")
        self.assertGreater(decision.new_boundary.reading_region_change, 0.5)

    def test_table_rows_keep_separate(self) -> None:
        prev = _block(
            "d",
            1,
            "Alpha One QX-4410 Nerulic",
            block_type="structured_record",
            source_table_id="grid_001",
        )
        cur = _block(
            "d",
            2,
            "Beta Two QX-5521 Dravish",
            block_type="structured_record",
            source_table_id="grid_001",
        )
        document = Document(id="d", source_path="d.pdf")
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")
        self.assertIsNotNone(decision.veto)

    def test_structured_cell_wrap_continuation_merges(self) -> None:
        prev = _block(
            "d",
            1,
            "Universal Customer Master (UCM) is enhanced to enable publishing to Oracle",
            page=4,
            block_type="structured_record",
            source_table_id="grid_001",
            roles=["PROMINENT", "BODY"],
        )
        cur = _block(
            "d",
            2,
            "regardless of their integration protocol (SOAP or REST).",
            page=5,
            block_type="structured_record",
            source_table_id="grid_001",
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 700, 520, 760),
                cur.source_block_ids[0]: BoundingBox(40, 60, 520, 100),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "MERGE")
        self.assertIsNone(decision.veto)

    def test_key_value_style_blocks_with_terminal_punctuation_stay_separate(self) -> None:
        prev = _block("d", 1, "Name: Alpha One.")
        cur = _block("d", 2, "Name: Beta Two.")
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 100, 300, 120),
                cur.source_block_ids[0]: BoundingBox(40, 140, 300, 160),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")

    def test_section_boundary_body_to_prominent(self) -> None:
        prev = _block(
            "d",
            1,
            "The previous section ends with a complete sentence.",
            roles=["BODY"],
        )
        cur = _block(
            "d",
            2,
            "Next Section Title",
            roles=["PROMINENT"],
            features={"prominence": 0.9, "marker_depth": 2.0, "rel_x0": 0.1, "gap_above_ratio": 2.0},
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 100, 520, 140),
                cur.source_block_ids[0]: BoundingBox(40, 180, 280, 205),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")

    def test_independent_blocks_similar_typography_stay_separate(self) -> None:
        prev = _block(
            "d",
            1,
            "First independent idea is fully stated here.",
            features={"prominence": 0.25, "mean_line_height_ratio": 1.0, "rel_x0": 0.1},
        )
        cur = _block(
            "d",
            2,
            "Second independent idea is also fully stated here.",
            features={
                "prominence": 0.25,
                "mean_line_height_ratio": 1.0,
                "rel_x0": 0.1,
                "gap_above_ratio": 2.2,
            },
        )
        document = _doc_with_boxes(
            "d",
            [prev, cur],
            {
                prev.source_block_ids[0]: BoundingBox(40, 100, 520, 150),
                cur.source_block_ids[0]: BoundingBox(40, 190, 520, 240),
            },
        )
        decision = self.consolidator.evaluate_pair(document, prev, cur)
        self.assertEqual(decision.decision, "KEEP_SEPARATE")
        # Finished sentences with a non-trivial gap must stay independent even
        # when body typography would otherwise look compatible.
        self.assertLess(decision.continuation.sentence_continuation_score, 0.4)
        self.assertGreater(decision.new_boundary.spacing_break, 0.4)

    def test_provenance_preserved_on_merge(self) -> None:
        a = _block("d", 1, "Opening clause that is incomplete")
        b = _block("d", 2, "and finishes on the next line.")
        document = _doc_with_boxes(
            "d",
            [a, b],
            {
                a.source_block_ids[0]: BoundingBox(40, 100, 500, 130),
                b.source_block_ids[0]: BoundingBox(40, 135, 500, 160),
            },
        )
        result = self.consolidator.consolidate(document, [a, b])
        self.assertEqual(len(result.blocks), 1)
        merged = result.blocks[0]
        self.assertEqual(merged.source_logical_block_ids, [a.id, b.id])
        self.assertEqual(
            set(merged.source_block_ids),
            {a.source_block_ids[0], b.source_block_ids[0]},
        )
        self.assertIsNotNone(merged.consolidation_evidence)


if __name__ == "__main__":
    unittest.main()
