"""Quoted capitalised entries must stay atomic (definition-list anti-merge)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from blockdiscovery.generic_discovery import (
    CandidateUnit,
    GenericDiscoveryEngine,
    _opens_quoted_capital,
    _starts_capitalised_unit,
)


def _cand(idx: int, text: str) -> CandidateUnit:
    return CandidateUnit(
        id=f"c{idx}",
        document_id="d",
        page_number=1,
        order=idx,
        text=text,
        raw_unit_ids=[f"r{idx}"],
        layout_class="text",
    )


class QuotedTermAtomicTests(unittest.TestCase):
    def test_helpers_look_past_curly_quotes(self) -> None:
        text = "\u201c Affiliate \u201d means any entity."
        self.assertTrue(_opens_quoted_capital(text))
        self.assertTrue(_starts_capitalised_unit(text))
        self.assertFalse(_opens_quoted_capital("Affiliate means any entity."))
        self.assertFalse(_opens_quoted_capital("\u201c affiliate \u201d means…"))

    def test_forced_split_between_quoted_definitions(self) -> None:
        engine = GenericDiscoveryEngine(backend=MagicMock())
        a = _cand(
            1,
            "\u201c Affiliate \u201d means any entity that directly or indirectly "
            "controls the subject entity.",
        )
        b = _cand(2, "\u201c Agreement \u201d means this Main Services Agreement.")
        c = _cand(
            3,
            "\u201c Beta Services \u201d means services clearly designated as beta.",
        )
        decisions, _ = engine._analyze([a, b, c], trace=None)
        self.assertEqual(len(decisions), 2)
        for d in decisions:
            self.assertEqual(d.decision, "START_NEW_LOGICAL_BLOCK")
            self.assertIn("quoted term", d.reason.lower())


if __name__ == "__main__":
    unittest.main()
