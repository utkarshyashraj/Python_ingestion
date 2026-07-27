"""Generate synthetic PDFs whose terminology the engine has never seen.

Every label, heading and column name below is invented nonsense. If discovery
still finds coherent structure in these documents, the algorithm is reading
layout evidence rather than recognising words.

Usage:
    python tools/make_synthetic_pdfs.py data/synthetic
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import pymupdf

PAGE_W, PAGE_H = 595.0, 842.0
MARGIN = 48.0


class Sheet:
    def __init__(self, doc: pymupdf.Document) -> None:
        self.doc = doc
        self.page = doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN

    def _space(self, amount: float) -> None:
        self.y += amount
        if self.y > PAGE_H - MARGIN:
            self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
            self.y = MARGIN

    def heading(self, text: str, size: float = 17.0) -> None:
        self._space(14)
        self.page.insert_text(
            (MARGIN, self.y), text, fontsize=size, fontname="hebo"
        )
        self.y += size + 6

    def paragraph(self, text: str, size: float = 10.0, width: float = PAGE_W - 2 * MARGIN) -> None:
        rect = pymupdf.Rect(MARGIN, self.y, MARGIN + width, self.y + 400)
        used = self.page.insert_textbox(rect, text, fontsize=size, fontname="helv", align=0)
        consumed = 400 - (used if used > 0 else 0)
        self._space(max(size * 2.0, consumed + 8))

    def table(self, rows: Sequence[Sequence[str]], size: float = 9.0) -> None:
        """Borderless, column-aligned rows — structure lives in the geometry."""
        if not rows:
            return
        columns = max(len(r) for r in rows)
        usable = PAGE_W - 2 * MARGIN
        col_w = usable / columns
        row_h = size * 1.9
        for r_i, row in enumerate(rows):
            if self.y + row_h > PAGE_H - MARGIN:
                self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
                self.y = MARGIN
            for c_i in range(columns):
                cell = row[c_i] if c_i < len(row) else ""
                if not cell:
                    continue
                self.page.insert_text(
                    (MARGIN + c_i * col_w, self.y + size),
                    cell[: max(6, int(col_w / (size * 0.5)))],
                    fontsize=size,
                    fontname="hebo" if r_i == 0 else "helv",
                )
            self.y += row_h
        self._space(14)

    def two_columns(self, left: str, right: str, size: float = 9.5) -> None:
        gutter = 18.0
        col_w = (PAGE_W - 2 * MARGIN - gutter) / 2
        top = self.y
        self.page.insert_textbox(
            pymupdf.Rect(MARGIN, top, MARGIN + col_w, top + 220), left, fontsize=size, fontname="helv"
        )
        self.page.insert_textbox(
            pymupdf.Rect(MARGIN + col_w + gutter, top, PAGE_W - MARGIN, top + 220),
            right,
            fontsize=size,
            fontname="helv",
        )
        self._space(232)

    def running_footer(self, text: str) -> None:
        for page in self.doc:
            page.insert_text((MARGIN, PAGE_H - 28), text, fontsize=8, fontname="helv")


# --------------------------------------------------------------------------- #
def build_grid_document(path: str) -> None:
    """Repeated 5-column records under invented headings."""
    doc = pymupdf.open()
    s = Sheet(doc)
    s.heading("Zorvax Quarnly Compendium")
    s.paragraph(
        "Threbular yannic plints are collated below. Each grintwold entry lists its "
        "own vorbish attributes and should be read independently of its neighbours."
    )
    s.heading("Plimwar Grintwolds", size=13.0)
    rows: List[List[str]] = [
        ["Grintwold", "Yannic", "Plimwar", "Vorbish Note", "Threb"],
        ["Alpha One", "QX-4410", "Nerulic", "Alpha One holds a narrow plint.", "Yes"],
        ["Beta Two", "QX-5521", "Dravish", "Beta Two extends the nerulic band.", "No"],
        ["Gamma Three", "QX-6632", "Nerulic", "Gamma Three folds the plint inward.", "Yes"],
        ["Delta Four", "QX-7743", "Sombric", "Delta Four resets every quarn cycle.", "No"],
        ["Epsilon Five", "QX-8854", "Dravish", "Epsilon Five is bound to Delta Four.", "Yes"],
    ]
    s.table(rows)
    s.heading("Sombric Grintwolds", size=13.0)
    s.table(
        [
            ["Grintwold", "Yannic", "Plimwar", "Vorbish Note", "Threb"],
            ["Zeta Six", "QX-9965", "Sombric", "Zeta Six mirrors Epsilon Five.", "No"],
            ["Eta Seven", "QX-1076", "Nerulic", "Eta Seven drifts across quarns.", "Yes"],
            ["Theta Eight", "QX-2187", "Dravish", "Theta Eight seals the plint.", "No"],
        ]
    )
    s.heading("Krellan Remarks", size=13.0)
    s.paragraph(
        "Krellan remarks accompany the grintwold tables. They describe the surrounding "
        "vorbish conditions in prose and carry no tabular structure of their own."
    )
    s.running_footer("Return to index")
    doc.save(path)
    doc.close()


def build_record_document(path: str) -> None:
    """Same information shaped as heading-plus-prose records, not a grid."""
    doc = pymupdf.open()
    s = Sheet(doc)
    s.heading("Wextor Plindrome Register")
    s.paragraph(
        "This register states each plindrome as a standalone stanza. There is no grid; "
        "the structure must be found from spacing and typography alone."
    )
    tiers = ("frunly", "brelt", "sombric")
    labels = ["X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q", "P", "N", "M", "K", "J", "H"]
    for i, label in enumerate(labels + [f"{c}2" for c in labels[:12]]):
        s.heading(f"Record {label}", size=12.5)
        s.paragraph(
            f"Value {label} occupies band {i + 1} of the plindrome register and is "
            f"stated without reference to its neighbours."
        )
        s.paragraph(f"Metadata {label}: {tiers[i % 3]}, tier {i % 4 + 1}", size=8.5)
    s.heading("Trailing Nook", size=12.5)
    s.running_footer("Return to index")
    doc.save(path)
    doc.close()


def build_mixed_document(path: str) -> None:
    """Multi-column prose, a narrow grid and reordered headings."""
    doc = pymupdf.open()
    s = Sheet(doc)
    s.heading("Ombric Fasculary")
    s.two_columns(
        "Left channel narrative. The fasculary drifts through several ombric stages, "
        "each one wider than the last. Nothing here repeats, so the discovery engine "
        "should treat this as flowing prose rather than a record series.",
        "Right channel narrative. Continuing the same ombric account across the gutter, "
        "this column keeps the identical typography while occupying a different band of "
        "the page, which tests alignment reasoning.",
    )
    s.heading("Narrow Plints", size=12.0)
    s.table(
        [
            ["Plint", "Quarn"],
            ["Nokta", "11"],
            ["Vresk", "12"],
            ["Jarlo", "13"],
            ["Umbek", "14"],
        ]
    )
    s.heading("Wide Plints", size=12.0)
    s.table(
        [
            ["Plint", "Quarn", "Fascule", "Drell", "Ombric", "Yannic", "Threb"],
            ["Nokta", "11", "aa", "bb", "cc", "dd", "ee"],
            ["Vresk", "12", "ff", "gg", "hh", "ii", "jj"],
            ["Jarlo", "13", "kk", "ll", "mm", "nn", "oo"],
        ]
    )
    s.heading("Empty Nook", size=12.0)
    s.running_footer("Return to index")
    doc.save(path)
    doc.close()


BUILDERS: Sequence[Tuple[str, object]] = (
    ("synthetic_grid.pdf", build_grid_document),
    ("synthetic_records.pdf", build_record_document),
    ("synthetic_mixed.pdf", build_mixed_document),
)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, builder in BUILDERS:
        path = os.path.join(out_dir, name)
        builder(path)  # type: ignore[operator]
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
