"""Generate structurally *different* sample PDFs to exercise the engine.

These synthetic release notes deliberately vary layout, headings, terminology
and ordering so we can validate that discovery does not depend on any of them:

* release_26_0 : explicit FEATURES / FIXES / BUGS headings
* release_26_1 : same categories, different wording and order
* release_26_2 : "New Capabilities" / "Resolved Issues" headings
* release_26_3 : NO headings at all, units separated by rules/whitespace
* release_26_4 : compact "Title / Description / Metadata" stacked layout

Semantically related content (authentication, billing, export crashes) recurs
across versions with different wording, so cross-document grouping has something
meaningful to discover.
"""

from __future__ import annotations

import os

import fitz

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

# Font aliases available in PyMuPDF base-14 set.
F_REG = "helv"
F_BOLD = "hebo"
F_ITAL = "heit"


class Cursor:
    def __init__(self, page, x=72, y=90, page_width=612, page_height=792):
        self.page = page
        self.x = x
        self.y = y
        self.pw = page_width
        self.ph = page_height

    def write(self, text, size=11, font=F_REG, gap_before=0, color=(0, 0, 0), x=None):
        self.y += gap_before
        if self.y > self.ph - 72:
            self.page = self.page.parent.new_page(width=self.pw, height=self.ph)
            self.y = 90
        self.page.insert_text((x if x is not None else self.x, self.y), text,
                              fontsize=size, fontname=font, color=color)
        self.y += size * 1.35
        return self


def _new_doc():
    return fitz.open()


def _feature(c, title, desc, meta, title_size=13, gap=18):
    c.write(title, size=title_size, font=F_BOLD, gap_before=gap)
    c.write(desc, size=10.5, font=F_REG, gap_before=6)
    if meta:
        c.write(meta, size=8.5, font=F_ITAL, gap_before=4)


def build_26_0(path):
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    c = Cursor(page)
    c.write("Release Notes 26.0", size=20, font=F_BOLD)
    c.write("FEATURES", size=16, font=F_BOLD, gap_before=24)
    _feature(c, "New authentication capability",
             "Users can now configure single sign-on for their organisation.",
             "Available in version 26.0.")
    _feature(c, "Dashboard customisation",
             "Widgets on the home dashboard can be rearranged and resized.",
             "Applies to all plans.")
    c.write("FIXES", size=16, font=F_BOLD, gap_before=28)
    _feature(c, "Resolved an issue with billing calculation",
             "The monthly total was incorrect when proration applied mid-cycle.",
             "Reported in 25.9.")
    c.write("BUGS", size=16, font=F_BOLD, gap_before=28)
    _feature(c, "Application crashes during export",
             "Large datasets cause the export process to run out of memory.",
             "Workaround: export in smaller batches.")
    doc.save(path)
    doc.close()


def build_26_1(path):
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    c = Cursor(page)
    c.write("Release Notes 26.1", size=20, font=F_BOLD)
    c.write("Features", size=16, font=F_BOLD, gap_before=24)
    _feature(c, "Multi-factor authentication support",
             "Accounts can enable MFA using an authenticator app or SMS code.",
             "Rolled out gradually in 26.1.")
    _feature(c, "Export to spreadsheet",
             "Reports can be exported directly to a spreadsheet format.",
             "Beta feature.")
    _feature(c, "Faster search indexing",
             "Search results now update within seconds of a change.",
             None)
    c.write("Fixes", size=16, font=F_BOLD, gap_before=28)
    _feature(c, "Fixed billing calculation error",
             "Corrected rounding when discounts and taxes were combined.",
             "Follow-up to 26.0.")
    _feature(c, "Login redirect loop resolved",
             "Some users were stuck in a redirect loop after signing in.",
             None)
    c.write("Bugs", size=16, font=F_BOLD, gap_before=28)
    _feature(c, "Export crash with large datasets",
             "Exporting very large reports can still terminate unexpectedly.",
             "Under investigation.")
    doc.save(path)
    doc.close()


def build_26_2(path):
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    c = Cursor(page)
    c.write("Product Update 26.2", size=20, font=F_BOLD)
    c.write("New Capabilities", size=15, font=F_BOLD, gap_before=24)
    _feature(c, "Enhanced login security",
             "Passwordless sign-in with hardware security keys is now supported.",
             "Enterprise plan.", title_size=12)
    _feature(c, "Custom report scheduling",
             "Users can schedule reports to be generated and emailed automatically.",
             "All plans.", title_size=12)
    c.write("Resolved Issues", size=15, font=F_BOLD, gap_before=28)
    _feature(c, "Billing totals now accurate for annual plans",
             "An error affecting annual subscription invoices has been corrected.",
             "Impacted 26.1 users.", title_size=12)
    _feature(c, "Export no longer fails on large files",
             "The export pipeline was rewritten to stream data and avoid crashes.",
             "Resolves long-standing report.", title_size=12)
    doc.save(path)
    doc.close()


def build_26_3(path):
    """No headings at all -- units separated by horizontal rules + whitespace."""
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    c = Cursor(page)
    c.write("Changelog 26.3", size=18, font=F_BOLD)

    def rule():
        c.y += 12
        c.page.draw_line((72, c.y), (540, c.y), color=(0.6, 0.6, 0.6))
        c.y += 12

    c.write("New authentication options", size=12, font=F_BOLD, gap_before=26)
    c.write("Users can now sign in with biometric methods on supported devices.",
            size=10.5, gap_before=6)
    rule()
    c.write("Resolved a billing miscalculation", size=12, font=F_BOLD, gap_before=6)
    c.write("Invoices generated during plan upgrades were occasionally wrong.",
            size=10.5, gap_before=6)
    rule()
    c.write("Crash when exporting large reports", size=12, font=F_BOLD, gap_before=6)
    c.write("Exporting datasets above a certain size could cause a hard crash.",
            size=10.5, gap_before=6)
    doc.save(path)
    doc.close()


def build_26_4(path):
    """Compact stacked title/description/metadata with no section headings."""
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    c = Cursor(page)
    c.write("Notes 26.4", size=18, font=F_BOLD)
    items = [
        ("Single sign-on improvements",
         "Authentication now supports additional identity providers.",
         "Category: capability"),
        ("Billing engine correction",
         "Fixed an inaccurate calculation affecting prorated charges.",
         "Category: correction"),
        ("Export stability",
         "Addressed crashes triggered by exporting large data sets.",
         "Category: correction"),
        ("Dark mode",
         "A dark colour theme is available in user preferences.",
         "Category: capability"),
    ]
    for title, desc, meta in items:
        c.write(title, size=12.5, font=F_BOLD, gap_before=20)
        c.write(desc, size=10.5, font=F_REG, gap_before=5)
        c.write(meta, size=8.5, font=F_ITAL, gap_before=3)
    doc.save(path)
    doc.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    builders = {
        "release_26_0.pdf": build_26_0,
        "release_26_1.pdf": build_26_1,
        "release_26_2.pdf": build_26_2,
        "release_26_3.pdf": build_26_3,
        "release_26_4.pdf": build_26_4,
    }
    for name, fn in builders.items():
        path = os.path.abspath(os.path.join(OUT_DIR, name))
        fn(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
