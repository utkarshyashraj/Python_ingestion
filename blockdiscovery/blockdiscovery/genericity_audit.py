"""Genericity audit.

Scans the engine's own source for constructs that would tie discovery to one
specific document: category vocabulary, section names, regex-driven structure
decisions, fixed coordinates, fixed page numbers and absolute font sizes.

Findings are classified as ``VALID_GENERIC`` (technical parsing that never
inspects document vocabulary) or ``DOCUMENT_SPECIFIC`` (must not exist).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Words that would betray knowledge of one document family. Stored here only so
# the auditor can prove they are absent from decision logic.
_CATEGORY_VOCABULARY = (
    "feature",
    "fix",
    "bug",
    "enhancement",
    "issue",
    "product name",
    "reference",
    "category",
    "description",
    "configuration instructions",
    "new features",
    "back to top",
    "release note",
)

# Modules whose regex use is pure machine-format or non-semantic tokenization.
_REGEX_ALLOWLIST: Dict[str, str] = {
    "utils.py": "filesystem slug generation (non-semantic)",
    "semantics.py": "non-semantic token splitting for embeddings",
    "normalization.py": "whitespace normalization",
    "cross_document.py": "parsing the engine's own role-signature serialization",
    "genericity_audit.py": "the auditor itself performs no discovery",
}

_STRUCTURE_MODULES = (
    "generic_discovery.py",
    "extraction_units.py",
    "boundaries.py",
    "relationships.py",
    "content_units.py",
    "logical_blocks.py",
    "patterns.py",
    "section_groups.py",
    "features.py",
    "fingerprints.py",
)


@dataclass
class Finding:
    file: str
    line: int
    kind: str
    classification: str
    snippet: str
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "kind": self.kind,
            "classification": self.classification,
            "snippet": self.snippet[:160],
            "note": self.note,
        }


@dataclass
class AuditReport:
    findings: List[Finding] = field(default_factory=list)
    files_scanned: int = 0

    def by_kind(self, kind: str, classification: Optional[str] = None) -> List[Finding]:
        return [
            f
            for f in self.findings
            if f.kind == kind and (classification is None or f.classification == classification)
        ]

    def counts(self) -> Dict[str, int]:
        kinds = (
            "hardcoded_semantic_category",
            "document_specific_section_rule",
            "regex_structure_discovery",
            "fixed_coordinate_rule",
            "fixed_page_rule",
            "absolute_font_size_rule",
        )
        return {k: len(self.by_kind(k, "DOCUMENT_SPECIFIC")) for k in kinds}

    def to_dict(self) -> Dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "violation_counts": self.counts(),
            "valid_generic": [f.to_dict() for f in self.findings if f.classification == "VALID_GENERIC"],
            "document_specific": [
                f.to_dict() for f in self.findings if f.classification == "DOCUMENT_SPECIFIC"
            ],
        }


def _string_constants(tree: ast.AST) -> List[ast.Constant]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _is_comparison_context(tree: ast.AST, node: ast.Constant) -> bool:
    """True when a string constant participates in a branch decision."""
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.Compare, ast.Set, ast.Dict)):
            for child in ast.walk(parent):
                if child is node:
                    return True
    return False


def _scan_regex(path: str, name: str, tree: ast.AST, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    allow_note = _REGEX_ALLOWLIST.get(name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        owner = func.value
        if not (isinstance(owner, ast.Name) and owner.id == "re"):
            continue
        if func.attr not in {
            "search",
            "match",
            "fullmatch",
            "findall",
            "finditer",
            "compile",
            "sub",
            "split",
        }:
            continue
        snippet = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
        if allow_note:
            findings.append(
                Finding(name, node.lineno, "regex_structure_discovery", "VALID_GENERIC", snippet, allow_note)
            )
        elif name in _STRUCTURE_MODULES:
            findings.append(
                Finding(
                    name,
                    node.lineno,
                    "regex_structure_discovery",
                    "DOCUMENT_SPECIFIC",
                    snippet,
                    "regex used inside a structure-discovery module",
                )
            )
        else:
            findings.append(
                Finding(name, node.lineno, "regex_structure_discovery", "VALID_GENERIC", snippet, "non-discovery module")
            )
    return findings


def _scan_vocabulary(name: str, tree: ast.AST, lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    if name == "genericity_audit.py":
        return findings
    for node in _string_constants(tree):
        low = node.value.strip().casefold()
        if not low or len(low) > 60:
            continue
        if low not in _CATEGORY_VOCABULARY:
            continue
        if not _is_comparison_context(tree, node):
            continue
        snippet = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
        kind = (
            "document_specific_section_rule"
            if name in ("section_groups.py", "generic_discovery.py")
            else "hardcoded_semantic_category"
        )
        findings.append(
            Finding(name, node.lineno, kind, "DOCUMENT_SPECIFIC", snippet, f"vocabulary literal {low!r} in a decision")
        )
    return findings


def _scan_numeric_rules(name: str, tree: ast.AST, lines: List[str]) -> List[Finding]:
    """Flag comparisons against absolute page coordinates, pages or font sizes."""
    findings: List[Finding] = []
    if name not in _STRUCTURE_MODULES:
        return findings
    coordinate_names = {"x0", "x1", "y0", "y1", "left", "top", "bottom", "right", "bbox"}
    page_names = {"page_number", "page_no", "page_index", "source_page"}
    size_names = {"font_size", "dominant_size", "size", "max_size", "min_size"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        target = None
        if isinstance(left, ast.Name):
            target = left.id
        elif isinstance(left, ast.Attribute):
            target = left.attr
        if target is None:
            continue
        literals = [
            c.value
            for c in node.comparators
            if isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
        ]
        if not literals:
            continue
        snippet = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
        if target in coordinate_names and any(abs(v) > 3 for v in literals):
            findings.append(
                Finding(name, node.lineno, "fixed_coordinate_rule", "DOCUMENT_SPECIFIC", snippet,
                        "absolute page coordinate in a decision")
            )
        elif target in page_names and any(v > 1 for v in literals):
            findings.append(
                Finding(name, node.lineno, "fixed_page_rule", "DOCUMENT_SPECIFIC", snippet,
                        "absolute page index in a decision")
            )
        elif target in size_names and any(v > 3 for v in literals):
            findings.append(
                Finding(name, node.lineno, "absolute_font_size_rule", "DOCUMENT_SPECIFIC", snippet,
                        "absolute font size in a decision")
            )
    return findings


def audit_package(package_dir: Optional[str] = None) -> AuditReport:
    """Audit every module of the engine and classify each finding."""
    base = package_dir or os.path.dirname(os.path.abspath(__file__))
    report = AuditReport()
    for entry in sorted(os.listdir(base)):
        if not entry.endswith(".py"):
            continue
        path = os.path.join(base, entry)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        lines = source.splitlines()
        report.files_scanned += 1
        report.findings.extend(_scan_regex(path, entry, tree, lines))
        report.findings.extend(_scan_vocabulary(entry, tree, lines))
        report.findings.extend(_scan_numeric_rules(entry, tree, lines))
    return report
