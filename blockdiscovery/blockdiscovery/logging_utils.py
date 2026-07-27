"""Dual logging: human-readable narrative + machine-readable structured events.

The engine emits *both* streams from a single call site so they never drift:

* Readable log  -> indented, sectioned narrative that explains the discovery
                   process to a developer (stdout and/or a ``.log`` file).
* Structured log -> one JSON object per line (JSONL) suitable for debugging,
                    testing, metrics, auditing, model evaluation and
                    reproducibility.

Every important algorithmic decision answers four questions:
    What happened?  Why did it happen?  What evidence supported it?
    What confidence was assigned?
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, Optional, TextIO


# Canonical list of required logging events (see spec section 13).
REQUIRED_EVENTS = [
    "document_started",
    "document_extraction_completed",
    "raw_extraction_completed",
    "page_processed",
    "raw_blocks_created",
    "block_normalization_completed",
    "candidate_unit_created",
    "possible_pattern_consolidation",
    "candidate_relationship_evaluated",
    "boundary_evaluated",
    "content_unit_decision",
    "content_unit_created",
    "content_unit_rejected",
    "content_unit_split",
    "content_unit_refined",
    "content_unit_merged",
    "table_structure_discovered",
    "record_boundary_evaluated",
    "record_boundary_detected",
    "structured_record_created",
    "heading_context_detected",
    "section_context_discovered",
    "semantic_representation_created",
    "structural_fingerprint_created",
    "pattern_discovered",
    "logical_block_created",
    "logical_block_merged",
    "logical_block_split",
    "cross_document_similarity_calculated",
    "logical_group_created",
    "logical_group_updated",
    "low_confidence_decision",
    "over_grouping_warning",
    "processing_completed",
]


class DiscoveryLogger:
    """Coordinates the readable and structured log streams."""

    def __init__(
        self,
        structured_path: Optional[str] = None,
        readable_path: Optional[str] = None,
        readable_stream: Optional[TextIO] = sys.stdout,
        readable_enabled: bool = True,
        low_confidence_threshold: float = 0.5,
    ) -> None:
        self._structured_file: Optional[TextIO] = (
            open(structured_path, "w", encoding="utf-8") if structured_path else None
        )
        self._readable_file: Optional[TextIO] = (
            open(readable_path, "w", encoding="utf-8") if readable_path else None
        )
        self._readable_stream = readable_stream
        self._readable_enabled = readable_enabled
        self._low_conf = low_confidence_threshold
        self._indent = 0
        self._counts: Dict[str, int] = {}
        self._start = time.time()

    # ------------------------------------------------------------------ #
    # Readable stream helpers
    # ------------------------------------------------------------------ #
    def _emit_readable(self, text: str) -> None:
        if not self._readable_enabled:
            return
        line = ("  " * self._indent) + text
        if self._readable_stream is not None:
            print(line, file=self._readable_stream)
        if self._readable_file is not None:
            self._readable_file.write(line + "\n")

    def section(self, tag: str, message: str = "") -> None:
        """Emit a top-level bracketed section header, e.g. ``[EXTRACTION]``."""
        self._indent = 0
        header = f"[{tag}]"
        if message:
            header += f" {message}"
        self._emit_readable("")
        self._emit_readable(header)
        self._indent = 1

    def line(self, text: str = "") -> None:
        self._emit_readable(text)

    def push(self) -> None:
        self._indent += 1

    def pop(self) -> None:
        self._indent = max(0, self._indent - 1)

    def kv(self, key: str, value: Any) -> None:
        self._emit_readable(f"{key}: {value}")

    def evidence_block(self, evidence: Dict[str, float], title: str = "Evidence:") -> None:
        self._emit_readable(title)
        self.push()
        for k, v in evidence.items():
            label = k.replace("_", " ").capitalize()
            self._emit_readable(f"{label}: {v:.2f}" if isinstance(v, (int, float)) else f"{label}: {v}")
        self.pop()

    # ------------------------------------------------------------------ #
    # Structured stream
    # ------------------------------------------------------------------ #
    def event(self, name: str, readable: Optional[str] = None, **fields: Any) -> None:
        """Emit a structured event (and optionally a one-line readable note).

        Automatically emits a companion ``low_confidence_decision`` event when a
        ``confidence`` field is present and falls below the configured threshold.
        """
        self._counts[name] = self._counts.get(name, 0) + 1
        record: Dict[str, Any] = {
            "ts": round(time.time(), 6),
            "elapsed_ms": round((time.time() - self._start) * 1000, 2),
            "event": name,
        }
        record.update(fields)
        if self._structured_file is not None:
            self._structured_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if readable:
            self._emit_readable(readable)

        conf = fields.get("confidence")
        if (
            conf is not None
            and name not in ("low_confidence_decision",)
            and isinstance(conf, (int, float))
            and conf < self._low_conf
        ):
            self.low_confidence(name, conf, fields)

    def low_confidence(self, source_event: str, confidence: float, context: Dict[str, Any]) -> None:
        self._counts["low_confidence_decision"] = self._counts.get("low_confidence_decision", 0) + 1
        record = {
            "ts": round(time.time(), 6),
            "elapsed_ms": round((time.time() - self._start) * 1000, 2),
            "event": "low_confidence_decision",
            "source_event": source_event,
            "confidence": round(float(confidence), 4),
            "context": {k: context.get(k) for k in ("document_id", "logical_block_id", "content_unit_id", "pattern_id", "group_id") if k in context},
        }
        if self._structured_file is not None:
            self._structured_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, int]:
        return dict(self._counts)

    def close(self) -> None:
        if self._structured_file is not None:
            self._structured_file.flush()
            self._structured_file.close()
        if self._readable_file is not None:
            self._readable_file.flush()
            self._readable_file.close()

    def __enter__(self) -> "DiscoveryLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
