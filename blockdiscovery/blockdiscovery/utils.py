"""Small shared helpers used by native and structured pipelines."""

from __future__ import annotations

import os
import re
from typing import List, Optional

from .models import TextBlock


def clip01(x: float) -> float:
    """Clamp a score into [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def document_slug(path: str) -> str:
    """Stable filesystem-safe id derived from a PDF basename."""
    base = os.path.splitext(os.path.basename(path))[0]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    return slug or "document"


def assign_roles(blocks: List[TextBlock], head_id: Optional[str] = None) -> List[str]:
    """Assign relative PROMINENT / BODY / META roles from local prominence."""
    if not blocks:
        return []
    if head_id is None:
        head_id = max(blocks, key=lambda b: b.features.get("prominence", 0.0)).id
    proms = [b.features.get("prominence", 0.0) for b in blocks]
    max_prom = max(proms) if proms else 0.0
    roles: List[str] = []
    for b in blocks:
        prom = b.features.get("prominence", 0.0)
        is_short = b.features.get("is_short", 0.0) >= 1.0
        if b.id == head_id:
            roles.append("PROMINENT")
        elif is_short and prom <= 0.25 * (max_prom + 1e-6) and b.char_count <= 40:
            roles.append("META")
        else:
            roles.append("BODY")
    return roles


def role_signature(roles: List[str]) -> str:
    """Compact role-count signature, e.g. P1B3M0."""
    return (
        f"P{roles.count('PROMINENT')}"
        f"B{roles.count('BODY')}"
        f"M{roles.count('META')}"
    )
