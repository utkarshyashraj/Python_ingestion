"""Generic, adaptive PDF logical block discovery engine.

Discovers meaningful logical content blocks from arbitrary PDFs using
evidence-first, document-relative reasoning -- no hardcoded section names,
coordinates, regex section detection, keyword classification or fixed layouts.
"""

from .config import (
    DEFAULT_CONFIG,
    EngineConfig,
    GroupSimilarityWeights,
    RelationshipWeights,
    Thresholds,
)
from .knowledge import KnowledgeBase
from .logging_utils import DiscoveryLogger, REQUIRED_EVENTS
from .models import (
    BoundingBox,
    ContentUnit,
    DiscoveredPattern,
    Document,
    Evidence,
    LogicalBlock,
    LogicalGroup,
    SectionGroup,
    TextBlock,
)
from .pipeline import DiscoveryEngine

__version__ = "0.1.0"

__all__ = [
    "DiscoveryEngine",
    "EngineConfig",
    "DEFAULT_CONFIG",
    "RelationshipWeights",
    "GroupSimilarityWeights",
    "Thresholds",
    "DiscoveryLogger",
    "REQUIRED_EVENTS",
    "KnowledgeBase",
    "BoundingBox",
    "TextBlock",
    "ContentUnit",
    "DiscoveredPattern",
    "LogicalBlock",
    "LogicalGroup",
    "SectionGroup",
    "Evidence",
    "Document",
]
