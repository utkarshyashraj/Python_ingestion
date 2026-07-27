# Adaptive PDF Logical Block Discovery Engine

Discovers **logical content blocks** in PDFs and groups related content — without
hardcoded section names, fixed coordinates, regex categories, or Feature/Fix/Bug
rules.

> Structure is discovered from evidence, not assumed.

Extraction backends supply evidence only. They never decide what a logical block
is — a table row is a strong *candidate*, and a heading is *evidence*, but both
must earn their status from geometry, typography, repetition and semantics.

---

## How the code works

The `structured` backend runs the generic evidence-first engine:

```text
PDF
  ▼
Extraction backend (evidence only)          extraction_units.py
  pymupdf4llm layout boxes → geometry, layout class, grid cells
  ▼
Raw extraction units → Candidate units      generic_discovery.py
  ▼
Structural feature extraction
  geometry · typography · spacing · alignment · reading order
  grid evidence · container evidence · repetition · semantics
  ▼
Relationship graph (weighted edges)
  ▼
Boundary detection  →  START_NEW_LOGICAL_BLOCK | CONTINUE_LOGICAL_BLOCK
  ▼
Candidate group discovery (component partition at an adaptive cut)
  ▼
Content unit refinement   MERGE · SPLIT · PRESERVE · REJECT
  ▼
Logical blocks → structural fingerprints → pattern discovery
  ▼
Cross-document similarity → KnowledgeBase
```

`--backend native` keeps the older PyMuPDF block-clustering path
(`extraction.py` → `relationships.py` → `boundaries.py` → `content_units.py`).

| Layer | Module | Role |
|---|---|---|
| Extraction (evidence only) | `extraction_units.py` | Layout boxes → raw units with geometry and grid cells |
| Generic discovery | `generic_discovery.py` | Features, relationship graph, boundaries, refinement, blocks, contexts |
| Native extract | `extraction.py` | PyMuPDF raw blocks, fonts, bboxes |
| Features | `features.py` | Document-relative geometry / typography |
| Semantics | `semantics.py` | Embeddings (hashing default) |
| Relationships / boundaries | `relationships.py`, `boundaries.py` | Native path: belong-together vs split-here |
| Patterns | `patterns.py`, `fingerprints.py` | Structure-led clustering of recurring shapes |
| Logical blocks | `logical_blocks.py` | Native-path block builder |
| Sections | `section_groups.py` | Native path: heading → items |
| Cross-document | `cross_document.py` | Recurring structures across PDFs |
| Genericity audit | `genericity_audit.py` | Scans the engine for document-specific logic |
| Reporting | `generic_log.py`, `validation_log.py` | Human-readable discovery narratives |
| Knowledge | `knowledge.py` | Search + provenance |
| Orchestration | `pipeline.py`, `cli.py`, `config.py`, `logging_utils.py` | Run, configure, log |

### Three ideas that keep discovery generic

**Form similarity is only evidence while it is surprising.** Matching geometry
and typography are discounted when a recurring template already predicts them,
or when the two units sit in different containers. This is what stops a table's
visual regularity from fusing its rows into one block, and it leaves semantic
coherence as the deciding signal.

**Signals earn weight from how much they vary here.** A signal that reads the
same on every edge cannot separate anything, so its share of the weight budget
is handed to signals that move. In a uniformly left-aligned document, alignment
automatically stops voting.

**Cut points come from the data.** Where a boundary falls is decided by
maximum-variance separation over the document's own edge-score distribution,
winsorized so a few page breaks cannot drag the threshold.

**PyMuPDF vs pymupdf4llm:** alternate ingest paths, not a sequence.
`--backend structured` uses pymupdf4llm (built on PyMuPDF) purely as an evidence
source. `--backend native` uses PyMuPDF block clustering only.

---

## Install

```bash
cd /path/to/pdf-block-discovery
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. Core: `PyMuPDF`, `numpy`, `pymupdf4llm`.

Optional: `docling` (layout AI), `sentence-transformers` (better embeddings).

---

## Commands

### Ingest PDFs (main command)

```bash
python -m blockdiscovery.cli ingest <pdfs...> [options]
```

| Option | Meaning |
|---|---|
| `--out DIR` | Output folder (default: `output`) |
| `--backend structured` | **Recommended** — tables/headings via pymupdf4llm |
| `--backend native` | Raw PyMuPDF block clustering |
| `--backend docling` | Docling if installed; else falls back to structured |
| `--max-pages N` | Process only the first N pages |
| `--verbose` | Log every candidate relationship (noisy; native) |
| `--quiet` | Suppress readable narrative on stdout (files still written) |

**What gets written under `--out`:**

| File | Purpose |
|---|---|
| `discovery.log` | Human-readable pipeline narrative |
| `validation_discovery_<doc_id>.log` | Validation log (extraction → units → boundaries → LBs → patterns) |
| `human_readable_<doc_id>.log` | Sections → items (easiest section view) |
| `events.jsonl` | Machine-readable decision events |
| `logical_blocks.json` | Discovered logical blocks |
| `section_groups.json` | Within-document sections |
| `patterns.json` | Recurring patterns |
| `logical_groups.json` | Cross-document (or within-doc) groups |
| `collection_tree.json` | Tree view of sections + groups |
| `documents.json` | Document metadata |

### 26.3 Update — first 15 pages (human-readable discovery log)

```bash
rm -rf output_26_3_structured_p1_15

python -m blockdiscovery.cli ingest data/samples/26.3_UPDATE.pdf \
  --backend structured --max-pages 15 --out output_26_3_structured_p1_15
```

Primary human-readable file:

- `output_26_3_structured_p1_15/generic_discovery_26_3_update.log`

It narrates extraction, discovered structures, the relationship graph, every
boundary decision with its evidence, content-unit refinement, logical blocks,
over-grouping analysis, contexts, patterns, cross-document structure and the
genericity audit.

Also written: `events.jsonl` (machine-readable decisions), `genericity_audit.json`,
`pattern_consolidation.json`, `discovery.log` and the JSON artefacts.

Native block-clustering path, for comparison:

```bash
python -m blockdiscovery.cli ingest data/samples/26.3_UPDATE.pdf \
  --backend native --max-pages 15 --out output_26_3_native_p1_15
```

### Genericity validation

Builds PDFs whose every label is invented, then runs the unmodified engine and
asserts structural expectations plus a zero-violation source audit:

```bash
python tools/make_synthetic_pdfs.py data/synthetic
python tools/validate_genericity.py
```

### Other useful ingest examples

```bash
# All six Siebel update PDFs (full docs — slow)
python -m blockdiscovery.cli ingest "data/samples/*.pdf" \
  --backend structured --out output_all_updates

# Unknown-terminology synthetic PDFs through the same code path
python tools/make_synthetic_pdfs.py data/synthetic
python -m blockdiscovery.cli ingest "data/synthetic/*.pdf" \
  --backend structured --out output_synthetic
```

### Search

Ingests the given PDFs, then runs a semantic query:

```bash
python -m blockdiscovery.cli search "authentication" "data/synthetic/*.pdf"
python -m blockdiscovery.cli search "encryption upgrade" data/samples/26.3_UPDATE.pdf --top-k 10
```

### Demo & tests

```bash
python scripts/demo.py              # narrated walkthrough on synthetic PDFs
python tests/test_engine.py         # end-to-end + events (no pytest required)
```

### Library API

```python
from blockdiscovery import DiscoveryEngine, DiscoveryLogger, EngineConfig

config = EngineConfig(ingestion_backend="structured", max_pages=15)
logger = DiscoveryLogger(
    structured_path="output/events.jsonl",
    readable_path="output/discovery.log",
)
engine = DiscoveryEngine(config=config, logger=logger)
kb = engine.run(["data/samples/26.3_UPDATE.pdf"])
engine.export(kb, "output")

kb.search("fixes")
kb.trace("26_3_update_logical_block_001")
```

---

## Logging & explainability

Two streams from one call site:

- **Readable** — `discovery.log` + `human_readable_*.log`
- **Structured** — `events.jsonl` (one JSON object per line)

Key events include: `content_unit_created`, `semantic_representation_created`,
`pattern_discovered`, `logical_block_created`, `logical_group_created`,
`low_confidence_decision`, `processing_completed`.

Confidence is always backed by an evidence bundle (signals + weights), never a
bare score.

---

## Configuration

Weights and thresholds live in `blockdiscovery/config.py`
(`RelationshipWeights`, `GroupSimilarityWeights`, `Thresholds`).

Optional post-hoc labels: `EngineConfig.optional_label_lexicon` (naming only;
discovery never depends on Feature/Fix/Bug words).

For stronger embeddings: install `sentence-transformers` and set
`EngineConfig.embedding_backend = "sentence-transformers"`.

---

## Sample data

| Path | What |
|---|---|
| `data/samples/26.*_UPDATE.pdf` | Real Siebel CRM update guides (~170–190 pages each) |
| `data/synthetic/*.pdf` | Tiny generated layouts for demos/tests |

---

## Scale note

Validated on six real ~170–190-page release-note PDFs. Cross-document similarity
uses vectorised NumPy and sparse neighbour clustering so it scales beyond a
naive O(n²) Python loop.
