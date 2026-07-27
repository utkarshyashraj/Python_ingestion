# Adaptive PDF Logical Block Discovery Engine

Discovers **logical content blocks** in PDFs and groups related content — without
hardcoded section names, fixed coordinates, regex categories, or Feature/Fix/Bug
rules.

> Structure is discovered from evidence, not assumed.

For Siebel-style release notes (`data/samples/26.*_UPDATE.pdf`), use the
**structured** backend (`pymupdf4llm`): table rows become content units directly.

---

## How the code works

```text
PDF
  │
  ├─ --backend structured (recommended for release notes)
  │     pymupdf4llm → markdown (headings + tables)
  │     → each table row / heading / paragraph = ContentUnit
  │     → section groups from headings
  │
  └─ --backend native (optional / research)
        PyMuPDF raw blocks → features → relationship + boundary scores
        → ContentUnits by clustering consecutive blocks
        │
        ▼
  Pattern discovery (recurring shapes → pattern_001, …)
        ▼
  Logical blocks (explainable units of meaning)
        ▼
  Cross-document groups + searchable KnowledgeBase
```

| Layer | Module | Role |
|---|---|---|
| Structured ingest | `structured_ingest.py` | pymupdf4llm markdown → units + sections |
| Native extract | `extraction.py` | PyMuPDF raw blocks, fonts, bboxes |
| Features | `features.py` | Document-relative geometry / typography |
| Semantics | `semantics.py` | Embeddings (hashing default) |
| Relationships / boundaries | `relationships.py`, `boundaries.py` | Native: belong-together vs split-here |
| Patterns | `patterns.py` | Unsupervised structural/semantic clusters |
| Logical blocks | `logical_blocks.py` | Primary output |
| Sections | `section_groups.py` | Within-doc heading → items |
| Cross-document | `cross_document.py` | Topic groups across PDFs |
| Knowledge | `knowledge.py` | Search + provenance |
| Orchestration | `pipeline.py`, `cli.py`, `config.py`, `logging_utils.py` | Run, configure, log |

**PyMuPDF vs pymupdf4llm:** they are alternate ingest paths, not a sequence.
`--backend structured` uses pymupdf4llm (built on PyMuPDF). `--backend native`
uses PyMuPDF block clustering only. For 26.1–26.6 update guides, use **structured only**.

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
| `human_readable_<doc_id>.log` | Sections → items (easiest to read) |
| `events.jsonl` | Machine-readable decision events |
| `logical_blocks.json` | Discovered logical blocks |
| `section_groups.json` | Within-document sections |
| `patterns.json` | Recurring patterns |
| `logical_groups.json` | Cross-document (or within-doc) groups |
| `collection_tree.json` | Tree view of sections + groups |
| `documents.json` | Document metadata |

### 26.3 Update — first 15 pages (human-readable logs)

```bash
python -m blockdiscovery.cli ingest data/samples/26.3_UPDATE.pdf \
  --backend structured --max-pages 15 --out output_26_3_structured_p1_15
```

Then open:

- `output_26_3_structured_p1_15/human_readable_26_3_update.log` — section/item view  
- `output_26_3_structured_p1_15/discovery.log` — full discovery narrative  

### Other useful ingest examples

```bash
# All six Siebel update PDFs (full docs — slow)
python -m blockdiscovery.cli ingest "data/samples/*.pdf" \
  --backend structured --out output_all_updates

# Native path (research / non-table PDFs)
python -m blockdiscovery.cli ingest data/samples/26.3_UPDATE.pdf \
  --backend native --max-pages 15 --out output_26_3_native_p1_15

# Synthetic demo PDFs
python scripts/generate_sample_pdfs.py
python -m blockdiscovery.cli ingest "data/synthetic/*.pdf" --backend native --out output
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
