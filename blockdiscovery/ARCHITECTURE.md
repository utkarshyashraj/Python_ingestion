# PDF Block Discovery — Architecture Guide (Team Briefing)

This document explains how the Adaptive PDF Logical Block Discovery Engine works end-to-end: code flow, why it is **document-agnostic** (Siebel, Salesforce, or any similarly structured PDF), what “no hardcoding / no regex” means in practice, which tools we use, and how to run ingestion.

> **Core principle:** Structure is *discovered from evidence*, not assumed from product vocabulary or fixed templates.

---

## 1. What the system does

Given one or more PDFs, the engine:

1. Extracts **physical evidence** from pages (text, geometry, layout roles, tables).
2. Discovers **logical blocks** (paragraphs, table rows, list items, titles).
3. Merges split continuations when evidence supports wrap/continuation.
4. Builds **nested section groups** (heading → children → items) from relative layout hierarchy.
5. Optionally finds **cross-document patterns** across multiple PDFs.
6. Writes human-readable logs + JSON artefacts for review and downstream use.

It does **not** require knowing whether the PDF is “Siebel release notes” or “Salesforce documentation.” Those names never enter discovery decisions.

---

## 2. High-level architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                              │
│   ingest / search  →  EngineConfig  →  DiscoveryEngine          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DiscoveryEngine (pipeline.py)                  │
│                                                                  │
│  ┌──────────────────┐    ┌────────────────────────────────────┐ │
│  │ Extraction        │    │ Generic discovery + grouping        │ │
│  │ (evidence only)   │───▶│ (structure decisions)               │ │
│  │ extraction_units  │    │ generic_discovery                   │ │
│  │ / extraction /    │    │ logical_block_consolidator          │ │
│  │   docling         │    │ section_groups                      │ │
│  └──────────────────┘    │ patterns · cross_document           │ │
│                           │ knowledge · export                  │ │
│                           └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Outputs: human_readable_*.log · section_groups.json ·          │
│  logical_blocks.json · discovery.log · genericity_audit.json …  │
└─────────────────────────────────────────────────────────────────┘
```

### Separation of concerns

| Layer | Responsibility | Must not do |
|-------|----------------|-------------|
| **Extraction backend** | Answer: *what is physically on the page?* | Decide logical sections / “this is a Feature” |
| **Generic discovery** | Group units into blocks from geometry, typography, grids, semantics | Use product vocabulary or section name lists |
| **Section hierarchy** | Nest headings by relative score (depth stack) | Hardcode product section titles |
| **Consolidator** | Merge true wrap/continuation rows | Merge unrelated titles |
| **Audit** | Prove source stays generic | — |

---

## 3. Code flow (structured backend — recommended)

When you run with `--backend structured` (recommended for table-heavy manuals):

```text
PDF file(s)
    │
    ▼
[1] extract_raw_units()                     extraction_units.py
    • pymupdf4llm / PyMuPDF layout boxes
    • Emits RawExtractionUnit:
        text, bbox, page, layout_class, grid_id / cells
    • Layout classes are extractor labels (section-header, page-header,
      table, list-item, text) — evidence, not final truth
    │
    ▼
[2] Fit embedding backend on corpus         semantics.py
    • Default: dependency-free hashing embeddings
    • Optional: sentence-transformers
    │
    ▼
[3] GenericDiscoveryEngine.run()            generic_discovery.py
    Evidence → Features → Relationships → Boundaries
      → Structure → Logical Blocks
    • Adaptive boundary cut from *this document’s* score distribution
    • Form similarity discounted when a template already predicts it
      (stops table rows from fusing into one blob)
    │
    ▼
[4] LogicalBlockConsolidator                logical_block_consolidator.py
    • Merge split table-cell wraps / anaphoric continuations
    • Keep adjacent title-like units separate (for nested headings)
    │
    ▼
[5] SectionGroupDiscovery                   section_groups.py
    • Find heading-like blocks (relative prominence / title shape)
    • Assign hierarchy scores from layout + adjacency (not keywords)
    • Stack-based nesting: parent_section_id / child_section_ids / depth
    • Example shape (any product):
          Umbrella heading
            └─ Release / chapter band
                 ├─ Leaf section (e.g. feature table)
                 └─ Leaf section (e.g. fix table)
    │
    ▼
[6] PatternDiscovery + CrossDocumentGrouping
    patterns.py · cross_document.py
    │
    ▼
[7] KnowledgeBase + export                  knowledge.py · pipeline.export()
    • JSON artefacts + human-readable logs
    • genericity_audit.json (source scan for document-specific logic)
```

### Alternate backends

| `--backend` | Extractor | When to use |
|-------------|-----------|-------------|
| `structured` | **pymupdf4llm** (on PyMuPDF) | Table-heavy docs, release notes — **recommended** |
| `native` | PyMuPDF block clustering | Lighter / comparison path |
| `docling` | IBM Docling (optional) | If installed; falls back to structured if missing |

Extraction backends are **swappable evidence sources**. Grouping logic stays the same.

---

## 4. Nested section model

`SectionGroup` carries hierarchy fields:

- `depth` — 0 = top-level peer section
- `parent_section_id` / `child_section_ids` — tree edges
- `member_logical_block_ids` — **direct** items under that heading

Levels are inferred from **relative evidence**, for example:

- Strong layout section-header + not opening onto a grid → peer (depth 0)
- Title immediately followed by another heading → container band
- Compact title above a multi-column table → leaf

No list of product section names is consulted.

Human logs indent the summary as:

```text
  • DiscoveredSection_010   depth=0  | <umbrella heading from PDF>
    ◦ DiscoveredSection_011 depth=1  | <band / chapter title>
      ▪ DiscoveredSection_012 depth=2 | <leaf section>
      ▪ DiscoveredSection_013 depth=2 | <leaf section>
```

(Headings shown are PDF text, not engine categories.)

---

## 5. Why Salesforce (or any PDF) works the same way

### Document vocabulary is not an input

Discovery never branches on strings like Siebel / Salesforce / Oracle, or Feature / Fix / Bug.

So a Salesforce admin guide or release PDF is ingested with the **same pipeline**. You only change the file path.

### What transfers across products

| Signal | Product-independent? |
|--------|----------------------|
| Bounding boxes, page order | Yes |
| Relative font size / bold / prominence | Yes |
| Table grids & column width continuity | Yes |
| Heading vs body shape (short, non-sentence) | Yes |
| Nested title before another title / table | Yes |
| Embeddings for semantic continuity | Yes (language-sensitive, not product-specific) |

### What to expect on a new family (e.g. Salesforce)

| Outcome | Meaning |
|---------|---------|
| Ingest completes + logs/JSON written | Always (given a readable PDF) |
| Useful blocks & section groups | Typical for manuals with headings/tables |
| Nesting as clean as current release-note samples | Depends on **layout similarity**, not product name |
| Flatter tree | Normal when hierarchy confidence is lower — safer than wrong deep nesting |

**Bottom line for the team:** supporting Salesforce is not a separate codebase path. Validate quality on Salesforce samples; if hierarchy needs improvement, retune **generic** relative heuristics — never add product-name branches.

---

## 6. “No hardcoding” and “no regex” — precise meaning

### No hardcoding (for structure discovery)

Means: **no document- or product-specific category rules** deciding groups.

Confirmed by `genericity_audit.py` (written to `genericity_audit.json` on each run):

- `hardcoded_semantic_category: 0`
- `document_specific_section_rule: 0`
- `document_specific: []`

Mentions of Feature / Fix / product names in comments or log banners are **explanatory only**.

### No regex for structure discovery

Means: **no regular expression decides sections, tables, or hierarchy.**

Regex *does* exist in a few **non-structure** helpers (allowed by the audit):

| File | Purpose |
|------|---------|
| `normalization.py` | Whitespace cleanup |
| `semantics.py` | Tokenize text for embeddings |
| `utils.py` | Filesystem ID slug |
| `cross_document.py` | Parse the engine’s own signature strings (`P1B2M3`) |

Those never classify content as Feature/Fix or invent section titles.

### Honest caveats (still generic, not “assumption-free”)

- **English wrap words** (`and`, `this`, …) aid continuation detection — language structure, not product taxonomy.
- **Numeric thresholds** (word-length caps, hierarchy score tiers) are layout heuristics for structured manuals; they are not product names.
- **Extractor layout labels** (`section-header`, `page-header`) are used as weak evidence; different extractors may label differently.

---

## 7. Tools & libraries

| Tool | Role |
|------|------|
| **Python 3.9+** | Runtime |
| **PyMuPDF (`fitz`)** | PDF parsing / native extraction |
| **pymupdf4llm** | Layout-aware extraction (tables, headers) for `--backend structured` |
| **NumPy** | Feature / score maths |
| **Optional: Docling** | AI layout/table backend |
| **Optional: sentence-transformers** | Higher-quality embeddings |
| **Optional: pytest** | Unit tests |

Install:

```bash
cd /path/to/pdf-block-discovery
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core from `requirements.txt`: `PyMuPDF`, `numpy`, `pymupdf4llm`.

---

## 8. Commands to run ingestion

Always set `PYTHONPATH` to the repo root (or install the package editable).

### Single PDF (full document)

```bash
cd /path/to/pdf-block-discovery
source .venv/bin/activate
export PYTHONPATH=.

python -m blockdiscovery.cli ingest data/samples/26.3_UPDATE.pdf \
  --backend structured \
  --out output_26_3_structured_full
```

### First N pages (faster iteration)

```bash
python -m blockdiscovery.cli ingest data/samples/26.3_UPDATE.pdf \
  --backend structured \
  --max-pages 15 \
  --out output_26_3_structured_p1_15
```

### Any PDF (e.g. Salesforce)

```bash
python -m blockdiscovery.cli ingest /path/to/salesforce_doc.pdf \
  --backend structured \
  --out output_salesforce
```

### Multiple PDFs / glob

```bash
python -m blockdiscovery.cli ingest "data/samples/*.pdf" \
  --backend structured \
  --out output_all_samples
```

### Semantic search after ingest

```bash
python -m blockdiscovery.cli search "authentication" \
  data/samples/26.3_UPDATE.pdf \
  --backend structured \
  --out output_search_demo
```

### CLI options

| Option | Meaning |
|--------|---------|
| `--out DIR` | Output directory |
| `--backend structured \| native \| docling` | Extraction backend |
| `--max-pages N` | Only first N pages |
| `--verbose` | Extra relationship logging |
| `--quiet` | Less stdout noise (files still written) |

---

## 9. Output artefacts (what to show the team)

Under `--out`:

| Artefact | Who reads it | Purpose |
|----------|--------------|---------|
| **`human_readable_<doc>.log`** | Everyone | Nested sections → items → full text (best demo) |
| `section_groups.json` | Engineers / integration | Machine-readable hierarchy (`depth`, parent/children) |
| `logical_blocks.json` | Engineers | Atomic discovered blocks + fields |
| `collection_tree.json` | Engineers | Tree summary |
| `generic_discovery_<doc>.log` | Engineers | Detailed boundary / evidence narrative |
| `block_consolidation_<doc>.log` | Engineers | Merge / keep-separate decisions |
| `discovery.log` / `events.jsonl` | Engineers | Pipeline narrative + machine events |
| `genericity_audit.json` | Compliance / architecture | Proves zero document-specific violations |
| `documents.json`, `patterns.json`, `logical_groups.json` | Downstream | Metadata, patterns, cross-doc groups |

**Team review tip:** start with `human_readable_*.log` → `[DISCOVERED SECTION SUMMARY]`, then drill into a section’s `[ITEM …]` list.

---

## 10. Key source modules (map for onboarding)

| Module | Role |
|--------|------|
| `cli.py` | CLI entry (`ingest`, `search`) |
| `pipeline.py` | Orchestrates extract → discover → consolidate → sections → export |
| `extraction_units.py` | Structured/docling raw units (evidence only) |
| `extraction.py` | Native PyMuPDF path |
| `generic_discovery.py` | Evidence-first block discovery |
| `logical_block_consolidator.py` | Wrap / title-separation consolidator |
| `section_groups.py` | Nested section groups + human section log |
| `patterns.py` / `fingerprints.py` | Recurring structure patterns |
| `cross_document.py` | Cross-PDF grouping |
| `semantics.py` | Embeddings backend |
| `genericity_audit.py` | Source scan for non-generic constructs |
| `knowledge.py` | Searchable knowledge base |
| `config.py` | Thresholds / backend config (no product lexicon by default) |
| `models.py` | `LogicalBlock`, `SectionGroup`, evidence models |

---

## 11. Genericity validation (optional)

```bash
python tools/make_synthetic_pdfs.py data/synthetic
python tools/validate_genericity.py
```

Synthetic PDFs use invented labels so the engine cannot “cheat” with real product vocab. The audit also scans the package for document-specific logic.

---

## 12. Talking points for stakeholders

1. **Evidence-first:** extractors supply geometry/layout; discovery decides structure.
2. **Product-agnostic:** Siebel and Salesforce use the same code path.
3. **No category hardcoding / no structure regex:** enforced and audited.
4. **Nested groups:** umbrella → band → leaf from relative hierarchy scores.
5. **Quality ≈ layout fit:** unusual layouts may produce flatter trees; fix generically.
6. **Demo artefact:** `human_readable_*.log` under the output folder.

---

## 13. Related docs

- `README.md` — install, quick commands, design notes
- `requirements.txt` — dependencies
- `genericity_audit.json` (per run) — live proof of generic constraints

## 14. Web UI (branch `feature/ingest-ui`)

A FastAPI UI lets you **drag-and-drop / upload** a PDF, run the same discovery engine, and view the **human-readable log** in the browser.

```bash
export PYTHONPATH=.
uvicorn blockdiscovery.web.app:app --host 127.0.0.1 --port 8000
```

See `blockdiscovery/web/README.md` for details.

---

*Last updated to reflect nested section discovery (`depth` / parent / children), structured ingestion via pymupdf4llm, multi-PDF full-document runs, and the ingest web UI.*
