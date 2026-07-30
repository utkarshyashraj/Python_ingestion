"""FastAPI UI for PDF ingest and human-readable discovery log viewing."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from blockdiscovery.config import EngineConfig
from blockdiscovery.logging_utils import DiscoveryLogger
from blockdiscovery.pipeline import DiscoveryEngine

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "blockdiscovery_web_jobs"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Block Discovery",
    description="Upload a PDF, run generic discovery, view the human-readable log.",
    version="0.1.0",
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="UI static files missing.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _find_human_log(out_dir: Path) -> Optional[Path]:
    logs = sorted(out_dir.glob("human_readable_*.log"))
    return logs[0] if logs else None


def _section_summary(section_groups_path: Path) -> List[Dict[str, Any]]:
    if not section_groups_path.is_file():
        return []
    import json

    sections = json.loads(section_groups_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for s in sections:
        out.append(
            {
                "id": s.get("id"),
                "heading": s.get("heading_text"),
                "depth": s.get("depth", 0),
                "items": s.get("item_count", 0),
                "pages": f"{s.get('page_start')}-{s.get('page_end')}",
                "parent_section_id": s.get("parent_section_id"),
                "child_count": len(s.get("child_section_ids") or []),
            }
        )
    return out


@app.post("/api/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    backend: str = Form("structured"),
    max_pages: Optional[int] = Form(None),
) -> Dict[str, Any]:
    """Accept a PDF upload, run discovery, return the human-readable log."""
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if backend not in {"structured", "native", "docling"}:
        raise HTTPException(status_code=400, detail="Invalid backend.")

    pages: Optional[int] = None
    if max_pages is not None:
        try:
            pages = int(max_pages)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="max_pages must be an integer.") from exc
        if pages <= 0:
            pages = None

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = job_dir / Path(filename).name
    out_dir = job_dir / "output"

    try:
        with pdf_path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()

    if pdf_path.stat().st_size <= 0:
        raise HTTPException(status_code=400, detail="Empty PDF upload.")

    events_path = out_dir / "events.jsonl"
    discovery_log = out_dir / "discovery.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = EngineConfig(
        ingestion_backend=backend,
        max_pages=pages,
        readable_log=False,
    )
    logger = DiscoveryLogger(
        structured_path=str(events_path),
        readable_path=str(discovery_log),
        readable_enabled=False,
        low_confidence_threshold=config.thresholds.low_confidence_flag,
    )

    try:
        engine = DiscoveryEngine(config=config, logger=logger)
        kb = engine.run([str(pdf_path)])
        paths = engine.export(kb, str(out_dir))
    except Exception as exc:  # noqa: BLE001 — surface ingest errors to UI
        logger.close()
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    finally:
        logger.close()

    human_path = _find_human_log(out_dir)
    if human_path is None:
        raise HTTPException(status_code=500, detail="Ingest succeeded but no human-readable log was written.")

    human_log = human_path.read_text(encoding="utf-8", errors="replace")
    docs = list(kb.documents.values())
    doc = docs[0] if docs else None

    return {
        "job_id": job_id,
        "filename": filename,
        "document_id": doc.id if doc else None,
        "page_count": doc.page_count if doc else None,
        "backend": backend,
        "max_pages": pages,
        "section_count": len(kb.section_groups),
        "logical_block_count": len(kb.logical_blocks),
        "sections": _section_summary(out_dir / "section_groups.json"),
        "human_readable_log": human_log,
        "artefact_names": sorted(paths.keys()),
    }


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str) -> Dict[str, Any]:
    job_dir = UPLOAD_ROOT / job_id / "output"
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")
    human_path = _find_human_log(job_dir)
    if human_path is None:
        raise HTTPException(status_code=404, detail="Human-readable log not found for job.")
    return {
        "job_id": job_id,
        "human_readable_log": human_path.read_text(encoding="utf-8", errors="replace"),
    }


def main() -> None:
    import uvicorn

    host = os.environ.get("BLOCKDISCOVERY_HOST", "127.0.0.1")
    port = int(os.environ.get("BLOCKDISCOVERY_PORT", "8000"))
    uvicorn.run("blockdiscovery.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
