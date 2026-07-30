# Web UI — PDF upload & human-readable log

Branch: `feature/ingest-ui`

## Run

From the repository root that contains the `blockdiscovery` package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

uvicorn blockdiscovery.web.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Usage

1. Drag and drop a PDF (or click **Choose PDF**).
2. Optionally set **Max pages** (leave blank for the full document).
3. Click **Run ingest**.
4. Review the nested section summary and the full **human-readable log**.

## API

- `GET /` — UI
- `GET /api/health` — health check
- `POST /api/ingest` — multipart form: `file` (PDF), optional `backend`, `max_pages`
- `GET /api/jobs/{job_id}/log` — re-fetch log for a completed job

The UI calls the same `DiscoveryEngine` used by the CLI (`--backend structured` by default). No product-specific hardcoding.
