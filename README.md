# SpendAI Matching Service

Quick start

Prerequisites: Docker & docker-compose OR a Python 3.11 environment.

Run with Docker Compose (recommended):

1. Build and start:

   docker-compose up --build

2. The API will be available locally on http://localhost:8000
3. The API is deployed through AWS and is publicly accessible on http://3.144.97.206:8000/

Endpoints:

- POST /historical/load — multipart file upload (CSV) to load historical rows and build TF-IDF index
- POST /match — JSON payload of a single incoming AP line, returns {incoming_ap_line_id, historical_ap_line_id|null, score}
- GET /health — basic health check

Batch prediction (local):

From the project root run:

python scripts/batch_predict.py C:\path\to\historical_ap_data.csv C:\path\to\incoming_ap_data.jsonl --out match_map.csv

This will produce `match_map.csv` with columns: incoming_ap_line_id, historical_ap_line_id, score.

Notes & next steps:

- Normalization rules are basic; consider expanding vendor and description cleaning, currency conversions, and improved candidate pruning.
- Thresholding for "no match" is currently 0.6 (mean of combined scores); tune with labeled data.
# How to run SpendAI (Docker Compose)

This document describes how to start the SpendAI matching service using Docker Compose and how to perform the common operational steps (load historical data, run batch predictions, inspect the API).

## Prerequisites
- Docker & Docker Compose installed
- Project root contains `docker-compose.yml` (this repository)

## Start the system
From the project root (where `docker-compose.yml` is):

```bash
# Build images and start services (API + PostgreSQL)
docker compose up --build
```

- The API service is exposed on http://localhost:8000 by default.
- The `db` service runs PostgreSQL; data is persisted to a named volume `spendai_pgdata`.

## Verify the service
- Health: GET http://localhost:8000/health should return `{ "status": "ok" }`.
- API docs (FastAPI): http://localhost:8000/docs

## Loading historical data (first run)
- Use the provided endpoint to upload a CSV file of historical AP lines:

```bash
curl -X POST "http://localhost:8000/historical/load" -F "file=@/path/to/historical_ap_data.csv"
```

The endpoint will ingest rows into the DB and build TF‑IDF artifacts under `app/data/` (vectorizer + matrix). Once the upload completes you can call `/match` or `/match/debug`.

## Important notes
- If you change the historical CSV or remove the TF‑IDF artifacts, you must re-run the historical load step to rebuild the index.
- If you run locally (not in Docker), you may use SQLite (default) or point `DATABASE_URL` to Postgres.

## Batch prediction (inside container or locally)
- The repository contains `scripts/batch_predict.py` to compute `match_map.csv`:

```bash
python scripts/batch_predict.py /path/to/historical_ap_data.csv /path/to/incoming_ap_data.jsonl --out match_map.csv
```

This will ensure the index exists, run matching for each incoming record, and write the CSV.

---
For more detailed developer instructions (running locally, debugging, and API endpoints) see `README.md`.
