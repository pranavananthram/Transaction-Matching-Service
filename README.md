# Transaction Matching Service

A service that matches incoming Accounts Payable (AP) line items to historical AP lines using fuzzy and vector techniques.

## Overview
- Language: Python 3.11, FastAPI
- DB: SQLite by default (or PostgreSQL if `DATABASE_URL` provided)
- Matching: TF‑IDF on `vendor_name + item_description` for candidate retrieval, combined scoring (text cosine + vendor fuzzy + description fuzzy + amount similarity) for ranking and thresholding.

## Quick start (Docker Compose)
1. From repository root:

```bash
docker compose up --build
```

2. API available: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Basic health check: GET http://localhost:8000/health

3. API hosted through AWS: http://3.144.97.206:8000/

## Run locally (without Docker)
Create and activate a Python 3.11 environment (conda or venv), then:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Loading historical data
Load a historical CSV using the API endpoint:

```bash
curl -X POST "http://localhost:8000/historical/load" -F "file=@/path/to/historical_ap_data.csv"
```

What this does:
- Normalizes common field aliases and stores rows in the DB (`spendai.db` by default)
- Builds and saves TF‑IDF vectorizer and matrix to `app/data/` (files: `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`)

Notes:
- If you do not upload data (and the TF‑IDF artifacts don't exist), `POST /match` will fail to load indices; use `/historical/load` first.

## Match one incoming record (real-time)
Endpoint: POST `/match`

Request JSON: any messy incoming record that contains at least `ap_line_id`; common fields used: `vendor_name`, `item_description`, `line_total_amount`. The API accepts a wide range of aliases.

Example curl:
```bash
curl -X 'POST' \
  'http://3.144.97.206:8000/match' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"ap_line_id": "APN-00000002", "purch_order_key": "NPO-2025-000001", "polnno": "20", "bill_identifier": "NINV-2025-000001", "memodetails": "capital spares - silverline REF-31", "supplier_code": "VEN0020", "suppliername": "SilverLine Precision", "supplier_country": "CA", "vendor_type": "Chemical Supplier", "productcatkey": "MRO-77", "material_category_nm": "Mechanical MRO", "itemitem_code": "VAL 12397", "material_description": "ATEX 3in EPDM Stainless Valve, 16 bar", "raw_uom": "BOX", "qty_ordered": "1", "eachrate": "7178.50", "lntotal": "7178.70", "discount-amount": "", "vat_cd": "TX", "vat_total": "344.00", "freight-amount": "", "line_itemsumamt": "7522.82", "raw_currency": "CAD", "purch_order_date": "2025-08-19", "promise_dt": "", "receivedDate": "", "invoice_dt": "", "pay_by_dt": "2025-11-05", "pay_terms": "2%10 NET30", "PAYMENT_STATUS": "PAID", "approval_status": "APPROVED", "plant_field": "PLT-MW2", "cccenter": "QC-310", "gl_ledger_account": "5210 MRO Mechanical", "requestor": "R. Kim", "deliver_todest": "PLT-MW2 Receiving 1", "lot_no_batch": "", "purchase_ordertotalamt": "52725.00", "invoicetotaltotal": "52717.30", "linestatus": "OPEN"}
'
```

Response:
```json
{
  "incoming_ap_line_id": "APN-00000002",
  "historical_ap_line_id": "AP-00007876",
  "score": 0.9965925942436968
}
```
`historical_ap_line_id` may be `null` when the system is not confident.

## Debug / UI
- There is a simple debug UI served at http://localhost:8000/ (paste JSON and click Match) that calls `POST /match/debug` and displays the canonicalized incoming fields and top-k candidates with scores.

## Batch prediction (generate `match_map.csv`)
Use the included script:

```bash
python scripts/batch_predict.py ../historical_ap_data.csv ../incoming_ap_data.jsonl --out match_map.csv
```

Output format (exact schema required):
```csv
incoming_ap_line_id,historical_ap_line_id
APN-00000022,AP-00000001
APN-00008976,AP-00000678
...
```
When no match is predicted, `historical_ap_line_id` is blank.

## How the matcher works
1. **Canonicalization:** incoming and historical rows are normalized (field alias mapping, simple cleaning). Deterministic keys like invoice IDs and PO numbers are intentionally not included in the `search_text` used for TF‑IDF.
2. **TF‑IDF index:** vectorizer built over `vendor_name + item_description` 
3. **Candidate retrieval:** `NearestNeighbors` (cosine) finds the top-K candidates quickly.
4. **Combined scoring:** per-candidate score is the mean of:
   - cosine similarity (1 - distance)
   - vendor fuzzy match (RapidFuzz token_sort_ratio)
   - description fuzzy match
   - amount similarity (robust relative comparison or neutral when missing)
5. **Thresholding:** default threshold is 0.6 for returning a match; otherwise `null`.

This combination uses TF‑IDF for efficient retrieval and multiple complementary signals for robust ranking.

## Important design choices
- **Do not rely on deterministic keys**: invoice numbers, PO numbers, vendor IDs are not used for matching (they may be stored but not used as signals). This is intentional given the problem spec.
- **Extensible scoring**: weights and scoring function can be tuned if labeled data is available.

## Troubleshooting
- If `/match` returns errors, verify that TF‑IDF artifacts exist: `app/data/tfidf_vectorizer.joblib` and `app/data/tfidf_matrix.joblib`.
- To rebuild the index, re-run `POST /historical/load` with the historical CSV.
- If you prefer to run locally without Postgres, set `DATABASE_URL` to a SQLite URL (default is `sqlite:///./spendai.db`).

## File layout (key files)
- `app/` — FastAPI app and core logic
  - `app/loader.py` — canonicalization and index building
  - `app/matcher.py` — TF‑IDF loading, NN retrieval, scoring
  - `app/api.py` — endpoints (`/match`, `/match/debug`, `/historical/load`)
  - `app/static/index.html` — simple debug UI
- `scripts/batch_predict.py` — produce `match_map.csv` in batch
- `historical_ap_data.csv`, `incoming_ap_data.jsonl` — sample inputs

## Next improvements (ideas)
- Currency conversion, date proximity, quantity/unit normalization, and learned ranking (if labeled matches exist).

---

# How to run (Docker Compose)

This section describes how to start the SpendAI matching service using Docker Compose and how to perform the common operational steps (load historical data, run batch predictions, inspect the API).

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
