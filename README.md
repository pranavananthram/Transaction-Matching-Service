# SpendAI Matching Service

Quick start

Prerequisites: Docker & docker-compose OR a Python 3.11 environment.

Run with Docker Compose (recommended):

1. Build and start:

   docker-compose up --build

2. The API will be available locally on http://localhost:8000
3. The API is deployed through AWS and is publically accessible on http://3.144.97.206:8000/

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
