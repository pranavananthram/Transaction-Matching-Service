import json
import csv
import os
from app.loader import load_historical_csv, TFIDF_PATH, MATRIX_PATH, DATA_DIR
from app.matcher import match_single, load_index


def ensure_index(historical_csv_path):
    # if index files not present, load the historical CSV to build them
    if not os.path.exists(TFIDF_PATH) or not os.path.exists(MATRIX_PATH):
        print("TF-IDF index not found, building from CSV...")
        load_historical_csv(historical_csv_path)
    load_index()


def batch_predict(historical_csv_path, incoming_jsonl_path, out_csv_path="match_map.csv"):
    ensure_index(historical_csv_path)

    with open(incoming_jsonl_path, "r", encoding="utf-8") as inf, open(out_csv_path, "w", newline='', encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=["incoming_ap_line_id", "historical_ap_line_id"])
        writer.writeheader()
        for line in inf:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            incoming_id = payload.get("ap_line_id") or payload.get("id")
            res = match_single(payload)
            if res is None:
                # write blank historical id when no match
                writer.writerow({"incoming_ap_line_id": incoming_id, "historical_ap_line_id": ""})
            else:
                writer.writerow({"incoming_ap_line_id": incoming_id, "historical_ap_line_id": res["ap_line_id"]})
    print(f"Wrote predictions to {out_csv_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("historical_csv", help="Path to historical_ap_data.csv")
    p.add_argument("incoming_jsonl", help="Path to incoming_ap_data.jsonl")
    p.add_argument("--out", default="match_map.csv")
    args = p.parse_args()

    batch_predict(args.historical_csv, args.incoming_jsonl, args.out)
