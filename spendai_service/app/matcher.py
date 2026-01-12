from sklearn.neighbors import NearestNeighbors
import joblib
from rapidfuzz import fuzz
from .loader import MATRIX_PATH, TFIDF_PATH, canonicalize_incoming
import numpy as np
from .db import SessionLocal
from .models import HistoricalAPLine
import threading

VEC = None
IDS = None
X = None
NN = None
HISTORICAL_CACHE = {}
_lock = threading.Lock()


def load_index():
    global VEC, IDS, X, NN, HISTORICAL_CACHE
    with _lock:
        if VEC is not None:  # Already loaded
            return
        VEC = joblib.load(TFIDF_PATH)
        IDS, X = joblib.load(MATRIX_PATH)
        NN = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
        NN.fit(X)
        # Preload historical data into cache for fast lookup
        session = SessionLocal()
        HISTORICAL_CACHE.clear()
        for row in session.query(HistoricalAPLine).all():
            HISTORICAL_CACHE[row.ap_line_id] = row
        session.close()





def match_single(incoming: dict, top_k: int = 5):
    if VEC is None:
        load_index()
    # Canonicalize incoming messy payloads
    c_in = canonicalize_incoming(incoming)
    text = c_in.get("search_text") or " "
    q = VEC.transform([text])
    dist, idx = NN.kneighbors(q, n_neighbors=top_k, return_distance=True)
    candidates = []
    for d, i in zip(dist[0], idx[0]):
        cand_id = IDS[i]
        # Fast lookup from in-memory cache instead of database query
        row = HISTORICAL_CACHE.get(cand_id)
        score = compute_score(c_in, row, d)
        candidates.append({"ap_line_id": cand_id, "score": score})
    # pick best score > threshold
    best = max(candidates, key=lambda x: x["score"])
    if best["score"] < 0.6:
        return None
    return best


def match_with_candidates(incoming: dict, top_k: int = 5):
    """Return ordered list of top-k candidate dicts including ap_line_id, score, vendor_name, item_description, line_total_amount.

    Candidates are returned sorted by the combined `score` (descending) and assigned a `rank` reflecting that order.
    """
    if VEC is None:
        load_index()
    c_in = canonicalize_incoming(incoming)
    text = c_in.get("search_text") or " "
    q = VEC.transform([text])
    dist, idx = NN.kneighbors(q, n_neighbors=top_k, return_distance=True)
    candidates = []
    for d, i in zip(dist[0], idx[0]):
        cand_id = IDS[i]
        row = HISTORICAL_CACHE.get(cand_id)
        score = compute_score(c_in, row, d)
        candidates.append({
            "ap_line_id": cand_id,
            "score": score,
            "vendor_name": row.vendor_name if row else None,
            "item_description": row.item_description if row else None,
            "line_total_amount": row.line_total_amount if row else None,
        })

    # Sort by combined score (descending) and reassign ranks
    candidates.sort(key=lambda c: c["score"], reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c["rank"] = rank

    return candidates


def compute_score(incoming, historical, cosine_dist):
    # combine cosine (1-dist), rapidfuzz on vendor and desc, and amount similarity
    scores = []
    scores.append(1 - cosine_dist)
    if historical is None:
        return 0.0
    v_score = fuzz.token_sort_ratio(str(incoming.get("vendor_name") or ""), str(historical.vendor_name or "")) / 100.0
    d_score = fuzz.token_sort_ratio(str(incoming.get("item_description") or ""), str(historical.item_description or "")) / 100.0
    scores.extend([v_score, d_score])
    # amount similarity - incoming should be canonicalized already
    a_in = incoming.get("line_total_amount")
    a_hist = historical.line_total_amount
    if a_in is not None and a_hist is not None:
        # robust relative similarity
        try:
            amt_score = 1 - abs(a_in - a_hist) / max(abs(a_in), abs(a_hist))
            amt_score = max(0.0, amt_score)
        except Exception:
            amt_score = 0.0
    else:
        amt_score = 0.5  # neutral when missing
    scores.append(amt_score)

    return float(np.mean(scores))


def try_float(x):
    try:
        return float(x)
    except Exception:
        return None
