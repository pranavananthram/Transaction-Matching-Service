from fastapi import APIRouter
from .matcher import match_single, match_with_candidates
from .loader import canonicalize_incoming

router = APIRouter()

@router.post("/match")
async def match_endpoint(payload: dict):
    incoming_id = payload.get("ap_line_id")
    res = match_single(payload)
    if res is None:
        return {"incoming_ap_line_id": incoming_id, "historical_ap_line_id": None}
    return {"incoming_ap_line_id": incoming_id, "historical_ap_line_id": res["ap_line_id"], "score": res["score"]}

@router.post("/match/debug")
async def match_debug_endpoint(payload: dict):
    """Return top-k candidate list for UI/debugging.
    Response: {incoming_ap_line_id, incoming: {canonical fields}, candidates: [{rank, ap_line_id, score, vendor_name, item_description, line_total_amount}, ...]}
    """
    incoming_id = payload.get("ap_line_id")
    top_k = int(payload.get("top_k", 5))
    incoming_canonical = canonicalize_incoming(payload)
    candidates = match_with_candidates(payload, top_k=top_k)
    # Determine the predicted best match using the same scoring logic
    best = match_single(payload)
    if best is None:
        predicted_best = None
    else:
        predicted_best = {"ap_line_id": best["ap_line_id"], "score": best["score"]}

    return {"incoming_ap_line_id": incoming_id, "incoming": incoming_canonical, "predicted_best": predicted_best, "candidates": candidates}
