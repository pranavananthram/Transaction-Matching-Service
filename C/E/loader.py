import pandas as pd
from sqlalchemy.exc import IntegrityError
from .db import SessionLocal, engine
from .models import Base, HistoricalAPLine
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

Base.metadata.create_all(bind=engine)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(DATA_DIR, "tfidf_matrix.joblib")

FIELD_MAP = {
    # map common historical column names
    "vendor_name": "vendor_name",
    "vendor_name_clean": "vendor_name",
    "item_description": "item_description",
    "item_desc": "item_description",
    "item_sku": "item_sku",
    "item_part_no": "item_sku",
}


def canonicalize_row(row: dict) -> dict:
    # pick a few canonical fields for storage
    vendor = row.get("vendor_name") or row.get("vendorlabel") or row.get("vendor_nm") or row.get("supplier_label") or row.get("vendorlabel")
    desc = row.get("item_description") or row.get("item_desc") or row.get("material_description") or row.get("product_description") or row.get("materialline_text")
    sku = row.get("item_sku") or row.get("item_sku") or row.get("material_part_no") or row.get("product_sku")
    return {
        "ap_line_id": row.get("ap_line_id"),
        "vendor_name": vendor,
        "vendor_id": row.get("vendor_id") or row.get("vendorid") or row.get("supplierid"),
        "item_description": desc,
        "item_sku": sku,
        "quantity": try_float(row.get("quantity") or row.get("qty") or row.get("units") or row.get("field_quantity")),
        "unit_price": try_float(row.get("unit_price") or row.get("eachprice") or row.get("unitrate") or row.get("each_rate")),
        "line_total_amount": try_float(row.get("line_total_amount") or row.get("line_item_total") or row.get("line_amount") or row.get("lineamt") or row.get("line_itemsumamount")),
        "currency": row.get("currency") or row.get("curr"),
        "invoice_date": row.get("invoice_date") or row.get("inv_dt") or row.get("billdate")
    }


# Utility: normalize text for consistent TF-IDF queries
import re

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def canonicalize_incoming(row: dict) -> dict:
    """Normalize incoming (messy) payloads into canonical fields used by matcher.
    This intentionally excludes deterministic keys such as invoice_id or purchase_order_id from the search text.
    """
    vendor = (
        row.get("vendor_name") or row.get("suppliername") or row.get("supplier_nm") or row.get("supplier_label") or
        row.get("payeename") or row.get("supplier_name") or row.get("supplierlabel") or row.get("payee_name") or row.get("vendor-name")
    )
    desc = (
        row.get("item_description") or row.get("material_description") or row.get("material_details") or row.get("itemdetails") or
        row.get("materialline_text") or row.get("memodetails") or row.get("productline_text") or row.get("product_desc") or row.get("material_description")
    )
    sku = (
        row.get("item_sku") or row.get("item_item_code") or row.get("itempart_no") or row.get("material_part_no") or row.get("productitem_code") or row.get("itemSku")
    )

    qty = try_float(
        row.get("quantity") or row.get("qty") or row.get("units") or row.get("quantity_field") or row.get("qty_ordered") or row.get("raw_quantity") or row.get("units")
    )
    unit_price = try_float(
        row.get("unit_price") or row.get("eachprice") or row.get("eachrate") or row.get("unitrate") or row.get("unit_cost") or row.get("each_unit_price") or row.get("per_cost")
    )
    line_total = try_float(
        row.get("line_total_amount") or row.get("line_itemsumamt") or row.get("lntotal") or row.get("lntotalvalue") or row.get("lineamt") or row.get("line_itemsumamount") or row.get("line_itemsumamt") or row.get("line_total") or row.get("line_item_value") or row.get("line_itemsumamt") or row.get("line_itemsumamt")
    )
    currency = row.get("currency") or row.get("raw_currency") or row.get("ccy") or row.get("curr")
    invoice_date = row.get("invoice_date") or row.get("invoice_dt") or row.get("invoice-date") or row.get("invoice_dt") or row.get("invoice_dt")

    return {
        "ap_line_id": row.get("ap_line_id") or row.get("id"),
        "vendor_name": vendor,
        "item_description": desc,
        "item_sku": sku,
        "quantity": qty,
        "unit_price": unit_price,
        "line_total_amount": line_total,
        "currency": currency,
        "invoice_date": invoice_date,
        "search_text": normalize_text(" ".join([str(vendor or ""), str(desc or "")])),
    }


def try_float(x):
    try:
        return float(x) if x not in [None, ""] else None
    except Exception:
        return None


def load_historical_csv(path: str):
    df = pd.read_csv(path)
    df = df.fillna("")

    session = SessionLocal()
    records = []
    texts = []
    ids = []
    for _, r in df.iterrows():
        row = r.to_dict()
        c = canonicalize_row(row)
        search_text = " ".join([str(c.get("vendor_name") or ""), str(c.get("item_description") or "")])
        obj = HistoricalAPLine(
            ap_line_id=c["ap_line_id"],
            vendor_name=c["vendor_name"],
            vendor_id=c["vendor_id"],
            item_description=c["item_description"],
            item_sku=c["item_sku"],
            quantity=c["quantity"],
            unit_price=c["unit_price"],
            line_total_amount=c["line_total_amount"],
            currency=c["currency"],
            invoice_date=c["invoice_date"],
            search_text=search_text,
        )
        session.merge(obj)
        texts.append(search_text)
        ids.append(c["ap_line_id"])

    session.commit()
    session.close()

    # build TF-IDF with optimized settings
    vec = TfidfVectorizer(
        max_features=3000,           # Reduce sparsity
        ngram_range=(1, 2),
        sublinear_tf=True,           # Dampen frequent terms
        min_df=2,                    # Ignore very rare terms
        max_df=0.8,                  # Ignore very common terms
        lowercase=True,              # Normalize case
        stop_words='english',        # Remove common words
        strip_accents='unicode'      # Normalize accents
    )
    X = vec.fit_transform(texts)
    joblib.dump(vec, TFIDF_PATH)
    joblib.dump((ids, X), MATRIX_PATH)
    return {"imported": len(ids)}
