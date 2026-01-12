from sqlalchemy import Column, String, Integer, Float, Date, Text
from .db import Base

class HistoricalAPLine(Base):
    __tablename__ = "historical_ap_lines"
    ap_line_id = Column(String, primary_key=True, index=True)
    vendor_name = Column(String, index=True)
    vendor_id = Column(String, index=True)
    item_description = Column(Text)
    item_sku = Column(String)
    quantity = Column(Float)
    unit_price = Column(Float)
    line_total_amount = Column(Float)
    currency = Column(String)
    invoice_date = Column(String)
    search_text = Column(Text)
