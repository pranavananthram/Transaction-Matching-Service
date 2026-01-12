from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from .loader import load_historical_csv
from .api import router as api_router
import os

app = FastAPI()
app.include_router(api_router, prefix="")

# Serve simple debug UI
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

@app.post("/historical/load")
async def load_historical(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = os.path.join(".", "tmp_historical.csv")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    res = load_historical_csv(temp_path)
    os.remove(temp_path)
    return res

@app.get("/health")
async def health():
    return {"status": "ok"}
