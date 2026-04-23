import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from inference import predictor


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RECYCLING_INFO_PATH = PROJECT_ROOT / "model" / "recycling_info.json"


def load_recycling_info():
    if not RECYCLING_INFO_PATH.exists():
        raise FileNotFoundError(f"recycling_info.json nicht gefunden: {RECYCLING_INFO_PATH}")

    with open(RECYCLING_INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("recycling_info.json ist ungültig")

    return data


app = FastAPI(title="Recycling API")


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "labels": predictor.labels,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(BytesIO(contents)).convert("RGB")

        result = predictor.predict(image)
        label = result["label"]

        recycling_info = load_recycling_info()
        info = recycling_info.get(label, {})

        return {
            "filename": file.filename,
            "message": "Image received successfully",
            "label": label,
            "material": info.get("material"),
            "confidence": result["confidence"],
            "more_info_url": info.get("more_info_url"),
            "recycle_url": info.get("recycle_url"),
        }

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")