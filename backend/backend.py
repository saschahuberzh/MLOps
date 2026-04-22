from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO

app = FastAPI(title="Recycling API")


@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # check if file exists
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # try to open image (validation)
        image = Image.open(BytesIO(contents)).convert("RGB")

        # 👉 NO MODEL YET — just dummy response
        return {
            "filename": file.filename,
            "message": "Image received successfully",
            "label": "unknown",
            "material": "unknown",
            "confidence": 0.0,
            "more_info_url": None,
            "recycle_url": None,
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")