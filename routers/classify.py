import uuid
import os
from datetime import datetime
from fastapi import APIRouter, Request, UploadFile, File, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from PIL import Image
import io
from database import get_db
from models import ClassificationResult
from inference import predict
from config import UPLOAD_DIR

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def convert_to_jpg(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@router.get("/classify")
async def classify_page(request: Request):
    return templates.TemplateResponse(request, "classify.html")


@router.post("/classify")
async def classify_image(
    request: Request,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    results = []

    for file in files:
        contents = await file.read()
        jpg_bytes = convert_to_jpg(contents)
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(jpg_bytes)

        image = Image.open(io.BytesIO(jpg_bytes))
        result = predict(image)

        db_result = ClassificationResult(
            image_path=filepath,
            predicted_class=result["class"],
            created_at=datetime.utcnow()
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)

        results.append({
            "id": db_result.id,
            "image_path": filepath,
            "predicted_class": result["class"],
            "confidence": result["confidence"],
            "created_at": db_result.created_at
        })

    if len(results) == 1:
        return templates.TemplateResponse(request, "result.html", {
            "result": results[0]
        })

    return templates.TemplateResponse(request, "results.html", {
        "results": results
    })


@router.get("/classify/result/{result_id}")
async def get_result(request: Request, result_id: int, db: Session = Depends(get_db)):
    result = db.query(ClassificationResult).filter(ClassificationResult.id == result_id).first()
    if not result:
        return templates.TemplateResponse(request, "error.html", {
            "message": "Result not found"
        })

    return templates.TemplateResponse(request, "result.html", {
        "result": {
            "id": result.id,
            "image_path": result.image_path,
            "predicted_class": result.predicted_class,
            "confidence": 0.0,
            "created_at": result.created_at
        }
    })
