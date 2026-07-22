import uuid
import os
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
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


@router.get("/train")
async def train_page(request: Request):
    return templates.TemplateResponse(request, "train.html")


@router.post("/train")
async def upload_for_training(request: Request, files: list[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_count = 0

    for file in files:
        contents = await file.read()
        jpg_bytes = convert_to_jpg(contents)
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(jpg_bytes)
        saved_count += 1

    return templates.TemplateResponse(request, "train.html", {
        "success": True,
        "count": saved_count
    })