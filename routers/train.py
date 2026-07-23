import os
import shutil
import tempfile
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from train_local import train_model

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/train")
async def train_page(request: Request):
    return templates.TemplateResponse(request, "train.html")


@router.post("/train")
async def train_local_model(request: Request):
    form = await request.form(max_files=10000, max_fields=10000)
    smile_folder = form.getlist("smile_folder")
    non_smile_folder = form.getlist("non_smile_folder")

    if not smile_folder or not non_smile_folder:
        return templates.TemplateResponse(request, "train.html", {
            "error": True,
            "message": "Please select both smile and non-smile folders."
        })

    temp_dir = tempfile.mkdtemp()

    try:
        smile_dir = os.path.join(temp_dir, "smile")
        non_smile_dir = os.path.join(temp_dir, "non_smile")
        os.makedirs(smile_dir)
        os.makedirs(non_smile_dir)

        for file in smile_folder:
            if file.filename:
                filepath = os.path.join(smile_dir, os.path.basename(file.filename))
                content = await file.read()
                with open(filepath, "wb") as f:
                    f.write(content)

        for file in non_smile_folder:
            if file.filename:
                filepath = os.path.join(non_smile_dir, os.path.basename(file.filename))
                content = await file.read()
                with open(filepath, "wb") as f:
                    f.write(content)

        result = train_model(smile_dir, non_smile_dir)

        if result["success"]:
            return templates.TemplateResponse(request, "train.html", {
                "success": True,
                "message": result["message"],
                "accuracy": result["accuracy"],
                "smile_count": result["smile_count"],
                "non_smile_count": result["non_smile_count"],
                "total_count": result["total_count"]
            })
        else:
            return templates.TemplateResponse(request, "train.html", {
                "error": True,
                "message": result["message"]
            })

    except Exception as e:
        return templates.TemplateResponse(request, "train.html", {
            "error": True,
            "message": f"Training failed: {str(e)}"
        })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
