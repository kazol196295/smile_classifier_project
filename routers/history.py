from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import get_db
from models import ClassificationResult

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/history")
async def history(request: Request, db: Session = Depends(get_db)):
    results = db.query(ClassificationResult).order_by(
        ClassificationResult.created_at.desc()
    ).all()

    return templates.TemplateResponse(request, "history.html", {
        "results": results
    })