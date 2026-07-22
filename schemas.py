from pydantic import BaseModel
from datetime import datetime


class ClassificationResultBase(BaseModel):
    image_path: str
    predicted_class: str


class ClassificationResultCreate(ClassificationResultBase):
    pass


class ClassificationResultResponse(ClassificationResultBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class TrainResponse(BaseModel):
    message: str
    images_saved: int


class ClassifyResponse(BaseModel):
    id: int
    image_path: str
    predicted_class: str
    confidence: float
    created_at: datetime