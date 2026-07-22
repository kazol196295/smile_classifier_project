from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from database import Base


class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)