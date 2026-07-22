import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smile_db.sqlite")
MODEL_PATH = os.getenv("MODEL_PATH", "model/smile_classifier.pkl")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "static/uploaded_images")