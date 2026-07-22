import pickle
import numpy as np
from PIL import Image
from config import MODEL_PATH

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((64, 64)).convert("L")
    return np.array(img).flatten().reshape(1, -1) / 255.0


def predict(image: Image.Image) -> dict:
    model = load_model()
    features = preprocess(image)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    label = "smiling" if prediction == 1 else "not_smiling"
    return {"class": label, "confidence": float(max(proba))}