# Smile Classifier - Implementation Plan

## Project Structure

```
smile_classifier_project/
├── kaggle/
│   ├── train.ipynb            # Kaggle notebook: training pipeline
│   └── inference.ipynb        # Kaggle notebook: inference demo
├── model/
│   └── smile_classifier.pkl   # Downloaded from Kaggle after training
├── main.py                    # FastAPI app entrypoint
├── config.py                  # DB URL, paths, constants
├── database.py                # SQLAlchemy engine, session, Base
├── models.py                  # DB model: ClassificationResult
├── schemas.py                 # Pydantic models for API responses
├── inference.py               # Local inference: load .pkl, predict
├── routers/
│   ├── home.py                # GET / → Home page
│   ├── train.py               # POST /train → upload images (no local training)
│   ├── classify.py            # POST /classify → upload + predict
│   └── history.py             # GET /history → classification history
├── templates/
│   ├── base.html              # Layout with nav: Home | Train | Classify | History
│   ├── home.html              # Explanation page
│   ├── train.html             # Multi-file upload form
│   ├── classify.html          # Single-file upload form
│   └── history.html           # Table: Image, Class, DateTime
├── static/
│   └── uploaded_images/       # Saved/uploaded images
├── docker-compose.yml         # PostgreSQL + app on YSDTP_B5_AI_YourID network
├── Dockerfile                 # App container
├── requirements.txt
└── .env                       # DB credentials (not committed)
```

---

## Phase 1: Kaggle Notebooks

### 1.1 `kaggle/train.ipynb`
- Add dataset: https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data
- Preprocessing: resize to 64×64, grayscale, flatten, normalize to [0,1]
- Train scikit-learn model (Logistic Regression or Random Forest)
- Evaluate: accuracy, classification report, confusion matrix
- Save model as `.pkl` → download to local `model/smile_classifier.pkl`

### 1.2 `kaggle/inference.ipynb`
- Load the `.pkl` model
- Test on sample images from the dataset
- Show predictions (class + confidence)
- Serves as documentation of how inference works

---

## Phase 2: Local Inference Module

### 2.1 `inference.py`
- Load `.pkl` model from `model/smile_classifier.pkl`
- Accept a PIL Image, preprocess same as Kaggle notebook (resize, grayscale, flatten, normalize)
- Return prediction: class (`smiling`/`not_smiling`) + confidence score
- Reusable by the Classify router

```python
import pickle
import numpy as np
from PIL import Image

MODEL_PATH = "model/smile_classifier.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

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
```

---

## Phase 3: Database Setup

### 3.1 SQLAlchemy config (`database.py`)
- Engine from env var: `DATABASE_URL` (default: `postgresql://user:pass@localhost:5432/smile_db`)
- `SessionLocal` factory for dependency injection
- `Base = declarative_base()`

### 3.2 Model (`models.py`)
```python
class ClassificationResult(Base):
    __tablename__ = "classification_results"
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)  # "smiling" or "not_smiling"
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 3.3 Create tables
- On app startup, call `Base.metadata.create_all(engine)` to auto-create tables

---

## Phase 4: FastAPI Application

### 4.1 App setup (`main.py`)
- Create `FastAPI(title="Smile Classifier - YourID")`
- Mount static files: `app.mount("/static", StaticFiles(...))`
- Include routers from `routers/`
- On startup: create tables, ensure upload directory exists

### 4.2 Home page (`routers/home.py` + `templates/home.html`)
- GET `/`
- Static content explaining:
  - Training done in Kaggle using `train.ipynb`
  - Framework: scikit-learn
  - Model: Logistic Regression / Random Forest
  - Why: simple, effective for image classification, fast training
  - Preprocessing: resize 64×64, grayscale, flatten, normalize

### 4.3 Train page (`routers/train.py` + `templates/train.html`)
- GET `/train` → render upload form (multi-file)
- POST `/train` → accept multiple image files:
  1. Save each to `static/uploaded_images/`
  2. Convert each to `.jpg` if not already (use Pillow)
  3. Return success message: "Images saved. Please retrain in Kaggle using `train.ipynb`."
  4. **No local training** — training happens in Kaggle

### 4.4 Classify page (`routers/classify.py` + `templates/classify.html`)
- GET `/classify` → render upload form (single file)
- POST `/classify` → accept single image:
  1. Save to `static/uploaded_images/`
  2. Convert to `.jpg` if needed
  3. Load `.pkl` model via `inference.py`, run prediction
  4. Save result to DB: image_path, predicted_class, datetime
  5. Render result page showing the image and class
- GET `/classify/result/{id}` → show classification result

### 4.5 History page (`routers/history.py` + `templates/history.html`)
- GET `/history`
- Query all `ClassificationResult` rows, ordered by `created_at DESC`
- Render table with columns: Image (thumbnail/path), Class, DateTime

---

## Phase 5: Docker Deployment

### 5.1 Dockerfile
- Base: `python:3.12-slim`
- Install dependencies from `requirements.txt`
- Copy app code + `model/` directory (with pre-trained `.pkl`)
- Expose port 8000
- CMD: `uvicorn main:app --host 0.0.0.0 --port 8000`

### 5.2 docker-compose.yml
- Service 1: `db` (postgres:15)
  - Environment: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
  - Volume: `pgdata:/var/lib/postgresql/data`
  - Network: `YSDTP_B5_AI_YourID`
- Service 2: `app`
  - Build from Dockerfile
  - Depends on: db
  - Environment: DATABASE_URL pointing to db service
  - Ports: `8000:8000`
  - Network: `YSDTP_B5_AI_YourID`
- Volumes: `pgdata`

### 5.3 Network
- Create external network: `docker network create YSDTP_B5_AI_YourID`
- Both services join this network

---

## Phase 6: Image Handling

### 6.1 Conversion to JPG
```python
from PIL import Image
import io

def convert_to_jpg(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
```

### 6.2 File naming
- Use `{uuid}.jpg` to avoid collisions
- Store path as `static/uploaded_images/{uuid}.jpg`

---

## Dependencies (`requirements.txt`)

```
fastapi
uvicorn[standard]
python-multipart
jinja2
sqlalchemy
psycopg2-binary
pillow
numpy
python-dotenv
```

Note: `scikit-learn` NOT needed in FastAPI app — inference uses pre-trained `.pkl` only.

---

## Execution Order

| Step | Action | Output |
|------|--------|--------|
| 1 | Create `kaggle/train.ipynb` | Train model in Kaggle, download `.pkl` |
| 2 | Create `kaggle/inference.ipynb` | Test inference in Kaggle |
| 3 | Place `.pkl` in `model/` | Ready for local app |
| 4 | Create project scaffold + `requirements.txt` | Structure |
| 5 | Implement `inference.py` | Local predict module |
| 6 | Implement `database.py` + `models.py` + `config.py` | DB layer |
| 7 | Implement `schemas.py` | Pydantic models |
| 8 | Implement `routers/` (home, train, classify, history) | API routes |
| 9 | Create `templates/` (base, home, train, classify, history) | HTML pages |
| 10 | Implement `main.py` | Wired app |
| 11 | Test locally (`uvicorn main:app --reload`) | Verify all pages |
| 12 | Create `Dockerfile` + `docker-compose.yml` | Containerized |
| 13 | Test with Docker (`docker-compose up --build`) | Full stack |

---

## Gotchas & Notes

- The dataset URL in the spec has a typo (`kagpgle.com` → `kaggle.com`)
- Uploaded images must be converted to JPG **before** saving to filesystem
- Train page: images are saved, user retrains in Kaggle (no local training)
- Classification results must be persisted to DB **with** the image path
- The `.pkl` model must be placed in `model/` before running the app
- Use `python-multipart` for file uploads in FastAPI
- PostgreSQL preferred but SQLite works for local dev (change `DATABASE_URL`)
- The `YourID` in project name and Docker network should be replaced with actual student ID
- `scikit-learn` is NOT a dependency of the FastAPI app — only needed in Kaggle