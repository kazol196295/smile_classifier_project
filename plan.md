# Smile Classifier - Implementation Plan

## Project Structure

```
smile_classifier_project/
├── model/
│   ├── smile_classifier.pkl   # Trained model
│   └── create_placeholder.py  # Creates dummy model
├── main.py                    # FastAPI app entrypoint
├── config.py                  # DB URL, paths, constants
├── database.py                # SQLAlchemy engine, session, Base
├── models.py                  # DB model: ClassificationResult
├── schemas.py                 # Pydantic models for API responses
├── inference.py               # Local inference: load .pkl, predict
├── train_local.py             # Local training: SVM model from uploaded images
├── routers/
│   ├── home.py                # GET / → Home page
│   ├── train.py               # POST /train → upload images
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

## Phase 1: Local Training Module

### 1.1 `train_local.py`
- Load images from two folders: smile and non-smile
- Preprocessing: resize to 64×64, grayscale, flatten, normalize to [0,1]
- Train SVM model with RBF kernel
- Evaluate: accuracy on test split (80/20)
- Save model as `.pkl` to `model/smile_classifier.pkl`

```python
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import MODEL_PATH

def load_images_from_folder(folder_path: str, label: int) -> tuple[list[np.ndarray], list[int]]:
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath)
                img = img.resize((64, 64)).convert("L")
                img_array = np.array(img).flatten() / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    return images, labels

def train_model(smile_folder: str, non_smile_folder: str) -> dict:
    smile_images, smile_labels = load_images_from_folder(smile_folder, label=1)
    non_smile_images, non_smile_labels = load_images_from_folder(non_smile_folder, label=0)
    X = np.array(smile_images + non_smile_images)
    y = np.array(smile_labels + non_smile_labels)
    if len(X) == 0:
        return {"success": False, "message": "No images found"}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else '.', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return {"success": True, "accuracy": round(accuracy * 100, 2), "smile_count": len(smile_images), "non_smile_count": len(non_smile_images)}
```

### 1.2 `model/create_placeholder.py`
- Creates a dummy SVM model for testing
- Useful when no training data is available

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
  - Training done locally using `train_local.py`
  - Framework: scikit-learn
  - Model: SVM (Support Vector Machine)
  - Why: effective for image classification, works well with small datasets
  - Preprocessing: resize 64×64, grayscale, flatten, normalize

### 4.3 Train page (`routers/train.py` + `templates/train.html`)
- GET `/train` → render upload form (multi-file)
- POST `/train` → accept multiple image files:
  1. Save each to `static/uploaded_images/`
  2. Convert each to `.jpg` if not already (use Pillow)
  3. Return success message: "Images saved. Run `train_local.py` to train the model."
  4. Local training available via `train_local.py`

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
scikit-learn
```

---

## Execution Order

| Step | Action | Output |
|------|--------|--------|
| 1 | Create project scaffold + `requirements.txt` | Structure |
| 2 | Implement `train_local.py` | Local training module |
| 3 | Implement `inference.py` | Local predict module |
| 4 | Implement `database.py` + `models.py` + `config.py` | DB layer |
| 5 | Implement `schemas.py` | Pydantic models |
| 6 | Implement `routers/` (home, train, classify, history) | API routes |
| 7 | Create `templates/` (base, home, train, classify, history) | HTML pages |
| 8 | Implement `main.py` | Wired app |
| 9 | Train model locally (`python train_local.py`) | Trained `.pkl` |
| 10 | Test locally (`uvicorn main:app --reload`) | Verify all pages |
| 11 | Create `Dockerfile` + `docker-compose.yml` | Containerized |
| 12 | Test with Docker (`docker-compose up --build`) | Full stack |

---

## Gotchas & Notes

- The dataset URL in the spec has a typo (`kagpgle.com` → `kaggle.com`)
- Uploaded images must be converted to JPG **before** saving to filesystem
- Train page: images are saved, training done locally via `train_local.py`
- Classification results must be persisted to DB **with** the image path
- The `.pkl` model must be placed in `model/` before running the app
- Use `python-multipart` for file uploads in FastAPI
- PostgreSQL preferred but SQLite works for local dev (change `DATABASE_URL`)
- The `YourID` in project name and Docker network should be replaced with actual student ID
- `scikit-learn` is needed for both training (`train_local.py`) and inference