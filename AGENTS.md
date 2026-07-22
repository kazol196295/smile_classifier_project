# Smile Classifier Project

## Project goal
Build a smile classifier (smiling / not smiling) with a FastAPI web app, training pipeline, inference API, database, and Docker deployment.

## Source of truth
`ai_framework_libraries_deployment_mini_project.md` — the single requirements document. All work must satisfy it.

## Key requirements

- **Dataset**: https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data
- **Training**: scikit-learn preferred, save model as `.pkl`
- **Inference**: load `.pkl` model, classify uploaded image
- **FastAPI project**: named `Smile Classifier - YourID`
- **Pages**: Home, Train, Classify, History
- **Train page**: upload multiple images, train model, save `.pkl`, delete uploaded images
- **Classify page**: upload single image, convert to JPG, classify, save result to DB
- **History page**: table with Image, Class, DateTime
- **Database**: PostgreSQL preferred, SQLAlchemy ORM preferred
- **Docker**: network named `YSDTP_B5_AI_YourID`, compose file for DB on same network

## Project structure

```
smile_classifier_project/
├── kaggle/
│   ├── train.ipynb            # Kaggle notebook: training pipeline
│   └── inference.ipynb        # Kaggle notebook: inference demo
├── model/
│   ├── smile_classifier.pkl   # Pre-trained model (from Kaggle)
│   └── create_placeholder.py  # Creates dummy model for testing
├── main.py                    # FastAPI app entrypoint
├── config.py                  # DB URL, paths, constants
├── database.py                # SQLAlchemy engine, session, Base
├── models.py                  # DB model: ClassificationResult
├── schemas.py                 # Pydantic models for API responses
├── inference.py               # Local inference: load .pkl, predict
├── routers/
│   ├── home.py                # GET / → Home page
│   ├── train.py               # POST /train → upload images
│   ├── classify.py            # POST /classify → upload + predict
│   └── history.py             # GET /history → classification history
├── templates/
│   ├── base.html              # Layout with nav
│   ├── home.html              # Explanation page
│   ├── train.html             # Multi-file upload form
│   ├── classify.html          # Single-file upload form
│   ├── result.html            # Classification result display
│   ├── history.html           # Table: Image, Class, DateTime
│   └── error.html             # Error page
├── static/
│   └── uploaded_images/       # Saved/uploaded images
├── docker-compose.yml         # PostgreSQL + app on YSDTP_B5_AI_YourID network
├── Dockerfile                 # App container
├── requirements.txt
└── plan.md                    # Detailed implementation plan
```

## Key implementation details

- **Training**: Done in Kaggle using `kaggle/train.ipynb`
- **Inference**: Local `inference.py` loads pre-trained `.pkl` model
- **Database**: SQLite for local dev, PostgreSQL for Docker
- **Image handling**: All images converted to JPG before saving
- **Train page**: Saves images only, user retrains in Kaggle (no local training)

## Running locally

```bash
# Install dependencies (use Python 3.12)
pip install -r requirements.txt

# Create placeholder model (if no trained model available)
python model/create_placeholder.py

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Running with Docker

```bash
# Build and start services
docker-compose up --build

# Access the app at http://localhost:8000
```