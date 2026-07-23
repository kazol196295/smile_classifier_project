# Smile Classifier - YourID

A web application that classifies whether a person in an image is smiling or not smiling using a machine learning model trained with scikit-learn.

## Project Overview

This project implements a full-stack smile classification system with:
- **FastAPI** web application with HTML templates
- **scikit-learn** machine learning model (SVM) for classification
- **PostgreSQL** database for storing classification results
- **Docker** deployment with multi-container setup
- **Local training** via `train_local.py`

## Features

- **Home Page**: Project overview and instructions
- **Train Page**: Upload multiple images for model training
- **Classify Page**: Upload a single image to classify as smiling/not smiling
- **History Page**: View all previous classification results

## Tech Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy ORM
- **Frontend**: Jinja2 templates, HTML/CSS
- **Database**: PostgreSQL (Docker), SQLite (local development)
- **ML**: scikit-learn (SVM), Pillow, NumPy
- **Deployment**: Docker, Docker Compose

## Project Structure

```
smile_classifier_project/
├── model/
│   ├── smile_classifier.pkl   # Pre-trained model
│   └── create_placeholder.py  # Creates dummy model
├── main.py                    # FastAPI app entrypoint
├── config.py                  # Configuration settings
├── database.py                # Database engine and session
├── models.py                  # SQLAlchemy models
├── schemas.py                 # Pydantic schemas
├── inference.py               # Model loading and prediction
├── train_local.py             # Local training: SVM model
├── routers/
│   ├── home.py                # Home page router
│   ├── train.py               # Training page router
│   ├── classify.py            # Classification page router
│   └── history.py             # History page router
├── templates/                 # HTML templates
├── static/                    # Static files
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker image definition
└── requirements.txt           # Python dependencies
```

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd smile_classifier_project

# Install dependencies
pip install -r requirements.txt

# Train the model (requires image folders)
python train_local.py

# Or create placeholder model (if no training data)
python model/create_placeholder.py

# Start the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the application at: http://localhost:8000

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Access the application at: http://localhost:8000
```

## Docker Image for Apple Silicon (ARM)

If you need to run this on Apple Silicon Macs:

```bash
# Build ARM-compatible image
docker buildx build --platform linux/arm64 -t smile-classifier:arm64 .

# Save image to file
docker save smile-classifier:arm64 -o smile-classifier-arm64.tar

# Load on Apple Silicon Mac
docker load -i smile-classifier-arm64.tar
docker-compose up
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./smile_db.sqlite` | Database connection string |
| `MODEL_PATH` | `model/smile_classifier.pkl` | Path to trained model |
| `UPLOAD_DIR` | `static/uploaded_images` | Directory for uploaded images |

## API Endpoints

- `GET /` - Home page
- `GET /train` - Training page
- `POST /train` - Upload images for training
- `GET /classify` - Classification page
- `POST /classify` - Upload and classify an image
- `GET /history` - Classification history

## Dataset

The model is trained on the [Smiling or Not Face Data](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data) dataset from Kaggle.

## License

This project is for educational purposes as part of the AI Frameworks Libraries Deployment mini-project.
