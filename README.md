# Planwise Recommendation System - Developer Guide

## Overview

This is a recommendation system for Planwise. It is a multi-model built for recommending places in Madrid based on user preferences. This guide will help explain how the code is structured and works.

## Prerequisites
Python 3.9 or above
Docker & Docker Compose
PostgreSQL (if running locally without Docker)
Streamlit

## Quickstart (Using Docker)
This is the easiest way to run the API, Database, and Streamlit UI locally:
git clone --recurse-submodules git@github.com:planwise-org/recommendation-system.git
cd recommendation-system

Start local dev environment
docker-compose -f docker-compose.dev.yml up --build

Then open:
http://localhost:8501 → Streamlit UI
http://localhost:8000/docs → FastAPI Swagger docs


## Manual Setup (For Dev Work in VS Code)
If you want to run services manually for debugging or development:
1. Clone the Repo and Init Submodules
git clone git@github.com:planwise-org/recommendation-system.git
cd recommendation-system

2. Create & Activate a Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

3. Set Up the Database (PostgreSQL)
You can run PostgreSQL via Docker or install it directly.
To use Docker:
docker run --name pg-planwise -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourpassword -p 5432:5432 -d postgres

Then update your .env file or settings to reflect:
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/postgres

Run migrations:
alembic upgrade head


4. Run the FastAPI Backend
cd api/
uvicorn src.main:app --reload

Access the docs at:
http://localhost:8000/docs


5. Run the Streamlit Frontend
cd reco/streamlit/
streamlit run app.py

Open:
http://localhost:8501


## How the code works
## Recommendation Engine
Autoencoder: Learns user-category affinities and reconstructs missing scores.

SVD: Matrix factorization model using surprise for collaborative filtering.

Transfer Learning (BiVAE): Uses movie preferences to map user embeddings to place types.

Each model outputs a ranked list of places → the ensemble module combines their scores.

## Data Pipeline
Found in data/userspipeline.py

Cleans, joins, and aggregates user-place interaction data.

Outputs clean CSVs used for training & inference.

## Model Files
Stored in models/

Use semantic versioning: ae_v1.2.pt, svd_model.pkl, etc.

Loaded by backend and UI for recommendations.


## Testing & CI
Lint, Type Check, and Tests:
flake8
mypy .
pytest

## CI/CD
GitHub Actions runs checks for every PR in .github/workflows/ci.yml


Task                        |                      Command
Run API locally             |                      uvicorn src.main:app --reload
Run Streamlit               |                      streamlit run app.py
Rebuild Docker containers   |                     docker-compose -f docker-compose.dev.yml up --build
Run Alembic migration       |                      alembic upgrade head
Activate venv               |                      source .venv/bin/activate


## Contributors
Els, Alexa, Anna, Ricardo, Rodrigo, David, Jose

