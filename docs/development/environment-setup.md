# Development Environment Setup

## Overview

This guide will help you set up a development environment for the Planwise project. The project consists of two main components:

1. **API Service**: FastAPI backend providing recommendation and data services
2. **Streamlit Web App**: Interactive frontend for user interaction

## Prerequisites

Before starting, ensure you have:

- [Python 3.8+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started) and Docker Compose
- [Git](https://git-scm.com/downloads)
- [Node.js](https://nodejs.org/) (optional, for frontend development tools)

## Clone the Repository

```bash
git clone https://github.com/yourusername/planwise.git
cd planwise
```

## Option 1: Docker Setup (Recommended)

The quickest way to get started is using Docker, which sets up all services automatically.

### Start the Docker Environment

```bash
# Start both API and Streamlit app
docker compose -f docker-compose.dev.yml up -d
```

This will:
- Start the API service on port 8080
- Start the Streamlit app on port 8501
- Set up PostgreSQL database automatically
- Configure Supabase locally

### Access the Services

- API: http://localhost:8080
- API Documentation: http://localhost:8080/docs
- Streamlit App: http://localhost:8501

## Option 2: Manual Setup

If you prefer to run the services directly on your machine, follow these steps.

### API Setup

1. **Create a virtual environment**

```bash
cd api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up Supabase locally**

```bash
# Install Supabase CLI
npm install -g supabase

# Initialize and start Supabase
supabase init
supabase start
```

4. **Configure environment variables**

Create a `.env` file in the `api` directory:

```
ENV=development
SECRET_KEY=your_development_secret_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

5. **Run the API**

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### Streamlit App Setup

1. **Create a virtual environment**

```bash
cd reco/streamlit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

## Recommendation Models Setup

For local development with the recommendation models:

1. **Install model dependencies**

```bash
cd reco
pip install -r requirements.txt
```

2. **Download model files**

```bash
# Create model directories
mkdir -p models

# Download pre-trained models (replace with actual download commands)
# Example:
# wget -O models/autoencoder.h5 https://example.com/models/autoencoder.h5
# wget -O models/scaler.save https://example.com/models/scaler.save
# wget -O models/madrid_place_embeddings.npz https://example.com/models/madrid_place_embeddings.npz
```

## Database Migrations

To apply database migrations:

```bash
cd api
alembic upgrade head
```

To create a new migration after changing models:

```bash
alembic revision --autogenerate -m "Description of changes"
```

## API Load Testing

For load testing the API:

```bash
pip install locust
cd tests/load
locust -f locustfile.py
```

Then open http://localhost:8089 to start a test.

## Development Tools

### Code Formatting

We use Black and isort for code formatting:

```bash
# Install formatting tools
pip install black isort

# Format code
black .
isort .
```

### Linting

We use flake8 for linting:

```bash
pip install flake8
flake8 .
```

### Type Checking

We use mypy for type checking:

```bash
pip install mypy
mypy .
```

## Troubleshooting

### Common Issues

1. **Database connection errors**:
   - Check that PostgreSQL is running
   - Verify DATABASE_URL environment variable is correct
   - Ensure database migrations have been applied

2. **Model loading errors**:
   - Verify model files are in the expected locations
   - Check TensorFlow/PyTorch versions match model requirements

3. **API authentication issues**:
   - Ensure SECRET_KEY environment variable is set
   - Check that JWT tokens haven't expired

### Getting Help

- Check the project issues on GitHub
- Consult the [API Documentation](../api/overview.md)
- Refer to the [Testing Guide](../testing/overview.md) 