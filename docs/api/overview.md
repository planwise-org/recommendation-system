# API Overview

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.io/)

## Introduction

The Planwise API provides a robust, scalable backend for our recommendation engine. Built with FastAPI, it handles user authentication, preference management, review collection, and recommendation generation. The API is designed to be both performant and developer-friendly, with comprehensive documentation and consistent patterns.

## Core Features

- **User Management**: Registration, authentication, and profile management
- **Preference Handling**: Store and retrieve user preferences across categories
- **Natural Language Processing**: Extract preferences from conversational text
- **Place Data**: Access and query information about places in Madrid
- **Review System**: Submit and retrieve user reviews of places
- **Recommendation Engine**: Generate personalized recommendations based on preferences
- **OAuth2 Authentication**: Secure JWT-based authentication

## Architecture

Our API follows a clean, layered architecture:

1. **Routes**: Handle HTTP requests and responses
2. **Services**: Implement business logic and model interaction
3. **Models**: Define data structures and database schema
4. **Middleware**: Process requests for authentication, logging, etc.

## Key Endpoints

The API is organized into several logical groups:

- **/api/token**: Authentication endpoint for obtaining JWT tokens
- **/api/users**: User management endpoints
- **/api/places**: Place data retrieval and management
- **/api/reviews**: Review submission and retrieval
- **/api/recommendations**: Recommendation generation endpoints
- **/api/preferences**: User preference management

## Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Setup and Deployment

### Local Development with Docker

#### Prerequisites

- Docker installed on your machine
- Supabase CLI

#### Setup Steps

1. **Navigate to API Directory**

   ```bash
   cd api
   ```

2. **Set up Supabase Locally**

   Install Supabase CLI:

   ```bash
   # MacOS
   brew install supabase/tap/supabase

   # Windows (with scoop)
   scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
   scoop install supabase
   ```

3. **Initialize and Start Supabase**

   ```bash
   supabase init
   supabase start
   ```

4. **Configure Database URL**
   - Copy the DB URL from Supabase CLI output
   - Paste it in the `DATABASE_URL` variable in `docker-compose.dev.yml`
   - ⚠️ **Important**: Replace `127.0.0.1` with `host.docker.internal` for proper Docker networking

5. **Build and Run Docker Container**

   ```bash
   docker compose -f docker-compose.dev.yml up -d --build
   ```

   This will:
   - Start the application on port 8080 (configurable)
   - Start PostgreSQL on port 5432 (configurable)
   - Run in detached mode

6. **Stop the Container**
   ```bash
   docker compose -f docker-compose.dev.yml down
   ```

### Manual Setup

#### Prerequisites
- Python 3.8+
- pip

#### Setup Steps

1. **Create Virtual Environment**
   ```bash
   cd api
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   uvicorn main:app --reload
   ```

### Environment Variables

Create a `.env` file in the `api` directory with the following variables:

```env
ENV=local
SECRET_KEY=<your_secret_key>
DATABASE_URL=<your_database_url>
SUPABASE_URL=<your_supabase_url>  # Only for production
SUPABASE_KEY=<your_supabase_key>  # Only for production
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 