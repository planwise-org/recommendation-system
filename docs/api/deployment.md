# API Deployment

## Overview

This document covers the deployment options for the Planwise API. We provide robust containerization with Docker and flexible hosting options to meet different deployment needs, from local development to cloud-based production.

## Docker Deployment

### Docker Configuration

The API comes with a production-ready `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

This configuration:
- Uses Python 3.10 slim image for minimal size
- Installs required dependencies
- Exposes the application on port 8080

### Building and Running the Container

```bash
# Build the image
docker build -t planwise-api .

# Run the container
docker run -d -p 8080:8080 \
  -e DATABASE_URL=postgresql://user:password@host:port/dbname \
  -e SECRET_KEY=your_secret_key \
  --name planwise-api \
  planwise-api
```

### Docker Compose for Local Development

For local development, we provide a `docker-compose.dev.yml` file:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
      - ENV=development
    volumes:
      - ./:/app
    command: uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

To start the development environment:

```bash
docker compose -f docker-compose.dev.yml up -d
```

### Docker Compose for Production

For production environments, use `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    restart: always
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
      - ENV=production
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Cloud Deployment Options

### Vercel Serverless

The API can be deployed to Vercel using the following `vercel.json` configuration:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "src/main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "src/main.py"
    }
  ]
}
```

Deploy with the Vercel CLI:

```bash
vercel --prod
```

### AWS Lambda

Deploy as a Lambda function using the AWS Serverless Application Model (SAM):

1. Create a `template.yaml` file
2. Configure API Gateway triggers
3. Deploy with the SAM CLI

### Kubernetes

For larger scale deployments, we provide Kubernetes manifests:

1. Deployment configuration
2. Service definition
3. Ingress rules
4. Horizontal Pod Autoscaler

## Database Setup

### Supabase Configuration

The API uses Supabase for managed PostgreSQL:

1. Create a Supabase project
2. Set up the required tables or use our migration scripts
3. Configure environment variables:
   - `DATABASE_URL`: PostgreSQL connection string
   - `SUPABASE_URL`: Supabase project URL
   - `SUPABASE_KEY`: Supabase service key

### Database Migrations

Database schema changes are managed with Alembic:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Environment Variables

Configure the following environment variables for deployment:

| Variable | Description | Example |
|----------|-------------|---------|
| `ENV` | Environment (development/production) | `production` |
| `SECRET_KEY` | JWT signing key | `your-secret-key-string` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:port/db` |
| `SUPABASE_URL` | Supabase project URL | `https://project.supabase.co` |
| `SUPABASE_KEY` | Supabase service key | `eyJhbGciOiJIUzI1NiIsInR...` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token expiration | `30` |

## SSL/TLS Configuration

For production deployments, always enable HTTPS:

1. Terminate SSL at the load balancer (recommended)
2. Or configure the application server with certificates

## Monitoring and Logging

The API integrates with standard monitoring tools:

1. Prometheus metrics endpoint at `/metrics`
2. Structured logging to stdout
3. Health check endpoint at `/health`

## CI/CD Pipeline

We use GitHub Actions for continuous integration and deployment:

1. Run tests on each pull request
2. Build and publish Docker images
3. Deploy to staging/production environments
4. Run database migrations 