# Continuous Integration and Deployment

## Overview

Planwise employs a robust CI/CD pipeline to automate testing, building, and deployment processes. This approach ensures code quality, reduces manual errors, and enables rapid iteration and delivery.

## CI/CD Architecture

Our pipeline is built using GitHub Actions and consists of the following stages:

1. **Code Linting & Style Checking**
2. **Unit and Integration Testing**
3. **Build & Package**
4. **Deployment**
5. **Post-Deployment Testing**

## Workflow Configuration

### Main Workflow

The primary workflow (`.github/workflows/ci.yml`) handles testing and deployment:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 black isort mypy
          pip install -r api/requirements.txt
          pip install -r reco/requirements.txt
      - name: Check formatting with Black
        run: black --check .
      - name: Check imports with isort
        run: isort --check-only --profile black .
      - name: Lint with flake8
        run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      - name: Type check with mypy
        run: mypy api/src reco/src

  test:
    needs: lint
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov
          pip install -r api/requirements.txt
          pip install -r reco/requirements.txt
      - name: Run API tests
        run: |
          cd api
          pytest --cov=src --cov-report=xml
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          SECRET_KEY: testing_secret_key
          ENV: test
      - name: Run recommendation model tests
        run: |
          cd reco
          python tests/run_tests.py
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./api/coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Build API Docker image
        uses: docker/build-push-action@v4
        with:
          context: ./api
          push: false
          tags: planwise-api:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Build Streamlit Docker image
        uses: docker/build-push-action@v4
        with:
          context: ./reco/streamlit
          push: false
          tags: planwise-streamlit:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        # Deployment steps here
        run: echo "Deployment would happen here"
```

### Pull Request Workflow

A separate workflow for pull requests:

```yaml
name: PR Checks

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  validate-pr:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate PR title
        uses: amannn/action-semantic-pull-request@v5
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: PR Size Labeler
        uses: codelytv/pr-size-labeler@v1
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Deployment Environments

We maintain three environments:

### Development

- **Trigger**: Push to `develop` branch
- **Purpose**: Testing new features and fixes
- **URL**: dev.planwise.example.com
- **Database**: Development database instance

### Staging

- **Trigger**: Manual promotion from development
- **Purpose**: Pre-production testing and validation
- **URL**: staging.planwise.example.com
- **Database**: Staging database (clone of production)

### Production

- **Trigger**: Push to `main` branch or manual promotion
- **Purpose**: Live application
- **URL**: planwise.example.com
- **Database**: Production database

## Deployment Process

### API Deployment

1. **Build Docker Image**: Package the FastAPI application
2. **Database Migrations**: Apply Alembic migrations
3. **Blue-Green Deployment**: Deploy new version alongside old version
4. **Health Check**: Verify the new deployment is functioning
5. **Traffic Switch**: Route traffic to new deployment
6. **Cleanup**: Remove old deployment

### Web App Deployment

1. **Build Streamlit Docker Image**: Package the Streamlit application
2. **Deploy to Container Service**: Update the container
3. **Verify Deployment**: Check app health and functionality

## Database Migrations

Database migrations are an integral part of our CI/CD process:

1. **Development**: Migrations are created and tested locally
2. **CI Pipeline**: Migrations are verified in test environment
3. **Deployment**: Migrations are applied automatically before code deployment
4. **Rollback Plan**: Each migration includes a down script for rollbacks

Example migration pipeline:

```yaml
jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r api/requirements.txt
      - name: Run migrations
        run: |
          cd api
          alembic upgrade head
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

## Monitoring and Alerting

Our CI/CD pipeline includes post-deployment monitoring:

1. **Health Checks**: Verify API endpoints after deployment
2. **Synthetic Tests**: Run automated workflows against deployed app
3. **Alerting**: Send notifications for failed deployments or tests
4. **Metrics Collection**: Track deployment frequency and success rate

## Rollback Procedure

If issues are detected after deployment:

1. **Automated Rollback**: Triggered by failed health checks
2. **Manual Rollback**: Via GitHub Actions interface
3. **Database Rollback**: Revert migrations if necessary
4. **Incident Report**: Document issue and resolution

## Security Considerations

Our CI/CD pipeline incorporates security best practices:

1. **Secret Management**: Sensitive values stored in GitHub Secrets
2. **Vulnerability Scanning**: Dependencies checked via Dependabot
3. **Container Scanning**: Docker images scanned for vulnerabilities
4. **Least Privilege**: CI/CD systems use minimal required permissions

## Continuous Improvement

We regularly review and improve our CI/CD process:

1. **Pipeline Metrics**: Track build time, success rate, and deployment frequency
2. **Post-Mortems**: Analyze and learn from deployment issues
3. **Automation Expansion**: Continuously automate manual processes
4. **Tool Evaluation**: Regularly assess new CI/CD tools and practices 