# Integration Testing

## Overview

Integration tests verify that different components of the Planwise system work together correctly. Unlike unit tests that test components in isolation, integration tests examine the interactions between components and identify issues that might only appear when parts of the system are combined.

## Integration Test Approach

Our integration testing strategy focuses on:

1. **API Endpoints**: Testing complete API routes with database interactions
2. **Recommendation Pipeline**: Testing the end-to-end recommendation flow
3. **Database Operations**: Verifying database schema, migrations, and queries
4. **Authentication Flow**: Testing the complete authentication and authorization process

## API Integration Tests

We use pytest and the FastAPI TestClient to test our API endpoints:

```python
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from src.main import app
from src.database import get_session
from src.models import User, Place, Review

# Test database setup
@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture(name="client")
def client_fixture(session):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_create_and_read_user(client, session):
    # Test data
    user_data = {
        "username": "testuser",
        "password": "password123",
        "full_name": "Test User",
        "role": "user"
    }
    
    # Create user
    response = client.post("/api/users/", json=user_data)
    assert response.status_code == 200
    created_user = response.json()
    assert created_user["username"] == user_data["username"]
    
    # Get authentication token
    auth_response = client.post(
        "/api/token",
        data={"username": user_data["username"], "password": user_data["password"]}
    )
    assert auth_response.status_code == 200
    token = auth_response.json()["access_token"]
    
    # Use token to access protected endpoint
    me_response = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert me_response.status_code == 200
    user_me = me_response.json()
    assert user_me["username"] == user_data["username"]
```

## Recommendation Pipeline Integration Tests

We test the complete recommendation flow from user preferences to final recommendations:

```python
def test_recommendation_pipeline():
    # 1. Set up test data
    user_preferences = {
        "museums": 4.5,
        "parks": 3.0,
        "cafes": 4.0
    }
    user_location = (40.4168, -3.7038)  # Madrid
    
    # 2. Initialize recommendation models
    auto_model = load_model("test_models/autoencoder.h5")
    scaler = joblib.load("test_models/scaler.save")
    
    # 3. Create the ensemble recommender with all sub-models
    ensemble = EnsembleRecommender()
    ensemble.initialize_models(
        auto_model=auto_model,
        scaler=scaler,
        places_df=test_places_df,
        category_to_place_types=test_category_mapping
    )
    
    # 4. Get recommendations
    recommendations = ensemble.get_recommendations(
        user_lat=user_location[0],
        user_lon=user_location[1],
        user_prefs=user_preferences,
        predicted_ratings_dict=user_preferences,
        num_recs=5
    )
    
    # 5. Validate recommendations
    assert len(recommendations) <= 5
    assert all(r.get('name') for r in recommendations)
    assert all(r.get('score') for r in recommendations)
    assert all(r.get('lat') for r in recommendations)
    assert all(r.get('lng') for r in recommendations)
    
    # 6. Validate recommendation ordering
    scores = [r['score'] for r in recommendations]
    assert scores == sorted(scores, reverse=True)  # Check descending order
```

## Database Integration Tests

We test database interactions including migrations and complex queries:

```python
def test_database_migrations(tmp_path):
    # Create a test database
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    # Set up Alembic environment
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", db_url)
    
    # Run migrations
    command.upgrade(config, "head")
    
    # Verify database schema
    engine = create_engine(db_url)
    inspector = inspect(engine)
    
    # Check tables exist
    assert "user" in inspector.get_table_names()
    assert "place" in inspector.get_table_names()
    assert "review" in inspector.get_table_names()
    assert "preference" in inspector.get_table_names()
    
    # Check columns in user table
    user_columns = {col["name"] for col in inspector.get_columns("user")}
    assert "id" in user_columns
    assert "username" in user_columns
    assert "hashed_password" in user_columns
    assert "full_name" in user_columns
    assert "role" in user_columns
```

## Authentication Integration Tests

We test the entire authentication flow including token generation and validation:

```python
def test_authentication_flow(client, session):
    # Create test user
    hashed_password = get_password_hash("testpassword")
    user = User(
        username="testauth",
        hashed_password=hashed_password,
        full_name="Test Auth",
        role="user"
    )
    session.add(user)
    session.commit()
    
    # Test incorrect password
    response = client.post(
        "/api/token",
        data={"username": "testauth", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    
    # Test correct password
    response = client.post(
        "/api/token",
        data={"username": "testauth", "password": "testpassword"}
    )
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    
    # Test accessing protected endpoint
    token = token_data["access_token"]
    response = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    user_data = response.json()
    assert user_data["username"] == "testauth"
    
    # Test invalid token
    response = client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer invalidtoken"}
    )
    assert response.status_code == 401
```

## End-to-End Workflow Tests

We test complete user workflows that involve multiple API calls:

```python
def test_preference_to_recommendation_workflow(client, session):
    # 1. Create test user
    response = client.post(
        "/api/users/",
        json={
            "username": "workflow_test",
            "password": "password123",
            "full_name": "Workflow Test",
            "role": "user"
        }
    )
    assert response.status_code == 200
    
    # 2. Login and get token
    response = client.post(
        "/api/token",
        data={"username": "workflow_test", "password": "password123"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Set user preferences
    preferences = [
        {"category": "museums", "rating": 4.5},
        {"category": "parks", "rating": 3.0},
        {"category": "cafes", "rating": 4.0}
    ]
    
    for pref in preferences:
        response = client.post(
            "/api/preferences/",
            headers=headers,
            json=pref
        )
        assert response.status_code == 200
    
    # 4. Get recommendations based on preferences
    response = client.get(
        "/api/recommendations/?lat=40.4168&lng=-3.7038&limit=5",
        headers=headers
    )
    assert response.status_code == 200
    recommendations = response.json()
    assert len(recommendations) > 0
    
    # 5. Submit a review for the first recommendation
    place_id = recommendations[0]["place_id"]
    review = {
        "place_id": place_id,
        "rating": 4.5,
        "comment": "Great place, loved it!"
    }
    
    response = client.post(
        "/api/reviews/",
        headers=headers,
        json=review
    )
    assert response.status_code == 200
    
    # 6. Verify the review was saved
    response = client.get(
        f"/api/reviews/user/{place_id}",
        headers=headers
    )
    assert response.status_code == 200
    saved_review = response.json()
    assert saved_review["rating"] == review["rating"]
    assert saved_review["comment"] == review["comment"]
```

## Running Integration Tests

To run all integration tests:

```bash
cd api
pytest tests/integration
```

To run a specific integration test file:

```bash
cd api
pytest tests/integration/test_auth_flow.py
```

## Integration Test Coverage

To check integration test coverage:

```bash
cd api
pytest tests/integration --cov=src --cov-report=term-missing
```

## Best Practices for Integration Tests

1. **Realistic Data**: Use realistic data that represents actual usage
2. **Complete Workflows**: Test end-to-end user workflows
3. **Test Boundaries**: Check interactions between different components
4. **Database State**: Be careful about test database state between tests
5. **Clean Up**: Make sure tests clean up after themselves
6. **Test Configuration**: Use a separate test configuration
7. **Performance**: Monitor test execution time to detect slow integrations 