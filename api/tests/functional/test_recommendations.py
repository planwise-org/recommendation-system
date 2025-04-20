import pytest
from fastapi.testclient import TestClient
from src.main import app
from sqlmodel import SQLModel, Session
from src.database import init_db, engine, drop_all_tables, get_session
from src.schemas.recommendation import RecommendationAlgorithm

client = TestClient(app)


@pytest.fixture
def created_user(test_user_data):
    response = client.post("/api/users/", json=test_user_data)
    assert response.status_code in [200, 201]
    return response.json()

@pytest.fixture
def created_place(test_place_data):
    response = client.post("/api/places/", json=test_place_data)
    assert response.status_code in [200, 201]
    return response.json()

@pytest.fixture
def test_recommendation_data(created_user, created_place):
    return {
        "user_id": created_user["id"],
        "place_id": created_place["id"],
        "algorithm": RecommendationAlgorithm.AUTOENCODER.value,
        "score": 0.95,
        "visited": False,
        "reviewed": False
    }

@pytest.fixture
def created_recommendation(test_recommendation_data):
    response = client.post("/api/recommendations/", json=test_recommendation_data)
    assert response.status_code in [200, 201]
    return response.json()

# CREATE RECOMMENDATION TESTS
def test_create_recommendation_success(test_recommendation_data):
    response = client.post("/api/recommendations/", json=test_recommendation_data)

    assert response.status_code in [200, 201]
    assert response.json()["user_id"] == test_recommendation_data["user_id"]
    assert response.json()["place_id"] == test_recommendation_data["place_id"]
    assert response.json()["algorithm"] == test_recommendation_data["algorithm"]
    assert response.json()["score"] == test_recommendation_data["score"]
    assert response.json()["visited"] == test_recommendation_data["visited"]
    assert response.json()["reviewed"] == test_recommendation_data["reviewed"]

def test_create_recommendation_invalid_data():
    invalid_data = {
        "user_id": 999999,  # Non-existent user
        "place_id": 999999,  # Non-existent place
        "score": 2.0,  # Invalid score (should be between 0 and 1)
        "algorithm": "invalid_algorithm"  # Invalid algorithm
    }
    response = client.post("/api/recommendations/", json=invalid_data)
    assert response.status_code in [400, 422]

# READ RECOMMENDATIONS TESTS
def test_get_recommendations(created_recommendation):
    response = client.get("/api/recommendations/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert any(rec["id"] == created_recommendation["id"] for rec in response.json())

def test_get_recommendations_by_user(created_recommendation):
    user_id = created_recommendation["user_id"]
    response = client.get(f"/api/recommendations/?user_id={user_id}")
    assert response.status_code == 200
    assert all(rec["user_id"] == user_id for rec in response.json())

def test_get_recommendations_by_algorithm(created_recommendation):
    algorithm = created_recommendation["algorithm"]
    response = client.get(f"/api/recommendations/?algorithm={algorithm}")
    assert response.status_code == 200
    assert all(rec["algorithm"] == algorithm for rec in response.json())

def test_get_recommendations_pagination(test_recommendation_data):
    # Create multiple recommendations
    for _ in range(5):
        client.post("/api/recommendations/", json=test_recommendation_data)

    response = client.get("/api/recommendations/?skip=1&limit=2")
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_get_recommendation_by_id(created_recommendation):
    recommendation_id = created_recommendation["id"]
    response = client.get(f"/api/recommendations/{recommendation_id}")
    assert response.status_code == 200
    assert response.json()["id"] == recommendation_id

def test_get_recommendation_not_found():
    response = client.get("/api/recommendations/999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Recommendation not found"

# GENERATE RECOMMENDATIONS TEST
def test_generate_user_recommendations(created_user):
    user_id = created_user["id"]
    response = client.post(f"/api/recommendations/generate/{user_id}")
    assert response.status_code in [200, 201]
    assert isinstance(response.json(), list)
    assert all(rec["user_id"] == user_id for rec in response.json())
    assert all(rec["algorithm"] == RecommendationAlgorithm.AUTOENCODER.value for rec in response.json())

def test_generate_user_recommendations_custom_algorithm(created_user):
    user_id = created_user["id"]
    algorithm = RecommendationAlgorithm.SVD.value
    response = client.post(f"/api/recommendations/generate/{user_id}?algorithm={algorithm}")
    assert response.status_code in [200, 201]
    assert all(rec["algorithm"] == algorithm for rec in response.json())

# UPDATE RECOMMENDATION TESTS
def test_update_recommendation(created_recommendation):
    recommendation_id = created_recommendation["id"]
    update_data = {
        "visited": True
    }
    response = client.put(f"/api/recommendations/{recommendation_id}", json=update_data)
    assert response.status_code == 200
    assert response.json()["visited"] == update_data["visited"]

def test_update_recommendation_not_found():
    update_data = {"visited": True}
    response = client.put("/api/recommendations/999999", json=update_data)
    assert response.status_code == 404
    assert response.json()["detail"] == "Recommendation not found"

def test_mark_recommendation_visited(created_recommendation):
    recommendation_id = created_recommendation["id"]
    response = client.put(f"/api/recommendations/{recommendation_id}/visited")
    assert response.status_code == 200
    assert response.json()["visited"] == True

def test_mark_recommendation_reviewed(created_recommendation):
    recommendation_id = created_recommendation["id"]
    response = client.put(f"/api/recommendations/{recommendation_id}/reviewed")
    assert response.status_code == 200
    assert response.json()["reviewed"] == True

# DELETE RECOMMENDATION TESTS
def test_delete_recommendation(created_recommendation):
    recommendation_id = created_recommendation["id"]
    response = client.delete(f"/api/recommendations/{recommendation_id}")
    assert response.status_code == 204

    # Verify recommendation is deleted
    response = client.get(f"/api/recommendations/{recommendation_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Recommendation not found"

def test_delete_recommendation_not_found():
    response = client.delete("/api/recommendations/999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Recommendation not found"
