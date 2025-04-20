import pytest
from fastapi.testclient import TestClient
from src.main import app
import logging
client = TestClient(app)


@pytest.fixture
def auth_headers(test_user_data):
    # Create user
    response = client.post("/api/users/", json=test_user_data)
    assert response.status_code in [200, 201]

    # Login
    login_data = {
        "username": test_user_data["username"],
        "password": test_user_data["password"]
    }
    response = client.post("/api/token", data=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def created_preference(test_preference_data, auth_headers):
    response = client.post("/api/preferences/", json=test_preference_data, headers=auth_headers)
    assert response.status_code in [200, 201]
    return response.json()

# TEXT EXTRACTION TESTS
def test_extract_preferences_success(auth_headers):
    text_input = {
        "text": "I love restaurants and cafes! The food is always amazing. I'm not a big fan of bars though."
    }
    response = client.post("/api/preferences/extract-preferences", json=text_input, headers=auth_headers)
    assert response.status_code == 200
    preferences = response.json()["preferences"]
    assert "restaurants" in preferences
    assert "cafes" in preferences
    assert "pubs/bars" in preferences
    assert preferences["restaurants"] > 3.0  # Positive sentiment
    assert preferences["cafes"] > 3.0  # Positive sentiment
    assert preferences["pubs/bars"] <= 3.0  # Negative sentiment

def test_extract_preferences_empty_text(auth_headers):
    text_input = {"text": ""}
    response = client.post("/api/preferences/extract-preferences", json=text_input, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["preferences"] == {}

def test_extract_preferences_unauthorized():
    text_input = {"text": "I love restaurants!"}
    response = client.post("/api/preferences/extract-preferences", json=text_input)
    assert response.status_code == 401

# CREATE PREFERENCE TESTS
def test_create_preference_success(test_preference_data, auth_headers):
    response = client.post("/api/preferences/", json=test_preference_data, headers=auth_headers)

    assert response.status_code in [200, 201]
    assert response.json()["category"] == test_preference_data["category"]
    assert response.json()["rating"] == test_preference_data["rating"]
    assert "created_at" in response.json()
    assert "updated_at" in response.json()

def test_create_preference_update_existing(created_preference, auth_headers):
    # Try to create the same preference again with different rating
    update_data = {
        "category": created_preference["category"],
        "rating": 3.5
    }
    response = client.post("/api/preferences/", json=update_data, headers=auth_headers)
    assert response.status_code in [200, 201]
    assert response.json()["rating"] == update_data["rating"]
    assert response.json()["id"] == created_preference["id"]

def test_create_preference_unauthorized():
    preference_data = {
        "category": "restaurants",
        "rating": 4.5
    }
    response = client.post("/api/preferences/", json=preference_data)
    assert response.status_code == 401

def test_create_preference_invalid_data(auth_headers):
    invalid_data = {
        "category": "restaurants",
        "rating": 6.0  # Invalid rating (should be between 0 and 5)
    }
    response = client.post("/api/preferences/", json=invalid_data, headers=auth_headers)
    assert response.status_code == 422

# READ PREFERENCES TESTS
def test_get_preferences(created_preference, auth_headers):
    response = client.get("/api/preferences/", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert response.json()[0]["id"] == created_preference["id"]

def test_get_preferences_unauthorized():
    response = client.get("/api/preferences/")
    assert response.status_code == 401

def test_get_preference_by_category(created_preference, auth_headers):
    category = created_preference["category"]
    response = client.get(f"/api/preferences/{category}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["category"] == category
    assert response.json()["rating"] == created_preference["rating"]

def test_get_preference_not_found(auth_headers):
    response = client.get("/api/preferences/nonexistent_category", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Preference not found"

def test_get_preference_unauthorized():
    response = client.get("/api/preferences/restaurants")
    assert response.status_code == 401

# UPDATE PREFERENCE TESTS
def test_update_preference_success(created_preference, auth_headers):
    category = created_preference["category"]
    update_data = {
        "rating": 3.5
    }
    response = client.put(f"/api/preferences/{category}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["rating"] == update_data["rating"]
    assert response.json()["id"] == created_preference["id"]

def test_update_preference_unauthorized(created_preference):
    category = created_preference["category"]
    update_data = {"rating": 3.5}
    response = client.put(f"/api/preferences/{category}", json=update_data)
    assert response.status_code == 401

def test_update_preference_not_found(auth_headers):
    update_data = {"rating": 3.5}
    response = client.put("/api/preferences/nonexistent_category", json=update_data, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Preference not found"

def test_update_preference_invalid_data(created_preference, auth_headers):
    category = created_preference["category"]
    invalid_data = {
        "rating": 6.0  # Invalid rating (should be between 0 and 5)
    }
    response = client.put(f"/api/preferences/{category}", json=invalid_data, headers=auth_headers)
    assert response.status_code == 422

# DELETE PREFERENCE TESTS
def test_delete_preference_success(created_preference, auth_headers):
    category = created_preference["category"]
    response = client.delete(f"/api/preferences/{category}", headers=auth_headers)
    assert response.status_code == 204

    # Verify preference is deleted
    response = client.get(f"/api/preferences/{category}", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Preference not found"

def test_delete_preference_unauthorized(created_preference):
    category = created_preference["category"]
    response = client.delete(f"/api/preferences/{category}")
    assert response.status_code == 401

def test_delete_preference_not_found(auth_headers):
    response = client.delete("/api/preferences/nonexistent_category", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Preference not found"

def test_multiple_preferences(test_preference_data, auth_headers):
    # Create multiple preferences
    preferences = [
        {"category": "restaurants", "rating": 4.5},
        {"category": "cafes", "rating": 4.0},
        {"category": "parks", "rating": 3.5}
    ]

    for pref in preferences:
        response = client.post("/api/preferences/", json=pref, headers=auth_headers)
        assert response.status_code in [200, 201]

    # Get all preferences
    response = client.get("/api/preferences/", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) == len(preferences)

    # Verify each preference
    received_prefs = {pref["category"]: pref["rating"] for pref in response.json()}
    for pref in preferences:
        assert pref["category"] in received_prefs
        assert received_prefs[pref["category"]] == pref["rating"]
