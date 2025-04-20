import pytest
from fastapi.testclient import TestClient
from src.main import app
from sqlmodel import SQLModel, Session
from src.database import init_db, engine, drop_all_tables, get_session

client = TestClient(app)


@pytest.fixture
def created_place(test_place_data, network_codes):
    response = client.post("/api/places/", json=test_place_data)
    assert response.status_code in network_codes["success"]
    return response.json()

# CREATE PLACE TESTS
def test_create_place_success(test_place_data, network_codes):
    response = client.post("/api/places/", json=test_place_data)

    assert response.status_code in network_codes["success"]
    assert response.json()["name"] == test_place_data["name"]
    assert response.json()["description"] == test_place_data["description"]
    assert response.json()["address"] == test_place_data["address"]
    assert response.json()["latitude"] == test_place_data["latitude"]
    assert response.json()["longitude"] == test_place_data["longitude"]
    assert response.json()["place_type"] == test_place_data["place_type"]

def test_create_place_invalid_data(network_codes):
    invalid_data = {
        "name": "",  # Empty name
        "description": "Test description",
        "latitude": 200,  # Invalid latitude
        "longitude": -200,  # Invalid longitude
        "place_type": "restaurant"
    }
    response = client.post("/api/places/", json=invalid_data)
    assert response.status_code in network_codes["error"]

# READ PLACES TESTS
def test_get_places(created_place, network_codes):
    response = client.get("/api/places/")
    assert response.status_code in network_codes["success"]
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert any(place["id"] == created_place["id"] for place in response.json())

def test_get_places_pagination(test_place_data, network_codes):
    # Create multiple places
    for i in range(5):
        place_data = test_place_data.copy()
        place_data["name"] = f"Test Place {i}"
        client.post("/api/places/", json=place_data)

    # Test pagination
    response = client.get("/api/places/?skip=1&limit=2")
    assert response.status_code in network_codes["success"]
    assert len(response.json()) == 2

def test_get_place_by_id(created_place, network_codes):
    place_id = created_place["id"]
    response = client.get(f"/api/places/{place_id}")
    assert response.status_code in network_codes["success"]
    assert response.json()["id"] == place_id
    assert response.json()["name"] == created_place["name"]

def test_get_place_not_found(network_codes):
    response = client.get("/api/places/999999")
    assert response.status_code in network_codes["error"]
    assert response.json()["detail"] == "Place not found"

# UPDATE PLACE TESTS
def test_update_place(created_place, network_codes):
    place_id = created_place["id"]
    update_data = {
        "name": "Updated Place Name",
        "description": "Updated description"
    }
    response = client.put(f"/api/places/{place_id}", json=update_data)
    assert response.status_code in network_codes["success"]
    assert response.json()["name"] == update_data["name"]
    assert response.json()["description"] == update_data["description"]
    # Check that other fields remain unchanged
    assert response.json()["address"] == created_place["address"]

def test_update_place_not_found(network_codes):
    update_data = {"name": "Updated Name"}
    response = client.put("/api/places/999999", json=update_data)
    assert response.status_code in network_codes["error"]
    assert response.json()["detail"] == "Place not found"

def test_update_place_invalid_data(created_place, network_codes):
    place_id = created_place["id"]
    invalid_data = {
        "latitude": 200,  # Invalid latitude
        "longitude": -200  # Invalid longitude
    }
    response = client.put(f"/api/places/{place_id}", json=invalid_data)
    assert response.status_code in network_codes["error"]

# DELETE PLACE TESTS
def test_delete_place(created_place, network_codes):
    place_id = created_place["id"]
    response = client.delete(f"/api/places/{place_id}")
    assert response.status_code in network_codes["success"]

    # Verify place is deleted
    response = client.get(f"/api/places/{place_id}")
    assert response.status_code in network_codes["error"]
    assert response.json()["detail"] == "Place not found"

def test_delete_place_not_found(network_codes):
    response = client.delete("/api/places/999999")
    assert response.status_code in network_codes["error"]
    assert response.json()["detail"] == "Place not found"
