import pytest
from fastapi.testclient import TestClient
from src.main import app

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
def created_place(test_place_data):
    response = client.post("/api/places/", json=test_place_data)
    assert response.status_code in [200, 201]
    return response.json()

@pytest.fixture
def test_review_data(created_place):
    return {
        "place_id": created_place["id"],
        "rating": 4.5,
        "comment": "Great place! Would visit again."
    }

@pytest.fixture
def created_review(test_review_data, auth_headers):
    response = client.post("/api/reviews/", json=test_review_data, headers=auth_headers)
    assert response.status_code in [200, 201]
    return response.json()

# CREATE REVIEW TESTS
def test_create_review_success(test_review_data, auth_headers):
    response = client.post("/api/reviews/", json=test_review_data, headers=auth_headers)

    assert response.status_code in [200, 201]
    assert response.json()["place_id"] == test_review_data["place_id"]
    assert response.json()["rating"] == test_review_data["rating"]
    assert response.json()["comment"] == test_review_data["comment"]
    assert "created_at" in response.json()
    assert "updated_at" in response.json()

def test_create_review_unauthorized():
    review_data = {
        "place_id": "some_place_id",
        "rating": 4.5,
        "comment": "Great place!"
    }
    response = client.post("/api/reviews/", json=review_data)
    assert response.status_code == 401

def test_create_review_invalid_data(auth_headers):
    invalid_data = {
        "place_id": "some_place_id",
        "rating": 6.0,  # Invalid rating (should be between 1 and 5)
        "comment": ""
    }
    response = client.post("/api/reviews/", json=invalid_data, headers=auth_headers)
    assert response.status_code == 422

# READ REVIEWS TESTS
def test_get_user_reviews(created_review, auth_headers):
    response = client.get("/api/reviews/", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert response.json()[0]["id"] == created_review["id"]

def test_get_user_reviews_unauthorized():
    response = client.get("/api/reviews/")
    assert response.status_code == 401

def test_get_place_reviews(created_review, created_place):
    place_id = created_place["id"]
    response = client.get(f"/api/reviews/{place_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert response.json()[0]["place_id"] == place_id

def test_get_user_place_review_exists(created_review, created_place, auth_headers):
    place_id = created_place["id"]
    response = client.get(f"/api/reviews/user/{place_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["rating"] == created_review["rating"]
    assert response.json()["comment"] == created_review["comment"]
    assert response.json()["submitted"] == True

def test_get_user_place_review_not_exists(created_place, auth_headers):
    place_id = created_place["id"]
    response = client.get(f"/api/reviews/user/{place_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["rating"] == 3.0
    assert response.json()["comment"] == ""
    assert response.json()["submitted"] == False

def test_get_user_place_review_unauthorized(created_place):
    place_id = created_place["id"]
    response = client.get(f"/api/reviews/user/{place_id}")
    assert response.status_code == 401

# UPDATE REVIEW TESTS
def test_update_review_success(created_review, auth_headers):
    review_id = created_review["id"]
    update_data = {
        "rating": 3.5,
        "comment": "Updated comment"
    }
    response = client.put(f"/api/reviews/{review_id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["rating"] == update_data["rating"]
    assert response.json()["comment"] == update_data["comment"]
    assert response.json()["id"] == review_id

def test_update_review_unauthorized(created_review):
    review_id = created_review["id"]
    update_data = {
        "rating": 3.5,
        "comment": "Updated comment"
    }
    response = client.put(f"/api/reviews/{review_id}", json=update_data)
    assert response.status_code == 401

def test_update_review_not_found(auth_headers):
    update_data = {
        "rating": 3.5,
        "comment": "Updated comment"
    }
    response = client.put("/api/reviews/999999", json=update_data, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Review not found"

def test_update_review_invalid_data(created_review, auth_headers):
    review_id = created_review["id"]
    invalid_data = {
        "rating": 6.0  # Invalid rating (should be between 1 and 5)
    }
    response = client.put(f"/api/reviews/{review_id}", json=invalid_data, headers=auth_headers)
    assert response.status_code == 422

def test_update_review_wrong_user(created_review, test_user_data):
    # Create another user and get their token
    new_user_data = test_user_data.copy()
    new_user_data["username"] = "another_user"
    response = client.post("/api/users/", json=new_user_data)
    assert response.status_code in [200, 201]

    login_data = {
        "username": new_user_data["username"],
        "password": new_user_data["password"]
    }
    response = client.post("/api/token", data=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Try to update the review created by the first user
    review_id = created_review["id"]
    update_data = {
        "rating": 3.5,
        "comment": "Updated comment"
    }
    response = client.put(f"/api/reviews/{review_id}", json=update_data, headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized to update this review"

# DELETE REVIEW TESTS
def test_delete_review_success(created_review, auth_headers):
    review_id = created_review["id"]
    response = client.delete(f"/api/reviews/{review_id}", headers=auth_headers)
    assert response.status_code == 204

    # Verify review is deleted
    response = client.get("/api/reviews/", headers=auth_headers)
    assert response.status_code == 200
    assert not any(review["id"] == review_id for review in response.json())

def test_delete_review_unauthorized(created_review):
    review_id = created_review["id"]
    response = client.delete(f"/api/reviews/{review_id}")
    assert response.status_code == 401

def test_delete_review_not_found(auth_headers):
    response = client.delete("/api/reviews/999999", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Review not found"

def test_delete_review_wrong_user(created_review, test_user_data):
    # Create another user and get their token
    new_user_data = test_user_data.copy()
    new_user_data["username"] = "another_user"
    response = client.post("/api/users/", json=new_user_data)
    assert response.status_code in [200, 201]

    login_data = {
        "username": new_user_data["username"],
        "password": new_user_data["password"]
    }
    response = client.post("/api/token", data=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Try to delete the review created by the first user
    review_id = created_review["id"]
    response = client.delete(f"/api/reviews/{review_id}", headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized to delete this review"
