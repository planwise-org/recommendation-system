from fastapi.testclient import TestClient
from src.main import app
import pytest
from src.models import UserRole, User, SQLModel
from src.database import init_db, Base, engine
from sqlmodel import Session

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Planwise API"}

# Test data
test_user_data = {
    "email": "test@example.com",
    "password": "testpassword123",
    "full_name": "Test User",
    "role": UserRole.USER
}

# Fixture for creating a test user
@pytest.fixture
def created_user():
    # Creates a function to create a user to be used for testing later on
    response = client.post("/api/users/", json=test_user_data)
    assert response.status_code == 201
    return response.json()

# CREATE USER TESTS
def test_create_user_success():
    response = client.post("/api/users/", json=test_user_data)
    print(response)
    assert response.status_code == 201
    assert response.json()["email"] == test_user_data["email"]
    assert response.json()["full_name"] == test_user_data["full_name"]
    assert "password" not in response.json()
    assert "hashed_password" not in response.json()


# def test_create_user_duplicate_email():
#     # Create first user
#     response = client.post("/api/users/", json=test_user_data)
#     assert response.status_code == 201

#     # Try to create user with same email
#     response = client.post("/api/users/", json=test_user_data)
#     assert response.status_code == 400
#     assert response.json()["detail"] == "Email already registered"

# def test_create_user_invalid_data():
#     invalid_data = {
#         "email": "not_an_email",
#         "password": "short",
#         "full_name": "",
#         "role": "invalid_role"
#     }
#     response = client.post("/api/users/", json=invalid_data)
#     assert response.status_code == 422

# # READ USERS TESTS
# def test_get_users(created_user):
#     response = client.get("/api/users/")
#     assert response.status_code == 200
#     assert isinstance(response.json(), list)
#     assert len(response.json()) > 0
#     assert response.json()[0]["email"] == created_user["email"]

# def test_get_users_pagination():
#     # Create multiple users
#     for i in range(5):
#         user_data = test_user_data.copy()
#         user_data["email"] = f"test{i}@example.com"
#         client.post("/api/users/", json=user_data)

#     # Test pagination
#     response = client.get("/api/users/?skip=1&limit=2")
#     assert response.status_code == 200
#     assert len(response.json()) == 2

# def test_get_user_by_id(created_user):
#     user_id = created_user["id"]
#     response = client.get(f"/api/users/{user_id}")
#     assert response.status_code == 200
#     assert response.json()["id"] == user_id
#     assert response.json()["email"] == created_user["email"]

# def test_get_user_not_found():
#     response = client.get("/api/users/999999")
#     assert response.status_code == 404
#     assert response.json()["detail"] == "User not found"

# # UPDATE USER TESTS
# def test_update_user(created_user):
#     user_id = created_user["id"]
#     update_data = {
#         "full_name": "Updated Name",
#         "password": "newpassword123"
#     }
#     response = client.put(f"/api/users/{user_id}", json=update_data)
#     assert response.status_code == 200
#     assert response.json()["full_name"] == "Updated Name"
#     assert "password" not in response.json()

# def test_update_user_not_found():
#     update_data = {"full_name": "Updated Name"}
#     response = client.put("/api/users/999999", json=update_data)
#     assert response.status_code == 404
#     assert response.json()["detail"] == "User not found"

# def test_update_user_invalid_data(created_user):
#     user_id = created_user["id"]
#     invalid_data = {"email": "not_an_email"}
#     response = client.put(f"/api/users/{user_id}", json=invalid_data)
#     assert response.status_code == 422

# # DELETE USER TESTS
# def test_delete_user(created_user):
#     user_id = created_user["id"]
#     response = client.delete(f"/api/users/{user_id}")
#     assert response.status_code == 204

#     # Verify user is deleted
#     response = client.get(f"/api/users/{user_id}")
#     assert response.status_code == 404

# def test_delete_user_not_found():
#     response = client.delete("/api/users/999999")
#     assert response.status_code == 404
#     assert response.json()["detail"] == "User not found"
