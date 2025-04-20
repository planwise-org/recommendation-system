import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.models import UserRole

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Planwise API"}


#  Fixture for creating a test user
@pytest.fixture
def created_user(test_user_data, network_codes):

    # Creates a function to create a user to be used for testing later on
    response = client.post("/api/users/", json=test_user_data)
    assert response.status_code in network_codes["success"]
    return response.json()


# CREATE USER TESTS
def test_create_user_success(test_user_data, network_codes):
    response = client.post("/api/users/", json=test_user_data)

    assert response.status_code in network_codes["success"]
    assert response.json()["username"] == test_user_data["username"]
    assert response.json()["full_name"] == test_user_data["full_name"]
    assert "password" not in response.json()
    assert "hashed_password" not in response.json()


def test_create_user_duplicate_username(test_user_data, network_codes):
    # Create first user
    response = client.post("/api/users/", json=test_user_data)
    assert response.status_code in network_codes["success"]

    response = client.post("/api/users/", json=test_user_data)
    assert response.status_code in network_codes["error"]
    assert response.json()["detail"] == "Username already registered"


def test_create_user_invalid_data(network_codes):
    invalid_data = {
        "username": "not_an_email",
        "password": "short",
        "full_name": "",
        "role": "invalid_role"
    }
    response = client.post("/api/users/", json=invalid_data)
    assert response.status_code in network_codes["error"]

# # READ USERS TESTS
def test_get_users(created_user, network_codes):
    response = client.get("/api/users/")
    assert response.status_code in network_codes["success"]
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

def test_get_users_pagination(test_user_data, network_codes):
    # Create multiple users
    for i in range(5):
        user_data = test_user_data.copy()
        user_data["username"] = f"test{i}@example.com"
        client.post("/api/users/", json=user_data)

    # Test pagination
    response = client.get("/api/users/?skip=1&limit=2")
    assert response.status_code in network_codes["success"]
    assert len(response.json()) == 2

def test_get_user_by_id(created_user, network_codes):
    user_id = created_user["id"]
    response = client.get(f"/api/users/{user_id}")
    assert response.status_code in network_codes["success"]
    assert response.json()["id"] == user_id
    assert response.json()["username"] == created_user["username"]

def test_get_user_not_found(network_codes):
    response = client.get("/api/users/999999")
    assert response.status_code in network_codes["error"]

def test_get_user_exists(created_user, network_codes):
    response = client.get(f"/api/users/{created_user['username']}/exists")
    assert response.status_code in network_codes["success"]
    assert response.json()["exists"] == True

def test_get_user_not_exists(network_codes):
    response = client.get("/api/users/999999/exists")
    assert response.status_code in network_codes["success"]
    assert response.json()["exists"] == False


# UPDATE USER TESTS
def test_update_user(created_user, network_codes):
    user_id = created_user["id"]
    update_data = {
        "full_name": "Updated Name",
        "password": "newpassword123"
    }
    response = client.put(f"/api/users/{user_id}", json=update_data)
    assert response.status_code in network_codes["success"]
    assert response.json()["full_name"] == "Updated Name"
    assert "password" not in response.json()

def test_update_user_not_found(network_codes):
    update_data = {"full_name": "Updated Name"}
    response = client.put("/api/users/999999", json=update_data)
    assert response.status_code in network_codes["error"]
