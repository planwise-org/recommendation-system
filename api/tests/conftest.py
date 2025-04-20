import pytest
from src.models import UserRole
from src.database import init_db, engine, drop_all_tables
from sqlmodel import SQLModel


@pytest.fixture(scope="function", autouse=True)
def setup_test_db():
    init_db()  # Create all tables in in-memory SQLite
    yield
    SQLModel.metadata.drop_all(engine)
    drop_all_tables()


@pytest.fixture
def test_user_data():
    return {
        "username": "user1",
        "password": "testpassword123",
        "full_name": "Test User",
        "role": UserRole.USER
    }

@pytest.fixture
def network_codes():
    return {
        "success": [200, 201],
        "error": [400, 401, 404, 422]
    }
