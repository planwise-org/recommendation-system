import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.models import UserRole



client = TestClient(app)
