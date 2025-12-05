import pytest
from fastapi.testclient import TestClient
from api.app.main import app
from database.database import Base, engine
import os

# Ensure DB is enabled and using SQLite for this test
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["DISABLE_DB"] = "0"
os.environ["API_KEY"] = "test_api_key"
os.environ["TESTING"] = "1"  # Ensure main.py respects test env vars


@pytest.fixture(scope="module")  # Changed scope to module for efficient client reuse
def client():
    """Create a TestClient for the FastAPI app."""
    # The TestClient will automatically handle the app's lifespan events (like model loading)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("endpoint", ["/", "/health"])
def test_app_starts_and_model_loads(endpoint, client):
    """Test that the app starts and model is loaded successfully (lifespan runs)."""
    response = client.get(endpoint)
    assert response.status_code == 200
    if endpoint == "/health":
        assert response.json()["status"] == "ok"
