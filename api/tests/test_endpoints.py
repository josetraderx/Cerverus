from fastapi.testclient import TestClient

from api.app.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Cerverus API is running" in response.json()["message"]
