# api/tests/test_endpoints.py
import pytest
from fastapi.testclient import TestClient

from api.app.main import app


# Test client fixture
@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Cerverus API is running"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        assert data["endpoints"]["health"] == "/health"
        assert data["endpoints"]["anomaly_detection"] == "/api/v1/anomaly"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "environment" in data
        assert data["version"] == "0.1.0"

    def test_anomaly_health_endpoint(self, client):
        """Test anomaly service health endpoint."""
        response = client.get("/api/v1/anomaly/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "anomaly detection service is healthy"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json(self, client):
        """Test with malformed JSON."""
        response = client.post(
            "/api/v1/anomaly/detect",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test with missing required fields."""
        response = client.post(
            "/api/v1/anomaly/detect",
            json={"contamination": 0.1},  # Missing 'data' field
        )
        assert response.status_code == 422

    def test_wrong_data_types(self, client):
        """Test with wrong data types."""
        response = client.post(
            "/api/v1/anomaly/detect",
            json={"data": "not a list", "contamination": "not a float"},
        )
        assert response.status_code == 422


class TestBasicAnomalyEndpoint:
    """Basic tests for anomaly detection endpoint without ML execution."""

    def test_detect_anomalies_empty_data(self, client):
        """Test anomaly detection with empty data."""
        response = client.post(
            "/api/v1/anomaly/detect", json={"data": [], "contamination": 0.1}
        )
        # This should fail gracefully, either 400 or 500
        assert response.status_code in [400, 500]

    def test_detect_anomalies_insufficient_data(self, client):
        """Test anomaly detection with insufficient data points."""
        response = client.post(
            "/api/v1/anomaly/detect",
            json={"data": [{"value": 1.0}], "contamination": 0.1},
        )
        # Should get error from insufficient data
        assert response.status_code == 500
        if "detail" in response.json():
            assert "Insufficient data for training" in response.json()["detail"]
