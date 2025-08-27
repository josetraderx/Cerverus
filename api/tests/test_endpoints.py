# api/tests/test_endpoints.py
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app.main import app


# Test client fixture
@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_valid_data():
    """Sample valid data for anomaly detection."""
    return {
        "data": [
            {"value": 1.2, "volume": 1000},
            {"value": 1.5, "volume": 1200},
            {"value": 10.5, "volume": 5000},  # This should be anomaly
            {"value": 1.3, "volume": 1100},
            {"value": 1.4, "volume": 1150},
        ],
        "contamination": 0.2,
    }


@pytest.fixture
def sample_invalid_data():
    """Sample invalid data (no numeric columns)."""
    return {
        "data": [
            {"name": "AAPL", "description": "Apple Inc"},
            {"name": "MSFT", "description": "Microsoft Corp"},
        ],
        "contamination": 0.1,
    }


@pytest.fixture
def sample_mixed_data():
    """Sample mixed data (numeric + non-numeric)."""
    return {
        "data": [
            {"symbol": "AAPL", "price": 150.0, "volume": 1000},
            {"symbol": "MSFT", "price": 300.0, "volume": 1200},
            {"symbol": "GOOGL", "price": 2500.0, "volume": 800},
        ],
        "contamination": 0.3,
    }


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


class TestAnomalyDetectionEndpoint:
    """Test anomaly detection functionality."""

    def test_detect_anomalies_valid_data(self, client, sample_valid_data):
        """Test anomaly detection with valid numeric data."""
        response = client.post("/api/v1/anomaly/detect", json=sample_valid_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "anomalies" in data
        assert "total_anomalies" in data
        assert "total_points" in data

        # Check data types
        assert isinstance(data["anomalies"], list)
        assert isinstance(data["total_anomalies"], int)
        assert isinstance(data["total_points"], int)

        # Check business logic
        assert data["total_points"] == len(sample_valid_data["data"])
        assert data["total_anomalies"] >= 0
        assert data["total_anomalies"] <= data["total_points"]

    def test_detect_anomalies_mixed_data(self, client, sample_mixed_data):
        """Test anomaly detection with mixed data types."""
        response = client.post("/api/v1/anomaly/detect", json=sample_mixed_data)

        assert response.status_code == 200
        data = response.json()

        # Should work with numeric columns only
        assert data["total_points"] == len(sample_mixed_data["data"])

        # Anomalies should contain all original columns
        if data["anomalies"]:
            anomaly = data["anomalies"][0]
            assert "symbol" in anomaly  # Non-numeric column preserved
            assert "price" in anomaly  # Numeric column preserved

    def test_detect_anomalies_no_numeric_data(self, client, sample_invalid_data):
        """Test anomaly detection with no numeric columns."""
        response = client.post("/api/v1/anomaly/detect", json=sample_invalid_data)

        assert response.status_code == 400
        assert "No numeric columns found" in response.json()["detail"]

    def test_detect_anomalies_empty_data(self, client):
        """Test anomaly detection with empty data."""
        response = client.post(
            "/api/v1/anomaly/detect", json={"data": [], "contamination": 0.1}
        )

        assert response.status_code == 500
        # Should get error from insufficient data

    def test_detect_anomalies_insufficient_data(self, client):
        """Test anomaly detection with insufficient data points."""
        response = client.post(
            "/api/v1/anomaly/detect",
            json={"data": [{"value": 1.0}], "contamination": 0.1},  # Only 1 data point
        )

        assert response.status_code == 500
        assert "Insufficient data for training" in response.json()["detail"]

    def test_detect_anomalies_invalid_contamination(self, client):
        """Test anomaly detection with invalid contamination values."""
        # Test contamination > 1
        response = client.post(
            "/api/v1/anomaly/detect",
            json={"data": [{"value": i} for i in range(20)], "contamination": 1.5},
        )
        assert response.status_code == 500

        # Test negative contamination
        response = client.post(
            "/api/v1/anomaly/detect",
            json={"data": [{"value": i} for i in range(20)], "contamination": -0.1},
        )
        assert response.status_code == 500

    def test_detect_anomalies_default_contamination(self, client):
        """Test anomaly detection with default contamination value."""
        response = client.post(
            "/api/v1/anomaly/detect",
            json={
                "data": [{"value": i, "volume": 1000 + i} for i in range(50)]
                # No contamination specified, should use default 0.01
            },
        )

        assert response.status_code == 200
        # With default contamination of 0.01 and 50 points,
        # should detect ~1 anomaly

    def test_detect_anomalies_large_dataset(self, client):
        """Test anomaly detection with larger dataset."""
        # Generate 1000 data points
        large_data = {
            "data": [
                {
                    "price": 100 + np.random.normal(0, 1),
                    "volume": 1000 + int(np.random.normal(0, 100)),
                }
                for _ in range(1000)
            ],
            "contamination": 0.05,
        }

        response = client.post("/api/v1/anomaly/detect", json=large_data)

        assert response.status_code == 200
        data = response.json()

        # With 1000 points and 0.05 contamination, expect ~50 anomalies
        assert data["total_points"] == 1000
        assert 30 <= data["total_anomalies"] <= 70  # Reasonable range


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


class TestIntegration:
    """Integration tests that verify end-to-end functionality."""

    def test_realistic_financial_data(self, client):
        """Test with realistic financial data structure."""
        financial_data = {
            "data": [
                {
                    "Open": 150.0,
                    "High": 155.0,
                    "Low": 149.0,
                    "Close": 152.0,
                    "Volume": 1000000,
                },
                {
                    "Open": 152.0,
                    "High": 154.0,
                    "Low": 151.0,
                    "Close": 153.0,
                    "Volume": 1100000,
                },
                {
                    "Open": 153.0,
                    "High": 200.0,
                    "Low": 152.0,
                    "Close": 195.0,
                    "Volume": 5000000,
                },  # Anomaly
                {
                    "Open": 153.0,
                    "High": 155.0,
                    "Low": 152.0,
                    "Close": 154.0,
                    "Volume": 1050000,
                },
                {
                    "Open": 154.0,
                    "High": 156.0,
                    "Low": 153.0,
                    "Close": 155.0,
                    "Volume": 1020000,
                },
            ],
            "contamination": 0.2,
        }

        response = client.post("/api/v1/anomaly/detect", json=financial_data)

        assert response.status_code == 200
        data = response.json()

        # Should detect the obvious anomaly (High=200, Close=195, Volume=5M)
        assert data["total_anomalies"] >= 1

        # Check that anomaly data contains all original columns
        if data["anomalies"]:
            anomaly = data["anomalies"][0]
            assert "Open" in anomaly
            assert "High" in anomaly
            assert "Close" in anomaly
            assert "Volume" in anomaly


# Performance tests (optional, can be slow)
@pytest.mark.slow
class TestPerformance:
    """Performance tests for anomaly detection."""

    def test_response_time_small_dataset(self, client):
        """Test response time with small dataset."""
        import time

        data = {"data": [{"value": i} for i in range(100)], "contamination": 0.1}

        start_time = time.time()
        response = client.post("/api/v1/anomaly/detect", json=data)
        end_time = time.time()

        assert response.status_code == 200
        assert end_time - start_time < 5.0  # Should complete within 5 seconds

    def test_memory_usage_large_dataset(self, client):
        """Test memory usage doesn't explode with larger datasets."""
        # This test would need memory profiling tools in a real scenario
        data = {
            "data": [{"value": i, "volume": i * 100} for i in range(10000)],
            "contamination": 0.01,
        }

        response = client.post("/api/v1/anomaly/detect", json=data)
        assert response.status_code == 200

        # Basic check that response is reasonable
        result = response.json()
        assert result["total_points"] == 10000
