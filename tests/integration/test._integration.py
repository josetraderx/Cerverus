import pytest
import requests
import time
import psycopg2
from redis import Redis

class TestCerverusIntegration:
    
    @pytest.fixture(scope="class")
    def setup_services(self):
        """Setup y teardown de servicios para tests."""
        # Wait for services to be ready
        time.sleep(30)
        yield
        
    def test_postgres_connection(self, setup_services):
        """Test PostgreSQL connectivity."""
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="cerverus", 
            user="cerverus_user",
            password="cerverus_pass"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        conn.close()
        
    def test_redis_connection(self, setup_services):
        """Test Redis connectivity."""
        r = Redis(host='localhost', port=6379, db=0)
        assert r.ping()
        
    def test_api_health_endpoint(self, setup_services):
        """Test API health endpoint."""
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    def test_airflow_health(self, setup_services):
        """Test Airflow webserver."""
        response = requests.get("http://localhost:8080/health")
        assert response.status_code == 200
        
    def test_ml_algorithms_import(self, setup_services):
        """Test ML algorithms can be imported."""
        import sys
        sys.path.append('src')
        
        from cerverus.models.isolation_forest_eda import CerverusIsolationForest
        from cerverus.algorithms.meta_learner import MetaLearner
        
        # Test instantiation
        detector = CerverusIsolationForest()
        meta_learner = MetaLearner()
        
        assert detector is not None
        assert meta_learner is not None
        
    def test_end_to_end_prediction(self, setup_services):
        """Test end-to-end ML prediction via API."""
        # Sample data for prediction
        sample_data = {
            "features": {
                "amount": 1000.0,
                "frequency": 5,
                "time_of_day": 14,
                "day_of_week": 2
            }
        }
        
        response = requests.post(
            "http://localhost:8000/predict", 
            json=sample_data
        )
        assert response.status_code == 200
        
        result = response.json()
        assert "anomaly_score" in result
        assert "prediction" in result
        assert 0 <= result["anomaly_score"] <= 1