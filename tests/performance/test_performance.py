import pytest
import requests
import time
import concurrent.futures

class TestCerverusPerformance:
    
    def test_api_response_time(self):
        """Test API response time under load."""
        def make_request():
            start_time = time.time()
            response = requests.get("http://localhost:8000/health")
            end_time = time.time()
            return response.status_code, end_time - start_time
            
        # Test concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [future.result() for future in futures]
            
        # Analyze results
        response_times = [result[1] for result in results]
        success_count = sum(1 for result in results if result[0] == 200)
        
        avg_response_time = sum(response_times) / len(response_times)
        
        assert success_count >= 95  # 95% success rate
        assert avg_response_time < 1.0  # Average response < 1 second
        
    def test_ml_prediction_performance(self):
        """Test ML prediction performance."""
        sample_data = {
            "features": {
                "amount": 1000.0,
                "frequency": 5,
                "time_of_day": 14,
                "day_of_week": 2
            }
        }
        
        # Test multiple predictions
        times = []
        for _ in range(50):
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/predict",
                json=sample_data
            )
            end_time = time.time()
            times.append(end_time - start_time)
            assert response.status_code == 200

        # Calculate average time after collecting all measurements
        avg_time = sum(times) / len(times)

        # ML predictions should be fast
        assert avg_time < 2.0  # Average prediction < 2 seconds
        assert max(times) < 5.0  # No prediction > 5 seconds


if __name__ == "__main__":
    pytest.main([__file__])