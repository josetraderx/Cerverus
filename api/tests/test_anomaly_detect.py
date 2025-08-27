import json

from fastapi.testclient import TestClient

from api.app.main import app
from tools.gen_sample_data import generate

client = TestClient(app)


def test_detect_endpoint():
    # generate small dataset with injected outliers
    import subprocess
    import sys

    payload = subprocess.check_output([sys.executable, "tools/gen_sample_data.py"])
    obj = json.loads(payload)

    res = client.post("/api/v1/anomaly/detect", json=obj)
    assert res.status_code == 200
    data = res.json()
    assert "total_points" in data
    assert data["total_points"] > 0
    # expect at least 1 anomaly detected in the generated payload
    assert data["total_anomalies"] >= 1
