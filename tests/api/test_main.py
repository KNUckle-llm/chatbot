from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    
    assert json_data["status"] == "ok"
    assert "server_name" in json_data
    assert "version" in json_data
    assert "timestamp" in json_data