from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "server_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        # timestamp는 동적으로 생성되므로 값만 확인
        "timestamp": response.json()["timestamp"]
    }
