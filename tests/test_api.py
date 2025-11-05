from fastapi.testclient import TestClient
from serve import app

def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200