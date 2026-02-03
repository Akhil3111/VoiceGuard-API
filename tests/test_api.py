from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Voice Guard API is running", "env": settings.APP_ENV}

def test_analyze_endpoint_no_auth():
    response = client.post("/api/v1/analyze", json={"audio_path": "test.wav"})
    assert response.status_code == 422 # Missing header

def test_analyze_endpoint_invalid_auth():
    response = client.post("/api/v1/analyze", json={"audio_path": "test.wav"}, headers={"x-api-key": "wrong"})
    assert response.status_code == 403

def test_analyze_endpoint_valid_auth():
    response = client.post(
        "/api/v1/analyze", 
        json={"audio_path": "test.wav"}, 
        headers={"x-api-key": settings.API_KEY}
    )
    assert response.status_code == 200
    assert response.json()["is_deepfake"] is False
