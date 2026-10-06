from fastapi.testclient import TestClient

from mlzero.api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_create_run_invalid_path():
    response = client.post("/runs", json={"dataset_path": "/etc/shadow"})
    assert response.status_code == 400
    assert "outside allowed data root" in response.json()["detail"]

def test_create_run_not_exist():
    response = client.post("/runs", json={"dataset_path": "tests/data/does_not_exist"})
    assert response.status_code == 400
    assert "Invalid dataset directory" in response.json()["detail"]

def test_get_run_episodes_unknown():
    resp = client.get("/runs/unknown-run-id-12345/episodes")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Run not found"

def test_create_run_and_get():
    # Submit run with mock LLM
    response = client.post("/runs", json={
        "dataset_path": "tests/data/tiny_classification",
        "options": {"mock_llm": True}
    })
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    
    # Get status
    resp2 = client.get(f"/runs/{run_id}")
    assert resp2.status_code == 200
    assert resp2.json()["run_id"] == run_id
    
    # Get artifacts
    resp3 = client.get(f"/runs/{run_id}/artifacts")
    assert resp3.status_code == 200

    # Get episodes - should be empty list initially since run just queued/started
    resp4 = client.get(f"/runs/{run_id}/episodes")
    assert resp4.status_code == 200
    data = resp4.json()
    assert data["run_id"] == run_id
    assert "episodes" in data
    assert isinstance(data["episodes"], list)

def test_api_mock_llm_propagation(monkeypatch):
    from mlzero.application.manager import run_manager
    passed_options = []
    
    original_submit = run_manager.submit_run
    def mock_submit(dataset_path, user_instruction=None, options=None):
        passed_options.append(options)
        return original_submit(dataset_path, user_instruction, options)
        
    monkeypatch.setattr(run_manager, "submit_run", mock_submit)
    
    response = client.post("/runs", json={
        "dataset_path": "tests/data/tiny_classification",
        "options": {"mock_llm": True}
    })
    assert response.status_code == 200
    assert len(passed_options) == 1
    assert passed_options[0].get("mock_llm") is True


def test_cors_middleware():
    response = client.options(
        "/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://example.com"


def test_upload_dataset_csv(tmp_path):
    csv_content = b"feature1,feature2,target\n1.0,2.0,0\n3.0,4.0,1\n"
    response = client.post(
        "/datasets/upload",
        files={"file": ("sample.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "dataset_path" in data
    assert "sample.csv" in data["files"]
    assert "train.csv" in data["files"]


def test_upload_dataset_invalid_extension():
    response = client.post(
        "/datasets/upload",
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_download_predictions_not_found():
    response = client.get("/runs/non-existent-run-id/predictions/download")
    assert response.status_code == 404

