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
