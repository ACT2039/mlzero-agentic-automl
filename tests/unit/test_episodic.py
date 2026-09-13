"""Tests for Episodic Memory module."""
from tempfile import TemporaryDirectory

from mlzero.memory.episodic import EpisodicMemory
from mlzero.memory.episodic_store import EpisodicStore
from mlzero.schemas.coder import CodeArtifact, ErrorContext, ExecutionResult
from mlzero.schemas.episodic import Episode, RunHistory
from mlzero.schemas.perception import PerceptualContext


def test_schemas():
    ep = Episode(
        episode_id="e1",
        run_id="r1",
        iteration=1,
        timestamp="2025-01-01T00:00:00Z",
        status="SUCCESS"
    )
    assert ep.status == "SUCCESS"
    
    run = RunHistory(
        run_id="r1",
        start_time="2025-01-01T00:00:00Z",
        episodes=[ep]
    )
    assert len(run.episodes) == 1


def test_episodic_store_persistence():
    with TemporaryDirectory() as tmpdir:
        store = EpisodicStore(storage_dir=tmpdir)
        
        ep = Episode(
            episode_id="e1",
            run_id="r1",
            iteration=1,
            timestamp="2025-01-01T00:00:00Z",
            status="FAIL"
        )
        run = RunHistory(
            run_id="r1",
            start_time="2025-01-01T00:00:00Z",
        )
        
        # Save run
        store.save_run_history(run)
        
        # Add episode
        store.add_episode(ep)
        
        # Verify persistence and retrieval
        loaded_run = store.get_run_history("r1")
        assert loaded_run is not None
        assert len(loaded_run.episodes) == 1
        assert loaded_run.episodes[0].status == "FAIL"
        
        # Get episode
        loaded_ep = store.get_episode("r1", 1)
        assert loaded_ep is not None
        assert loaded_ep.episode_id == "e1"
        
        # Latest episode
        assert store.get_latest_episode("r1").episode_id == "e1"
        
        # List episodes
        assert len(store.list_episodes("r1")) == 1
        
        # Clear run
        store.clear_run("r1")
        assert store.get_run_history("r1") is None


def test_episodic_memory_service(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        from mlzero.core.config import settings
        monkeypatch.setattr(settings.episodic, "storage_dir", tmpdir)
        monkeypatch.setattr(settings.episodic, "max_episodes_in_context", 2)
        monkeypatch.setattr(settings.episodic, "max_stdout_stderr_chars", 10)
        
        store = EpisodicStore(storage_dir=tmpdir)
        mem = EpisodicMemory(store=store)
        
        run_id = "test_run"
        pctx = PerceptualContext()
        mem.start_run(run_id, pctx)
        
        # Record fail
        artifact = CodeArtifact(code="print('fail')", language="python")
        result = ExecutionResult(success=False, return_code=1, stdout="too long stdout that should be truncated", stderr="")
        error_ctx = ErrorContext(iteration=1, error_category="test", error_message="msg", stderr_excerpt="", suggested_fix="fix")
        
        mem.record_iteration(run_id, 1, "FAIL", artifact, result, error_ctx)
        
        # Record success
        mem.record_iteration(run_id, 2, "SUCCESS")
        
        # End run
        mem.end_run(run_id, "SUCCESS")
        
        # Verify latest
        assert mem.get_latest_failure(run_id).iteration == 1
        assert mem.get_latest_success(run_id).iteration == 2
        
        # Verify bounded context
        ctx = mem.get_bounded_context(run_id)
        assert ctx["run_id"] == run_id
        assert ctx["total_iterations_so_far"] == 2
        assert len(ctx["recent_history"]) == 2
        assert ctx["latest_failure"]["iteration"] == 1
        
        # Verify truncation
        fail_ep = mem.get_latest_failure(run_id)
        assert fail_ep.execution_result_summary["stdout_excerpt"] == "too long s..."
