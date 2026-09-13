"""
Local persistent storage for Episodic Memory.
"""
import json
from pathlib import Path

from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.schemas.episodic import Episode, RunHistory

logger = setup_logger(__name__)


class EpisodicStore:
    """Local JSON-based store for RunHistory and Episodes."""
    
    def __init__(self, storage_dir: str | Path | None = None) -> None:
        if storage_dir is None:
            self.storage_dir = Path(settings.episodic.storage_dir)
        else:
            self.storage_dir = Path(storage_dir)
            
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_run_path(self, run_id: str) -> Path:
        return self.storage_dir / f"{run_id}.json"
        
    def save_run_history(self, run: RunHistory) -> None:
        """Save a complete run history to disk."""
        path = self._get_run_path(run.run_id)
        try:
            path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save run history {run.run_id}: {e}")
            
    def get_run_history(self, run_id: str) -> RunHistory | None:
        """Load a run history from disk."""
        path = self._get_run_path(run_id)
        if not path.exists():
            return None
            
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RunHistory(**data)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load run history {run_id}: {e}")
            return None

    def add_episode(self, episode: Episode) -> None:
        """Add an episode to a run's history."""
        run = self.get_run_history(episode.run_id)
        if not run:
            logger.error(f"Cannot add episode: Run {episode.run_id} does not exist.")
            return
            
        run.episodes.append(episode)
        self.save_run_history(run)
        
    def get_episode(self, run_id: str, iteration: int) -> Episode | None:
        """Get a specific episode by iteration number."""
        run = self.get_run_history(run_id)
        if not run:
            return None
            
        for ep in run.episodes:
            if ep.iteration == iteration:
                return ep
        return None
        
    def list_episodes(self, run_id: str) -> list[Episode]:
        """List all episodes for a run."""
        run = self.get_run_history(run_id)
        return run.episodes if run else []
        
    def get_latest_episode(self, run_id: str) -> Episode | None:
        """Get the most recent episode for a run."""
        run = self.get_run_history(run_id)
        if not run or not run.episodes:
            return None
        return run.episodes[-1]
        
    def clear_run(self, run_id: str) -> None:
        """Delete a run's history."""
        path = self._get_run_path(run_id)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to clear run {run_id}: {e}")
