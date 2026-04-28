from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from ..models import build_model
from ..specs import TaskSpec


@dataclass
class TaskArtifact:
    task: TaskSpec
    directory: Optional[Path]
    checkpoint_path: Optional[Path]
    summary: Dict

    @property
    def available(self) -> bool:
        return self.checkpoint_path is not None and self.checkpoint_path.exists()


class ArtifactCatalog:
    def __init__(self, model_root: Path):
        self.model_root = Path(model_root)
        self._by_task_id: Dict[str, TaskArtifact] = {}

    def scan(self, task_catalog) -> None:
        discovered = {}
        if self.model_root.exists():
            for task_json in self.model_root.rglob("task.json"):
                try:
                    payload = json.loads(task_json.read_text(encoding="utf-8"))
                except Exception:
                    continue
                task_id = payload.get("task_id")
                if task_id:
                    directory = task_json.parent
                    summary_path = directory / "summary.json"
                    if summary_path.exists():
                        try:
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        except Exception:
                            summary = {}
                    else:
                        summary = {}
                    checkpoint_path = directory / "best_model.pt"
                    discovered[task_id] = (directory, checkpoint_path if checkpoint_path.exists() else None, summary)

        for task in task_catalog:
            directory, checkpoint, summary = discovered.get(task.task_id, (None, None, {}))
            self._by_task_id[task.task_id] = TaskArtifact(
                task=task,
                directory=directory,
                checkpoint_path=checkpoint,
                summary=summary,
            )

    def get(self, task_id: str) -> Optional[TaskArtifact]:
        return self._by_task_id.get(task_id)

    def checkpoint_index(self) -> Dict[str, str]:
        return {
            task_id: str(artifact.checkpoint_path)
            for task_id, artifact in self._by_task_id.items()
            if artifact.available
        }


class CheckpointRunner:
    def __init__(self, artifact: TaskArtifact, device: torch.device):
        if not artifact.available:
            raise FileNotFoundError(f"Checkpoint not found for {artifact.task.task_id}")
        self.artifact = artifact
        self.device = device
        self.model = self._load_model().to(device)
        self.model.eval()

    def _load_model(self):
        checkpoint = torch.load(self.artifact.checkpoint_path, map_location=self.device)
        model = build_model(self.artifact.task)
        model.load_state_dict(checkpoint["state_dict"])
        return model
