from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"
    MEASUREMENT = "measurement"
    MULTIMODAL = "multimodal"


@dataclass
class TaskSpec:
    task_id: str
    domain: str
    title: str
    task_type: TaskType
    dataset_names: List[str]
    labels: List[str] = field(default_factory=list)
    description: str = ""
    train_samples: List[Dict[str, Any]] = field(default_factory=list)
    val_samples: List[Dict[str, Any]] = field(default_factory=list)
    test_samples: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ready"
    skip_reason: str = ""
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and bool(self.train_samples)

    @property
    def sample_count(self) -> int:
        return len(self.train_samples) + len(self.val_samples) + len(self.test_samples)

    @property
    def output_name(self) -> str:
        return self.task_id.replace("/", "_")

    @property
    def primary_metric(self) -> str:
        if self.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            return "macro_f1"
        if self.task_type == TaskType.SEGMENTATION:
            return "dice"
        if self.task_type == TaskType.DETECTION:
            return "mean_iou"
        return "mae"

    @property
    def primary_mode(self) -> str:
        return "min" if self.primary_metric == "mae" else "max"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "title": self.title,
            "task_type": self.task_type.value,
            "dataset_names": list(self.dataset_names),
            "labels": list(self.labels),
            "description": self.description,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "notes": list(self.notes),
            "extras": dict(self.extras),
            "counts": {
                "train": len(self.train_samples),
                "val": len(self.val_samples),
                "test": len(self.test_samples),
                "total": self.sample_count,
            },
        }
