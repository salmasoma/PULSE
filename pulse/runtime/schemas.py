from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..specs import TaskSpec


@dataclass
class AnalysisContext:
    request_id: str
    prompt: str
    primary_image: Path
    extra_images: Dict[str, Path]
    data_root: Path
    model_root: Path
    runtime_root: Path
    task_catalog: List[TaskSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain_hint: Optional[str] = None
    detected_domain: Optional[str] = None
    detected_anatomy: Optional[str] = None
    routing_task_id: Optional[str] = None
    routing_label: Optional[str] = None
    domain_scores: Dict[str, float] = field(default_factory=dict)
    task_hints: List[str] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    plan: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool_name: str
    task_id: Optional[str]
    status: str
    summary: str
    domain: Optional[str] = None
    confidence: Optional[float] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "domain": self.domain,
            "confidence": self.confidence,
            "outputs": dict(self.outputs),
            "artifacts": dict(self.artifacts),
            "errors": list(self.errors),
        }
