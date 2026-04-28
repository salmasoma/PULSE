from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..specs import TaskSpec, TaskType
from .schemas import ToolResult

FETAL_VIEW_DISPLAY = {
    "abdomen": "fetal abdomen",
    "brain": "fetal brain",
    "femur": "fetal femur",
    "heart": "fetal heart",
    "cervix": "maternal cervix",
    "kidney": "fetal kidney",
    "lips_nose": "fetal lips/nose",
    "profile_patient": "fetal profile",
    "spine": "fetal spine",
}

FETAL_VIEW_TASK_IDS = {
    "brain": ["fetal/hc_measurement"],
    "abdomen": [],
    "heart": [],
    "femur": [],
    "cervix": [],
}

def _task_matches_intents(task: TaskSpec, intents: Sequence[str]) -> bool:
    if not intents:
        return True
    if task.task_type.value in intents:
        return True
    return "measurement" in intents and task.task_type in {TaskType.MEASUREMENT, TaskType.REGRESSION}


def fetal_view_from_result(result: ToolResult | None) -> str | None:
    if result is None or result.status != "completed":
        return None
    label_key = result.outputs.get("label_key")
    if isinstance(label_key, str) and label_key:
        return label_key
    label = str(result.outputs.get("label", "")).strip().lower()
    for key, display in FETAL_VIEW_DISPLAY.items():
        if label == display:
            return key
    return None


def build_fetal_view_plan(
    task_catalog: Sequence[TaskSpec],
    intents: Sequence[str],
    fetal_routing_results: Sequence[ToolResult],
    fetalnet_ready: bool = False,
) -> Tuple[str | None, List[TaskSpec], List[ToolResult], List[str]]:
    plane_result = next(
        (
            result
            for result in fetal_routing_results
            if result.task_id in {"fetalclip/plane_zero_shot", "fetalnet/view_classification"}
        ),
        None,
    )
    view_key = fetal_view_from_result(plane_result)
    if view_key is None:
        return None, [], [], []

    by_id = {task.task_id: task for task in task_catalog}
    actual_tasks: List[TaskSpec] = []
    plan_entries: List[str] = []
    virtual_results: List[ToolResult] = []

    task_ids = list(FETAL_VIEW_TASK_IDS.get(view_key, []))
    if fetalnet_ready and view_key == "brain":
        task_ids = [task_id for task_id in task_ids if task_id != "fetal/hc_measurement"]

    for task_id in task_ids:
        task = by_id.get(task_id)
        if task is None or not task.is_ready:
            continue
        if not _task_matches_intents(task, intents):
            continue
        actual_tasks.append(task)
        plan_entries.append(f"{task.title} [{task.task_type.value}]")

    return view_key, actual_tasks, virtual_results, plan_entries
