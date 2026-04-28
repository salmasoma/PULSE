from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from ..specs import TaskSpec, TaskType

PRIMARY_ROUTER_TASK_IDS = [
    "system/domain_classification",
]

FALLBACK_ROUTER_TASK_IDS = [
    "cardiac/butterfly_view_classification",
    "cardiac/butterfly_cardiac_screening",
    "abdominal/organ_classification",
    "carotid/lumen_segmentation",
]

DOMAIN_KEYWORDS = {
    "cardiac": ["cardiac", "heart", "echo", "ventricle", "apical", "plax"],
    "breast": ["breast", "lesion", "mass", "busi", "tumor"],
    "thyroid": ["thyroid", "nodule", "neck"],
    "fetal": ["fetal", "fetus", "obstetric", "pregnancy", "head circumference", "plane"],
    "abdominal": ["abdominal", "abdomen", "organ", "gallbladder", "portal", "spleen"],
    "liver": ["liver", "hepatic"],
    "kidney": ["kidney", "renal", "transplant kidney"],
    "pcos": ["pcos", "ovary", "ovarian"],
    "carotid": ["carotid", "imt", "vascular", "lumen"],
}

INTENT_KEYWORDS = {
    "segmentation": ["segment", "segmentation", "outline", "mask"],
    "detection": ["detect", "detection", "localize", "locate", "box"],
    "classification": ["classify", "classification", "benign", "malignant", "normal", "abnormal"],
    "measurement": ["measure", "measurement", "circumference", "imt", "thickness", "ga"],
    "multimodal": ["multimodal", "cdfi", "elastic", "elastography"],
}


def _keyword_matches(lowered: str, keyword: str) -> bool:
    keyword = keyword.lower().strip()
    if not keyword:
        return False
    if " " in keyword:
        return keyword in lowered
    return re.search(rf"\b{re.escape(keyword)}\b", lowered) is not None


def detect_domain_from_prompt(prompt: str) -> Tuple[str | None, Dict[str, float]]:
    lowered = prompt.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = float(sum(1 for keyword in keywords if _keyword_matches(lowered, keyword)))
        if score > 0:
            scores[domain] = score
    if not scores:
        return None, {}
    best_domain = max(scores.items(), key=lambda item: item[1])[0]
    return best_domain, scores


def detect_intents(prompt: str) -> List[str]:
    lowered = prompt.lower()
    intents = []
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(_keyword_matches(lowered, keyword) for keyword in keywords):
            intents.append(intent)
    return intents


def modality_requirements_satisfied(task: TaskSpec, available_modalities: Sequence[str]) -> bool:
    if task.task_type != TaskType.MULTIMODAL:
        return True
    required = set(task.extras.get("modalities", []))
    return required <= set(available_modalities)


def get_domain_router_tasks(tasks: Sequence[TaskSpec], available_modalities: Sequence[str]) -> List[TaskSpec]:
    by_id = {task.task_id: task for task in tasks if task.is_ready}
    selected = []
    for task_id in PRIMARY_ROUTER_TASK_IDS:
        task = by_id.get(task_id)
        if task and modality_requirements_satisfied(task, available_modalities):
            selected.append(task)

    for task_id in FALLBACK_ROUTER_TASK_IDS:
        task = by_id.get(task_id)
        if task and task not in selected and modality_requirements_satisfied(task, available_modalities):
            selected.append(task)
    return selected


def build_execution_plan(
    tasks: Sequence[TaskSpec],
    domain_hint: str | None,
    intents: Sequence[str],
    available_modalities: Sequence[str],
) -> List[TaskSpec]:
    scoped = [task for task in tasks if task.is_ready and task.domain != "system"]
    if domain_hint:
        domain_tasks = [task for task in scoped if task.domain == domain_hint]
        if domain_tasks:
            scoped = domain_tasks

    if intents:
        intent_matched = []
        for task in scoped:
            if task.task_type.value in intents:
                intent_matched.append(task)
            elif "measurement" in intents and task.task_type == TaskType.REGRESSION:
                intent_matched.append(task)
        if intent_matched:
            scoped = intent_matched

    scoped = [task for task in scoped if modality_requirements_satisfied(task, available_modalities)]

    task_priority = {
        TaskType.CLASSIFICATION: 0,
        TaskType.DETECTION: 1,
        TaskType.SEGMENTATION: 2,
        TaskType.MULTIMODAL: 3,
        TaskType.MEASUREMENT: 4,
        TaskType.REGRESSION: 4,
    }
    scoped.sort(key=lambda task: (task_priority.get(task.task_type, 9), task.task_id))
    return scoped
