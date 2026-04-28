#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulse.discovery import discover_all_tasks
from pulse.runtime.service import PULSEInferenceService, RuntimeConfig
from pulse.specs import TaskSpec, TaskType

DEFAULT_PROMPT = "Analyze this ultrasound scan and run all applicable specialist models."
CASE_SPLIT_ORDER = ("test", "val", "train")
CLASS_TASK_SOURCE_EXCLUSIONS = {"system/domain_classification"}
CURATED_LABEL_SAMPLE_BASENAMES = {
    ("fetal/plane_classification", "Maternal cervix"): "Patient00333_Plane4_1_of_1.png",
}


@dataclass
class CaseRequest:
    case_id: str
    domain: str
    source_task_id: str
    source_task_title: str
    source_task_type: str
    source_split: str
    source_label: str | None
    primary_image: Path
    extra_images: Dict[str, Path]
    metadata: Dict[str, Any]
    expected: Dict[str, Any]


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "case"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sample_splits(task: TaskSpec) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
    for split_name in CASE_SPLIT_ORDER:
        yield split_name, list(getattr(task, f"{split_name}_samples"))


def _sample_has_mask(sample: Dict[str, Any]) -> bool:
    if sample.get("mask_path"):
        return True
    if sample.get("mask_paths"):
        return bool(sample["mask_paths"])
    if sample.get("mask_polygons"):
        return bool(sample["mask_polygons"])
    return False


def _pick_label_sample(task: TaskSpec, label_index: int, label_name: str) -> Tuple[str, Dict[str, Any]] | None:
    preferred_basename = CURATED_LABEL_SAMPLE_BASENAMES.get((task.task_id, label_name))
    for split_name, samples in _sample_splits(task):
        matching = [sample for sample in samples if int(sample.get("label", -1)) == label_index]
        if not matching:
            continue
        if task.task_type == TaskType.MULTIMODAL:
            matching = [sample for sample in matching if sample.get("modalities", {}).get("bmode")]
        if preferred_basename:
            preferred = [
                sample for sample in matching
                if Path(str(sample.get("image", ""))).name == preferred_basename
            ]
            if preferred:
                return split_name, preferred[0]
        if matching:
            return split_name, matching[0]
    return None


def _pick_representative_sample(task: TaskSpec) -> Tuple[str, Dict[str, Any]] | None:
    for split_name, samples in _sample_splits(task):
        if not samples:
            continue
        if task.task_type == TaskType.SEGMENTATION:
            annotated = [sample for sample in samples if _sample_has_mask(sample)]
            if annotated:
                return split_name, annotated[0]
        if task.task_type == TaskType.DETECTION:
            boxed = [sample for sample in samples if sample.get("bbox")]
            if boxed:
                return split_name, boxed[0]
        return split_name, samples[0]
    return None


def _sample_inputs(sample: Dict[str, Any]) -> Tuple[Path, Dict[str, Path], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    if "modalities" in sample:
        modalities = {name: Path(path).resolve() for name, path in sample["modalities"].items()}
        primary = modalities["bmode"]
        extra = {name: path for name, path in modalities.items() if name != "bmode"}
    else:
        primary = Path(sample["image"]).resolve()
        extra = {}
    if "pixel_size" in sample:
        metadata["pixel_spacing_mm"] = sample["pixel_size"]
    return primary, extra, metadata


def _expected_payload(task: TaskSpec, sample: Dict[str, Any], split_name: str, source_label: str | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source_task_id": task.task_id,
        "source_task_title": task.title,
        "source_task_type": task.task_type.value,
        "source_domain": task.domain,
        "source_split": split_name,
        "source_label": source_label,
        "patient_id": sample.get("patient_id"),
    }
    if "label_name" in sample:
        payload["source_label_name"] = sample["label_name"]
    if "target" in sample:
        payload["target"] = sample["target"]
    if "pixel_size" in sample:
        payload["pixel_spacing_mm"] = sample["pixel_size"]
    if "bbox" in sample:
        payload["bbox"] = sample["bbox"]
    if sample.get("mask_path"):
        payload["has_mask_path"] = True
    if sample.get("mask_paths"):
        payload["mask_path_count"] = len(sample["mask_paths"])
    if sample.get("mask_polygons"):
        payload["mask_polygon_count"] = len(sample["mask_polygons"])
    if sample.get("modalities"):
        payload["modalities"] = sorted(sample["modalities"].keys())
    return payload


def build_case_requests(tasks: Sequence[TaskSpec]) -> List[CaseRequest]:
    cases: List[CaseRequest] = []
    for task in tasks:
        if not task.is_ready:
            continue
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL} and task.labels:
            if task.task_id in CLASS_TASK_SOURCE_EXCLUSIONS:
                continue
            for label_index, label_name in enumerate(task.labels):
                picked = _pick_label_sample(task, label_index, label_name)
                if picked is None:
                    continue
                split_name, sample = picked
                primary_image, extra_images, metadata = _sample_inputs(sample)
                cases.append(
                    CaseRequest(
                        case_id=f"{task.output_name}__{_slugify(label_name)}",
                        domain=task.domain,
                        source_task_id=task.task_id,
                        source_task_title=task.title,
                        source_task_type=task.task_type.value,
                        source_split=split_name,
                        source_label=label_name,
                        primary_image=primary_image,
                        extra_images=extra_images,
                        metadata=metadata,
                        expected=_expected_payload(task, sample, split_name, label_name),
                    )
                )
            continue

        picked = _pick_representative_sample(task)
        if picked is None:
            continue
        split_name, sample = picked
        primary_image, extra_images, metadata = _sample_inputs(sample)
        cases.append(
            CaseRequest(
                case_id=f"{task.output_name}__example",
                domain=task.domain,
                source_task_id=task.task_id,
                source_task_title=task.title,
                source_task_type=task.task_type.value,
                source_split=split_name,
                source_label=None,
                primary_image=primary_image,
                extra_images=extra_images,
                metadata=metadata,
                expected=_expected_payload(task, sample, split_name, None),
            )
        )
    return cases


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _artifact_source_from_reference(reference: str, request_root: Path, request_id: str) -> Path | None:
    ref = str(reference)
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    generated_prefix = f"/generated/{request_id}/"
    if ref.startswith(generated_prefix):
        candidate = request_root / ref[len(generated_prefix):]
        if candidate.exists():
            return candidate
    return None


def _copy_runtime_output(request_root: Path, output_root: Path, payload: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if not request_root.exists():
        return
    core_files = [
        request_root / "response.json",
        request_root / "clinical_report.md",
    ]
    for item in core_files:
        if item.exists():
            _copy_file(item, output_root / item.name)

    request_id = str(payload.get("request_id", ""))
    artifact_sources: List[Path] = []
    for result in payload.get("results", []):
        for reference in result.get("artifacts", {}).values():
            source = _artifact_source_from_reference(str(reference), request_root, request_id)
            if source is not None:
                artifact_sources.append(source)

    for source in sorted(set(artifact_sources)):
        if source.exists() and source.is_file():
            _copy_file(source, output_root / source.relative_to(request_root))


def _list_relative_files(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )


def _build_case_record(
    case: CaseRequest,
    case_dir: Path,
    prompt: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    output_root = case_dir / "output"
    task_ids = [item.get("task_id") for item in payload.get("results", []) if item.get("task_id")]
    status_counts: Dict[str, int] = {}
    for item in payload.get("results", []):
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "case_id": case.case_id,
        "case_folder": str(case_dir),
        "prompt": prompt,
        "source": {
            "domain": case.domain,
            "task_id": case.source_task_id,
            "task_title": case.source_task_title,
            "task_type": case.source_task_type,
            "split": case.source_split,
            "label": case.source_label,
        },
        "expected": _json_ready(case.expected),
        "inputs": {
            "primary_image": "input/primary" + case.primary_image.suffix.lower(),
            "extra_images": {
                name: f"input/extra/{_slugify(name)}{path.suffix.lower()}"
                for name, path in case.extra_images.items()
            },
        },
        "runtime": {
            "request_id": payload.get("request_id"),
            "detected_domain": payload.get("detected_domain"),
            "detected_anatomy": payload.get("detected_anatomy"),
            "routing_task_id": payload.get("routing_task_id"),
            "routing_label": payload.get("routing_label"),
            "domain_scores": payload.get("domain_scores"),
            "plan": payload.get("plan"),
            "result_task_ids": task_ids,
            "status_counts": status_counts,
        },
        "files": {
            "runtime_response": "output/response.json",
            "clinical_report": "output/clinical_report.md",
            "all_output_files": _list_relative_files(output_root),
        },
    }


def export_casebook(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    model_root = Path(args.model_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = discover_all_tasks(data_root, seed=args.seed)
    cases = build_case_requests(tasks)
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    service = PULSEInferenceService(
        RuntimeConfig(
            data_root=str(data_root),
            model_root=str(model_root),
            runtime_root=str(runtime_root),
            device=args.device,
            seed=args.seed,
            fetalclip_enabled=not args.disable_fetalclip,
            fetalclip_weights=args.fetalclip_weights,
            fetalnet_enabled=not args.disable_fetalnet,
            fetalnet_repo_root=args.fetalnet_repo,
            fetalnet_weights=args.fetalnet_weights,
            roboflow_brain_enabled=not args.disable_roboflow,
            roboflow_api_url=args.roboflow_api_url,
            roboflow_api_key=args.roboflow_api_key,
            roboflow_brain_model_id=args.roboflow_model_id,
        )
    )

    index_rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        prompt = args.prompt
        case_dir = output_root / case.domain / case.case_id
        if case_dir.exists() and args.overwrite:
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        input_root = case_dir / "input"
        _copy_file(case.primary_image, input_root / f"primary{case.primary_image.suffix.lower()}")
        for name, path in case.extra_images.items():
            _copy_file(path, input_root / "extra" / f"{_slugify(name)}{path.suffix.lower()}")

        print(f"[{index:03d}/{len(cases):03d}] {case.case_id}", flush=True)
        payload = service.analyze(
            primary_image=case.primary_image,
            prompt=prompt,
            extra_images=case.extra_images,
            metadata=case.metadata,
        )

        request_root = runtime_root / str(payload["request_id"])
        _copy_runtime_output(request_root, case_dir / "output", payload)

        case_record = _build_case_record(case, case_dir, prompt, payload)
        with open(case_dir / "case.json", "w", encoding="utf-8") as handle:
            json.dump(_json_ready(case_record), handle, indent=2)

        index_rows.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "source_task_id": case.source_task_id,
                "source_split": case.source_split,
                "source_label": case.source_label or "",
                "detected_domain": payload.get("detected_domain") or "",
                "routing_task_id": payload.get("routing_task_id") or "",
                "case_folder": str(case_dir),
                "result_task_ids": ";".join(item for item in case_record["runtime"]["result_task_ids"] if item),
                "completed": case_record["runtime"]["status_counts"].get("completed", 0),
                "skipped": case_record["runtime"]["status_counts"].get("skipped", 0),
                "error": case_record["runtime"]["status_counts"].get("error", 0),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "model_root": str(model_root),
        "runtime_root": str(runtime_root),
        "output_root": str(output_root),
        "prompt": args.prompt,
        "case_count": len(index_rows),
        "cases": index_rows,
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with open(output_root / "index.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "domain",
                "source_task_id",
                "source_split",
                "source_label",
                "detected_domain",
                "routing_task_id",
                "case_folder",
                "result_task_ids",
                "completed",
                "skipped",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a casebook of inference examples across PULSE tasks.")
    parser.add_argument("--data-root", default="Datasets")
    parser.add_argument("--model-root", default="runs/pulse_retrain_new")
    parser.add_argument("--runtime-root", default="runs/pulse_casebook_runtime")
    parser.add_argument("--output-dir", default="exports/pulse_inference_casebook")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-fetalclip", action="store_true")
    parser.add_argument("--fetalclip-weights", default=None)
    parser.add_argument("--disable-fetalnet", action="store_true")
    parser.add_argument("--fetalnet-repo", default=None)
    parser.add_argument("--fetalnet-weights", default=None)
    parser.add_argument("--disable-roboflow", action="store_true")
    parser.add_argument("--roboflow-api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--roboflow-api-key", default=None)
    parser.add_argument("--roboflow-model-id", default="fetal-brain-abnormalities-ultrasound/1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_casebook(args)


if __name__ == "__main__":
    main()
