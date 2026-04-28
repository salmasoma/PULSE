#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402

from pulse.coreml_export import convert_checkpoint_to_coreml, load_checkpoint_record  # noqa: E402
from pulse.data import build_dataloaders  # noqa: E402
from pulse.discovery import discover_all_tasks  # noqa: E402
from pulse.metrics import classification_metrics, detection_metrics, regression_metrics, segmentation_metrics  # noqa: E402
from pulse.specs import TaskSpec, TaskType  # noqa: E402
from pulse.trainer import PULSETrainer, TrainConfig  # noqa: E402


CLASSIFICATION_KEYS = ["loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]
SEGMENTATION_KEYS = ["loss", "dice", "iou"]
DETECTION_KEYS = ["loss", "mean_iou"]
REGRESSION_KEYS = ["loss", "mae", "rmse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a baseline/distilled/quantized comparison report for PULSE tasks.")
    parser.add_argument("--data-root", default="Datasets", help="Dataset root used for task discovery.")
    parser.add_argument("--teacher-root", default="runs/pulse_retrain_new", help="Teacher checkpoint run directory.")
    parser.add_argument("--distilled-root", default="Distilled/distilled", help="Directory containing distilled checkpoints and distillation reports.")
    parser.add_argument(
        "--quantized-export-dir",
        default="tmp/quantized_coreml_eval",
        help="Directory used to write temporary float16 Core ML exports for evaluation.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/model_compression",
        help="Directory where LaTeX/JSON/CSV outputs are written.",
    )
    parser.add_argument("--minimum-ios", type=int, default=16, help="Minimum iOS target used for Core ML export.")
    parser.add_argument(
        "--compute-precision",
        choices=["float16", "float32"],
        default="float16",
        help="Core ML compute precision used for the quantized comparison stage.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers for evaluation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for Core ML evaluation. Use 1 for deterministic comparison.")
    parser.add_argument("--limit-tasks", nargs="*", default=None, help="Optional subset of task ids to evaluate.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_tasks(data_root: Path) -> Dict[str, TaskSpec]:
    tasks = {task.task_id: task for task in discover_all_tasks(data_root)}
    return tasks


def load_distillation_reports(distilled_root: Path) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    for path in sorted(distilled_root.glob("*/distillation_report.json")):
        payload = load_json(path)
        reports[str(payload["task_id"])] = payload
    return reports


def load_teacher_summaries(teacher_root: Path) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for path in sorted(teacher_root.glob("*/summary.json")):
        payload = load_json(path)
        task_id = str(payload.get("task_id", ""))
        if task_id:
            summaries[task_id] = payload
    return summaries


def package_size_mb(path: Path) -> float:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total / (1024.0 * 1024.0)


def export_quantized_package(
    task_id: str,
    distilled_root: Path,
    export_dir: Path,
    minimum_ios: int,
    compute_precision: str,
) -> Path:
    checkpoint_path = distilled_root / task_id.replace("/", "_")
    if not checkpoint_path.exists():
        checkpoint_path = None
        for candidate in distilled_root.glob("*/student_best.pt"):
            if candidate.parent.name == task_id.replace("/", "_") or candidate.parent.name == task_id.replace("/", "_").replace("__", "_"):
                checkpoint_path = candidate
                break
    else:
        checkpoint_path = checkpoint_path / "student_best.pt"
    if checkpoint_path is None or not checkpoint_path.exists():
        normalized = task_id.replace("/", "_")
        fallback = distilled_root / normalized / "student_best.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Student checkpoint not found for {task_id}")

    record = load_checkpoint_record(checkpoint_path)
    output_path = export_dir / f"{record.task.output_name}.mlpackage"
    if output_path.exists():
        return output_path
    export_dir.mkdir(parents=True, exist_ok=True)
    convert_checkpoint_to_coreml(
        record,
        output_dir=export_dir,
        minimum_ios_version=minimum_ios,
        compute_precision=compute_precision,
    )
    return output_path


def _load_rgb(path: str, image_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)


def build_coreml_inputs(task: TaskSpec, meta: Dict[str, Any], image_size: int) -> Dict[str, Image.Image]:
    if task.task_type == TaskType.MULTIMODAL:
        names = list(task.extras.get("modalities", []))
        return {name: _load_rgb(meta["modalities"][name], image_size) for name in names}
    return {"image": _load_rgb(meta["image"], image_size)}


def slice_batch_value(value: Any, sample_index: int) -> Any:
    if torch.is_tensor(value):
        return value[sample_index : sample_index + 1]
    if isinstance(value, dict):
        return {key: slice_batch_value(item, sample_index) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [value[sample_index]]
    return value


def coreml_output_to_torch(task: TaskSpec, prediction: Dict[str, Any], batch: Dict[str, Any]):
    key = next(iter(prediction.keys()))
    array = np.asarray(prediction[key], dtype=np.float32)
    tensor = torch.from_numpy(array)

    if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    if task.task_type == TaskType.SEGMENTATION:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    if task.task_type == TaskType.DETECTION:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return {
            "objectness": tensor[:, :1],
            "bbox": tensor[:, 1:5],
        }

    if tensor.ndim == 0:
        tensor = tensor.view(1)
    elif tensor.ndim > 1:
        tensor = tensor.reshape(batch["target"].shape)
    return tensor


def evaluate_coreml_task(
    task: TaskSpec,
    package_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    model = ct.models.MLModel(str(package_path))
    config = TrainConfig(batch_size=batch_size, image_size=image_size, num_workers=num_workers, device="cpu")
    trainer = PULSETrainer(config)
    class_weights = trainer._classification_class_weights(task, config)
    _, _, test_loader = build_dataloaders(
        task,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        balanced_sampling=False,
    )
    if test_loader is None:
        return {}

    total_loss = 0.0
    total_items = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    segmentation_totals = {"dice": 0.0, "iou": 0.0}
    segmentation_batches = 0
    detection_ious: List[float] = []
    regression_preds: List[torch.Tensor] = []
    regression_targets: List[torch.Tensor] = []

    for batch in test_loader:
        batch_size_actual = 1
        metas = batch["meta"]
        outputs_list = []
        for sample_index, meta in enumerate(metas):
            inputs = build_coreml_inputs(task, meta, image_size)
            prediction = model.predict(inputs)
            sample_batch = {
                key: slice_batch_value(value, sample_index)
                for key, value in batch.items()
                if key != "meta"
            }
            sample_batch["meta"] = [meta]
            outputs_list.append(coreml_output_to_torch(task, prediction, sample_batch))

        if task.task_type == TaskType.DETECTION:
            outputs = {
                "objectness": torch.cat([item["objectness"] for item in outputs_list], dim=0),
                "bbox": torch.cat([item["bbox"] for item in outputs_list], dim=0),
            }
        else:
            outputs = torch.cat(outputs_list, dim=0)

        loss = trainer._compute_loss(task, outputs, batch, class_weights=class_weights, config=config)
        if task.task_type == TaskType.MULTIMODAL:
            batch_size_actual = batch["label"].shape[0]
        else:
            for value in batch.values():
                if torch.is_tensor(value):
                    batch_size_actual = value.shape[0]
                    break
        total_loss += float(loss.item()) * batch_size_actual
        total_items += batch_size_actual

        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch["label"].tolist())
        elif task.task_type == TaskType.SEGMENTATION:
            metrics = segmentation_metrics(
                outputs,
                batch["mask"],
                task.extras.get("segmentation_mode", "binary"),
                len(task.extras.get("mask_class_names", [])),
            )
            segmentation_totals["dice"] += metrics["dice"]
            segmentation_totals["iou"] += metrics["iou"]
            segmentation_batches += 1
        elif task.task_type == TaskType.DETECTION:
            metrics = detection_metrics(outputs["bbox"], batch["bbox"])
            detection_ious.append(metrics["mean_iou"])
        else:
            regression_preds.append(outputs.detach().cpu())
            regression_targets.append(batch["target"].detach().cpu())

    metrics: Dict[str, float] = {"loss": total_loss / max(total_items, 1)}
    if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
        metrics.update(classification_metrics(all_preds, all_targets, len(task.labels)))
    elif task.task_type == TaskType.SEGMENTATION:
        metrics.update(
            {
                "dice": segmentation_totals["dice"] / max(segmentation_batches, 1),
                "iou": segmentation_totals["iou"] / max(segmentation_batches, 1),
            }
        )
    elif task.task_type == TaskType.DETECTION:
        metrics["mean_iou"] = sum(detection_ious) / max(len(detection_ious), 1)
    else:
        preds = torch.cat(regression_preds) if regression_preds else torch.empty(0)
        targets = torch.cat(regression_targets) if regression_targets else torch.empty(0)
        metrics.update(regression_metrics(preds, targets))
    return metrics


def metric_keys_for_task(task: TaskSpec) -> List[str]:
    if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
        return CLASSIFICATION_KEYS
    if task.task_type == TaskType.SEGMENTATION:
        return SEGMENTATION_KEYS
    if task.task_type == TaskType.DETECTION:
        return DETECTION_KEYS
    return REGRESSION_KEYS


def primary_metric_direction(task: TaskSpec) -> str:
    if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL, TaskType.SEGMENTATION, TaskType.DETECTION}:
        return "up"
    return "down"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def pretty_metric_name(key: str) -> str:
    mapping = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "weighted_f1": "Weighted-F1",
        "balanced_accuracy": "Balanced Acc.",
        "dice": "Dice",
        "iou": "IoU",
        "mean_iou": "Mean IoU",
        "mae": "MAE",
        "rmse": "RMSE",
    }
    return mapping.get(key, key.replace("_", " ").title())


def detailed_table_section(title: str, task_rows: List[Dict[str, Any]], metric_keys: List[str]) -> str:
    if not task_rows:
        return ""

    column_spec = "l" + "r" * (len(metric_keys) * 3)
    header_metrics = " & ".join([f"\\multicolumn{{3}}{{c}}{{{latex_escape(pretty_metric_name(key))}}}" for key in metric_keys])
    sub_header = "Task " + "".join([" & T & S & Q" for _ in metric_keys])

    lines = [
        "\\begin{landscape}",
        f"\\subsection*{{{latex_escape(title)}}}",
        "\\begin{center}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        f" & {header_metrics} \\\\",
        f"{sub_header} \\\\",
        "\\midrule",
    ]
    for row in task_rows:
        values: List[str] = [latex_escape(row["task_id"])]
        for key in metric_keys:
            values.extend(
                [
                    fmt(row["teacher_metrics"].get(key)),
                    fmt(row["student_metrics"].get(key)),
                    fmt(row["quantized_metrics"].get(key)),
                ]
            )
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{center}",
            "\\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def build_tex_report(results: List[Dict[str, Any]], output_dir: Path, compute_precision: str) -> str:
    summary_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{array}",
        r"\usepackage{graphicx}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\title{PULSE Compression Comparison Report}",
        r"\author{Generated from local checkpoint artifacts}",
        r"\date{\today}",
        r"\maketitle",
        r"\section*{Protocol}",
        (
            "Baseline metrics are taken from the teacher checkpoints before distillation. "
            "Distilled metrics are taken from the student checkpoints after knowledge distillation. "
            f"Quantized metrics are measured after Core ML export using {latex_escape(compute_precision)} compute precision "
            "and evaluating the exported models on the same test splits."
        ),
        (
            f"Full machine-readable metrics are written to \\texttt{{{latex_escape(str((output_dir / 'model_compression_comparison.json').resolve()))}}}."
        ),
        r"\section*{Summary}",
        r"\begin{landscape}",
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\rowcolors{2}{gray!6}{white}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Task & Type & Primary & Teacher & Student & Quantized & Student $\Delta$ & Quantized $\Delta$ & Q Size (MB) \\",
        r"\midrule",
    ]

    for row in results:
        direction = primary_metric_direction(row["task"])
        key = row["primary_metric"]
        teacher = float(row["teacher_metrics"].get(key, float("nan")))
        student = float(row["student_metrics"].get(key, float("nan")))
        quant = float(row["quantized_metrics"].get(key, float("nan")))
        if direction == "down":
            student_delta = teacher - student
            quant_delta = teacher - quant
        else:
            student_delta = student - teacher
            quant_delta = quant - teacher
        summary_lines.append(
            " & ".join(
                [
                    latex_escape(row["task_id"]),
                    latex_escape(row["task_type"]),
                    latex_escape(pretty_metric_name(key)),
                    fmt(teacher),
                    fmt(student),
                    fmt(quant),
                    fmt(student_delta),
                    fmt(quant_delta),
                    fmt(row["quantized_size_mb"], 3),
                ]
            )
            + r" \\"
        )

    summary_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{center}",
            r"\end{landscape}",
            "",
        ]
    )

    cls_rows = [row for row in results if row["task"].task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}]
    seg_rows = [row for row in results if row["task"].task_type == TaskType.SEGMENTATION]
    det_rows = [row for row in results if row["task"].task_type == TaskType.DETECTION]
    reg_rows = [row for row in results if row["task"].task_type not in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL, TaskType.SEGMENTATION, TaskType.DETECTION}]

    summary_lines.append(detailed_table_section("Classification and Multimodal Tasks", cls_rows, CLASSIFICATION_KEYS))
    summary_lines.append(detailed_table_section("Segmentation Tasks", seg_rows, SEGMENTATION_KEYS))
    summary_lines.append(detailed_table_section("Detection Tasks", det_rows, DETECTION_KEYS))
    summary_lines.append(detailed_table_section("Regression and Measurement Tasks", reg_rows, REGRESSION_KEYS))
    summary_lines.extend([r"\end{document}", ""])
    return "\n".join(summary_lines)


def write_full_metrics_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    metric_keys = sorted(
        {
            key
            for row in results
            for source in ("teacher_metrics", "student_metrics", "quantized_metrics")
            for key in row[source].keys()
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "task_id",
            "task_type",
            "primary_metric",
            "teacher_size_mb",
            "student_size_mb",
            "quantized_size_mb",
        ]
        for prefix in ("teacher", "student", "quantized"):
            header.extend([f"{prefix}_{key}" for key in metric_keys])
        writer.writerow(header)
        for row in results:
            record = [
                row["task_id"],
                row["task_type"],
                row["primary_metric"],
                row["size_mb"].get("teacher"),
                row["size_mb"].get("student"),
                row["quantized_size_mb"],
            ]
            for prefix in ("teacher_metrics", "student_metrics", "quantized_metrics"):
                record.extend([row[prefix].get(key) for key in metric_keys])
            writer.writerow(record)


def main() -> int:
    args = parse_args()
    data_root = (REPO_ROOT / args.data_root).resolve()
    teacher_root = (REPO_ROOT / args.teacher_root).resolve()
    distilled_root = (REPO_ROOT / args.distilled_root).resolve()
    export_dir = (REPO_ROOT / args.quantized_export_dir).resolve()
    report_dir = (REPO_ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(data_root)
    distillation_reports = load_distillation_reports(distilled_root)
    teacher_summaries = load_teacher_summaries(teacher_root)

    available_task_ids = sorted(set(distillation_reports).intersection(tasks))
    if args.limit_tasks:
        requested = set(args.limit_tasks)
        available_task_ids = [task_id for task_id in available_task_ids if task_id in requested]

    if not available_task_ids:
        raise SystemExit("No overlapping task ids were found across discovery and distillation reports.")

    results: List[Dict[str, Any]] = []

    for index, task_id in enumerate(available_task_ids, start=1):
        task = tasks[task_id]
        report = distillation_reports[task_id]
        summary = teacher_summaries.get(task_id, {})
        print(f"[{index}/{len(available_task_ids)}] {task_id}")
        package_path = export_quantized_package(
            task_id=task_id,
            distilled_root=distilled_root,
            export_dir=export_dir,
            minimum_ios=args.minimum_ios,
            compute_precision=args.compute_precision,
        )
        quantized_metrics = evaluate_coreml_task(
            task=task,
            package_path=package_path,
            image_size=int(task.extras.get("image_size", 224)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results.append(
            {
                "task_id": task_id,
                "task": task,
                "task_type": task.task_type.value,
                "primary_metric": task.primary_metric,
                "teacher_metrics": report["metrics"]["teacher"]["test"],
                "student_metrics": report["metrics"]["student"]["test"],
                "quantized_metrics": quantized_metrics,
                "teacher_summary_metrics": summary.get("test_metrics", {}),
                "parameters": report.get("parameters", {}),
                "size_mb": report.get("size_mb", {}),
                "quantized_size_mb": package_size_mb(package_path),
                "quantized_package": str(package_path),
            }
        )

    json_payload = {
        "name": "PULSE compression comparison",
        "teacher_root": str(teacher_root),
        "distilled_root": str(distilled_root),
        "quantized_export_dir": str(export_dir),
        "quantization": {
            "format": "Core ML mlprogram",
            "compute_precision": args.compute_precision,
            "minimum_ios": args.minimum_ios,
        },
        "tasks": [
            {
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "primary_metric": row["primary_metric"],
                "teacher_metrics": row["teacher_metrics"],
                "student_metrics": row["student_metrics"],
                "quantized_metrics": row["quantized_metrics"],
                "teacher_summary_metrics": row["teacher_summary_metrics"],
                "parameters": row["parameters"],
                "size_mb": row["size_mb"],
                "quantized_size_mb": row["quantized_size_mb"],
                "quantized_package": row["quantized_package"],
            }
            for row in results
        ],
    }

    json_path = report_dir / "model_compression_comparison.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    csv_path = report_dir / "model_compression_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "task_type",
                "primary_metric",
                "teacher_primary",
                "student_primary",
                "quantized_primary",
                "teacher_size_mb",
                "student_size_mb",
                "quantized_size_mb",
            ]
        )
        for row in results:
            primary = row["primary_metric"]
            writer.writerow(
                [
                    row["task_id"],
                    row["task_type"],
                    primary,
                    row["teacher_metrics"].get(primary),
                    row["student_metrics"].get(primary),
                    row["quantized_metrics"].get(primary),
                    row["size_mb"].get("teacher"),
                    row["size_mb"].get("student"),
                    row["quantized_size_mb"],
                ]
            )

    full_csv_path = report_dir / "model_compression_all_metrics.csv"
    write_full_metrics_csv(results, full_csv_path)

    tex_path = report_dir / "model_compression_comparison.tex"
    tex_path.write_text(build_tex_report(results, report_dir, args.compute_precision), encoding="utf-8")

    print(f"\nWrote:\n- {json_path}\n- {csv_path}\n- {full_csv_path}\n- {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
