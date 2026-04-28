#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pulse.coreml_export import task_spec_from_dict  # noqa: E402
from pulse.data import ClassificationDataset, SegmentationDataset, _pulse_collate  # noqa: E402
from pulse.metrics import segmentation_metrics  # noqa: E402
from pulse.models import build_model  # noqa: E402
from pulse.specs import TaskSpec  # noqa: E402
from pulse.trainer import PULSETrainer, TrainConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate teacher and student checkpoints on the New_test_Datasets bundle.")
    parser.add_argument(
        "--datasets-root",
        default="/Users/salma.hassan/Downloads/New_test_Datasets",
        help="Root directory containing the unseen datasets.",
    )
    parser.add_argument(
        "--teacher-root",
        default="/Users/salma.hassan/Ultrasound_Project/runs/pulse_retrain_new",
        help="Root directory containing teacher checkpoints.",
    )
    parser.add_argument(
        "--student-root",
        default="/Users/salma.hassan/Ultrasound_Project/Distilled/distilled",
        help="Root directory containing student checkpoints.",
    )
    parser.add_argument(
        "--report-json",
        default="/Users/salma.hassan/Ultrasound_Project/reports/external_eval/new_test_datasets_eval.json",
        help="Where to save the machine-readable evaluation report.",
    )
    parser.add_argument(
        "--summary-csv",
        default="/Users/salma.hassan/Ultrasound_Project/reports/external_eval/new_test_datasets_eval_summary.csv",
        help="Where to save the summary CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--device", default="cpu", help="Torch device to use: cpu, cuda, mps, or auto.")
    parser.add_argument("--threads", type=int, default=1, help="Torch / BLAS thread count for reproducible evaluation.")
    return parser.parse_args()


def configure_reproducibility(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    torch.set_num_threads(max(1, threads))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, threads))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def is_mask_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_tumor") or "_other" in stem or stem.endswith("_mask")


def load_task_bundle(checkpoint_path: Path, device: str, batch_size: int, num_workers: int) -> tuple[dict[str, Any], TaskSpec, int, PULSETrainer, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task = task_spec_from_dict(dict(checkpoint["task"]))
    config = dict(checkpoint.get("config", {}))
    student_width = checkpoint.get("student_width")
    if student_width is not None:
        extras = dict(task.extras)
        extras["encoder_width"] = int(student_width)
        task.extras = extras
    elif "encoder_width" not in task.extras and config.get("encoder_width") is not None:
        extras = dict(task.extras)
        extras["encoder_width"] = int(config["encoder_width"])
        task.extras = extras
    image_size = int(config.get("image_size") or 224)
    trainer = PULSETrainer(
        TrainConfig(
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
    )
    model = build_model(task).to(trainer.device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, task, image_size, trainer, model


def build_classification_samples(task: TaskSpec, class_dirs: dict[str, list[Path]]) -> list[dict[str, Any]]:
    label_to_index = {label.lower(): index for index, label in enumerate(task.labels)}
    samples: list[dict[str, Any]] = []
    for label_name, directories in class_dirs.items():
        normalized = label_name.lower()
        if normalized not in label_to_index:
            raise ValueError(f"Label {label_name!r} not present in checkpoint labels {task.labels}")
        label_index = label_to_index[normalized]
        for directory in directories:
            subset = str(directory)
            for image_path in sorted(directory.glob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                samples.append(
                    {
                        "image": str(image_path),
                        "label": label_index,
                        "patient_id": f"external_{directory.name}_{image_path.stem}",
                        "subset": subset,
                        "label_name": normalized,
                    }
                )
    return samples


def classification_metrics_supported(preds: list[int], targets: list[int], num_classes: int) -> dict[str, float]:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for pred, target in zip(preds, targets):
        confusion[int(target), int(pred)] += 1.0
    total = float(confusion.sum().item())
    correct = float(torch.trace(confusion).item())
    supported = [index for index in range(num_classes) if float(confusion[index, :].sum().item()) > 0.0]
    per_class_f1: list[float] = []
    per_class_recall: list[float] = []
    for index in supported:
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        denom = (2 * tp) + fp + fn
        per_class_f1.append(float((2 * tp / denom).item()) if denom > 0 else 0.0)
        recall_denom = tp + fn
        per_class_recall.append(float((tp / recall_denom).item()) if recall_denom > 0 else 0.0)
    return {
        "accuracy": correct / max(total, 1.0),
        "macro_f1_supported": sum(per_class_f1) / max(len(per_class_f1), 1),
        "balanced_accuracy_supported": sum(per_class_recall) / max(len(per_class_recall), 1),
        "support": total,
    }


def evaluate_classification(
    trainer: PULSETrainer,
    task: TaskSpec,
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    dataset = ClassificationDataset(samples, image_size=image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_pulse_collate,
    )
    total_loss = 0.0
    total_items = 0
    preds: list[int] = []
    targets: list[int] = []
    subset_targets: dict[str, list[int]] = defaultdict(list)
    subset_preds: dict[str, list[int]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            metas = batch["meta"]
            batch = trainer._move_batch(batch)
            outputs = trainer._forward(task, model, batch)
            loss = trainer._compute_loss(task, outputs, batch, class_weights=None, config=trainer.config)
            batch_size_current = batch["label"].shape[0]
            total_loss += loss.item() * batch_size_current
            total_items += batch_size_current
            batch_preds = torch.argmax(outputs, dim=1).detach().cpu().tolist()
            batch_targets = batch["label"].detach().cpu().tolist()
            preds.extend(batch_preds)
            targets.extend(batch_targets)
            for meta, pred, target in zip(metas, batch_preds, batch_targets):
                subset_key = str(meta.get("subset", "unknown"))
                subset_preds[subset_key].append(int(pred))
                subset_targets[subset_key].append(int(target))

    overall = classification_metrics_supported(preds, targets, len(task.labels))
    overall["loss"] = total_loss / max(total_items, 1)

    by_subset: dict[str, Any] = {}
    for subset_key in sorted(subset_targets):
        metrics = classification_metrics_supported(subset_preds[subset_key], subset_targets[subset_key], len(task.labels))
        metrics["sample_count"] = len(subset_targets[subset_key])
        by_subset[subset_key] = metrics

    return {
        "sample_count": len(samples),
        "metrics": overall,
        "by_subset": by_subset,
    }


def build_breast_segmentation_samples(dataset_root: Path, include_empty_negatives: bool) -> list[dict[str, Any]]:
    files = sorted(path for path in dataset_root.glob("*.png"))
    mask_index: dict[str, list[str]] = {}
    image_paths: list[Path] = []
    for path in files:
        if is_mask_file(path):
            prefix = path.stem.split("_", 1)[0]
            mask_index.setdefault(prefix, []).append(str(path))
        else:
            image_paths.append(path)
    samples: list[dict[str, Any]] = []
    for image_path in image_paths:
        key = image_path.stem
        mask_paths = sorted(mask_index.get(key, []))
        sample: dict[str, Any] = {"image": str(image_path), "patient_id": f"external_{key}"}
        if mask_paths:
            sample["mask_paths"] = mask_paths
            samples.append(sample)
        elif include_empty_negatives:
            samples.append(sample)
    return samples


def evaluate_segmentation(
    trainer: PULSETrainer,
    task: TaskSpec,
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "metrics": {}}
    mode = task.extras.get("segmentation_mode", "binary")
    dataset = SegmentationDataset(samples, image_size=image_size, train=False, segmentation_mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_pulse_collate,
    )
    total_loss = 0.0
    total_items = 0
    dice_total = 0.0
    iou_total = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = trainer._move_batch(batch)
            outputs = trainer._forward(task, model, batch)
            loss = trainer._compute_loss(task, outputs, batch, class_weights=None, config=trainer.config)
            batch_size_current = batch["mask"].shape[0]
            total_loss += loss.item() * batch_size_current
            total_items += batch_size_current
            outputs_cpu = outputs.cpu()
            masks_cpu = batch["mask"].cpu()
            for index in range(batch_size_current):
                metrics = segmentation_metrics(
                    outputs_cpu[index : index + 1],
                    masks_cpu[index : index + 1],
                    task.extras.get("segmentation_mode", "binary"),
                    len(task.extras.get("mask_class_names", [])),
                )
                dice_total += float(metrics["dice"])
                iou_total += float(metrics["iou"])
                sample_count += 1

    return {
        "sample_count": len(samples),
        "metrics": {
            "loss": total_loss / max(total_items, 1),
            "dice": dice_total / max(sample_count, 1),
            "iou": iou_total / max(sample_count, 1),
        },
    }


def ensure_extracted_pcos(zip_path: Path, output_root: Path) -> Path:
    extract_root = output_root / "pcos_dataverse"
    marker = extract_root / "testing" / "normal"
    if marker.exists():
        return extract_root
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)
    candidates = sorted(path for path in extract_root.iterdir() if path.is_dir())
    if len(candidates) == 1 and (candidates[0] / "testing").exists():
        return candidates[0]
    return extract_root


def checkpoint_path(root: Path, run_name: str, filename: str) -> Path:
    return root / run_name / filename


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.threads)
    datasets_root = Path(args.datasets_root).resolve()
    teacher_root = Path(args.teacher_root).resolve()
    student_root = Path(args.student_root).resolve()
    report_path = Path(args.report_json).resolve()
    csv_path = Path(args.summary_csv).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = report_path.parent / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "datasets_root": str(datasets_root),
        "teacher_root": str(teacher_root),
        "student_root": str(student_root),
        "evaluations": {},
    }
    summary_rows: list[dict[str, Any]] = []

    def record_row(dataset: str, task: str, metric: str, teacher_value: float, student_value: float) -> None:
        summary_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "metric": metric,
                "teacher": teacher_value,
                "student": student_value,
            }
        )

    # 1. Multi-organ domain routing.
    multi_organ_root = datasets_root / "multi organ" / "train_datasets"
    domain_dirs = {
        "breast": multi_organ_root / "breast",
        "carotid": multi_organ_root / "carotid",
        "kidney": multi_organ_root / "kidney",
        "thyroid": multi_organ_root / "thyroid",
        "liver": multi_organ_root / "liver",
    }
    router_specs = {
        "all": {label: [root / "high_quality", root / "low_quality"] for label, root in domain_dirs.items()},
        "high_quality": {label: [root / "high_quality"] for label, root in domain_dirs.items()},
        "low_quality": {label: [root / "low_quality"] for label, root in domain_dirs.items()},
    }

    for variant, ckpt_root, filename in [
        ("teacher", teacher_root, "best_model.pt"),
        ("student", student_root, "student_best.pt"),
    ]:
        ckpt_path = checkpoint_path(ckpt_root, "system_domain_classification", filename)
        print(f"[router] loading {variant}: {ckpt_path}", flush=True)
        checkpoint, task, image_size, trainer, model = load_task_bundle(
            ckpt_path, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        if task.task_id != "system/domain_classification":
            raise SystemExit(f"Unexpected task for router checkpoint: {task.task_id}")
        variant_results: dict[str, Any] = {}
        for subset_name, class_dirs in router_specs.items():
            print(f"[router] evaluating {variant} on {subset_name}", flush=True)
            samples = build_classification_samples(task, class_dirs)
            variant_results[subset_name] = evaluate_classification(
                trainer=trainer,
                task=task,
                model=model,
                samples=samples,
                image_size=image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        results["evaluations"].setdefault("multi_organ_domain_routing", {})[variant] = {
            "checkpoint": str(ckpt_path),
            "task_id": task.task_id,
            "labels": task.labels,
            "evaluation": variant_results,
            "checkpoint_config": checkpoint.get("config", {}),
        }

    for subset_name in ["all", "high_quality", "low_quality"]:
        teacher_metrics = results["evaluations"]["multi_organ_domain_routing"]["teacher"]["evaluation"][subset_name]["metrics"]
        student_metrics = results["evaluations"]["multi_organ_domain_routing"]["student"]["evaluation"][subset_name]["metrics"]
        record_row(
            dataset=f"multi organ/train_datasets ({subset_name})",
            task="system/domain_classification",
            metric="accuracy",
            teacher_value=float(teacher_metrics["accuracy"]),
            student_value=float(student_metrics["accuracy"]),
        )

    # 2. PCOS binary classification on testing split only.
    pcos_zip = datasets_root / "dataverse_files_pcos.zip"
    pcos_root = ensure_extracted_pcos(pcos_zip, tmp_root)
    pcos_class_dirs = {
        "negative": [
            pcos_root / "train copy" / "normal",
            pcos_root / "validation" / "normal",
            pcos_root / "testing" / "normal",
        ],
        "positive": [
            pcos_root / "train copy" / "pco",
            pcos_root / "validation" / "pco",
            pcos_root / "testing" / "pco",
        ],
    }

    for variant, ckpt_root, filename in [
        ("teacher", teacher_root, "best_model.pt"),
        ("student", student_root, "student_best.pt"),
    ]:
        ckpt_path = checkpoint_path(ckpt_root, "pcos_binary_classification", filename)
        print(f"[pcos] loading {variant}: {ckpt_path}", flush=True)
        checkpoint, task, image_size, trainer, model = load_task_bundle(
            ckpt_path, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        if task.task_id != "pcos/binary_classification":
            raise SystemExit(f"Unexpected task for PCOS checkpoint: {task.task_id}")
        samples = build_classification_samples(task, pcos_class_dirs)
        print(f"[pcos] evaluating {variant} on full archive", flush=True)
        results["evaluations"].setdefault("pcos_full_archive", {})[variant] = {
            "checkpoint": str(ckpt_path),
            "task_id": task.task_id,
            "labels": task.labels,
            "evaluation": evaluate_classification(
                trainer=trainer,
                task=task,
                model=model,
                samples=samples,
                image_size=image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            ),
            "checkpoint_config": checkpoint.get("config", {}),
        }

    record_row(
        dataset="dataverse_files_pcos.zip (all splits)",
        task="pcos/binary_classification",
        metric="macro_f1_supported",
        teacher_value=float(results["evaluations"]["pcos_full_archive"]["teacher"]["evaluation"]["metrics"]["macro_f1_supported"]),
        student_value=float(results["evaluations"]["pcos_full_archive"]["student"]["evaluation"]["metrics"]["macro_f1_supported"]),
    )

    # 3. External breast lesion segmentation.
    breast_root = datasets_root / "BrEaST-Lesions_USG-images_and_masks"
    for variant, ckpt_root, filename in [
        ("teacher", teacher_root, "best_model.pt"),
        ("student", student_root, "student_best.pt"),
    ]:
        ckpt_path = checkpoint_path(ckpt_root, "breast_lesion_segmentation", filename)
        print(f"[breast] loading {variant}: {ckpt_path}", flush=True)
        checkpoint, task, image_size, trainer, model = load_task_bundle(
            ckpt_path, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        if task.task_id != "breast/lesion_segmentation":
            raise SystemExit(f"Unexpected task for breast segmentation checkpoint: {task.task_id}")
        lesion_only_samples = build_breast_segmentation_samples(breast_root, include_empty_negatives=False)
        all_samples = build_breast_segmentation_samples(breast_root, include_empty_negatives=True)
        print(f"[breast] evaluating {variant} lesion-only ({len(lesion_only_samples)})", flush=True)
        lesion_only_eval = evaluate_segmentation(
            trainer=trainer,
            task=task,
            model=model,
            samples=lesion_only_samples,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"[breast] evaluating {variant} including empty ({len(all_samples)})", flush=True)
        all_eval = evaluate_segmentation(
            trainer=trainer,
            task=task,
            model=model,
            samples=all_samples,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results["evaluations"].setdefault("breast_lesion_segmentation", {})[variant] = {
            "checkpoint": str(ckpt_path),
            "task_id": task.task_id,
            "dataset_summary": {
                "base_images": len(all_samples),
                "lesion_annotated_images": len(lesion_only_samples),
                "empty_mask_images": len(all_samples) - len(lesion_only_samples),
                "multi_mask_cases": sum(1 for sample in lesion_only_samples if len(sample.get("mask_paths", [])) > 1),
                "mask_union_rule": "All *_tumor and *_otherN masks for the same case are unioned into one binary lesion mask.",
            },
            "evaluation": {
                "lesion_only_pairs": lesion_only_eval,
                "including_empty_mask_images": all_eval,
            },
            "checkpoint_config": checkpoint.get("config", {}),
        }

    record_row(
        dataset="BrEaST-Lesions_USG-images_and_masks (lesion-only)",
        task="breast/lesion_segmentation",
        metric="dice",
        teacher_value=float(results["evaluations"]["breast_lesion_segmentation"]["teacher"]["evaluation"]["lesion_only_pairs"]["metrics"]["dice"]),
        student_value=float(results["evaluations"]["breast_lesion_segmentation"]["student"]["evaluation"]["lesion_only_pairs"]["metrics"]["dice"]),
    )
    record_row(
        dataset="BrEaST-Lesions_USG-images_and_masks (including empty masks)",
        task="breast/lesion_segmentation",
        metric="dice",
        teacher_value=float(results["evaluations"]["breast_lesion_segmentation"]["teacher"]["evaluation"]["including_empty_mask_images"]["metrics"]["dice"]),
        student_value=float(results["evaluations"]["breast_lesion_segmentation"]["student"]["evaluation"]["including_empty_mask_images"]["metrics"]["dice"]),
    )

    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "task", "metric", "teacher", "student"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({"report_json": str(report_path), "summary_csv": str(csv_path), "rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
