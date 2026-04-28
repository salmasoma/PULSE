#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pulse.coreml_export import task_spec_from_dict  # noqa: E402
from pulse.data import SegmentationDataset, _pulse_collate  # noqa: E402
from pulse.models import build_model  # noqa: E402
from pulse.specs import TaskSpec  # noqa: E402
from pulse.metrics import segmentation_metrics  # noqa: E402
from pulse.trainer import PULSETrainer, TrainConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the teacher breast lesion segmentation model on an external flat image/mask dataset.")
    parser.add_argument(
        "--dataset-root",
        default="/Users/salma.hassan/Downloads/BrEaST-Lesions_USG-images_and_masks",
        help="Directory containing flat case images and matching *_tumor / *_otherN mask files.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/Users/salma.hassan/Ultrasound_Project/runs/pulse_retrain_new/breast_lesion_segmentation/best_model.pt",
        help="Teacher checkpoint to evaluate.",
    )
    parser.add_argument("--image-size", type=int, default=0, help="Resize used for evaluation. Defaults to the checkpoint config image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda, mps, or auto.")
    parser.add_argument("--threads", type=int, default=1, help="Torch / BLAS thread count for reproducible evaluation.")
    parser.add_argument(
        "--report-json",
        default="/Users/salma.hassan/Ultrasound_Project/reports/external_eval/breast_lesion_segmentation_teacher_breast_lesions_usg.json",
        help="Where to save the evaluation JSON report.",
    )
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


def build_external_samples(dataset_root: Path, include_empty_negatives: bool) -> list[dict[str, Any]]:
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
        sample: dict[str, Any] = {
            "image": str(image_path),
            "patient_id": f"external_{key}",
        }
        if mask_paths:
            sample["mask_paths"] = mask_paths
            samples.append(sample)
        elif include_empty_negatives:
            samples.append(sample)
    return samples


def build_eval_loader(task: TaskSpec, samples: list[dict[str, Any]], image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    mode = task.extras.get("segmentation_mode", "binary")
    dataset = SegmentationDataset(samples, image_size=image_size, train=False, segmentation_mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_pulse_collate,
    )


def evaluate_split(
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
    loader = build_eval_loader(task, samples, image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    total_loss = 0.0
    total_items = 0
    dice_total = 0.0
    iou_total = 0.0
    sample_count = 0

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
    metrics = {
        "loss": total_loss / max(total_items, 1),
        "dice": dice_total / max(sample_count, 1),
        "iou": iou_total / max(sample_count, 1),
    }
    return {
        "sample_count": len(samples),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.threads)
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

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
    image_size = int(args.image_size or config.get("image_size") or 224)

    if task.task_id != "breast/lesion_segmentation":
        raise SystemExit(f"Checkpoint task_id is {task.task_id}, not breast/lesion_segmentation")

    trainer = PULSETrainer(
        TrainConfig(
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    )
    model = build_model(task).to(trainer.device)
    model.load_state_dict(checkpoint["state_dict"])

    lesion_only_samples = build_external_samples(dataset_root, include_empty_negatives=False)
    all_samples = build_external_samples(dataset_root, include_empty_negatives=True)
    empty_negative_count = len(all_samples) - len(lesion_only_samples)

    lesion_only = evaluate_split(
        trainer=trainer,
        task=task,
        model=model,
        samples=lesion_only_samples,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    with_empty_negatives = evaluate_split(
        trainer=trainer,
        task=task,
        model=model,
        samples=all_samples,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    report = {
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint_path),
        "task_id": task.task_id,
        "task_title": task.title,
        "variant": "student" if checkpoint_path.name == "student_best.pt" else "teacher",
        "encoder_width": int(task.extras.get("encoder_width", 24)),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "dataset_summary": {
            "base_images": len(all_samples),
            "lesion_annotated_images": len(lesion_only_samples),
            "empty_mask_images": empty_negative_count,
            "multi_mask_cases": sum(1 for sample in lesion_only_samples if len(sample.get("mask_paths", [])) > 1),
            "mask_union_rule": "All *_tumor and *_otherN masks for the same case are unioned into one binary lesion mask.",
        },
        "evaluation": {
            "lesion_only_pairs": lesion_only,
            "including_empty_mask_images": with_empty_negatives,
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
