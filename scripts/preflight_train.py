#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from pulse.data import build_dataloaders
from pulse.discovery import discover_all_tasks
from pulse.models import build_model
from pulse.trainer import PULSETrainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test PULSE tasks before launching full sequential training.")
    parser.add_argument("--data-root", default="Datasets", help="Path to the dataset root directory.")
    parser.add_argument("--only", nargs="*", default=[], help="Optional task IDs or domain names to test.")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size for smoke tests.")
    parser.add_argument("--image-size", type=int, default=128, help="Resize for the smoke-test forward pass.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task discovery and trainer setup.")
    parser.add_argument("--device", default="cpu", help="Device for the smoke-test forward pass.")
    return parser.parse_args()


def matches_filter(task_id: str, domain: str, only: Sequence[str]) -> bool:
    requested = {item.strip() for item in only if item.strip()}
    if not requested:
        return True
    return task_id in requested or domain in requested


def patient_set(samples: Iterable[Dict]) -> set[str]:
    patients = set()
    for sample in samples:
        patient_id = sample.get("patient_id")
        if patient_id is not None:
            patients.add(str(patient_id))
    return patients


def audit_split_overlap(task) -> Tuple[List[str], Dict[str, int]]:
    train_patients = patient_set(task.train_samples)
    val_patients = patient_set(task.val_samples)
    test_patients = patient_set(task.test_samples)

    findings = []
    for left_name, left_set, right_name, right_set in [
        ("train", train_patients, "val", val_patients),
        ("train", train_patients, "test", test_patients),
        ("val", val_patients, "test", test_patients),
    ]:
        overlap = sorted(left_set & right_set)
        if overlap:
            preview = ", ".join(overlap[:5])
            findings.append(f"{left_name}/{right_name} overlap ({len(overlap)}): {preview}")

    counts = {
        "train": len(train_patients),
        "val": len(val_patients),
        "test": len(test_patients),
    }
    return findings, counts


def smoke_loader(trainer: PULSETrainer, task, loader, split_name: str) -> None:
    if loader is None:
        return
    batch = next(iter(loader))
    batch = trainer._move_batch(batch)
    model = build_model(task).to(trainer.device)
    model.eval()
    with torch.no_grad():
        outputs = trainer._forward(task, model, batch)
        loss = trainer._compute_loss(task, outputs, batch)
    if not float(loss.detach().cpu().item()) == float(loss.detach().cpu().item()):
        raise RuntimeError(f"{split_name} loss is NaN")


def main():
    args = parse_args()
    tasks = discover_all_tasks(Path(args.data_root).resolve(), seed=args.seed)
    trainer = PULSETrainer(
        TrainConfig(
            epochs=1,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
        )
    )

    failures: List[Tuple[str, str]] = []
    checked = 0
    for task in tasks:
        if not matches_filter(task.task_id, task.domain, args.only):
            continue

        if not task.is_ready:
            print(f"[SKIP] {task.task_id} ({task.status}) {task.skip_reason}")
            continue

        checked += 1
        overlaps, patient_counts = audit_split_overlap(task)
        if overlaps:
            failures.append((task.task_id, "; ".join(overlaps)))
            print(f"[FAIL] {task.task_id} split leakage detected: {'; '.join(overlaps)}")
            continue

        try:
            train_loader, val_loader, test_loader = build_dataloaders(
                task,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
            )
            smoke_loader(trainer, task, train_loader, "train")
            smoke_loader(trainer, task, val_loader, "val")
            smoke_loader(trainer, task, test_loader, "test")
            print(
                f"[OK] {task.task_id} "
                f"patients(train/val/test)={patient_counts['train']}/{patient_counts['val']}/{patient_counts['test']} "
                f"samples(train/val/test)={len(task.train_samples)}/{len(task.val_samples)}/{len(task.test_samples)}"
            )
        except Exception as exc:
            failures.append((task.task_id, str(exc)))
            print(f"[FAIL] {task.task_id} {exc}")

    print(f"\nChecked {checked} ready task(s).")
    if failures:
        print(f"Failures: {len(failures)}")
        for task_id, reason in failures:
            print(f"- {task_id}: {reason}")
        raise SystemExit(1)

    print("All checked tasks passed split and one-batch smoke tests.")


if __name__ == "__main__":
    main()
