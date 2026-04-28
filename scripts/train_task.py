#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.agents import DatasetInspectorAgent
from pulse.trainer import PULSETrainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train one PULSE task.")
    parser.add_argument("--task-id", required=True, help="Task ID, for example `breast/lesion_segmentation`.")
    parser.add_argument("--data-root", default="Datasets", help="Path to the dataset root directory.")
    parser.add_argument("--output-dir", default="runs/pulse_single", help="Directory for this task run.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    inspector = DatasetInspectorAgent(Path(args.data_root).resolve(), seed=args.seed)
    tasks = {task.task_id: task for task in inspector.inspect()}
    if args.task_id not in tasks:
        raise SystemExit(f"Unknown task ID: {args.task_id}")

    trainer = PULSETrainer(
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            patience=args.patience,
            seed=args.seed,
        )
    )
    summary = trainer.train(tasks[args.task_id], Path(args.output_dir).resolve() / tasks[args.task_id].output_name)
    print(summary)


if __name__ == "__main__":
    main()
