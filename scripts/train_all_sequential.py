#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.agents import DatasetInspectorAgent, PlanningAgent, ReportingAgent, TrainingAgent
from pulse.training_policy import AgenticRLTrainingPolicyAgent
from pulse.trainer import PULSETrainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Launch all ready PULSE training jobs sequentially.")
    parser.add_argument("--data-root", default="Datasets", help="Path to the dataset root directory.")
    parser.add_argument("--output-dir", default="runs/pulse", help="Directory for checkpoints, logs, and reports.")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per task.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per task.")
    parser.add_argument("--image-size", type=int, default=224, help="Square resize used by the lightweight models.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional task IDs or domain names to run, for example `thyroid` or `thyroid/nodule_detection`.",
    )
    parser.add_argument("--include-unavailable", action="store_true", help="Include unavailable tasks in the final report.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and plan without training.")
    parser.add_argument(
        "--disable-rl-policy",
        action="store_true",
        help="Disable the planner/critic/curriculum/recovery RL controller and use fixed sequential training.",
    )
    parser.add_argument("--policy-state", default="", help="Optional path for the persistent RL policy state JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    inspector = DatasetInspectorAgent(data_root=data_root, seed=args.seed)
    tasks = inspector.inspect()
    print(inspector.to_markdown(tasks))

    planner = PlanningAgent()
    plan = planner.build_plan(tasks, only=args.only, include_unavailable=args.include_unavailable)
    if not plan:
        raise SystemExit("No tasks matched the current filters.")

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
    policy_agent = None
    if not args.disable_rl_policy:
        policy_state = Path(args.policy_state).resolve() if args.policy_state else (output_dir / "training_policy_state.json")
        policy_agent = AgenticRLTrainingPolicyAgent(policy_state, seed=args.seed)
    training_agent = TrainingAgent(trainer)
    results = training_agent.run(plan, output_root=output_dir, dry_run=args.dry_run, policy_agent=policy_agent)

    reporter = ReportingAgent()
    reporter.write(plan, results, output_dir / "PULSE_run_report.md")
    print(f"\nReport written to {output_dir / 'PULSE_run_report.md'}")


if __name__ == "__main__":
    main()
