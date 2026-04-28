from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

from .discovery import discover_all_tasks
from .specs import TaskSpec

if TYPE_CHECKING:
    from .trainer import PULSETrainer
    from .training_policy import AgenticRLTrainingPolicyAgent


class DatasetInspectorAgent:
    def __init__(self, data_root: Path, seed: int = 42):
        self.data_root = Path(data_root)
        self.seed = seed

    def inspect(self) -> List[TaskSpec]:
        return discover_all_tasks(self.data_root, seed=self.seed)

    def to_markdown(self, tasks: Sequence[TaskSpec]) -> str:
        lines = [
            "# PULSE Task Discovery",
            "",
            f"Data root: `{self.data_root}`",
            "",
        ]
        for task in tasks:
            counts = task.to_dict()["counts"]
            if task.is_ready:
                status = "READY"
            elif task.status == "unavailable":
                status = "UNAVAILABLE"
            else:
                status = "EMPTY"
            lines.append(
                f"- `{task.task_id}` [{status}] train={counts['train']} val={counts['val']} "
                f"test={counts['test']} datasets={', '.join(task.dataset_names)}"
            )
            if task.skip_reason:
                lines.append(f"  reason: {task.skip_reason}")
        return "\n".join(lines)


class PlanningAgent:
    def build_plan(
        self,
        tasks: Sequence[TaskSpec],
        only: Optional[Sequence[str]] = None,
        include_unavailable: bool = False,
    ) -> List[TaskSpec]:
        requested = {item.strip() for item in (only or []) if item.strip()}
        plan = []
        for task in tasks:
            if requested and task.task_id not in requested and task.domain not in requested:
                continue
            if task.is_ready or include_unavailable:
                plan.append(task)
        return plan


class TrainingAgent:
    def __init__(self, trainer: "PULSETrainer"):
        self.trainer = trainer

    def run(
        self,
        tasks: Iterable[TaskSpec],
        output_root: Path,
        dry_run: bool = False,
        policy_agent: "AgenticRLTrainingPolicyAgent | None" = None,
    ):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        results = []
        pending = list(tasks)
        while pending:
            if policy_agent is not None:
                task, decision = policy_agent.select_next_task(pending, self.trainer.config)
            else:
                task = pending[0]
                decision = None
            print(f"\n==> {task.task_id}")
            if decision is not None:
                print(f"    policy={decision.profile} priority={decision.priority:.2f}")
                for note in decision.rationale:
                    print(f"    note: {note}")
            if dry_run:
                payload = {"task_id": task.task_id, "status": "planned"}
                if decision is not None:
                    payload["training_policy"] = decision.to_dict()
                results.append(payload)
                pending.remove(task)
                continue
            task_output = output_root / task.output_name
            result = self.trainer.train(
                task,
                task_output,
                overrides=decision.overrides if decision is not None else None,
                policy_agent=policy_agent,
                policy_decision=decision,
            )
            if decision is not None:
                result["training_policy"] = decision.to_dict()
                policy_agent.update(task, decision, result)
            results.append(result)
            pending.remove(task)
        return results


class ReportingAgent:
    def render(self, tasks: Sequence[TaskSpec], results: Sequence[dict]) -> str:
        lines = [
            "# PULSE Run Report",
            "",
            "## Summary",
            "",
        ]
        by_id = {result["task_id"]: result for result in results if "task_id" in result}
        for task in tasks:
            result = by_id.get(task.task_id, {"status": "not_run"})
            lines.append(f"- `{task.task_id}`: {result['status']}")
            if result["status"] == "completed":
                metric_name = result.get("primary_metric", task.primary_metric)
                lines.append(f"  best `{metric_name}`: {result.get('best_metric', 0.0):.4f}")
                policy = result.get("training_policy")
                if isinstance(policy, dict) and policy.get("profile"):
                    lines.append(f"  policy: `{policy['profile']}`")
                trace = result.get("policy_trace")
                if isinstance(trace, list) and trace:
                    lines.append(f"  epoch actions: {', '.join(str(item.get('action', 'unknown')) for item in trace[:5])}")
            elif task.skip_reason:
                lines.append(f"  reason: {task.skip_reason}")
        return "\n".join(lines)

    def write(self, tasks: Sequence[TaskSpec], results: Sequence[dict], output_path: Path) -> None:
        report = self.render(tasks, results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        summary_path = output_path.with_suffix(".json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump({"tasks": [task.to_dict() for task in tasks], "results": list(results)}, handle, indent=2)
