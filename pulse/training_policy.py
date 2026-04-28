from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence

from .specs import TaskSpec, TaskType
from .trainer import TrainConfig


@dataclass
class PolicyDecision:
    profile: str
    priority: float
    rationale: List[str]
    overrides: Dict[str, object]
    agents: List[str] = field(default_factory=lambda: ["planner", "critic"])

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EpochControlDecision:
    action: str
    state_key: str
    q_value: float
    exploration_rate: float
    rationale: List[str]
    overrides: Dict[str, object]
    agents: List[str] = field(default_factory=lambda: ["curriculum", "critic", "recovery"])

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PROFILE_LIBRARY: Dict[str, Dict[str, object]] = {
    "stable": {
        "learning_rate_scale": 0.75,
        "patience_bonus": 1,
        "epoch_bonus": 0,
        "focal_gamma": 0.5,
        "class_weight_power": 0.55,
        "label_smoothing": 0.04,
    },
    "balanced": {
        "learning_rate_scale": 1.0,
        "patience_bonus": 2,
        "epoch_bonus": 2,
        "focal_gamma": 1.5,
        "class_weight_power": 0.8,
        "label_smoothing": 0.02,
    },
    "aggressive": {
        "learning_rate_scale": 1.15,
        "patience_bonus": 3,
        "epoch_bonus": 4,
        "focal_gamma": 2.0,
        "class_weight_power": 1.0,
        "label_smoothing": 0.0,
    },
}


CLASSIFICATION_ACTIONS: Dict[str, Dict[str, object]] = {
    "stabilize": {
        "sampling_strategy": "balanced",
        "classification_loss": "weighted_ce",
        "focal_gamma": 0.0,
        "label_smoothing": 0.05,
        "train_augmentation": "light",
        "learning_rate_scale": 0.85,
    },
    "minority_focus": {
        "sampling_strategy": "minority_focus",
        "classification_loss": "weighted_focal",
        "focal_gamma": 2.2,
        "label_smoothing": 0.02,
        "train_augmentation": "standard",
        "learning_rate_scale": 0.9,
    },
    "hard_mining": {
        "sampling_strategy": "hybrid_hard",
        "classification_loss": "weighted_focal",
        "focal_gamma": 1.8,
        "label_smoothing": 0.01,
        "train_augmentation": "strong",
        "learning_rate_scale": 1.0,
    },
    "explore": {
        "sampling_strategy": "balanced",
        "classification_loss": "weighted_focal",
        "focal_gamma": 1.2,
        "label_smoothing": 0.01,
        "train_augmentation": "strong",
        "learning_rate_scale": 1.08,
    },
    "recover": {
        "sampling_strategy": "balanced",
        "classification_loss": "weighted_ce",
        "focal_gamma": 0.0,
        "label_smoothing": 0.06,
        "train_augmentation": "light",
        "learning_rate_scale": 0.65,
        "restore_best": True,
    },
}

NON_CLASSIFICATION_ACTIONS: Dict[str, Dict[str, object]] = {
    "stabilize": {
        "learning_rate_scale": 0.85,
        "train_augmentation": "light",
    },
    "explore": {
        "learning_rate_scale": 1.05,
        "train_augmentation": "standard",
    },
    "recover": {
        "learning_rate_scale": 0.7,
        "train_augmentation": "light",
        "restore_best": True,
    },
}


class AgenticRLTrainingPolicyAgent:
    def __init__(
        self,
        state_path: Path,
        seed: int = 42,
        exploration: float = 0.8,
        q_alpha: float = 0.35,
        q_gamma: float = 0.8,
        epoch_exploration: float = 0.45,
        min_epoch_exploration: float = 0.08,
    ):
        self.state_path = Path(state_path)
        self.seed = seed
        self.rng = Random(seed)
        self.exploration = exploration
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.epoch_exploration = epoch_exploration
        self.min_epoch_exploration = min_epoch_exploration
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, object]:
        if not self.state_path.exists():
            return {
                "families": {},
                "history": [],
                "epoch_q_table": {},
                "epoch_state_visits": {},
                "epoch_history": [],
            }
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        return {
            "families": loaded.get("families", {}),
            "history": loaded.get("history", []),
            "epoch_q_table": loaded.get("epoch_q_table", {}),
            "epoch_state_visits": loaded.get("epoch_state_visits", {}),
            "epoch_history": loaded.get("epoch_history", []),
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _task_family(self, task: TaskSpec) -> str:
        if task.extras.get("router_task"):
            return "router"
        return f"{task.domain}:{task.task_type.value}"

    def _class_counts(self, task: TaskSpec) -> List[int]:
        if task.task_type not in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            return []
        counts = [0 for _ in task.labels]
        for sample in task.train_samples:
            label = sample.get("label")
            if isinstance(label, int) and 0 <= label < len(counts):
                counts[label] += 1
        return counts

    def _imbalance_ratio(self, task: TaskSpec) -> float:
        counts = [count for count in self._class_counts(task) if count > 0]
        if not counts:
            return 1.0
        return max(counts) / max(min(counts), 1)

    def _family_profile_stats(self, family: str, profile: str) -> tuple[float, int]:
        family_state = self.state.setdefault("families", {}).setdefault(family, {})
        profile_state = family_state.setdefault(profile, {"count": 0, "mean_reward": 0.0})
        return float(profile_state["mean_reward"]), int(profile_state["count"])

    def _ucb_score(self, family: str, profile: str, total_trials: int) -> float:
        mean_reward, count = self._family_profile_stats(family, profile)
        if count == 0:
            return 1e6
        return mean_reward + self.exploration * math.sqrt(math.log(max(total_trials, 1) + 1) / count)

    def _choose_profile(self, task: TaskSpec) -> str:
        family = self._task_family(task)
        imbalance = self._imbalance_ratio(task)
        if task.extras.get("router_task"):
            candidates = ["balanced", "aggressive"]
        elif task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL} and imbalance >= 2.5:
            candidates = ["balanced", "aggressive", "stable"]
        elif task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            candidates = ["balanced", "stable", "aggressive"]
        else:
            candidates = ["stable", "balanced"]
        total_trials = sum(
            int(stats.get("count", 0))
            for stats in self.state.setdefault("families", {}).setdefault(family, {}).values()
            if isinstance(stats, dict)
        )
        return max(candidates, key=lambda profile: self._ucb_score(family, profile, total_trials))

    def decide(self, task: TaskSpec, base_config: TrainConfig) -> PolicyDecision:
        profile = self._choose_profile(task)
        profile_config = PROFILE_LIBRARY[profile]
        imbalance = self._imbalance_ratio(task)
        rationale: List[str] = []

        overrides: Dict[str, object] = {}
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            overrides["balanced_sampling"] = True
            overrides["classification_loss"] = "weighted_focal"
            overrides["focal_gamma"] = profile_config["focal_gamma"]
            overrides["class_weight_power"] = profile_config["class_weight_power"]
            overrides["label_smoothing"] = profile_config["label_smoothing"]
            overrides["learning_rate"] = float(base_config.learning_rate) * float(profile_config["learning_rate_scale"])
            overrides["epochs"] = int(base_config.epochs) + int(profile_config["epoch_bonus"])
            overrides["patience"] = int(base_config.patience) + int(profile_config["patience_bonus"])
            rationale.append("planner agent selected an imbalance-aware supervised profile")
            if imbalance >= 2.5:
                rationale.append(f"critic flagged strong class imbalance ({imbalance:.2f}x), so minority-aware sampling is prioritized")
        else:
            overrides["learning_rate"] = float(base_config.learning_rate) * min(float(profile_config["learning_rate_scale"]), 1.0)
            overrides["patience"] = int(base_config.patience) + int(profile_config["patience_bonus"])
            rationale.append("planner agent selected a conservative dense-task profile")

        if task.extras.get("router_task"):
            overrides["epochs"] = max(int(overrides.get("epochs", base_config.epochs)), int(base_config.epochs) + 6)
            overrides["patience"] = max(int(overrides.get("patience", base_config.patience)), int(base_config.patience) + 3)
            overrides["learning_rate"] = min(float(overrides.get("learning_rate", base_config.learning_rate)), float(base_config.learning_rate))
            rationale.append("router receives extra budget because all runtime agent routing depends on it")

        family_reward = self._recent_family_reward(self._task_family(task))
        priority = 0.0
        if task.extras.get("router_task"):
            priority += 5.0
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            priority += 2.0
        priority += min(imbalance, 5.0)
        priority += min(task.sample_count / 2000.0, 3.0)
        if family_reward < 0:
            priority += min(abs(family_reward), 1.5)
            rationale.append("critic boosted priority because this task family has underperformed recently")

        return PolicyDecision(profile=profile, priority=priority, rationale=rationale, overrides=overrides)

    def select_next_task(self, pending: Sequence[TaskSpec], base_config: TrainConfig) -> tuple[TaskSpec, PolicyDecision]:
        ranked = []
        for task in pending:
            decision = self.decide(task, base_config)
            ranked.append((decision.priority, self.rng.random(), task, decision))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, task, decision = ranked[0]
        return task, decision

    def _metric_sign(self, task: TaskSpec) -> float:
        return -1.0 if task.primary_mode == "min" else 1.0

    def _epoch_stage(self, epoch: int, total_epochs: int) -> str:
        ratio = epoch / max(total_epochs, 1)
        if ratio <= 0.33:
            return "early"
        if ratio <= 0.66:
            return "mid"
        return "late"

    def _task_scale_bucket(self, task: TaskSpec) -> str:
        if task.sample_count < 1000:
            return "small"
        if task.sample_count < 5000:
            return "medium"
        return "large"

    def _imbalance_bucket(self, task: TaskSpec) -> str:
        ratio = self._imbalance_ratio(task)
        if ratio < 1.5:
            return "balanced"
        if ratio < 3.0:
            return "moderate"
        return "severe"

    def _trend_bucket(self, task: TaskSpec, history_rows: Sequence[Dict[str, object]]) -> str:
        metric_key = f"val_{task.primary_metric}"
        values = [float(row[metric_key]) for row in history_rows if metric_key in row]
        if len(values) < 2:
            return "cold_start"
        delta = values[-1] - values[-2]
        if task.primary_mode == "min":
            delta = -delta
        if delta > 0.02:
            return "improving"
        if delta < -0.02:
            return "degrading"
        return "stalled"

    def _gap_bucket(self, history_rows: Sequence[Dict[str, object]]) -> str:
        if not history_rows:
            return "unknown"
        row = history_rows[-1]
        train_loss = float(row.get("train_loss", 0.0))
        val_loss = float(row.get("val_loss", 0.0))
        gap = val_loss - train_loss
        if gap < 0.05:
            return "tight"
        if gap < 0.2:
            return "moderate"
        return "wide"

    def _quality_bucket(self, task: TaskSpec, history_rows: Sequence[Dict[str, object]]) -> str:
        metric_key = f"val_{task.primary_metric}"
        values = [float(row[metric_key]) for row in history_rows if metric_key in row]
        if not values:
            return "unknown"
        value = values[-1]
        if task.primary_mode == "min":
            if value < 0.1:
                return "strong"
            if value < 0.2:
                return "fair"
            return "weak"
        if value >= 0.8:
            return "strong"
        if value >= 0.55:
            return "fair"
        return "weak"

    def _recent_family_reward(self, family: str) -> float:
        epoch_history = self.state.get("epoch_history", [])
        rewards = [float(entry["reward"]) for entry in epoch_history[-30:] if entry.get("family") == family]
        if not rewards:
            return 0.0
        return sum(rewards) / max(len(rewards), 1)

    def _state_key(
        self,
        task: TaskSpec,
        base_config: TrainConfig,
        epoch: int,
        history_rows: Sequence[Dict[str, object]],
        epochs_without_improvement: int,
    ) -> str:
        components = [
            self._epoch_stage(epoch, base_config.epochs),
            self._task_scale_bucket(task),
            self._imbalance_bucket(task),
            self._trend_bucket(task, history_rows),
            self._gap_bucket(history_rows),
            self._quality_bucket(task, history_rows),
            "plateau" if epochs_without_improvement >= 2 else "active",
        ]
        return "|".join(components)

    def _action_space(self, task: TaskSpec) -> Dict[str, Dict[str, object]]:
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            return CLASSIFICATION_ACTIONS
        return NON_CLASSIFICATION_ACTIONS

    def _ensure_q_state(self, family: str, state_key: str, action_space: Dict[str, Dict[str, object]]) -> Dict[str, float]:
        family_table = self.state.setdefault("epoch_q_table", {}).setdefault(family, {})
        action_values = family_table.setdefault(state_key, {})
        for action_name in action_space:
            action_values.setdefault(action_name, 0.0)
        return action_values

    def _state_visits(self, family: str, state_key: str) -> int:
        family_visits = self.state.setdefault("epoch_state_visits", {}).setdefault(family, {})
        return int(family_visits.get(state_key, 0))

    def _bump_state_visits(self, family: str, state_key: str) -> int:
        family_visits = self.state.setdefault("epoch_state_visits", {}).setdefault(family, {})
        family_visits[state_key] = int(family_visits.get(state_key, 0)) + 1
        return int(family_visits[state_key])

    def decide_epoch(
        self,
        task: TaskSpec,
        task_decision: PolicyDecision | None,
        base_config: TrainConfig,
        epoch: int,
        history_rows: Sequence[Dict[str, object]],
        best_metric: float,
        epochs_without_improvement: int,
    ) -> EpochControlDecision:
        del best_metric
        family = self._task_family(task)
        state_key = self._state_key(task, base_config, epoch, history_rows, epochs_without_improvement)
        action_space = self._action_space(task)
        action_values = self._ensure_q_state(family, state_key, action_space)
        visits = self._bump_state_visits(family, state_key)
        exploration_rate = max(self.min_epoch_exploration, self.epoch_exploration / math.sqrt(visits))

        rationale: List[str] = []
        if task_decision is not None:
            rationale.append(f"planner profile `{task_decision.profile}` remains active")
        rationale.append(f"curriculum state `{state_key}` detected")

        force_recover = epochs_without_improvement >= max(2, base_config.patience // 2)
        if force_recover and "recover" in action_space:
            action = "recover"
            rationale.append("recovery agent forced a rollback-aware step after repeated stagnation")
        else:
            if self.rng.random() < exploration_rate:
                unexplored = [name for name, value in action_values.items() if abs(value) < 1e-6]
                pool = unexplored or list(action_space.keys())
                action = self.rng.choice(pool)
                rationale.append("critic requested exploration to improve the epoch policy value estimates")
            else:
                action = max(action_values, key=lambda name: action_values[name])
                rationale.append("critic selected the highest-value action for the current training state")

        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            if self._imbalance_ratio(task) >= 3.0 and action != "recover":
                rationale.append("minority protection remains important for this task family")
        else:
            rationale.append("dense task uses a smaller action space to avoid destabilizing convergence")

        return EpochControlDecision(
            action=action,
            state_key=state_key,
            q_value=float(action_values.get(action, 0.0)),
            exploration_rate=exploration_rate,
            rationale=rationale,
            overrides=dict(action_space[action]),
        )

    def _minority_f1(self, history_rows: Sequence[Dict[str, object]]) -> float:
        if not history_rows:
            return 0.0
        row = history_rows[-1]
        values = [float(value) for key, value in row.items() if key.startswith("val_f1_class_")]
        if not values:
            return 0.0
        return min(values)

    def _epoch_reward(
        self,
        task: TaskSpec,
        history_rows: Sequence[Dict[str, object]],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        improved: bool,
        epochs_without_improvement: int,
    ) -> float:
        current = float(val_metrics.get(task.primary_metric, 0.0))
        previous = None
        if len(history_rows) >= 2:
            previous = float(history_rows[-2].get(f"val_{task.primary_metric}", current))
        sign = self._metric_sign(task)
        delta = 0.0 if previous is None else sign * (current - previous)

        reward = 4.0 * delta
        reward -= 0.08 * float(val_metrics.get("loss", train_metrics.get("loss", 0.0)))
        generalization_gap = max(float(val_metrics.get("loss", 0.0)) - float(train_metrics.get("loss", 0.0)), 0.0)
        reward -= 0.2 * generalization_gap
        if improved:
            reward += 0.4
        if epochs_without_improvement >= 2:
            reward -= 0.2

        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            current_minority = self._minority_f1(history_rows)
            previous_minority = 0.0
            if len(history_rows) >= 2:
                previous_minority = self._minority_f1(history_rows[:-1])
            reward += 1.2 * (current_minority - previous_minority)
            reward += 0.3 * float(val_metrics.get("balanced_accuracy", 0.0))
        return reward

    def update_epoch(
        self,
        task: TaskSpec,
        task_decision: PolicyDecision | None,
        epoch_decision: EpochControlDecision,
        base_config: TrainConfig,
        epoch: int,
        history_rows: Sequence[Dict[str, object]],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        improved: bool,
        best_metric: float,
        epochs_without_improvement: int,
    ) -> float:
        del task_decision, best_metric
        family = self._task_family(task)
        action_space = self._action_space(task)
        current_values = self._ensure_q_state(family, epoch_decision.state_key, action_space)
        reward = self._epoch_reward(task, history_rows, train_metrics, val_metrics, improved, epochs_without_improvement)

        next_state_key = self._state_key(task, base_config, epoch + 1, history_rows, epochs_without_improvement)
        next_values = self._ensure_q_state(family, next_state_key, action_space)
        best_next = max(next_values.values()) if next_values else 0.0

        old_value = float(current_values.get(epoch_decision.action, 0.0))
        updated_value = old_value + self.q_alpha * ((reward + (self.q_gamma * best_next)) - old_value)
        current_values[epoch_decision.action] = updated_value

        self.state.setdefault("epoch_history", []).append(
            {
                "family": family,
                "task_id": task.task_id,
                "epoch": epoch,
                "state_key": epoch_decision.state_key,
                "action": epoch_decision.action,
                "reward": reward,
                "updated_q": updated_value,
                "primary_metric": val_metrics.get(task.primary_metric),
            }
        )
        self._save_state()
        return reward

    def update(self, task: TaskSpec, decision: PolicyDecision, result: Dict[str, object]) -> None:
        family = self._task_family(task)
        family_state = self.state.setdefault("families", {}).setdefault(family, {})
        profile_state = family_state.setdefault(decision.profile, {"count": 0, "mean_reward": 0.0})

        reward = 0.0
        if result.get("status") == "completed":
            best_metric = float(result.get("best_metric", 0.0))
            reward = -best_metric if task.primary_mode == "min" else best_metric
            val_metrics = result.get("val_metrics", {})
            if isinstance(val_metrics, dict) and "loss" in val_metrics:
                reward -= 0.05 * float(val_metrics["loss"])
            policy_trace = result.get("policy_trace", [])
            if isinstance(policy_trace, list) and policy_trace:
                reward += 0.1 * sum(float(item.get("reward", 0.0)) for item in policy_trace) / max(len(policy_trace), 1)
        elif result.get("status") == "skipped":
            reward = -0.25
        else:
            reward = -1.0

        count = int(profile_state["count"]) + 1
        previous_mean = float(profile_state["mean_reward"])
        profile_state["count"] = count
        profile_state["mean_reward"] = previous_mean + ((reward - previous_mean) / count)

        history = self.state.setdefault("history", [])
        history.append(
            {
                "task_id": task.task_id,
                "family": family,
                "profile": decision.profile,
                "reward": reward,
                "best_metric": result.get("best_metric"),
                "status": result.get("status"),
            }
        )
        self._save_state()


BanditTrainingPolicyAgent = AgenticRLTrainingPolicyAgent
