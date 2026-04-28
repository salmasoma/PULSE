from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .data import build_dataloaders
from .metrics import classification_metrics, detection_metrics, regression_metrics, segmentation_metrics
from .models import build_model
from .specs import TaskSpec, TaskType


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    image_size: int = 224
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    patience: int = 4
    seed: int = 42
    device: str = "auto"
    min_epochs: int = 3
    gradient_clip_norm: float = 1.0
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    min_learning_rate: float = 1e-6
    balanced_sampling: bool = True
    classification_loss: str = "weighted_focal"
    focal_gamma: float = 1.5
    class_weight_power: float = 0.75
    class_weight_max: float = 6.0
    label_smoothing: float = 0.02


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PULSETrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        _set_seed(config.seed)
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

    def _resolved_config(self, overrides: Optional[Dict[str, object]] = None) -> TrainConfig:
        if not overrides:
            return self.config
        valid = {field_name for field_name in TrainConfig.__dataclass_fields__}
        filtered = {key: value for key, value in overrides.items() if key in valid}
        return replace(self.config, **filtered)

    def _classification_class_counts(self, task: TaskSpec) -> torch.Tensor:
        labels = [int(sample["label"]) for sample in task.train_samples if "label" in sample]
        if not labels:
            return torch.zeros(len(task.labels), dtype=torch.float32)
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=max(len(task.labels), 1)).float()
        return counts

    def _classification_class_weights(self, task: TaskSpec, config: TrainConfig) -> torch.Tensor | None:
        if task.task_type not in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL} or not task.labels:
            return None
        counts = self._classification_class_counts(task)
        if counts.numel() == 0 or torch.count_nonzero(counts).item() <= 1:
            return None
        counts = counts.clamp_min(1.0)
        weights = (counts.mean() / counts).pow(float(config.class_weight_power))
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights.clamp(max=float(config.class_weight_max))
        return weights.to(self.device)

    def _classification_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None,
        config: TrainConfig,
    ) -> torch.Tensor:
        return self._classification_losses(outputs, targets, class_weights, config).mean()

    def _classification_losses(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None,
        config: TrainConfig,
    ) -> torch.Tensor:
        if config.classification_loss == "weighted_focal" and float(config.focal_gamma) > 0:
            log_probs = nn.functional.log_softmax(outputs, dim=1)
            probs = log_probs.exp()
            ce = nn.functional.nll_loss(log_probs, targets, weight=class_weights, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
            focal_term = (1.0 - pt).pow(float(config.focal_gamma))
            return focal_term * ce
        return nn.functional.cross_entropy(
            outputs,
            targets,
            weight=class_weights,
            reduction="none",
            label_smoothing=float(config.label_smoothing),
        )

    def _segmentation_loss(self, logits: torch.Tensor, masks: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "binary":
            bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
            probs = torch.sigmoid(logits)
            intersection = (probs * masks).sum(dim=(1, 2, 3))
            denom = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
            return bce + dice_loss

        ce = nn.functional.cross_entropy(logits, masks)
        one_hot = nn.functional.one_hot(masks, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        intersection = (probs * one_hot).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
        return ce + dice_loss

    def _compute_loss(
        self,
        task: TaskSpec,
        outputs,
        batch,
        class_weights: torch.Tensor | None = None,
        config: TrainConfig | None = None,
    ) -> torch.Tensor:
        config = config or self.config
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            return self._classification_loss(outputs, batch["label"], class_weights, config)
        if task.task_type == TaskType.SEGMENTATION:
            return self._segmentation_loss(outputs, batch["mask"], task.extras.get("segmentation_mode", "binary"))
        if task.task_type == TaskType.DETECTION:
            objectness_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["objectness"],
                torch.ones_like(outputs["objectness"]),
            )
            bbox_loss = nn.functional.smooth_l1_loss(outputs["bbox"], batch["bbox"])
            return objectness_loss + bbox_loss
        return nn.functional.smooth_l1_loss(outputs, batch["target"])

    def _move_batch(self, batch):
        moved = {}
        for key, value in batch.items():
            if key == "modalities":
                moved[key] = {name: tensor.to(self.device) for name, tensor in value.items()}
            elif torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _forward(self, task: TaskSpec, model: nn.Module, batch):
        if task.task_type == TaskType.MULTIMODAL:
            return model(batch["modalities"])
        return model(batch["image"])

    def _evaluate_loader(
        self,
        task: TaskSpec,
        model: nn.Module,
        loader,
        class_weights: torch.Tensor | None = None,
        config: TrainConfig | None = None,
    ) -> Dict[str, float]:
        if loader is None:
            return {}
        config = config or self.config

        model.eval()
        total_loss = 0.0
        total_items = 0
        all_preds = []
        all_targets = []
        binary_dice: Dict[str, float] = {"dice": 0.0, "iou": 0.0}
        segmentation_batches = 0
        detection_ious = []
        regression_pred = []
        regression_target = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)
                outputs = self._forward(task, model, batch)
                loss = self._compute_loss(task, outputs, batch, class_weights=class_weights, config=config)
                batch_size = next(iter(batch.values())).shape[0] if task.task_type != TaskType.MULTIMODAL else batch["label"].shape[0]
                total_loss += loss.item() * batch_size
                total_items += batch_size

                if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(batch["label"].cpu().tolist())
                elif task.task_type == TaskType.SEGMENTATION:
                    metrics = segmentation_metrics(
                        outputs.cpu(),
                        batch["mask"].cpu(),
                        task.extras.get("segmentation_mode", "binary"),
                        len(task.extras.get("mask_class_names", [])),
                    )
                    binary_dice["dice"] += metrics["dice"]
                    binary_dice["iou"] += metrics["iou"]
                    segmentation_batches += 1
                elif task.task_type == TaskType.DETECTION:
                    metrics = detection_metrics(outputs["bbox"].cpu(), batch["bbox"].cpu())
                    detection_ious.append(metrics["mean_iou"])
                else:
                    regression_pred.append(outputs.detach().cpu())
                    regression_target.append(batch["target"].detach().cpu())

        metrics: Dict[str, float] = {
            "loss": total_loss / max(total_items, 1),
        }
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            metrics.update(classification_metrics(all_preds, all_targets, len(task.labels)))
        elif task.task_type == TaskType.SEGMENTATION:
            metrics.update(
                {
                    "dice": binary_dice["dice"] / max(segmentation_batches, 1),
                    "iou": binary_dice["iou"] / max(segmentation_batches, 1),
                }
            )
        elif task.task_type == TaskType.DETECTION:
            metrics["mean_iou"] = sum(detection_ious) / max(len(detection_ious), 1)
        else:
            preds = torch.cat(regression_pred) if regression_pred else torch.empty(0)
            targets = torch.cat(regression_target) if regression_target else torch.empty(0)
            metrics.update(regression_metrics(preds, targets))
        return metrics

    def _update_sample_difficulty(self, batch, outputs, per_item_losses: torch.Tensor, difficulty_scores: torch.Tensor | None) -> None:
        if difficulty_scores is None or "sample_index" not in batch:
            return
        sample_index = batch["sample_index"].detach().cpu().long().flatten()
        if sample_index.numel() == 0:
            return
        losses = per_item_losses.detach().cpu().float().flatten()
        if outputs.ndim == 2 and "label" in batch:
            preds = torch.argmax(outputs.detach(), dim=1).cpu()
            targets = batch["label"].detach().cpu()
            losses = losses * (1.0 + (preds != targets).float())
        current = difficulty_scores.index_select(0, sample_index)
        updated = (0.85 * current) + (0.15 * losses)
        difficulty_scores.index_copy_(0, sample_index, updated)

    def train(
        self,
        task: TaskSpec,
        output_dir: Path,
        overrides: Optional[Dict[str, object]] = None,
        policy_agent=None,
        policy_decision=None,
    ) -> Dict[str, object]:
        config = self._resolved_config(overrides)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "task.json", "w", encoding="utf-8") as handle:
            json.dump(task.to_dict(), handle, indent=2)

        if not task.is_ready:
            summary = {
                "task_id": task.task_id,
                "status": "skipped",
                "reason": task.skip_reason or "Task is not trainable.",
            }
            with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            return summary

        _, val_loader, test_loader = build_dataloaders(
            task,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            balanced_sampling=False,
            sampling_strategy="natural",
            train_augmentation="light",
        )

        model = build_model(task).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if task.primary_mode == "min" else "max",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.min_learning_rate,
        )
        class_weights = self._classification_class_weights(task, config)
        class_counts = self._classification_class_counts(task)

        best_metric = float("inf") if task.primary_mode == "min" else float("-inf")
        best_path = output_dir / "best_model.pt"
        best_epoch = 0
        epochs_without_improvement = 0
        history_rows = []
        difficulty_scores = None
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL} and task.train_samples:
            difficulty_scores = torch.ones(len(task.train_samples), dtype=torch.float32)
        policy_trace = []

        for epoch in range(1, config.epochs + 1):
            epoch_decision = None
            epoch_config = config
            train_augmentation = "standard"
            sampling_strategy = "balanced" if config.balanced_sampling else "natural"
            if policy_agent is not None and hasattr(policy_agent, "decide_epoch"):
                epoch_decision = policy_agent.decide_epoch(
                    task=task,
                    task_decision=policy_decision,
                    base_config=config,
                    epoch=epoch,
                    history_rows=history_rows,
                    best_metric=best_metric,
                    epochs_without_improvement=epochs_without_improvement,
                )
                epoch_overrides = dict(getattr(epoch_decision, "overrides", {}) or {})
                train_augmentation = str(epoch_overrides.pop("train_augmentation", train_augmentation))
                sampling_strategy = str(epoch_overrides.pop("sampling_strategy", sampling_strategy))
                restore_best = bool(epoch_overrides.pop("restore_best", False))
                lr_scale = float(epoch_overrides.pop("learning_rate_scale", 1.0))
                merged_overrides = dict(overrides or {})
                merged_overrides.update(epoch_overrides)
                epoch_config = self._resolved_config(merged_overrides)
                if restore_best and best_path.exists():
                    checkpoint = torch.load(best_path, map_location=self.device)
                    model.load_state_dict(checkpoint["state_dict"])
                if lr_scale != 1.0:
                    for param_group in optimizer.param_groups:
                        current_lr = float(param_group["lr"])
                        param_group["lr"] = max(
                            epoch_config.min_learning_rate,
                            min(current_lr * lr_scale, config.learning_rate * 1.5),
                        )

            train_loader, _, _ = build_dataloaders(
                task,
                batch_size=epoch_config.batch_size,
                num_workers=epoch_config.num_workers,
                image_size=epoch_config.image_size,
                balanced_sampling=epoch_config.balanced_sampling,
                sampling_strategy=sampling_strategy,
                sample_score_overrides=difficulty_scores if sampling_strategy in {"hard_example", "hybrid_hard"} else None,
                train_augmentation=train_augmentation,
            )
            model.train()
            train_loss = 0.0
            seen = 0
            for batch in train_loader:
                batch = self._move_batch(batch)
                outputs = self._forward(task, model, batch)
                per_item_losses = None
                if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
                    per_item_losses = self._classification_losses(outputs, batch["label"], class_weights, epoch_config)
                    loss = per_item_losses.mean()
                else:
                    loss = self._compute_loss(task, outputs, batch, class_weights=class_weights, config=epoch_config)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if epoch_config.gradient_clip_norm and epoch_config.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(epoch_config.gradient_clip_norm))
                optimizer.step()
                if per_item_losses is not None:
                    self._update_sample_difficulty(batch, outputs, per_item_losses, difficulty_scores)
                batch_size = batch["label"].shape[0] if "label" in batch else batch["image"].shape[0]
                train_loss += loss.item() * batch_size
                seen += batch_size

            train_metrics = {"loss": train_loss / max(seen, 1)}
            val_metrics = self._evaluate_loader(task, model, val_loader, class_weights=class_weights, config=epoch_config)
            monitored = val_metrics.get(task.primary_metric, train_metrics["loss"])
            scheduler.step(monitored)
            improved = monitored < best_metric if task.primary_mode == "min" else monitored > best_metric
            if improved:
                best_metric = monitored
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "task": task.to_dict(),
                        "config": asdict(epoch_config),
                        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1

            history_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics.get("loss", 0.0),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            for key, value in val_metrics.items():
                if key != "loss":
                    history_row[f"val_{key}"] = value
            if epoch_decision is not None:
                history_row["policy_action"] = getattr(epoch_decision, "action", "unknown")
                history_row["policy_state"] = getattr(epoch_decision, "state_key", "unknown")
                history_row["policy_sampling"] = sampling_strategy
                history_row["policy_augmentation"] = train_augmentation
            epoch_reward = None
            if policy_agent is not None and epoch_decision is not None and hasattr(policy_agent, "update_epoch"):
                epoch_reward = policy_agent.update_epoch(
                    task=task,
                    task_decision=policy_decision,
                    epoch_decision=epoch_decision,
                    base_config=config,
                    epoch=epoch,
                    history_rows=history_rows + [history_row],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    improved=improved,
                    best_metric=best_metric,
                    epochs_without_improvement=epochs_without_improvement,
                )
                history_row["policy_reward"] = float(epoch_reward)
                policy_trace.append(epoch_decision.to_dict() | {"reward": float(epoch_reward)})
            history_rows.append(history_row)

            print(
                f"[{task.task_id}] epoch {epoch:02d}/{config.epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val={task.primary_metric}:{val_metrics.get(task.primary_metric, 0.0):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                + (
                    f" action={getattr(epoch_decision, 'action', 'none')} reward={epoch_reward:.3f}"
                    if epoch_decision is not None and epoch_reward is not None
                    else ""
                )
            )

            if epoch >= epoch_config.min_epochs and epochs_without_improvement >= epoch_config.patience:
                break

        with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history_rows for key in row.keys()}))
            writer.writeheader()
            for row in history_rows:
                writer.writerow(row)

        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])

        test_metrics = self._evaluate_loader(task, model, test_loader, class_weights=class_weights, config=config)
        val_metrics = self._evaluate_loader(task, model, val_loader, class_weights=class_weights, config=config)
        summary = {
            "task_id": task.task_id,
            "status": "completed",
            "best_epoch": best_epoch,
            "primary_metric": task.primary_metric,
            "primary_mode": task.primary_mode,
            "best_metric": best_metric,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoint": str(best_path),
            "train_config": asdict(config),
            "class_counts": {
                task.labels[index] if index < len(task.labels) else str(index): int(value)
                for index, value in enumerate(class_counts.tolist())
            }
            if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}
            else {},
            "class_weights": {
                task.labels[index] if index < len(task.labels) else str(index): float(value)
                for index, value in enumerate((class_weights.detach().cpu().tolist() if class_weights is not None else []))
            }
            if class_weights is not None
            else {},
            "policy_trace": policy_trace,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
