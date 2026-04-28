from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .data import IMAGENET_MEAN, IMAGENET_STD
from .models import build_model
from .specs import TaskSpec, TaskType

try:
    import coremltools as ct
except ModuleNotFoundError:  # pragma: no cover
    ct = None


EXCLUDED_RUNTIME_TASKS = {
    "fetal/plane_classification",
    "fetal/hc_measurement",
}

EXCLUDED_BUNDLED_TASKS = set(EXCLUDED_RUNTIME_TASKS)


@dataclass
class CheckpointRecord:
    checkpoint_path: Path
    task_dir: Path
    task: TaskSpec
    config: Dict[str, Any]
    state_dict: Dict[str, Any]
    variant: str = "teacher"
    student_width: int | None = None


@dataclass
class ExportRecord:
    task_id: str
    output_name: str
    domain: str
    title: str
    task_type: str
    labels: List[str]
    modalities: List[str]
    image_size: int
    runtime_enabled: bool
    output_semantics: str
    input_names: List[str]
    output_names: List[str]
    source_checkpoint: str
    model_variant: str
    student_width: int | None
    coreml_path: str | None
    exported: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "output_name": self.output_name,
            "domain": self.domain,
            "title": self.title,
            "task_type": self.task_type,
            "labels": list(self.labels),
            "modalities": list(self.modalities),
            "image_size": self.image_size,
            "runtime_enabled": self.runtime_enabled,
            "output_semantics": self.output_semantics,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "source_checkpoint": self.source_checkpoint,
            "model_variant": self.model_variant,
            "student_width": self.student_width,
            "coreml_path": self.coreml_path,
            "exported": self.exported,
            "reason": self.reason,
        }


def task_spec_from_dict(payload: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_id=str(payload["task_id"]),
        domain=str(payload["domain"]),
        title=str(payload["title"]),
        task_type=TaskType(str(payload["task_type"])),
        dataset_names=list(payload.get("dataset_names", [])),
        labels=list(payload.get("labels", [])),
        description=str(payload.get("description", "")),
        status=str(payload.get("status", "ready")),
        skip_reason=str(payload.get("skip_reason", "")),
        notes=list(payload.get("notes", [])),
        extras=dict(payload.get("extras", {})),
    )


def find_checkpoint_files(model_root: Path, checkpoint_name: str = "best_model.pt") -> List[Path]:
    model_root = Path(model_root).resolve()
    return sorted(model_root.rglob(checkpoint_name))


def find_checkpoint_dirs(model_root: Path, checkpoint_name: str = "best_model.pt") -> List[Path]:
    return [path.parent for path in find_checkpoint_files(model_root, checkpoint_name=checkpoint_name)]


def load_checkpoint_record(checkpoint_path: Path) -> CheckpointRecord:
    checkpoint_path = Path(checkpoint_path).resolve()
    task_dir = checkpoint_path.parent
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task = task_spec_from_dict(dict(checkpoint["task"]))
    config = dict(checkpoint.get("config", {}))
    student_width = checkpoint.get("student_width")
    variant = "student" if checkpoint_path.name == "student_best.pt" else "teacher"
    if student_width is not None:
        extras = dict(task.extras)
        extras["encoder_width"] = int(student_width)
        task.extras = extras
    elif "encoder_width" not in task.extras and config.get("encoder_width") is not None:
        extras = dict(task.extras)
        extras["encoder_width"] = int(config["encoder_width"])
        task.extras = extras
    return CheckpointRecord(
        checkpoint_path=checkpoint_path,
        task_dir=task_dir,
        task=task,
        config=config,
        state_dict=dict(checkpoint["state_dict"]),
        variant=variant,
        student_width=int(student_width) if student_width is not None else None,
    )


class ChannelwiseNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class SingleImageExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, task_type: TaskType):
        super().__init__()
        self.normalize = ChannelwiseNormalize()
        self.model = model
        self.task_type = task_type

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize(image)
        output = self.model(normalized)
        if self.task_type == TaskType.DETECTION:
            return torch.cat([output["objectness"], output["bbox"]], dim=1)
        return output


class DualModalExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, names: Sequence[str]):
        super().__init__()
        self.normalize = ChannelwiseNormalize()
        self.model = model
        self.names = list(names)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.model(
            {
                self.names[0]: self.normalize(first),
                self.names[1]: self.normalize(second),
            }
        )


class TripleModalExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, names: Sequence[str]):
        super().__init__()
        self.normalize = ChannelwiseNormalize()
        self.model = model
        self.names = list(names)

    def forward(self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        return self.model(
            {
                self.names[0]: self.normalize(first),
                self.names[1]: self.normalize(second),
                self.names[2]: self.normalize(third),
            }
        )


class QuadModalExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, names: Sequence[str]):
        super().__init__()
        self.normalize = ChannelwiseNormalize()
        self.model = model
        self.names = list(names)

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        third: torch.Tensor,
        fourth: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            {
                self.names[0]: self.normalize(first),
                self.names[1]: self.normalize(second),
                self.names[2]: self.normalize(third),
                self.names[3]: self.normalize(fourth),
            }
        )


def _output_semantics(task: TaskSpec) -> str:
    if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
        return "logits"
    if task.task_type == TaskType.SEGMENTATION:
        if task.extras.get("segmentation_mode", "binary") == "binary":
            return "binary_mask_logits"
        return "mask_class_logits"
    if task.task_type == TaskType.DETECTION:
        return "objectness_cxcywh"
    return "scalar_regression"


def _output_names(task: TaskSpec) -> List[str]:
    if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
        return ["logits"]
    if task.task_type == TaskType.SEGMENTATION:
        return ["segmentation_logits"]
    if task.task_type == TaskType.DETECTION:
        return ["detection_head"]
    return ["value"]


def _modalities(task: TaskSpec) -> List[str]:
    if task.task_type == TaskType.MULTIMODAL:
        return list(task.extras.get("modalities", []))
    return ["image"]


def _runtime_enabled(task: TaskSpec) -> bool:
    return task.task_id not in EXCLUDED_RUNTIME_TASKS


def export_record_template(record: CheckpointRecord, image_size: int) -> ExportRecord:
    task = record.task
    return ExportRecord(
        task_id=task.task_id,
        output_name=task.output_name,
        domain=task.domain,
        title=task.title,
        task_type=task.task_type.value,
        labels=list(task.labels),
        modalities=_modalities(task),
        image_size=image_size,
        runtime_enabled=_runtime_enabled(task),
        output_semantics=_output_semantics(task),
        input_names=_modalities(task),
        output_names=_output_names(task),
        source_checkpoint=str(record.checkpoint_path),
        model_variant=record.variant,
        student_width=record.student_width,
        coreml_path=None,
        exported=False,
    )


def build_export_module(record: CheckpointRecord) -> Tuple[nn.Module, Tuple[torch.Tensor, ...], List[str], List[str]]:
    task = record.task
    model = build_model(task)
    model.load_state_dict(record.state_dict)
    model.eval()
    image_size = int(record.config.get("image_size", 224))
    if task.task_type == TaskType.MULTIMODAL:
        modalities = list(task.extras.get("modalities", []))
        example_inputs = tuple(torch.rand(1, 3, image_size, image_size, dtype=torch.float32) for _ in modalities)
        if len(modalities) == 2:
            wrapper = DualModalExportWrapper(model, modalities)
        elif len(modalities) == 3:
            wrapper = TripleModalExportWrapper(model, modalities)
        elif len(modalities) == 4:
            wrapper = QuadModalExportWrapper(model, modalities)
        else:
            raise ValueError(f"Unsupported multimodal arity for Core ML export: {len(modalities)}")
        return wrapper, example_inputs, modalities, _output_names(task)

    wrapper = SingleImageExportWrapper(model, task.task_type)
    example_inputs = (torch.rand(1, 3, image_size, image_size, dtype=torch.float32),)
    return wrapper, example_inputs, ["image"], _output_names(task)


def _image_input_type(name: str, example: torch.Tensor):
    kwargs: Dict[str, Any] = {
        "name": name,
        "shape": tuple(example.shape),
        "scale": 1.0 / 255.0,
    }
    if hasattr(ct, "colorlayout"):
        kwargs["color_layout"] = ct.colorlayout.RGB
    return ct.ImageType(**kwargs)


def convert_checkpoint_to_coreml(
    record: CheckpointRecord,
    output_dir: Path,
    minimum_ios_version: int = 16,
    compute_precision: str = "float16",
) -> ExportRecord:
    if ct is None:
        raise ModuleNotFoundError(
            "coremltools is not installed. Install the optional Core ML export requirements first."
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export = export_record_template(record, image_size=int(record.config.get("image_size", 224)))
    module, example_inputs, input_names, output_names = build_export_module(record)
    module.eval()
    traced = torch.jit.trace(module, example_inputs, strict=False)
    inputs = [_image_input_type(name, example) for name, example in zip(input_names, example_inputs)]
    outputs = [ct.TensorType(name=name) for name in output_names]
    target = getattr(ct.target, f"iOS{minimum_ios_version}")
    precision = None
    if compute_precision.lower() == "float16" and hasattr(ct, "precision"):
        precision = ct.precision.FLOAT16
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=target,
        compute_precision=precision,
    )
    if hasattr(mlmodel, "author"):
        mlmodel.author = "PULSE"
    if hasattr(mlmodel, "short_description"):
        mlmodel.short_description = f"{record.task.title} exported from PULSE for on-device iOS inference."
    if hasattr(mlmodel, "user_defined_metadata"):
        mlmodel.user_defined_metadata["pulse.task_id"] = record.task.task_id
        mlmodel.user_defined_metadata["pulse.task_type"] = record.task.task_type.value
        mlmodel.user_defined_metadata["pulse.domain"] = record.task.domain
        mlmodel.user_defined_metadata["pulse.labels"] = json.dumps(record.task.labels)
        mlmodel.user_defined_metadata["pulse.output_semantics"] = export.output_semantics
    model_path = output_dir / f"{record.task.output_name}.mlpackage"
    mlmodel.save(str(model_path))
    export.coreml_path = str(model_path)
    export.exported = True
    return export


def manifest_payload(
    source_model_root: Path,
    exports: Sequence[ExportRecord],
) -> Dict[str, Any]:
    exported = [item for item in exports if item.exported]
    skipped = [item for item in exports if not item.exported]
    return {
        "name": "PULSE Core ML Export Manifest",
        "source_model_root": str(Path(source_model_root).resolve()),
        "model_count": len(exports),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "variants": sorted({item.model_variant for item in exports}),
        "models": [item.to_dict() for item in exports],
        "external_specialists": [
            {
                "name": "MobileFetalCLIP",
                "status": "on_device",
                "reason": "Exported separately as a bundled Core ML image encoder with prompt-bank routing metadata.",
            },
            {
                "name": "FetalNet",
                "status": "on_device",
                "reason": "Exported separately as a bundled Core ML dual-head fetal measurement model.",
            },
            {
                "name": "Roboflow fetal brain abnormality classifier",
                "status": "not_on_device",
                "reason": "Current integration remains remote and is not packaged into the offline iPhone build.",
            },
        ],
    }
