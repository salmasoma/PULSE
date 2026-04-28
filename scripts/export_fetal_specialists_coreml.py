#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: E402

from pulse.runtime.fetalclip import (  # noqa: E402
    BRAIN_SUBPLANE_PROMPTS,
    FIVE_PLANE_DISPLAY,
    FIVE_PLANE_PROMPTS,
    FetalCLIPAdapter,
)


MOBILE_FETAL_CLIP_MODEL_NAME = "mobile_fetal_clip_image_encoder"
FETALNET_MODEL_NAME = "fetalnet_dual_head"
SPECIALIST_MANIFEST_NAME = "fetal_specialists_manifest.json"


class MobileFetalCLIPImageEncoderWrapper(nn.Module):
    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.mean) / self.std
        embedding = self.visual(normalized)
        return embedding / embedding.norm(dim=-1, keepdim=True)


class FetalNetDualHeadWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[1] == 3:
            gray = (
                (0.299 * image[:, 0:1]) +
                (0.587 * image[:, 1:2]) +
                (0.114 * image[:, 2:3])
            )
        else:
            gray = image[:, :1]
        sequence = gray.unsqueeze(1)
        class_logits, mask_logits = self.model(sequence)
        return class_logits, mask_logits


@dataclass
class SpecialistExportResult:
    name: str
    exported: bool
    model_path: str | None = None
    reason: str = ""
    extra: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "exported": self.exported,
            "model_path": self.model_path,
            "reason": self.reason,
        }
        if self.extra:
            payload.update(self.extra)
        return payload


def image_input_type(name: str, shape: tuple[int, ...]):
    kwargs: Dict[str, Any] = {
        "name": name,
        "shape": shape,
        "scale": 1.0 / 255.0,
    }
    if hasattr(ct, "colorlayout"):
        kwargs["color_layout"] = ct.colorlayout.RGB
    return ct.ImageType(**kwargs)


def convert_and_save(
    module: nn.Module,
    example_input: torch.Tensor,
    output_dir: Path,
    model_name: str,
    output_names: list[str],
    minimum_ios_version: int = 16,
) -> Path:
    traced = torch.jit.trace(module.eval(), (example_input,), strict=False)
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[image_input_type("image", tuple(example_input.shape))],
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=getattr(ct.target, f"iOS{minimum_ios_version}"),
        compute_precision=ct.precision.FLOAT16 if hasattr(ct, "precision") else None,
    )
    mlmodel.author = "PULSE"
    model_path = output_dir / f"{model_name}.mlpackage"
    mlmodel.save(str(model_path))
    return model_path


def export_mobile_fetal_clip(
    output_dir: Path,
    weights_path: Path,
    config_path: Path | None,
) -> SpecialistExportResult:
    adapter = FetalCLIPAdapter(
        device=torch.device("cpu"),
        search_roots=[REPO_ROOT],
        enabled=True,
        weights_path=str(weights_path),
        config_path=str(config_path) if config_path else None,
    )
    adapter._ensure_loaded()

    wrapper = MobileFetalCLIPImageEncoderWrapper(adapter.model.visual).eval()
    image_size = 256
    example = torch.rand(1, 3, image_size, image_size, dtype=torch.float32)
    exported_program = torch.export.export(wrapper, (example,))
    exported_program = exported_program.run_decompositions({})
    mlmodel = ct.convert(
        exported_program,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[image_input_type("image", tuple(example.shape))],
        outputs=[ct.TensorType(name="image_embedding")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16 if hasattr(ct, "precision") else None,
    )
    model_path = output_dir / f"{MOBILE_FETAL_CLIP_MODEL_NAME}.mlpackage"
    mlmodel.save(str(model_path))

    plane_labels, plane_embeddings = adapter._text_embeddings("five_plane_export", FIVE_PLANE_PROMPTS)
    brain_labels, brain_embeddings = adapter._text_embeddings("brain_subplane_export", BRAIN_SUBPLANE_PROMPTS)
    prompt_manifest = {
        "mobile_fetal_clip": {
            "model_name": MOBILE_FETAL_CLIP_MODEL_NAME,
            "input_size": image_size,
            "embedding_dim": int(plane_embeddings.shape[-1]),
            "prompt_sets": {
                "five_plane": {
                    "labels": plane_labels,
                    "display_map": FIVE_PLANE_DISPLAY,
                    "embeddings": plane_embeddings.cpu().tolist(),
                },
                "brain_subplane": {
                    "labels": brain_labels,
                    "display_map": {},
                    "embeddings": brain_embeddings.cpu().tolist(),
                },
            },
        }
    }
    return SpecialistExportResult(
        name="MobileFetalCLIP",
        exported=True,
        model_path=str(model_path),
        extra=prompt_manifest,
    )


def import_fetalnet_model(repo_root: Path) -> nn.Module:
    module_path = repo_root / "model" / "FetalNet.py"
    spec = importlib.util.spec_from_file_location("pulse_ios_fetalnet_upstream", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import FetalNet from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YNet(1, 64, 1)


def export_fetalnet(output_dir: Path, repo_root: Path, weights_path: Path) -> SpecialistExportResult:
    model = import_fetalnet_model(repo_root)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = FetalNetDualHeadWrapper(model).eval()
    image_size = 224
    example = torch.rand(1, 3, image_size, image_size, dtype=torch.float32)
    model_path = convert_and_save(
        wrapper,
        example,
        output_dir=output_dir,
        model_name=FETALNET_MODEL_NAME,
        output_names=["view_logits", "mask_logits"],
    )
    return SpecialistExportResult(
        name="FetalNet",
        exported=True,
        model_path=str(model_path),
        extra={
            "fetalnet": {
                "model_name": FETALNET_MODEL_NAME,
                "input_size": image_size,
                "class_labels": ["head", "abdomen", "femur", "background"],
            }
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileFetalCLIP and FetalNet to Core ML for the iPhone app.")
    parser.add_argument(
        "--output-dir",
        default="/Users/salma.hassan/Ultrasound_Project/ios/PULSEOnDevice/PULSEOnDevice/Resources/Models",
        help="Directory where .mlpackage exports and the specialist manifest are written.",
    )
    parser.add_argument(
        "--mobile-fetal-clip-weights",
        default="/Users/salma.hassan/Ultrasound_Project/runs/mobile_fetal_clip_weights.pt",
        help="Path to the MobileFetalCLIP checkpoint.",
    )
    parser.add_argument(
        "--mobile-fetal-clip-config",
        default="/Users/salma.hassan/Ultrasound_Project/third_party/MobileFetalCLIP/configs/model/mobileclip2_s0_fetal.json",
        help="Path to the MobileFetalCLIP JSON config.",
    )
    parser.add_argument(
        "--fetalnet-repo-root",
        default="/Users/salma.hassan/Ultrasound_Project/external/FetalNet",
        help="Path to the FetalNet source checkout.",
    )
    parser.add_argument(
        "--fetalnet-weights",
        default="/Users/salma.hassan/Ultrasound_Project/runs/fuvai_weights.pt",
        help="Path to the FetalNet checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[SpecialistExportResult] = []
    specialist_manifest: Dict[str, Any] = {
        "name": "PULSE Fetal Specialist Manifest",
        "mobile_fetal_clip": None,
        "fetalnet": None,
    }

    try:
        result = export_mobile_fetal_clip(
            output_dir=output_dir,
            weights_path=Path(args.mobile_fetal_clip_weights).expanduser().resolve(),
            config_path=Path(args.mobile_fetal_clip_config).expanduser().resolve(),
        )
        results.append(result)
        specialist_manifest["mobile_fetal_clip"] = result.extra["mobile_fetal_clip"] if result.extra else None
        print(f"Exported {result.name} -> {Path(result.model_path).name}")
    except Exception as exc:
        results.append(SpecialistExportResult(name="MobileFetalCLIP", exported=False, reason=str(exc)))
        print(f"Failed MobileFetalCLIP export: {exc}")

    try:
        result = export_fetalnet(
            output_dir=output_dir,
            repo_root=Path(args.fetalnet_repo_root).expanduser().resolve(),
            weights_path=Path(args.fetalnet_weights).expanduser().resolve(),
        )
        results.append(result)
        specialist_manifest["fetalnet"] = result.extra["fetalnet"] if result.extra else None
        print(f"Exported {result.name} -> {Path(result.model_path).name}")
    except Exception as exc:
        results.append(SpecialistExportResult(name="FetalNet", exported=False, reason=str(exc)))
        print(f"Failed FetalNet export: {exc}")

    specialist_manifest["results"] = [item.to_dict() for item in results]
    manifest_path = output_dir / SPECIALIST_MANIFEST_NAME
    manifest_path.write_text(json.dumps(specialist_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    return 0 if all(item.exported for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
