from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageDraw = None

from ..geometry import ellipse_circumference
from .schemas import ToolResult

INPUT_SIZE = 224
CLASS_LABELS = ["head", "abdomen", "femur", "background"]
VIEW_TO_CLASS = {
    "brain": "head",
    "abdomen": "abdomen",
    "femur": "femur",
}
TASK_ID_BY_CLASS = {
    "head": "fetalnet/head_biometry",
    "abdomen": "fetalnet/abdominal_circumference",
    "femur": "fetalnet/femur_length",
}
VIEW_CLASSIFICATION_TASK_ID = "fetalnet/view_classification"
TITLE_BY_CLASS = {
    "head": "FetalNet head biometry",
    "abdomen": "FetalNet abdominal circumference",
    "femur": "FetalNet femur length",
}
VIEW_LABEL_BY_CLASS = {
    "head": "fetal brain",
    "abdomen": "fetal abdomen",
    "femur": "fetal femur",
}
VIEW_KEY_BY_CLASS = {
    "head": "brain",
    "abdomen": "abdomen",
    "femur": "femur",
}


def _square_pad_gray(image: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    image = image.convert("L")
    width, height = image.size
    side = max(width, height)
    padded = Image.new("L", (side, side), 0)
    offset = ((side - width) // 2, (side - height) // 2)
    padded.paste(image, offset)
    return padded, offset


@dataclass
class FetalNetStatus:
    enabled: bool
    available: bool
    reason: str
    repo_root: str | None
    weights_path: str | None

    def to_dict(self) -> Dict[str, str | bool | None]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "reason": self.reason,
            "repo_root": self.repo_root,
            "weights_path": self.weights_path,
        }


class FetalNetAdapter:
    def __init__(
        self,
        device: torch.device,
        search_roots: Sequence[Path],
        enabled: bool = True,
        repo_root: str | None = None,
        weights_path: str | None = None,
    ):
        self.device = device
        self.search_roots = [Path(root).resolve() for root in search_roots]
        self.enabled = enabled
        self.repo_root = self._resolve_repo_root(repo_root)
        self.weights_path = self._resolve_weights_path(weights_path)
        self.status = self._build_status()
        self.model = None
        self._load_error: str | None = None

    def _candidate_paths(self, explicit: str | None, relative_candidates: Sequence[str]) -> List[Path]:
        candidates: List[Path] = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        for root in self.search_roots:
            for relative in relative_candidates:
                candidates.append(root / relative)
        return candidates

    def _resolve_repo_root(self, explicit: str | None) -> Path | None:
        candidates = self._candidate_paths(
            explicit,
            [
                "external/FetalNet",
                "third_party/FetalNet",
                "FetalNet",
            ],
        )
        for candidate in candidates:
            model_file = candidate / "model" / "FetalNet.py"
            if model_file.exists():
                return candidate.resolve()
        return None

    def _resolve_weights_path(self, explicit: str | None) -> Path | None:
        candidates = self._candidate_paths(
            explicit,
            [
                "runs/fuvai_weights.pt",
                "runs/fetalnet_weights.pt",
                "fetalnet_weights.pt",
                "weights/fetalnet_weights.pt",
                "weights/FetalNet_weights.pt",
                "models/fetalnet_weights.pt",
            ],
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _build_status(self) -> FetalNetStatus:
        if not self.enabled:
            return FetalNetStatus(False, False, "FetalNet integration is disabled.", None, None)
        if np is None or Image is None or ImageDraw is None:
            return FetalNetStatus(
                True,
                False,
                "FetalNet integration requires `numpy` and `pillow`.",
                str(self.repo_root) if self.repo_root else None,
                str(self.weights_path) if self.weights_path else None,
            )
        if self.repo_root is None:
            return FetalNetStatus(True, False, "FetalNet source checkout was not found.", None, str(self.weights_path) if self.weights_path else None)
        if self.weights_path is None:
            return FetalNetStatus(True, False, "FetalNet weights were not found.", str(self.repo_root), None)
        return FetalNetStatus(True, True, "ready", str(self.repo_root), str(self.weights_path))

    def availability(self) -> Dict[str, str | bool | None]:
        return self.status.to_dict()

    def supports_view(self, fetal_view_key: str | None) -> bool:
        return self.status.available and fetal_view_key in VIEW_TO_CLASS

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _ensure_loaded(self) -> None:
        if not self.status.available:
            raise RuntimeError(self.status.reason)
        if self.model is not None:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)

        try:
            module_path = self.repo_root / "model" / "FetalNet.py"
            spec = importlib.util.spec_from_file_location("pulse_fetalnet_upstream", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not import FetalNet from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.YNet(1, 64, 1)
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            raise RuntimeError(self._load_error) from exc

    def analyze(
        self,
        image_path: Path,
        fetal_view_key: str,
        pixel_spacing_mm: float | None = None,
        artifact_root: Path | None = None,
    ) -> ToolResult:
        expected_class = VIEW_TO_CLASS.get(fetal_view_key)
        if expected_class is None:
            return ToolResult(
                tool_name="fetalnet/measurement",
                task_id=None,
                status="skipped",
                summary="FetalNet only supports fetal head, abdomen, and femur measurements.",
                domain="fetal",
            )
        if not self.status.available:
            return ToolResult(
                tool_name=TASK_ID_BY_CLASS[expected_class],
                task_id=TASK_ID_BY_CLASS[expected_class],
                status="unavailable",
                summary=self.status.reason,
                domain="fetal",
                outputs={
                    "model_name": "FetalNet",
                    "target_class": expected_class,
                },
            )

        try:
            self._ensure_loaded()
            tensor, original_size, pad_offset = self._preprocess(image_path)
            with torch.inference_mode():
                with self._autocast():
                    class_logits, mask_logits = self.model(tensor)
            class_probs = torch.softmax(class_logits, dim=1)[0].detach().cpu()
            top_index = int(torch.argmax(class_probs).item())
            predicted_class = CLASS_LABELS[top_index]
            confidence = float(class_probs[top_index].item())
            top_probs, top_indices = torch.topk(class_probs, k=min(4, class_probs.numel()))
            top_predictions = [
                {"label": CLASS_LABELS[int(index)], "probability": float(prob)}
                for prob, index in zip(top_probs.tolist(), top_indices.tolist())
            ]
            expected_confidence = float(class_probs[CLASS_LABELS.index(expected_class)].item())
            if predicted_class != expected_class and expected_confidence < 0.35:
                return ToolResult(
                    tool_name=TASK_ID_BY_CLASS[expected_class],
                    task_id=TASK_ID_BY_CLASS[expected_class],
                    status="skipped",
                    summary=(
                        f"FetalNet classification is not consistent with the expected `{expected_class}` plane. "
                        f"Top class is `{predicted_class}`."
                    ),
                    domain="fetal",
                    confidence=confidence,
                    outputs={
                        "model_name": "FetalNet",
                        "predicted_class": predicted_class,
                        "expected_class": expected_class,
                        "expected_confidence": expected_confidence,
                        "top_predictions": top_predictions,
                    },
                )

            mask_prob = torch.sigmoid(mask_logits)[0, 0].detach().cpu().float().numpy()
            restored_mask = self._restore_mask(mask_prob, original_size, pad_offset)
            binary_mask = restored_mask > 0.6
            if binary_mask.sum() < 20:
                return ToolResult(
                    tool_name=TASK_ID_BY_CLASS[expected_class],
                    task_id=TASK_ID_BY_CLASS[expected_class],
                    status="error",
                    summary="FetalNet did not produce a usable segmentation mask for measurement.",
                    domain="fetal",
                    confidence=confidence,
                    outputs={
                        "model_name": "FetalNet",
                        "predicted_class": predicted_class,
                        "expected_class": expected_class,
                        "expected_confidence": expected_confidence,
                        "top_predictions": top_predictions,
                    },
                )

            measurement = self._measure(binary_mask, expected_class, pixel_spacing_mm)
            artifacts = {}
            if artifact_root is not None:
                artifact_root.mkdir(parents=True, exist_ok=True)
                mask_path = artifact_root / f"fetalnet_{expected_class}_mask.png"
                overlay_path = artifact_root / f"fetalnet_{expected_class}_overlay.png"
                self._save_artifacts(image_path, binary_mask, measurement, mask_path, overlay_path)
                artifacts = {
                    "mask": str(mask_path),
                    "overlay": str(overlay_path),
                }

            summary = self._summary(expected_class, measurement)
            outputs = {
                "model_name": "FetalNet",
                "predicted_class": predicted_class,
                "expected_class": expected_class,
                "expected_confidence": expected_confidence,
                "top_predictions": top_predictions,
                "positive_fraction": float(binary_mask.mean()),
                **measurement,
            }
            return ToolResult(
                tool_name=TASK_ID_BY_CLASS[expected_class],
                task_id=TASK_ID_BY_CLASS[expected_class],
                status="completed",
                summary=summary,
                domain="fetal",
                confidence=expected_confidence,
                outputs=outputs,
                artifacts=artifacts,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=TASK_ID_BY_CLASS[expected_class],
                task_id=TASK_ID_BY_CLASS[expected_class],
                status="error",
                summary="FetalNet inference failed.",
                domain="fetal",
                outputs={"model_name": "FetalNet", "expected_class": expected_class},
                errors=[str(exc)],
            )

    def classify_view(self, image_path: Path) -> ToolResult:
        if not self.status.available:
            return ToolResult(
                tool_name=VIEW_CLASSIFICATION_TASK_ID,
                task_id=VIEW_CLASSIFICATION_TASK_ID,
                status="unavailable",
                summary=self.status.reason,
                domain="fetal",
                outputs={"model_name": "FetalNet"},
            )

        try:
            self._ensure_loaded()
            tensor, _, _ = self._preprocess(image_path)
            with torch.inference_mode():
                with self._autocast():
                    class_logits, _ = self.model(tensor)
            class_probs = torch.softmax(class_logits, dim=1)[0].detach().cpu()
            top_index = int(torch.argmax(class_probs).item())
            predicted_class = CLASS_LABELS[top_index]
            confidence = float(class_probs[top_index].item())
            top_probs, top_indices = torch.topk(class_probs, k=min(4, class_probs.numel()))
            top_predictions = [
                {"label": CLASS_LABELS[int(index)], "probability": float(prob)}
                for prob, index in zip(top_probs.tolist(), top_indices.tolist())
            ]
            if predicted_class == "background":
                return ToolResult(
                    tool_name=VIEW_CLASSIFICATION_TASK_ID,
                    task_id=VIEW_CLASSIFICATION_TASK_ID,
                    status="skipped",
                    summary="FetalNet did not identify a supported head, abdomen, or femur plane.",
                    domain="fetal",
                    confidence=confidence,
                    outputs={
                        "model_name": "FetalNet",
                        "predicted_class": predicted_class,
                        "top_predictions": top_predictions,
                    },
                )
            return ToolResult(
                tool_name=VIEW_CLASSIFICATION_TASK_ID,
                task_id=VIEW_CLASSIFICATION_TASK_ID,
                status="completed",
                summary=f"FetalNet view classification favors `{VIEW_LABEL_BY_CLASS[predicted_class]}`.",
                domain="fetal",
                confidence=confidence,
                outputs={
                    "model_name": "FetalNet",
                    "predicted_class": predicted_class,
                    "label_key": VIEW_KEY_BY_CLASS[predicted_class],
                    "label": VIEW_LABEL_BY_CLASS[predicted_class],
                    "top_predictions": [
                        {
                            "label": VIEW_LABEL_BY_CLASS.get(item["label"], item["label"]),
                            "label_key": VIEW_KEY_BY_CLASS.get(item["label"]),
                            "probability": item["probability"],
                        }
                        for item in top_predictions
                        if item["label"] != "background"
                    ],
                },
            )
        except Exception as exc:
            return ToolResult(
                tool_name=VIEW_CLASSIFICATION_TASK_ID,
                task_id=VIEW_CLASSIFICATION_TASK_ID,
                status="error",
                summary="FetalNet view classification failed.",
                domain="fetal",
                outputs={"model_name": "FetalNet"},
                errors=[str(exc)],
            )

    def _preprocess(self, image_path: Path) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        image = Image.open(image_path).convert("L")
        original_size = image.size
        padded, pad_offset = _square_pad_gray(image)
        resized = padded.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, original_size, pad_offset

    def _restore_mask(
        self,
        mask_prob: "np.ndarray",
        original_size: tuple[int, int],
        pad_offset: tuple[int, int],
    ) -> "np.ndarray":
        width, height = original_size
        side = max(width, height)
        mask_image = Image.fromarray((mask_prob * 255.0).clip(0, 255).astype("uint8"), mode="L")
        square_mask = mask_image.resize((side, side), Image.BILINEAR)
        crop_box = (pad_offset[0], pad_offset[1], pad_offset[0] + width, pad_offset[1] + height)
        cropped = square_mask.crop(crop_box)
        return np.asarray(cropped, dtype=np.float32) / 255.0

    def _measure(
        self,
        binary_mask: "np.ndarray",
        expected_class: str,
        pixel_spacing_mm: float | None,
    ) -> Dict[str, Any]:
        coords_yx = np.column_stack(np.where(binary_mask > 0))
        coords_xy = np.stack([coords_yx[:, 1], coords_yx[:, 0]], axis=1).astype(np.float64)
        center = coords_xy.mean(axis=0)
        centered = coords_xy - center
        if centered.shape[0] < 3:
            eigvecs = np.eye(2, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                projections = centered @ eigvecs
        else:
            cov = np.cov(centered, rowvar=False)
            if not np.all(np.isfinite(cov)):
                eigvecs = np.eye(2, dtype=np.float64)
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    projections = centered @ eigvecs
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, order]
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    projections = centered @ eigvecs
                if not np.all(np.isfinite(projections)):
                    eigvecs = np.eye(2, dtype=np.float64)
                    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                        projections = centered @ eigvecs
        min_proj = projections.min(axis=0)
        max_proj = projections.max(axis=0)
        major_axis_px = float(max_proj[0] - min_proj[0] + 1.0)
        minor_axis_px = float(max_proj[1] - min_proj[1] + 1.0)
        scale = float(pixel_spacing_mm) if pixel_spacing_mm and pixel_spacing_mm > 0 else 1.0
        unit = "mm" if pixel_spacing_mm and pixel_spacing_mm > 0 else "pixels"

        measurement: Dict[str, Any] = {
            "measurement_unit": unit,
            "major_axis": major_axis_px * scale,
            "minor_axis": minor_axis_px * scale,
            "major_axis_pixels": major_axis_px,
            "minor_axis_pixels": minor_axis_px,
            "center_xy": [float(center[0]), float(center[1])],
            "principal_axes": eigvecs.tolist(),
            "projection_bounds": {"min": min_proj.tolist(), "max": max_proj.tolist()},
        }
        if expected_class == "head":
            hc_value = float(ellipse_circumference(major_axis_px, minor_axis_px) * scale)
            bpd_value = float(minor_axis_px * scale)
            measurement.update(
                {
                    "hc_value": hc_value,
                    "bpd_value": bpd_value,
                    "hc_unit": unit,
                    "bpd_unit": unit,
                }
            )
        elif expected_class == "abdomen":
            ac_value = float(ellipse_circumference(major_axis_px, minor_axis_px) * scale)
            measurement.update(
                {
                    "value": ac_value,
                    "measurement_name": "AC",
                    "unit": unit,
                }
            )
        elif expected_class == "femur":
            fl_value = float(major_axis_px * scale)
            measurement.update(
                {
                    "value": fl_value,
                    "measurement_name": "FL",
                    "unit": unit,
                }
            )
        return measurement

    def _summary(self, expected_class: str, measurement: Dict[str, Any]) -> str:
        if expected_class == "head":
            return (
                "FetalNet head biometry estimated "
                f"HC = {measurement['hc_value']:.2f} {measurement['hc_unit']} and "
                f"BPD = {measurement['bpd_value']:.2f} {measurement['bpd_unit']}."
            )
        return (
            f"FetalNet estimated {measurement['measurement_name']} = "
            f"{measurement['value']:.2f} {measurement['unit']}."
        )

    def _save_artifacts(
        self,
        image_path: Path,
        binary_mask: "np.ndarray",
        measurement: Dict[str, Any],
        mask_path: Path,
        overlay_path: Path,
    ) -> None:
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.fromarray(binary_mask.astype("uint8") * 255, mode="L")
        mask_image.save(mask_path)

        overlay = Image.new("RGBA", image.size, (0, 255, 0, 0))
        overlay_pixels = overlay.load()
        mask_pixels = mask_image.load()
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if mask_pixels[x, y] > 0:
                    overlay_pixels[x, y] = (0, 255, 0, 84)
        composed = Image.alpha_composite(image.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(composed)
        center = measurement["center_xy"]
        major_axis_px = float(measurement["major_axis_pixels"])
        minor_axis_px = float(measurement["minor_axis_pixels"])
        eigvecs = np.asarray(measurement["principal_axes"], dtype=np.float32)
        if measurement.get("measurement_name") == "FL":
            min_major = float(measurement["projection_bounds"]["min"][0])
            max_major = float(measurement["projection_bounds"]["max"][0])
            axis = eigvecs[:, 0]
            start = (center[0] + axis[0] * min_major, center[1] + axis[1] * min_major)
            end = (center[0] + axis[0] * max_major, center[1] + axis[1] * max_major)
            draw.line([start, end], fill=(255, 215, 0, 255), width=3)
        else:
            points: List[tuple[float, float]] = []
            angles = np.linspace(0.0, 2.0 * np.pi, num=96, endpoint=False)
            for angle in angles:
                local = np.array(
                    [
                        (major_axis_px / 2.0) * np.cos(angle),
                        (minor_axis_px / 2.0) * np.sin(angle),
                    ],
                    dtype=np.float32,
                )
                rotated = eigvecs @ local
                points.append((float(center[0] + rotated[0]), float(center[1] + rotated[1])))
            draw.line(points + [points[0]], fill=(255, 215, 0, 255), width=3)

            if "bpd_value" in measurement:
                axis = eigvecs[:, 1]
                start = (center[0] - axis[0] * (minor_axis_px / 2.0), center[1] - axis[1] * (minor_axis_px / 2.0))
                end = (center[0] + axis[0] * (minor_axis_px / 2.0), center[1] + axis[1] * (minor_axis_px / 2.0))
                draw.line([start, end], fill=(255, 0, 0, 255), width=3)
        composed.save(overlay_path)
