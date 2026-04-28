from __future__ import annotations

import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

try:
    import open_clip
except ModuleNotFoundError:  # pragma: no cover
    open_clip = None

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None

from .schemas import ToolResult

INPUT_SIZE = 224
OPEN_CLIP_COMPAT_VERSION = "2.26.1"

FIVE_PLANE_PROMPTS = {
    "abdomen": [
        "Fetal ultrasound image centered on the abdomen with abdominal anatomy in view.",
        "Obstetric ultrasound focused on the fetal abdominal region and organ contours.",
        "Prenatal ultrasound showing a standard fetal abdominal plane.",
    ],
    "brain": [
        "Fetal ultrasound image centered on the brain with intracranial anatomy in view.",
        "Obstetric ultrasound focused on the fetal head and brain structures.",
        "Prenatal ultrasound showing a standard fetal brain plane.",
    ],
    "femur": [
        "Fetal ultrasound image centered on the femur and long-bone anatomy.",
        "Obstetric ultrasound focused on the fetal femur and skeletal measurement view.",
        "Prenatal ultrasound showing a standard fetal femur plane.",
    ],
    "heart": [
        "Fetal ultrasound image centered on the heart with cardiac anatomy in view.",
        "Obstetric ultrasound focused on the fetal thorax and heart structures.",
        "Prenatal ultrasound showing a standard fetal cardiac plane.",
    ],
    "kidney": [
        "Fetal ultrasound image centered on the kidney and retroperitoneal anatomy.",
        "Obstetric ultrasound focused on the fetal kidney and renal structures.",
        "Prenatal ultrasound showing a fetal renal plane.",
    ],
    "lips_nose": [
        "Fetal ultrasound image centered on the lips and nose with facial anatomy in view.",
        "Obstetric ultrasound focused on the fetal lips and nasal profile.",
        "Prenatal ultrasound showing a fetal lips and nose view.",
    ],
    "profile_patient": [
        "Fetal ultrasound image centered on the facial profile.",
        "Obstetric ultrasound focused on the fetal sagittal profile and face.",
        "Prenatal ultrasound showing a fetal profile view.",
    ],
    "spine": [
        "Fetal ultrasound image centered on the spine and vertebral alignment.",
        "Obstetric ultrasound focused on the fetal spine and posterior anatomy.",
        "Prenatal ultrasound showing a fetal spinal plane.",
    ],
    "cervix": [
        "Obstetric ultrasound image centered on the maternal cervix.",
        "Prenatal ultrasound focused on the cervical canal and cervix.",
        "Transabdominal or transvaginal ultrasound showing the maternal cervix.",
    ],
}

BRAIN_SUBPLANE_PROMPTS = {
    "trans-thalamic": [
        "Fetal brain ultrasound in a trans-thalamic plane.",
        "Prenatal ultrasound showing the transverse thalamic brain view.",
        "Fetal cranial ultrasound focused on the trans-thalamic plane.",
    ],
    "trans-cerebellum": [
        "Fetal brain ultrasound in a trans-cerebellar plane.",
        "Prenatal ultrasound showing the transverse cerebellar brain view.",
        "Fetal cranial ultrasound focused on the trans-cerebellum plane.",
    ],
    "trans-ventricular": [
        "Fetal brain ultrasound in a trans-ventricular plane.",
        "Prenatal ultrasound showing the transverse ventricular brain view.",
        "Fetal cranial ultrasound focused on the trans-ventricular plane.",
    ],
}

FIVE_PLANE_DISPLAY = {
    "abdomen": "fetal abdomen",
    "brain": "fetal brain",
    "femur": "fetal femur",
    "heart": "fetal heart",
    "kidney": "fetal kidney",
    "lips_nose": "fetal lips/nose",
    "profile_patient": "fetal profile",
    "spine": "fetal spine",
    "cervix": "maternal cervix",
}
def _make_square_with_zero_padding(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    if image.mode == "L":
        padded = Image.new("L", (max_side, max_side), 0)
    else:
        padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
    offset = ((max_side - width) // 2, (max_side - height) // 2)
    padded.paste(image, offset)
    return padded.convert("RGB")


@dataclass
class FetalCLIPStatus:
    enabled: bool
    available: bool
    reason: str
    weights_path: str | None
    config_path: str | None

    def to_dict(self) -> Dict[str, str | bool | None]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "reason": self.reason,
            "weights_path": self.weights_path,
            "config_path": self.config_path,
        }


class FetalCLIPAdapter:
    def __init__(
        self,
        device: torch.device,
        search_roots: Sequence[Path],
        enabled: bool = True,
        weights_path: str | None = None,
        config_path: str | None = None,
    ):
        self.device = device
        self.search_roots = [Path(root).resolve() for root in search_roots]
        self.enabled = enabled
        self.weights_path = self._resolve_weights_path(weights_path)
        self.config_path = self._resolve_config_path(config_path)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._prompt_cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}
        self.model_family = "fetalclip"
        self.model_label = "FetalCLIP"
        self.status = self._build_status()
        self._load_error: str | None = None

    def _candidate_paths(self, explicit: str | None, relative_candidates: Sequence[str]) -> List[Path]:
        candidates: List[Path] = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        for root in self.search_roots:
            for relative in relative_candidates:
                candidates.append(root / relative)
        return candidates

    def _resolve_weights_path(self, explicit: str | None) -> Path | None:
        candidates = self._candidate_paths(
            explicit,
            [
                "FetalCLIP_weights.pt",
                "runs/mobile_fetal_clip_weights.pt",
                "mobile_fetal_clip_weights.pt",
                "external/FetalCLIP/FetalCLIP_weights.pt",
                "third_party/FetalCLIP/FetalCLIP_weights.pt",
                "models/FetalCLIP_weights.pt",
                "weights/FetalCLIP_weights.pt",
            ],
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _resolve_config_path(self, explicit: str | None) -> Path | None:
        mobile_candidates = [
            "third_party/MobileFetalCLIP/configs/model/mobileclip2_s0_fetal.json",
            "pulse/assets/fetalclip/mobileclip2_s0_fetal.json",
        ]
        fetal_candidates = [
            "pulse/assets/fetalclip/FetalCLIP_config.json",
            "FetalCLIP_config.json",
            "external/FetalCLIP/FetalCLIP_config.json",
            "third_party/FetalCLIP/FetalCLIP_config.json",
        ]
        preferred = mobile_candidates if self.weights_path and "mobile_fetal_clip" in self.weights_path.name.lower() else fetal_candidates + mobile_candidates
        candidates = self._candidate_paths(
            explicit,
            preferred,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _build_status(self) -> FetalCLIPStatus:
        if not self.enabled:
            return FetalCLIPStatus(False, False, "FetalCLIP integration is disabled.", None, None)
        if open_clip is None:
            return FetalCLIPStatus(True, False, "open-clip-torch is not installed.", str(self.weights_path) if self.weights_path else None, str(self.config_path) if self.config_path else None)
        if Image is None:
            return FetalCLIPStatus(True, False, "Pillow is not installed.", str(self.weights_path) if self.weights_path else None, str(self.config_path) if self.config_path else None)
        if self.weights_path is None:
            return FetalCLIPStatus(True, False, "FetalCLIP weights were not found.", None, str(self.config_path) if self.config_path else None)
        return FetalCLIPStatus(True, True, "ready", str(self.weights_path), str(self.config_path))

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

        checkpoint = self._load_checkpoint_object()
        state_dict = self._extract_state_dict(checkpoint)
        family = self._detect_checkpoint_family(state_dict)
        self.model_family = family
        self.model_label = "MobileFetalCLIP" if family == "mobile_fetal_clip" else "FetalCLIP"

        try:
            if family == "mobile_fetal_clip":
                model, preprocess, tokenizer = self._load_mobile_fetal_clip(state_dict)
            else:
                model, preprocess, tokenizer = self._load_fetalclip(state_dict)
        except Exception as exc:
            message = str(exc)
            if "text pos_embed width changed" not in message and "size mismatch" not in message:
                self._load_error = self._format_load_error(exc)
                raise RuntimeError(self._load_error) from exc

            try:
                model, preprocess, tokenizer = self._load_with_inferred_config(state_dict)
            except Exception as fallback_exc:
                self._load_error = self._format_load_error(fallback_exc, original_error=message)
                raise RuntimeError(self._load_error) from fallback_exc

        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def _format_load_error(self, exc: Exception, original_error: str | None = None) -> str:
        version = getattr(open_clip, "__version__", "unknown") if open_clip is not None else "missing"
        parts = []
        if original_error:
            parts.append(f"Direct {self.model_label} load failed: {original_error}.")
        parts.append(f"{self.model_label} checkpoint load failed: {exc}.")
        parts.append(
            f"Check that you are using the correct checkpoint family and `open-clip-torch=={OPEN_CLIP_COMPAT_VERSION}`."
        )
        parts.append(f"Detected open_clip version: {version}.")
        return " ".join(parts)

    def _load_checkpoint_object(self) -> Any:
        return torch.load(self.weights_path, map_location="cpu")

    def _extract_state_dict(self, checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "module", "network"):
                value = checkpoint.get(key)
                if isinstance(value, dict) and value:
                    checkpoint = value
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format.")

        state_dict: Dict[str, torch.Tensor] = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            cleaned = str(key)
            if cleaned.startswith("module."):
                cleaned = cleaned[len("module.") :]
            state_dict[cleaned] = value
        if not state_dict:
            raise ValueError("No tensor state dict was found in the checkpoint.")
        return state_dict

    def _detect_checkpoint_family(self, state_dict: Dict[str, torch.Tensor]) -> str:
        if any(key.startswith("visual.trunk.patch_embed.") for key in state_dict):
            return "mobile_fetal_clip"
        if any(key.startswith("image_encoder.model.") for key in state_dict):
            return "mobile_fetal_clip"
        return "fetalclip"

    def _load_fetalclip(self, state_dict: Dict[str, torch.Tensor]):
        if self.config_path is None:
            raise RuntimeError("FetalCLIP config file was not found.")
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config
        model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=str(self.weights_path))
        tokenizer = open_clip.get_tokenizer("FetalCLIP")
        return model, preprocess, tokenizer

    def _import_mobile_fetal_clip_factory(self):
        try:
            from mobile_fetal_clip.models.factory import (  # type: ignore
                create_fetal_clip_model,
                get_tokenizer,
                load_pretrained_weights,
            )
            return create_fetal_clip_model, get_tokenizer, load_pretrained_weights
        except ModuleNotFoundError:
            pass

        candidate_dirs = []
        for root in self.search_roots:
            candidate_dirs.append(root / "third_party" / "MobileFetalCLIP" / "src")
            candidate_dirs.append(root / "MobileFetalCLIP" / "src")

        for directory in candidate_dirs:
            if not directory.exists():
                continue
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            try:
                from mobile_fetal_clip.models.factory import (  # type: ignore
                    create_fetal_clip_model,
                    get_tokenizer,
                    load_pretrained_weights,
                )
                return create_fetal_clip_model, get_tokenizer, load_pretrained_weights
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "MobileFetalCLIP source was not found. Expected `third_party/MobileFetalCLIP/src` or an installed `mobile_fetal_clip` package."
        )

    def _load_mobile_fetal_clip(self, state_dict: Dict[str, torch.Tensor]):
        if self.config_path is None:
            raise RuntimeError("MobileFetalCLIP config file was not found.")
        create_fetal_clip_model, get_tokenizer, load_pretrained_weights = self._import_mobile_fetal_clip_factory()
        model, _, preprocess = create_fetal_clip_model(
            str(self.config_path),
            pretrained=None,
            precision="fp32",
            device=str(self.device),
        )
        load_pretrained_weights(model, str(self.weights_path), strict=False)
        tokenizer = get_tokenizer(str(self.config_path))
        return model, preprocess, tokenizer

    def _infer_config_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        token_embedding = state_dict["token_embedding.weight"]
        positional_embedding = state_dict["positional_embedding"]
        text_projection = state_dict["text_projection"]
        conv1 = state_dict["visual.conv1.weight"]
        visual_positional_embedding = state_dict["visual.positional_embedding"]

        vocab_size, text_width = token_embedding.shape
        context_length = positional_embedding.shape[0]
        if text_projection.ndim != 2:
            raise ValueError("Unsupported `text_projection` shape in FetalCLIP checkpoint.")
        embed_dim = int(text_projection.shape[1])

        text_layers = sorted(
            {
                int(key.split(".")[2])
                for key in state_dict
                if key.startswith("transformer.resblocks.") and key.split(".")[2].isdigit()
            }
        )
        visual_layers = sorted(
            {
                int(key.split(".")[3])
                for key in state_dict
                if key.startswith("visual.transformer.resblocks.") and key.split(".")[3].isdigit()
            }
        )
        if not text_layers or not visual_layers:
            raise ValueError("Unable to infer CLIP transformer depth from checkpoint keys.")

        vision_width = int(conv1.shape[0])
        patch_size = int(conv1.shape[-1])
        num_visual_tokens = int(visual_positional_embedding.shape[0]) - 1
        grid_size = int(round(math.sqrt(max(num_visual_tokens, 1))))
        image_size = int(grid_size * patch_size)

        return {
            "embed_dim": embed_dim,
            "vision_cfg": {
                "image_size": image_size,
                "layers": int(max(visual_layers) + 1),
                "width": vision_width,
                "patch_size": patch_size,
            },
            "text_cfg": {
                "context_length": int(context_length),
                "vocab_size": int(vocab_size),
                "width": int(text_width),
                "heads": max(1, int(text_width // 64)),
                "layers": int(max(text_layers) + 1),
            },
        }

    def _load_with_inferred_config(self, state_dict: Dict[str, torch.Tensor]):
        inferred_config = self._infer_config_from_state_dict(state_dict)
        inferred_name = "PULSE_FetalCLIP_Auto"
        open_clip.factory._MODEL_CONFIGS[inferred_name] = inferred_config
        model, _, preprocess = open_clip.create_model_and_transforms(inferred_name, pretrained=None)
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.endswith("attn_mask")]
        missing = [key for key in incompatible.missing_keys if not key.endswith("attn_mask")]
        if unexpected or missing:
            raise RuntimeError(
                f"Incompatible FetalCLIP checkpoint. Missing keys: {missing[:6]}; unexpected keys: {unexpected[:6]}"
            )
        tokenizer = open_clip.get_tokenizer(inferred_name)
        return model, preprocess, tokenizer

    def _encode_image(self, image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._ensure_loaded()
        image = Image.open(image_path)
        original_size = image.size
        prepared = _make_square_with_zero_padding(image)
        tensor = self.preprocess(prepared).unsqueeze(0).to(self.device)

        with torch.no_grad(), self._autocast():
            features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features, original_size

    def _text_embeddings(self, cache_key: str, prompt_bank: Dict[str, List[str]]) -> Tuple[List[str], torch.Tensor]:
        self._ensure_loaded()
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        labels: List[str] = []
        features: List[torch.Tensor] = []
        with torch.no_grad(), self._autocast():
            for label, prompts in prompt_bank.items():
                text_tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_feature = text_features.mean(dim=0, keepdim=True)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                labels.append(label)
                features.append(text_feature)

        stacked = torch.cat(features, dim=0)
        self._prompt_cache[cache_key] = (labels, stacked)
        return labels, stacked

    def _classify_with_prompt_bank(
        self,
        image_features: torch.Tensor,
        prompt_bank: Dict[str, List[str]],
        cache_key: str,
        display_map: Dict[str, str] | None = None,
    ) -> Dict:
        labels, text_features = self._text_embeddings(cache_key, prompt_bank)
        logits = (100.0 * image_features @ text_features.T)[0]
        probs = torch.softmax(logits, dim=-1).cpu()
        topk = min(3, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=topk)
        best_index = int(top_indices[0].item())
        best_key = labels[best_index]
        best_label = display_map.get(best_key, best_key) if display_map else best_key
        top_predictions = []
        for probability, index in zip(top_probs.tolist(), top_indices.tolist()):
            label_key = labels[int(index)]
            label_name = display_map.get(label_key, label_key) if display_map else label_key
            top_predictions.append(
                {
                    "label": label_name,
                    "label_key": label_key,
                    "probability": float(probability),
                }
            )
        return {
            "label": best_label,
            "label_key": best_key,
            "confidence": float(probs[best_index].item()),
            "probabilities": probs.tolist(),
            "top_predictions": top_predictions,
        }

    def availability(self) -> Dict[str, str | bool | None]:
        return self.status.to_dict()

    def analyze(self, image_path: Path) -> List[ToolResult]:
        if not self.status.available:
            return [
                ToolResult(
                    tool_name="fetalclip/plane_zero_shot",
                    task_id="fetalclip/plane_zero_shot",
                    status="unavailable",
                    summary=self.status.reason,
                    domain="fetal",
                )
            ]
        try:
            image_features, _ = self._encode_image(image_path)
            plane_output = self._classify_with_prompt_bank(
                image_features,
                FIVE_PLANE_PROMPTS,
                cache_key="five_plane",
                display_map=FIVE_PLANE_DISPLAY,
            )
            results = [
                ToolResult(
                    tool_name="fetalclip/plane_zero_shot",
                    task_id="fetalclip/plane_zero_shot",
                    status="completed",
                    summary=f"{self.model_label} zero-shot plane analysis favors `{plane_output['label']}`.",
                    domain="fetal",
                    confidence=plane_output["confidence"],
                    outputs={**plane_output, "model_name": self.model_label},
                )
            ]

            if plane_output["label_key"] == "brain":
                brain_output = self._classify_with_prompt_bank(
                    image_features,
                    BRAIN_SUBPLANE_PROMPTS,
                    cache_key="brain_subplane",
                )
                results.append(
                    ToolResult(
                        tool_name="fetalclip/brain_subplane_zero_shot",
                        task_id="fetalclip/brain_subplane_zero_shot",
                        status="completed",
                        summary=f"{self.model_label} brain sub-plane analysis favors `{brain_output['label']}`.",
                        domain="fetal",
                        confidence=brain_output["confidence"],
                        outputs={**brain_output, "model_name": self.model_label},
                    )
                )

            return results
        except Exception as exc:
            return [
                ToolResult(
                    tool_name="fetalclip/plane_zero_shot",
                    task_id="fetalclip/plane_zero_shot",
                    status="error",
                    summary="FetalCLIP inference failed.",
                    domain="fetal",
                    errors=[str(exc)],
                )
            ]
