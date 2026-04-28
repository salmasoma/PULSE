from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

try:
    from inference_sdk import InferenceHTTPClient
except ModuleNotFoundError:  # pragma: no cover
    InferenceHTTPClient = None

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None

from .schemas import ToolResult

DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_FETAL_BRAIN_MODEL_ID = "fetal-brain-abnormalities-ultrasound/1"


@dataclass
class RoboflowStatus:
    enabled: bool
    available: bool
    reason: str
    api_url: str
    model_id: str
    deployment_mode: str
    api_key_configured: bool

    def to_dict(self) -> Dict[str, str | bool]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "reason": self.reason,
            "api_url": self.api_url,
            "model_id": self.model_id,
            "deployment_mode": self.deployment_mode,
            "api_key_configured": self.api_key_configured,
        }


class RoboflowBrainAdapter:
    def __init__(
        self,
        enabled: bool = True,
        api_url: str = DEFAULT_ROBOFLOW_API_URL,
        api_key: str | None = None,
        model_id: str = DEFAULT_FETAL_BRAIN_MODEL_ID,
    ):
        self.enabled = enabled
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id
        self.status = self._build_status()

    def _deployment_mode(self) -> str:
        parsed = urlparse(self.api_url)
        hostname = (parsed.hostname or "").lower()
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return "local"
        return "hosted"

    def _build_status(self) -> RoboflowStatus:
        deployment_mode = self._deployment_mode()
        api_key_configured = bool(self.api_key)
        if not self.enabled:
            return RoboflowStatus(
                False,
                False,
                "Roboflow fetal-brain specialist is disabled.",
                self.api_url,
                self.model_id,
                deployment_mode,
                api_key_configured,
            )
        if InferenceHTTPClient is None:
            if requests is None:
                return RoboflowStatus(
                    True,
                    False,
                    "Neither inference-sdk nor requests is installed.",
                    self.api_url,
                    self.model_id,
                    deployment_mode,
                    api_key_configured,
                )
        if deployment_mode == "hosted" and not self.api_key:
            return RoboflowStatus(
                True,
                False,
                "Roboflow API key is required for the hosted serverless endpoint.",
                self.api_url,
                self.model_id,
                deployment_mode,
                api_key_configured,
            )
        reason = "ready"
        if InferenceHTTPClient is None:
            reason = "ready (requests fallback mode)"
        if deployment_mode == "local" and not self.api_key:
            reason = "ready (local server mode without an API key; ensure the model is already available to the local server)."
        return RoboflowStatus(True, True, reason, self.api_url, self.model_id, deployment_mode, api_key_configured)

    def availability(self) -> Dict[str, str | bool]:
        return self.status.to_dict()

    def _client(self) -> InferenceHTTPClient:
        kwargs = {"api_url": self.api_url}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return InferenceHTTPClient(**kwargs)

    def _split_model_id(self) -> tuple[str, str]:
        if "/" not in self.model_id:
            raise ValueError(f"Roboflow model_id must have the form '<project>/<version>', got: {self.model_id}")
        project_id, model_version = self.model_id.rsplit("/", 1)
        return project_id, model_version

    def _infer_via_requests(self, image_path: Path) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests is not installed.")
        project_id, model_version = self._split_model_id()
        image_bytes = Path(image_path).read_bytes()
        if self.status.deployment_mode == "local":
            import base64

            payload = {
                "model_id": self.model_id,
                "image": {
                    "type": "base64",
                    "value": base64.b64encode(image_bytes).decode("ascii"),
                },
            }
            if self.api_key:
                payload["api_key"] = self.api_key
            response = requests.post(
                f"{self.api_url.rstrip('/')}/infer/classification",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()

        import base64

        response = requests.post(
            f"https://classify.roboflow.com/{project_id}/{model_version}?api_key={self.api_key}",
            data=base64.b64encode(image_bytes),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _infer(self, image_path: Path) -> Dict[str, Any]:
        if InferenceHTTPClient is not None:
            return self._client().infer(str(Path(image_path).resolve()), model_id=self.model_id)
        return self._infer_via_requests(image_path)

    def analyze(self, image_path: Path) -> ToolResult:
        if not self.status.available:
            return ToolResult(
                tool_name="roboflow/fetal_brain_abnormality_classification",
                task_id="roboflow/fetal_brain_abnormality_classification",
                status="unavailable",
                summary=self.status.reason,
                domain="fetal",
                outputs={
                    "model_name": "Roboflow fetal brain abnormality classifier",
                    "model_id": self.model_id,
                    "api_url": self.api_url,
                    "deployment_mode": self.status.deployment_mode,
                },
            )

        try:
            result = self._infer(image_path)
            predictions = self._parse_predictions(result)
            predicted_classes = self._predicted_classes(result, predictions)
            abnormal_classes = [label for label in predicted_classes if label != "normal"]
            confidence = max((item["probability"] for item in predictions), default=0.0)
            top_label = predictions[0]["label"] if predictions else (predicted_classes[0] if predicted_classes else None)
            normal_confidence = next(
                (float(item["probability"]) for item in predictions if str(item["label"]).lower() == "normal"),
                None,
            )

            if abnormal_classes:
                summary = "Roboflow fetal brain abnormality classifier flags " + ", ".join(f"`{label}`" for label in abnormal_classes) + "."
            elif predicted_classes:
                summary = f"Roboflow fetal brain abnormality classifier favors `{predicted_classes[0]}`."
            else:
                summary = "Roboflow fetal brain abnormality classifier returned no positive labels."

            return ToolResult(
                tool_name="roboflow/fetal_brain_abnormality_classification",
                task_id="roboflow/fetal_brain_abnormality_classification",
                status="completed",
                summary=summary,
                domain="fetal",
                confidence=confidence,
                outputs={
                    "model_name": "Roboflow fetal brain abnormality classifier",
                    "model_id": self.model_id,
                    "api_url": self.api_url,
                    "deployment_mode": self.status.deployment_mode,
                    "predicted_classes": predicted_classes,
                    "abnormal_classes": abnormal_classes,
                    "top_label": top_label,
                    "normal_confidence": normal_confidence,
                    "top_predictions": predictions[:5],
                    "raw_response": result,
                },
            )
        except Exception as exc:
            return ToolResult(
                tool_name="roboflow/fetal_brain_abnormality_classification",
                task_id="roboflow/fetal_brain_abnormality_classification",
                status="error",
                summary="Roboflow fetal brain abnormality inference failed.",
                domain="fetal",
                outputs={
                    "model_name": "Roboflow fetal brain abnormality classifier",
                    "model_id": self.model_id,
                    "api_url": self.api_url,
                    "deployment_mode": self.status.deployment_mode,
                },
                errors=[str(exc)],
            )

    def _parse_predictions(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions = response.get("predictions", {})
        parsed: List[Dict[str, Any]] = []
        if isinstance(predictions, dict):
            for label, payload in predictions.items():
                if isinstance(payload, dict):
                    confidence = float(payload.get("confidence", 0.0))
                else:
                    confidence = float(payload)
                parsed.append({"label": str(label), "probability": confidence})
        elif isinstance(predictions, list):
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                label = item.get("class", item.get("label"))
                confidence = float(item.get("confidence", 0.0))
                if label is None:
                    continue
                parsed.append({"label": str(label), "probability": confidence})
        parsed.sort(key=lambda item: item["probability"], reverse=True)
        return parsed

    def _predicted_classes(self, response: Dict[str, Any], predictions: List[Dict[str, Any]]) -> List[str]:
        predicted_classes = response.get("predicted_classes", [])
        if isinstance(predicted_classes, list):
            labels = [str(label) for label in predicted_classes if label]
            if labels:
                return labels
        top = response.get("top")
        if isinstance(top, str) and top:
            return [top]
        if predictions:
            return [predictions[0]["label"]]
        return []
