from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from ..data import _load_image, _normalize, _pil_to_tensor
from ..discovery import discover_all_tasks
from ..specs import TaskSpec, TaskType
from .artifacts import ArtifactCatalog, CheckpointRunner
from .fetalclip import FetalCLIPAdapter
from .fetalnet import FetalNetAdapter
from .fetal_workflow import FETAL_VIEW_DISPLAY, build_fetal_view_plan, fetal_view_from_result
from .planner import build_execution_plan, detect_domain_from_prompt, detect_intents, get_domain_router_tasks
from .roboflow import DEFAULT_FETAL_BRAIN_MODEL_ID, DEFAULT_ROBOFLOW_API_URL, RoboflowBrainAdapter
from .reporting import build_markdown_report, build_technical_appendix
from .schemas import AnalysisContext, ToolResult


@dataclass
class RuntimeConfig:
    data_root: str = "Datasets"
    model_root: str = "runs/pulse_retrain_new"
    runtime_root: str = "runs/pulse_runtime"
    image_size: int = 224
    device: str = "auto"
    seed: int = 42
    fetalclip_enabled: bool = True
    fetalclip_weights: str | None = None
    fetalclip_config: str | None = None
    fetalnet_enabled: bool = True
    fetalnet_repo_root: str | None = None
    fetalnet_weights: str | None = None
    roboflow_brain_enabled: bool = True
    roboflow_api_url: str = DEFAULT_ROBOFLOW_API_URL
    roboflow_api_key: str | None = None
    roboflow_brain_model_id: str = DEFAULT_FETAL_BRAIN_MODEL_ID


class PULSEInferenceService:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.data_root = Path(config.data_root).resolve()
        self.model_root = Path(config.model_root).resolve()
        self.runtime_root = Path(config.runtime_root).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.task_catalog = discover_all_tasks(self.data_root, seed=config.seed)
        self.artifacts = ArtifactCatalog(self.model_root)
        self.artifacts.scan(self.task_catalog)
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.fetalclip = FetalCLIPAdapter(
            device=self.device,
            search_roots=[self.data_root.parent, self.model_root.parent, Path.cwd()],
            enabled=config.fetalclip_enabled,
            weights_path=config.fetalclip_weights,
            config_path=config.fetalclip_config,
        )
        self.fetalnet = FetalNetAdapter(
            device=self.device,
            search_roots=[self.data_root.parent, self.model_root.parent, Path.cwd()],
            enabled=config.fetalnet_enabled,
            repo_root=config.fetalnet_repo_root,
            weights_path=config.fetalnet_weights,
        )
        self.roboflow_brain = RoboflowBrainAdapter(
            enabled=config.roboflow_brain_enabled,
            api_url=config.roboflow_api_url,
            api_key=config.roboflow_api_key,
            model_id=config.roboflow_brain_model_id,
        )

    def create_request_workspace(self) -> Tuple[str, Path, Path]:
        request_id = uuid.uuid4().hex[:12]
        request_root = self.runtime_root / request_id
        input_root = request_root / "inputs"
        artifact_root = request_root / "artifacts"
        input_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        return request_id, input_root, artifact_root

    def dependency_status(self) -> Dict[str, bool]:
        fetalclip_status = self.fetalclip.availability()
        fetalnet_status = self.fetalnet.availability()
        roboflow_status = self.roboflow_brain.availability()
        return {
            "torch": True,
            "numpy": np is not None,
            "pillow": Image is not None,
            "open_clip": fetalclip_status["available"] or fetalclip_status["reason"] != "open-clip-torch is not installed.",
            "fetalnet": fetalnet_status["available"] or fetalnet_status["reason"] not in {"FetalNet source checkout was not found.", "FetalNet weights were not found."},
            "roboflow_client": roboflow_status["reason"] != "Neither inference-sdk nor requests is installed.",
        }

    def task_inventory(self) -> List[Dict[str, Any]]:
        inventory: List[Dict[str, Any]] = []
        for task in self.task_catalog:
            artifact = self.artifacts.get(task.task_id)
            inventory.append(
                {
                    **task.to_dict(),
                    "checkpoint_available": bool(artifact and artifact.available),
                    "checkpoint_path": str(artifact.checkpoint_path) if artifact and artifact.available else None,
                }
            )
        return inventory

    def health(self) -> Dict[str, Any]:
        ready_tasks = sum(1 for task in self.task_catalog if task.is_ready)
        checkpoints = self.artifacts.checkpoint_index()
        dependencies = self.dependency_status()
        status = "ok" if dependencies["numpy"] and dependencies["pillow"] else "degraded"
        return {
            "status": status,
            "device": str(self.device),
            "dependencies": dependencies,
            "fetalclip": self.fetalclip.availability(),
            "fetalnet": self.fetalnet.availability(),
            "roboflow_brain": self.roboflow_brain.availability(),
            "data_root": str(self.data_root),
            "model_root": str(self.model_root),
            "runtime_root": str(self.runtime_root),
            "tasks_total": len(self.task_catalog),
            "tasks_ready": ready_tasks,
            "checkpoints_available": len(checkpoints),
        }

    def analyze(
        self,
        primary_image: Path,
        prompt: str = "",
        extra_images: Dict[str, Path] | None = None,
        domain_hint: str | None = None,
        task_hints: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict:
        extra_images = extra_images or {}
        task_hints = task_hints or []
        metadata = metadata or {}
        primary_image = Path(primary_image).resolve()
        if not primary_image.exists():
            raise FileNotFoundError(f"Input image not found: {primary_image}")
        request_id = primary_image.parent.parent.name if primary_image.parent.name == "inputs" else uuid.uuid4().hex[:12]
        intents = list(dict.fromkeys([*task_hints, *detect_intents(prompt)]))
        context = AnalysisContext(
            request_id=request_id,
            prompt=prompt,
            primary_image=primary_image,
            extra_images=extra_images,
            data_root=self.data_root,
            model_root=self.model_root,
            runtime_root=self.runtime_root,
            task_catalog=self.task_catalog,
            metadata=metadata,
            domain_hint=None,
            task_hints=intents,
        )
        context.quality = self._assess_image(primary_image)

        request_root = self.runtime_root / request_id
        artifact_root = request_root / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        available_modalities = ["bmode", *sorted(extra_images.keys())]
        prompt_domain_hint, prompt_scores = detect_domain_from_prompt(prompt)
        detected_domain, routing_result, domain_scores, prompt_domain_scores = self._route_domain(
            context,
            available_modalities,
            artifact_root / "routing",
        )
        context.detected_domain = detected_domain
        context.domain_scores = domain_scores or prompt_domain_scores
        if routing_result is not None:
            context.routing_task_id = routing_result.task_id
            context.routing_label = str(routing_result.outputs.get("label", ""))
            context.detected_anatomy = self._detected_anatomy(detected_domain, routing_result)

        if detected_domain is not None:
            plan = build_execution_plan(self.task_catalog, detected_domain, intents, available_modalities)
        elif prompt_domain_hint:
            plan = build_execution_plan(self.task_catalog, prompt_domain_hint, intents, available_modalities)
            context.domain_scores = prompt_scores
            context.detected_domain = prompt_domain_hint
        else:
            plan = []
        plan = self._filter_runnable_plan(plan)
        if context.detected_domain == "fetal":
            plan = [task for task in plan if task.task_id != "fetal/plane_classification"]
        context.plan = [f"{task.title} [{task.task_type.value}]" for task in plan]

        fetalclip_results: List[ToolResult] = []
        fetal_virtual_results: List[ToolResult] = []
        fetalnet_results: List[ToolResult] = []
        roboflow_results: List[ToolResult] = []
        fetal_view_key: str | None = None
        if self._should_run_fetalclip(context, plan, prompt_domain_hint):
            fetalclip_results = self._run_fetalclip(context)
            fetal_routing_results: List[ToolResult] = [result for result in fetalclip_results if result.status != "unavailable"]
            plane_result = next(
                (
                    result
                    for result in fetalclip_results
                    if result.task_id == "fetalclip/plane_zero_shot" and result.status == "completed"
                ),
                None,
            )
            if plane_result is None and context.detected_domain == "fetal":
                fetalnet_view_result = self.fetalnet.classify_view(context.primary_image)
                fetalnet_results.append(fetalnet_view_result)
                fetal_routing_results.append(fetalnet_view_result)
                if fetalnet_view_result.status == "completed":
                    plane_result = fetalnet_view_result
            if plane_result is not None:
                fetal_view_key = fetal_view_from_result(plane_result)
                if fetal_view_key is not None:
                    context.detected_domain = "fetal"
                    context.domain_scores = {"fetal": float(plane_result.confidence or 0.0)}
                    context.routing_task_id = plane_result.task_id
                    context.routing_label = str(plane_result.outputs.get("label", ""))
                    context.detected_anatomy = str(plane_result.outputs.get("label", "")) or "fetal"
                elif not context.detected_anatomy:
                    context.detected_anatomy = str(plane_result.outputs.get("label", "")) or "fetal"
                if fetal_view_key is not None:
                    _, fetal_plan, fetal_virtual_results, fetal_plan_entries = build_fetal_view_plan(
                        self.task_catalog,
                        intents,
                        fetal_routing_results,
                        fetalnet_ready=self.fetalnet.supports_view(fetal_view_key),
                    )
                    plan = self._filter_runnable_plan(fetal_plan)
                    context.plan = (
                        self._fetalclip_plan_entries(fetal_routing_results)
                        + fetal_plan_entries
                    )
                    context.metadata["fetal_view_key"] = fetal_view_key
                    context.metadata["fetal_view_label"] = FETAL_VIEW_DISPLAY.get(fetal_view_key, fetal_view_key)
                else:
                    context.plan = self._fetalclip_plan_entries(fetal_routing_results) + context.plan
            else:
                context.plan = self._fetalclip_plan_entries(fetal_routing_results) + context.plan

        if self._should_run_fetalnet(fetal_view_key, intents):
            context.plan = context.plan + self._fetalnet_plan_entries(fetal_view_key)
            fetalnet_results.extend(
                self._run_fetalnet(context, fetal_view_key, self._pixel_spacing_mm(context), artifact_root)
            )

        if self._should_run_roboflow_brain(context, fetal_view_key, intents):
            context.plan = context.plan + self._roboflow_plan_entries()
            roboflow_results = self._run_roboflow_brain(context)

        results: List[ToolResult] = []
        results.append(
            ToolResult(
                tool_name="quality_assessment",
                task_id=None,
                status="completed",
                summary=f"Input image assessed as {context.quality.get('quality_label', 'unknown')}.",
                outputs=context.quality,
            )
        )
        results.extend(fetalclip_results)
        results.extend(fetalnet_results)
        results.extend(roboflow_results)

        routing_cache = {routing_result.task_id: routing_result} if routing_result and routing_result.task_id else {}
        for task in plan:
            if task.task_id in routing_cache:
                results.append(routing_cache[task.task_id])
            else:
                results.append(self._run_task(task, context, artifact_root))
        results.extend(fetal_virtual_results)
        results = [result for result in results if result.status != "unavailable"]

        report = build_markdown_report(context, results, self.artifacts.checkpoint_index())
        report_path = request_root / "clinical_report.md"
        report_path.write_text(report, encoding="utf-8")
        technical_appendix = build_technical_appendix(context, results, self.artifacts.checkpoint_index())
        technical_report_path = request_root / "technical_appendix.md"
        technical_report_path.write_text(technical_appendix, encoding="utf-8")

        result_payload = {
            "request_id": request_id,
            "detected_domain": context.detected_domain,
            "detected_anatomy": context.detected_anatomy,
            "routing_task_id": context.routing_task_id,
            "routing_label": context.routing_label,
            "domain_scores": context.domain_scores,
            "intents": intents,
            "quality": context.quality,
            "plan": context.plan,
            "results": [result.to_dict() for result in results],
            "artifact_index": self.artifacts.checkpoint_index(),
            "report_markdown": report,
            "technical_appendix_markdown": technical_appendix,
            "report_path": str(report_path),
            "report_url": self._artifact_url(report_path),
            "technical_report_path": str(technical_report_path),
            "technical_report_url": self._artifact_url(technical_report_path),
            "response_url": self._artifact_url(request_root / "response.json"),
        }
        with open(request_root / "response.json", "w", encoding="utf-8") as handle:
            json.dump(result_payload, handle, indent=2)
        return result_payload

    def _should_run_fetalclip(
        self,
        context: AnalysisContext,
        plan: List[TaskSpec],
        prompt_domain_hint: str | None,
    ) -> bool:
        if context.detected_domain == "fetal":
            return True
        if prompt_domain_hint == "fetal":
            return True
        return any(task.domain == "fetal" for task in plan)

    def _run_fetalclip(self, context: AnalysisContext) -> List[ToolResult]:
        return self.fetalclip.analyze(context.primary_image)

    def _should_run_fetalnet(self, fetal_view_key: str | None, intents: List[str]) -> bool:
        if fetal_view_key not in {"brain", "abdomen", "femur"}:
            return False
        if intents and "measurement" not in intents:
            return False
        return self.fetalnet.supports_view(fetal_view_key)

    def _run_fetalnet(
        self,
        context: AnalysisContext,
        fetal_view_key: str,
        pixel_spacing_mm: float | None,
        artifact_root: Path,
    ) -> List[ToolResult]:
        result = self.fetalnet.analyze(
            context.primary_image,
            fetal_view_key=fetal_view_key,
            pixel_spacing_mm=pixel_spacing_mm,
            artifact_root=artifact_root,
        )
        normalized_artifacts: Dict[str, str] = {}
        for key, value in result.artifacts.items():
            try:
                normalized_artifacts[key] = self._artifact_url(Path(value))
            except Exception:
                normalized_artifacts[key] = value
        result.artifacts = normalized_artifacts
        return [result]

    def _should_run_roboflow_brain(
        self,
        context: AnalysisContext,
        fetal_view_key: str | None,
        intents: List[str],
    ) -> bool:
        if not self.roboflow_brain.availability().get("available", False):
            return False
        if intents and "classification" not in intents:
            return False
        if fetal_view_key == "brain":
            return True
        if context.detected_domain != "fetal":
            return False
        prompt_text = f"{context.prompt} {context.detected_anatomy or ''}".lower()
        return any(token in prompt_text for token in ("brain", "head", "ventricle", "cranial", "cns"))

    def _run_roboflow_brain(self, context: AnalysisContext) -> List[ToolResult]:
        return [self.roboflow_brain.analyze(context.primary_image)]

    def _filter_runnable_plan(self, tasks: List[TaskSpec]) -> List[TaskSpec]:
        runnable: List[TaskSpec] = []
        for task in tasks:
            artifact = self.artifacts.get(task.task_id)
            if artifact is None or not artifact.available:
                continue
            runnable.append(task)
        return runnable

    def _pixel_spacing_mm(self, context: AnalysisContext) -> float | None:
        pixel_spacing_mm = None
        for key in ("pixel_spacing_mm", "pixel_spacing", "spacing_mm_per_pixel"):
            value = context.metadata.get(key)
            if value in (None, ""):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                pixel_spacing_mm = parsed
                break
        return pixel_spacing_mm

    def _fetalclip_plan_entries(self, results: List[ToolResult]) -> List[str]:
        labels = {
            "fetalclip/plane_zero_shot": "zero-shot plane analysis",
            "fetalclip/brain_subplane_zero_shot": "zero-shot brain sub-plane analysis",
            "fetalnet/view_classification": "view classification",
        }
        entries = []
        for result in results:
            if result.task_id not in labels:
                continue
            model_name = str(result.outputs.get("model_name", "Fetal foundation model"))
            entries.append(f"{model_name} {labels[result.task_id]} [foundation]")
        return entries

    def _fetalnet_plan_entries(self, fetal_view_key: str | None) -> List[str]:
        labels = {
            "brain": "FetalNet head biometry [measurement]",
            "abdomen": "FetalNet abdominal circumference [measurement]",
            "femur": "FetalNet femur length [measurement]",
        }
        if fetal_view_key in labels:
            return [labels[fetal_view_key]]
        return []

    def _roboflow_plan_entries(self) -> List[str]:
        return ["Roboflow fetal brain abnormality classification [classification]"]

    def _route_domain(
        self,
        context: AnalysisContext,
        available_modalities: List[str],
        routing_root: Path,
    ) -> Tuple[str | None, ToolResult | None, Dict[str, float], Dict[str, float]]:
        routing_root.mkdir(parents=True, exist_ok=True)
        router_tasks = get_domain_router_tasks(self.task_catalog, available_modalities)
        routing_results: Dict[str, ToolResult] = {}
        for task in router_tasks:
            result = self._run_task(task, context, routing_root)
            routing_results[task.task_id] = result

        prompt_domain_hint, prompt_scores = detect_domain_from_prompt(context.prompt)
        fallback_domain, fallback_result, fallback_scores = self._route_from_fallback(routing_results)
        global_router = routing_results.get("system/domain_classification")
        if global_router is not None and global_router.status == "completed":
            routed_domain, domain_scores = self._route_from_global_router(global_router)
            fallback_domain, fallback_result, fallback_scores = self._route_from_fallback(
                routing_results,
                global_domain=routed_domain,
            )
            merged_scores = self._merge_domain_scores(domain_scores, fallback_scores)
            preferred_domain, preferred_result, preferred_scores = self._prefer_specialist_router(
                routed_domain=routed_domain,
                routed_result=global_router,
                routed_scores=merged_scores,
                fallback_domain=fallback_domain,
                fallback_result=fallback_result,
                fallback_scores=merged_scores,
            )
            if preferred_domain is not None:
                return preferred_domain, preferred_result, preferred_scores, prompt_scores
            if prompt_domain_hint:
                return prompt_domain_hint, None, merged_scores, prompt_scores
            return None, global_router, merged_scores, prompt_scores

        if fallback_domain is None and prompt_domain_hint:
            return prompt_domain_hint, None, fallback_scores, prompt_scores
        return fallback_domain, fallback_result, fallback_scores, prompt_scores

    def _merge_domain_scores(self, primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
        merged = dict(primary)
        for domain, score in secondary.items():
            merged[domain] = max(float(score), float(merged.get(domain, 0.0)))
        return merged

    def _prefer_specialist_router(
        self,
        routed_domain: str | None,
        routed_result: ToolResult | None,
        routed_scores: Dict[str, float],
        fallback_domain: str | None,
        fallback_result: ToolResult | None,
        fallback_scores: Dict[str, float],
    ) -> Tuple[str | None, ToolResult | None, Dict[str, float]]:
        if fallback_domain is None or fallback_result is None:
            return routed_domain, routed_result, routed_scores
        if routed_domain is None:
            return fallback_domain, fallback_result, fallback_scores
        if fallback_domain == routed_domain:
            if fallback_result.task_id != "system/domain_classification":
                return fallback_domain, fallback_result, fallback_scores
            return routed_domain, routed_result, routed_scores
        specialist_domains = {"cardiac", "abdominal", "thyroid", "carotid", "liver", "kidney"}
        if fallback_domain in specialist_domains:
            return fallback_domain, fallback_result, fallback_scores
        return routed_domain, routed_result, routed_scores

    def _route_from_global_router(self, result: ToolResult) -> Tuple[str | None, Dict[str, float]]:
        if result.status != "completed":
            return None, {}

        top_predictions = result.outputs.get("top_predictions", [])
        if not top_predictions:
            label = result.outputs.get("label")
            confidence = float(result.outputs.get("confidence", result.confidence or 0.0))
            if isinstance(label, str) and label:
                return label, {label: confidence}
            return None, {}

        domain_scores = {
            str(item.get("label")): float(item.get("probability", 0.0))
            for item in top_predictions
            if item.get("label") is not None
        }
        best_label = str(top_predictions[0].get("label", ""))
        best_score = float(top_predictions[0].get("probability", 0.0))
        second_score = float(top_predictions[1].get("probability", 0.0)) if len(top_predictions) > 1 else 0.0
        dynamic_floor = max(0.32, (1.0 / max(len(top_predictions), 1)) + 0.10)
        if best_label and (best_score >= 0.55 and (best_score - second_score) >= 0.08):
            return best_label, domain_scores
        if best_label and best_score >= dynamic_floor and (best_score - second_score) >= 0.04:
            return best_label, domain_scores
        return None, domain_scores

    def _route_from_fallback(
        self,
        routing_results: Dict[str, ToolResult],
        global_domain: str | None = None,
    ) -> Tuple[str | None, ToolResult | None, Dict[str, float]]:
        domain_scores: Dict[str, float] = {}
        evidence: Dict[str, ToolResult] = {}

        butterfly_view = routing_results.get("cardiac/butterfly_view_classification")
        cardiac_screen = routing_results.get("cardiac/butterfly_cardiac_screening")
        abdominal_organ = routing_results.get("abdominal/organ_classification")
        carotid_segmentation = routing_results.get("carotid/lumen_segmentation")

        if butterfly_view is not None and butterfly_view.status == "completed":
            label = str(butterfly_view.outputs.get("label", "")).lower()
            confidence = float(butterfly_view.outputs.get("confidence", butterfly_view.confidence or 0.0))
            screen_label = ""
            screen_confidence = 0.0
            if cardiac_screen is not None and cardiac_screen.status == "completed":
                screen_label = str(cardiac_screen.outputs.get("label", "")).lower()
                screen_confidence = float(cardiac_screen.outputs.get("confidence", cardiac_screen.confidence or 0.0))

            allow_butterfly_noncardiac = global_domain in {None, "cardiac"}
            if allow_butterfly_noncardiac and label in {"thyroid", "carotid"} and confidence >= 0.60 and screen_label == "non_cardiac" and screen_confidence >= 0.95:
                domain_scores[label] = confidence
                evidence[label] = butterfly_view
            elif allow_butterfly_noncardiac and label in {"bladder", "ivc", "morisons_pouch"} and confidence >= 0.85 and screen_label == "non_cardiac" and screen_confidence >= 0.95:
                domain_scores["abdominal"] = confidence
                evidence["abdominal"] = butterfly_view
            elif label in {"2ch", "4ch", "plax"} and confidence >= 0.80:
                screen_label = ""
                screen_confidence = 0.0
                if cardiac_screen is not None and cardiac_screen.status == "completed":
                    screen_label = str(cardiac_screen.outputs.get("label", "")).lower()
                    screen_confidence = float(cardiac_screen.outputs.get("confidence", cardiac_screen.confidence or 0.0))
                if screen_label == "cardiac" and screen_confidence >= 0.70:
                    domain_scores["cardiac"] = (confidence + screen_confidence) / 2.0
                    evidence["cardiac"] = butterfly_view

        if abdominal_organ is not None and abdominal_organ.status == "completed":
            label = str(abdominal_organ.outputs.get("label", "")).lower()
            confidence = float(abdominal_organ.outputs.get("confidence", abdominal_organ.confidence or 0.0))
            if label in {"liver", "hepatic"} and confidence >= 0.40:
                domain_scores["liver"] = max(domain_scores.get("liver", 0.0), confidence)
                evidence["liver"] = abdominal_organ
            elif label == "kidney" and confidence >= 0.40:
                domain_scores["kidney"] = max(domain_scores.get("kidney", 0.0), confidence)
                evidence["kidney"] = abdominal_organ

        if carotid_segmentation is not None and carotid_segmentation.status == "completed":
            positive_fraction = float(carotid_segmentation.outputs.get("positive_fraction", 0.0))
            confidence = float(carotid_segmentation.confidence or 0.0)
            if positive_fraction >= 0.05 and confidence >= 0.85:
                carotid_score = confidence * min(1.0, positive_fraction / 0.10)
                domain_scores["carotid"] = max(domain_scores.get("carotid", 0.0), carotid_score)
                evidence["carotid"] = carotid_segmentation

        if not domain_scores:
            return None, None, {}

        ranked = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
        best_domain, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        domain_min_score = {
            "kidney": 0.40,
            "liver": 0.40,
            "thyroid": 0.60,
            "carotid": 0.60,
            "abdominal": 0.80,
            "cardiac": 0.75,
        }.get(best_domain, 0.75)
        domain_min_margin = {
            "kidney": 0.05,
            "liver": 0.05,
            "thyroid": 0.05,
            "carotid": 0.05,
        }.get(best_domain, 0.08)
        if best_score < domain_min_score or margin < domain_min_margin:
            return None, None, domain_scores
        return best_domain, evidence.get(best_domain), domain_scores

    def _detected_anatomy(self, detected_domain: str | None, routing_result: ToolResult) -> str | None:
        if detected_domain is None:
            return None
        label = routing_result.outputs.get("label")
        if detected_domain in {"liver", "kidney"}:
            return detected_domain
        if isinstance(label, str) and label:
            return label
        if detected_domain:
            return detected_domain
        return None

    def _preprocess_tensor(self, image_path: Path) -> torch.Tensor:
        image = _load_image(str(image_path), self.config.image_size)
        tensor = _normalize(_pil_to_tensor(image)).unsqueeze(0).to(self.device)
        return tensor

    def _build_multimodal_inputs(self, context: AnalysisContext, task: TaskSpec) -> Dict[str, torch.Tensor] | None:
        modalities = {}
        required = task.extras.get("modalities", [])
        for modality in required:
            if modality == "bmode":
                modalities[modality] = self._preprocess_tensor(context.primary_image)
            else:
                path = context.extra_images.get(modality)
                if path is None:
                    return None
                modalities[modality] = self._preprocess_tensor(path)
        return modalities

    def _run_task(self, task: TaskSpec, context: AnalysisContext, artifact_root: Path) -> ToolResult:
        artifact = self.artifacts.get(task.task_id)
        if artifact is None or not artifact.available:
            return ToolResult(
                tool_name=task.task_id,
                task_id=task.task_id,
                status="unavailable",
                summary="No trained checkpoint is available for this task.",
                domain=task.domain,
            )

        try:
            runner = CheckpointRunner(artifact, self.device)
            with torch.no_grad():
                if task.task_type == TaskType.MULTIMODAL:
                    inputs = self._build_multimodal_inputs(context, task)
                    if inputs is None:
                        return ToolResult(
                            tool_name=task.task_id,
                            task_id=task.task_id,
                            status="skipped",
                            summary="Required multimodal inputs are missing for this task.",
                            domain=task.domain,
                        )
                    outputs = runner.model(inputs)
                else:
                    tensor = self._preprocess_tensor(context.primary_image)
                    outputs = runner.model(tensor)
            return self._summarize_output(task, outputs, context, artifact_root)
        except Exception as exc:
            return ToolResult(
                tool_name=task.task_id,
                task_id=task.task_id,
                status="error",
                summary="Inference failed for this task.",
                domain=task.domain,
                errors=[str(exc)],
            )

    def _summarize_output(self, task: TaskSpec, outputs, context: AnalysisContext, artifact_root: Path) -> ToolResult:
        if task.task_type in {TaskType.CLASSIFICATION, TaskType.MULTIMODAL}:
            probs = torch.softmax(outputs, dim=1)[0].cpu()
            label_index = int(torch.argmax(probs).item())
            confidence = float(probs[label_index].item())
            label_name = task.labels[label_index] if task.labels else str(label_index)
            topk = min(3, probs.numel())
            top_probs, top_indices = torch.topk(probs, k=topk)
            top_predictions = [
                {
                    "label": task.labels[int(index)] if task.labels else str(int(index)),
                    "probability": float(prob),
                }
                for prob, index in zip(top_probs.tolist(), top_indices.tolist())
            ]
            return ToolResult(
                tool_name=task.task_id,
                task_id=task.task_id,
                status="completed",
                summary=f"Predicted `{label_name}`.",
                domain=task.domain,
                confidence=confidence,
                outputs={
                    "label": label_name,
                    "confidence": confidence,
                    "probabilities": probs.tolist(),
                    "top_predictions": top_predictions,
                },
            )

        if task.task_type == TaskType.DETECTION:
            objectness = float(torch.sigmoid(outputs["objectness"][0]).item())
            bbox = outputs["bbox"][0].cpu().tolist()
            artifact_path = artifact_root / f"{task.output_name}_detection.png"
            self._draw_detection(context.primary_image, bbox, artifact_path)
            return ToolResult(
                tool_name=task.task_id,
                task_id=task.task_id,
                status="completed",
                summary="Detection head produced a single localization proposal.",
                domain=task.domain,
                confidence=objectness,
                outputs={"bbox_cxcywh": bbox, "objectness": objectness},
                artifacts={"overlay": self._artifact_url(artifact_path)},
            )

        if task.task_type == TaskType.SEGMENTATION:
            if task.extras.get("segmentation_mode", "binary") == "binary":
                probs = torch.sigmoid(outputs)[0, 0].cpu()
                mask = (probs > 0.5).byte()
                segmentation_confidence = float(((probs * mask.float()) + ((1.0 - probs) * (1.0 - mask.float()))).mean().item())
                class_fractions = {}
            else:
                probs = torch.softmax(outputs, dim=1)[0].cpu()
                max_probs, predicted = torch.max(probs, dim=0)
                mask = predicted.byte()
                segmentation_confidence = float(max_probs.mean().item())
                class_names = task.extras.get("mask_class_names", [])
                class_fractions = {}
                total_pixels = float(mask.numel())
                for class_index, class_name in enumerate(class_names[1:], start=1):
                    class_fraction = float((mask == class_index).float().sum().item() / max(total_pixels, 1.0))
                    if class_fraction > 0:
                        class_fractions[class_name] = class_fraction
            mask_path = artifact_root / f"{task.output_name}_mask.png"
            overlay_path = artifact_root / f"{task.output_name}_overlay.png"
            self._save_mask_and_overlay(context.primary_image, mask, mask_path, overlay_path)
            positive_fraction = float(mask.float().mean().item())
            return ToolResult(
                tool_name=task.task_id,
                task_id=task.task_id,
                status="completed",
                summary="Segmentation mask generated.",
                domain=task.domain,
                confidence=segmentation_confidence,
                outputs={"positive_fraction": positive_fraction, "class_fractions": class_fractions},
                artifacts={
                    "mask": self._artifact_url(mask_path),
                    "overlay": self._artifact_url(overlay_path),
                },
            )

        value = float(outputs.squeeze().item())
        unit = task.extras.get("target_name", "value")
        return ToolResult(
            tool_name=task.task_id,
            task_id=task.task_id,
            status="completed",
            summary=f"Estimated `{unit}` = {value:.3f}.",
            domain=task.domain,
            outputs={"value": value, "unit": unit},
        )

    def _assess_image(self, image_path: Path) -> Dict[str, float | int | str]:
        if Image is None or np is None:
            return {"quality_label": "unknown"}

        image = Image.open(image_path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        gx = np.diff(array, axis=1)
        gy = np.diff(array, axis=0)
        blur_score = float((gx.var() + gy.var()) / 2.0)
        brightness = float(array.mean())
        contrast = float(array.std())

        if blur_score < 0.002 or contrast < 0.08:
            quality_label = "low"
        elif blur_score < 0.006 or contrast < 0.12:
            quality_label = "medium"
        else:
            quality_label = "good"

        return {
            "width": image.width,
            "height": image.height,
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "quality_label": quality_label,
        }

    def _artifact_url(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.runtime_root)
        return f"/generated/{relative.as_posix()}"

    def _save_mask_and_overlay(self, image_path: Path, mask: torch.Tensor, mask_path: Path, overlay_path: Path) -> None:
        if Image is None or np is None:
            return
        base = Image.open(image_path).convert("RGB")
        mask_image = Image.fromarray((mask.numpy() > 0).astype("uint8") * 255, mode="L").resize(base.size, Image.NEAREST)
        mask_image.save(mask_path)

        overlay = base.copy()
        alpha = Image.new("RGBA", base.size, (255, 0, 0, 0))
        alpha_pixels = alpha.load()
        mask_pixels = mask_image.load()
        for y in range(base.size[1]):
            for x in range(base.size[0]):
                if mask_pixels[x, y] > 0:
                    alpha_pixels[x, y] = (255, 0, 0, 96)
        composed = Image.alpha_composite(base.convert("RGBA"), alpha)
        composed.save(overlay_path)

    def _draw_detection(self, image_path: Path, bbox_cxcywh: List[float], artifact_path: Path) -> None:
        if Image is None or ImageDraw is None:
            return
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        cx, cy, bw, bh = bbox_cxcywh
        x1 = int((cx - (bw / 2.0)) * width)
        y1 = int((cy - (bh / 2.0)) * height)
        x2 = int((cx + (bw / 2.0)) * width)
        y2 = int((cy + (bh / 2.0)) * height)
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        image.save(artifact_path)
