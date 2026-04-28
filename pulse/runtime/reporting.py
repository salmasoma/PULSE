from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .schemas import AnalysisContext, ToolResult

EXAM_TITLES = {
    "cardiac": "Cardiac / Multi-POCUS Ultrasound",
    "breast": "Breast Ultrasound",
    "thyroid": "Thyroid Ultrasound",
    "fetal": "Fetal Ultrasound",
    "abdominal": "Abdominal Ultrasound",
    "liver": "Liver Ultrasound",
    "kidney": "Renal Ultrasound",
    "pcos": "Pelvic / Ovarian Ultrasound",
    "carotid": "Carotid Ultrasound",
}


def _fmt_float(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _clean_label(value: Any) -> str:
    return str(value).replace("_", " ").replace("-", " ").strip()


def _fmt_label_list(values: Iterable[Any]) -> str:
    labels = [f"`{_clean_label(value)}`" for value in values if str(value).strip()]
    return ", ".join(labels)


def _quality_impression(label: str) -> str:
    mapping = {
        "good": "Diagnostic image quality.",
        "medium": "Mildly limited image quality.",
        "low": "Limited image quality.",
        "unknown": "Image quality could not be fully characterized.",
    }
    return mapping.get(label, "Image quality could not be fully characterized.")


def _specialist_results(results: Iterable[ToolResult]) -> List[ToolResult]:
    return [result for result in results if result.task_id]


def _top_predictions(result: ToolResult) -> List[Dict[str, Any]]:
    raw = result.outputs.get("top_predictions", [])
    return raw if isinstance(raw, list) else []


def _classification_finding(result: ToolResult) -> str:
    label = result.outputs.get("label", "unknown")
    confidence = result.outputs.get("confidence", result.confidence)
    if result.task_id == "fetalnet/view_classification":
        return f"FetalNet view classification favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "fetalnet/head_biometry":
        return (
            "FetalNet head biometry estimates "
            f"HC {_fmt_float(result.outputs.get('hc_value'))} {result.outputs.get('hc_unit', '')} and "
            f"BPD {_fmt_float(result.outputs.get('bpd_value'))} {result.outputs.get('bpd_unit', '')}."
        )
    if result.task_id == "fetalnet/abdominal_circumference":
        return (
            "FetalNet abdominal biometry estimates "
            f"AC {_fmt_float(result.outputs.get('value'))} {result.outputs.get('unit', '')}."
        )
    if result.task_id == "fetalnet/femur_length":
        return (
            "FetalNet femur biometry estimates "
            f"FL {_fmt_float(result.outputs.get('value'))} {result.outputs.get('unit', '')}."
        )
    if result.task_id == "fetalclip/plane_zero_shot":
        model_name = result.outputs.get("model_name", "Fetal foundation model")
        return f"{model_name} zero-shot fetal plane analysis favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "fetalclip/brain_subplane_zero_shot":
        model_name = result.outputs.get("model_name", "Fetal foundation model")
        return f"{model_name} zero-shot brain sub-plane analysis favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "roboflow/fetal_brain_abnormality_classification":
        model_name = result.outputs.get("model_name", "Roboflow fetal brain abnormality classifier")
        abnormal_classes = result.outputs.get("abnormal_classes", [])
        predicted_classes = result.outputs.get("predicted_classes", [])
        top_label = result.outputs.get("top_label", label)
        if abnormal_classes:
            return (
                f"{model_name} flags research abnormality labels: {_fmt_label_list(abnormal_classes)}. "
                f"Highest returned score {_fmt_float(confidence)}."
            )
        if predicted_classes:
            return (
                f"{model_name} does not flag an abnormal label on this image; the leading returned class is "
                f"`{_clean_label(top_label)}` with score {_fmt_float(confidence)}."
            )
        return f"{model_name} returned no positive brain-abnormality labels."
    if result.task_id == "cardiac/butterfly_view_classification":
        return f"View classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "cardiac/butterfly_cardiac_screening":
        return f"Cardiac screening classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "abdominal/organ_classification":
        return f"Organ classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "abdominal/anomaly_detection":
        return f"Abdominal anomaly classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "breast/lesion_classification":
        return f"Breast lesion classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "thyroid/benign_malignant_classification":
        return f"Thyroid classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "thyroid/multimodal_fusion_classification":
        return f"Multimodal thyroid fusion classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "fetal/plane_classification":
        return f"Fetal plane classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "liver/pathology_classification":
        return f"Liver classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "kidney/normal_abnormal_classification":
        return f"Renal classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    if result.task_id == "pcos/binary_classification":
        return f"PCOS classifier favors `{label}` with confidence {_fmt_float(confidence)}."
    return f"Classifier output favors `{label}` with confidence {_fmt_float(confidence)}."


def _segmentation_finding(result: ToolResult) -> str:
    positive_fraction = result.outputs.get("positive_fraction")
    class_fractions = result.outputs.get("class_fractions", {})
    if result.task_id == "carotid/lumen_segmentation":
        if positive_fraction is not None:
            return f"Automated lumen segmentation delineates a carotid lumen region occupying approximately {_fmt_pct(float(positive_fraction))} of the image field."
        return "Automated lumen segmentation was generated."
    if result.task_id == "breast/lesion_segmentation":
        if positive_fraction is not None:
            return f"Automated lesion segmentation identifies a target region occupying approximately {_fmt_pct(float(positive_fraction))} of the image field."
        return "Automated breast lesion segmentation was generated."
    if result.task_id == "liver/segmentation":
        if class_fractions:
            named = ", ".join(f"{key} {_fmt_pct(float(value))}" for key, value in class_fractions.items() if value > 0)
            return f"Automated liver segmentation produced labeled regions with estimated field coverage: {named}."
        return "Automated liver segmentation was generated."
    if result.task_id == "kidney/anatomy_segmentation":
        if class_fractions:
            named = ", ".join(f"{key} {_fmt_pct(float(value))}" for key, value in class_fractions.items() if value > 0)
            return f"Automated renal anatomy segmentation identified the following labeled regions: {named}."
        return "Automated renal anatomy segmentation was generated."
    if positive_fraction is not None:
        return f"Segmentation produced a positive region occupying approximately {_fmt_pct(float(positive_fraction))} of the image field."
    return "Segmentation output was generated."


def _detection_finding(result: ToolResult) -> str:
    objectness = result.outputs.get("objectness", result.confidence)
    if result.task_id == "thyroid/nodule_detection":
        return f"Automated detection localizes a thyroid nodule candidate with confidence {_fmt_float(objectness)}."
    return f"Automated detection generated a localization proposal with confidence {_fmt_float(objectness)}."


def _measurement_finding(result: ToolResult) -> str:
    value = result.outputs.get("value")
    unit = result.outputs.get("unit", "value")
    if result.task_id == "fetalnet/head_biometry":
        return (
            "FetalNet head biometry estimates "
            f"HC {_fmt_float(result.outputs.get('hc_value'))} {result.outputs.get('hc_unit', '')} and "
            f"BPD {_fmt_float(result.outputs.get('bpd_value'))} {result.outputs.get('bpd_unit', '')}. "
            "This is a segmentation-derived research estimate."
        )
    if result.task_id == "fetalnet/abdominal_circumference":
        return (
            f"FetalNet abdominal biometry estimates AC {_fmt_float(value)} {unit}. "
            "This is a segmentation-derived research estimate."
        )
    if result.task_id == "fetalnet/femur_length":
        return (
            f"FetalNet femur biometry estimates FL {_fmt_float(value)} {unit}. "
            "This is a segmentation-derived research estimate."
        )
    if result.task_id == "carotid/imt_measurement":
        return f"Automated intima-media thickness proxy measures {_fmt_float(value)} {unit}. This is a pixel-space estimate unless physical spacing metadata is provided."
    if result.task_id == "fetal/hc_measurement":
        return f"Automated fetal head circumference estimate measures {_fmt_float(value)} {unit}."
    return f"Automated quantitative estimate: {_fmt_float(value)} {unit}."


def _result_finding(result: ToolResult) -> str:
    if result.status != "completed":
        return result.summary
    if result.task_id in {"fetalnet/head_biometry", "fetalnet/abdominal_circumference", "fetalnet/femur_length"}:
        return _measurement_finding(result)
    if "label" in result.outputs:
        return _classification_finding(result)
    if "bbox_cxcywh" in result.outputs:
        return _detection_finding(result)
    if "value" in result.outputs:
        return _measurement_finding(result)
    if "positive_fraction" in result.outputs or "class_fractions" in result.outputs:
        return _segmentation_finding(result)
    return result.summary


def _build_impression(context: AnalysisContext, results: List[ToolResult]) -> List[str]:
    lines = []
    if context.detected_domain:
        exam = EXAM_TITLES.get(context.detected_domain, context.detected_domain.title())
        lines.append(f"Uploaded image is most consistent with `{exam}`.")
    if context.detected_anatomy:
        lines.append(f"Most likely routed anatomy/site: `{context.detected_anatomy}`.")

    for result in results:
        if result.status != "completed":
            continue
        if result.task_id == "thyroid/benign_malignant_classification":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"Thyroid classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "breast/lesion_classification":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"Breast lesion classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "liver/pathology_classification":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"Liver classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "kidney/normal_abnormal_classification":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"Renal classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "abdominal/anomaly_detection":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"Abdominal anomaly classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "pcos/binary_classification":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"PCOS classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "cardiac/butterfly_cardiac_screening":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"Cardiac screening classifier favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "thyroid/nodule_detection":
            lines.append("A thyroid nodule candidate was localized by the detection model.")
        elif result.task_id == "carotid/lumen_segmentation":
            lines.append("Carotid lumen segmentation was successfully generated.")
        elif result.task_id == "breast/lesion_segmentation":
            lines.append("Breast lesion segmentation was successfully generated.")
        elif result.task_id == "carotid/imt_measurement":
            value = result.outputs.get("value")
            unit = result.outputs.get("unit", "value")
            lines.append(f"IMT proxy: {_fmt_float(value)} {unit}.")
        elif result.task_id == "fetal/hc_measurement":
            value = result.outputs.get("value")
            unit = result.outputs.get("unit", "value")
            lines.append(f"Head circumference estimate: {_fmt_float(value)} {unit}.")
        elif result.task_id == "fetalnet/head_biometry":
            lines.append(
                "FetalNet head biometry estimates "
                f"HC {_fmt_float(result.outputs.get('hc_value'))} {result.outputs.get('hc_unit', '')} and "
                f"BPD {_fmt_float(result.outputs.get('bpd_value'))} {result.outputs.get('bpd_unit', '')}."
            )
        elif result.task_id == "fetalnet/abdominal_circumference":
            lines.append(
                f"FetalNet abdominal biometry estimates AC {_fmt_float(result.outputs.get('value'))} {result.outputs.get('unit', '')}."
            )
        elif result.task_id == "fetalnet/femur_length":
            lines.append(
                f"FetalNet femur biometry estimates FL {_fmt_float(result.outputs.get('value'))} {result.outputs.get('unit', '')}."
            )
        elif result.task_id == "fetalnet/view_classification":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            lines.append(f"FetalNet view classification favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "fetalclip/plane_zero_shot":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            model_name = result.outputs.get("model_name", "Fetal foundation model")
            lines.append(f"{model_name} zero-shot plane analysis favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "fetalclip/brain_subplane_zero_shot":
            label = result.outputs.get("label")
            confidence = result.outputs.get("confidence", result.confidence)
            model_name = result.outputs.get("model_name", "Fetal foundation model")
            lines.append(f"{model_name} zero-shot brain sub-plane analysis favors `{label}` with confidence {_fmt_float(confidence)}.")
        elif result.task_id == "roboflow/fetal_brain_abnormality_classification":
            abnormal_classes = result.outputs.get("abnormal_classes", [])
            top_label = result.outputs.get("top_label", "unknown")
            confidence = result.outputs.get("confidence", result.confidence)
            if abnormal_classes:
                lines.append(
                    "Roboflow fetal brain abnormality specialist flags research labels "
                    + _fmt_label_list(abnormal_classes)
                    + f" with highest score {_fmt_float(confidence)}."
                )
            else:
                lines.append(
                    "Roboflow fetal brain abnormality specialist does not flag an abnormal label; "
                    f"top returned class is `{_clean_label(top_label)}` with score {_fmt_float(confidence)}."
                )

    if context.quality.get("quality_label") == "low":
        lines.append("Interpretation is limited by low image quality.")
    elif context.quality.get("quality_label") == "medium":
        lines.append("Interpretation is mildly limited by image quality.")
    return lines[:5]


def build_markdown_report(
    context: AnalysisContext,
    results: Iterable[ToolResult],
    artifact_index: Dict[str, str],
) -> str:
    results = list(results)
    specialist_results = _specialist_results(results)
    completed_results = [result for result in specialist_results if result.status == "completed"]
    pending_results = [result for result in specialist_results if result.status == "skipped"]
    exam_title = EXAM_TITLES.get(context.detected_domain or context.domain_hint or "", "Ultrasound Study")
    indication = context.prompt.strip() or "Uploaded image submitted for automated review."
    quality_label = context.quality.get("quality_label", "unknown")

    lines: List[str] = [
        "# PULSE Clinical Report",
        "",
        f"**Exam:** {exam_title}",
        f"**Request ID:** `{context.request_id}`",
        f"**Indication:** {indication}",
        f"**Technique:** Single uploaded ultrasound image analyzed with automatic domain routing and all eligible specialist models for the detected domain. When fetal analysis is triggered and fetal foundation weights are available, zero-shot fetal foundation-model outputs are included; supported fetal head, abdomen, and femur views can also call the optional FetalNet biometric specialist, and fetal brain cases can call the optional Roboflow abnormality specialist. Extra modalities: {', '.join(sorted(context.extra_images)) or 'none'}.",
        f"**Image Quality:** {_quality_impression(quality_label)}",
        "",
        "## AI Triage",
        "",
    ]

    if context.detected_domain:
        lines.append(f"- Detected domain: `{context.detected_domain}`")
    if context.detected_anatomy:
        lines.append(f"- Routed anatomy / label: `{context.detected_anatomy}`")
    if context.routing_task_id:
        lines.append(f"- Routing model: `{context.routing_task_id}`")
    if context.domain_scores:
        ordered_scores = sorted(context.domain_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        lines.append(
            "- Top routing scores: "
            + ", ".join(f"`{domain}` {_fmt_float(score)}" for domain, score in ordered_scores)
        )
    fetal_tool = next((result for result in specialist_results if result.task_id and result.task_id.startswith("fetalclip/")), None)
    if fetal_tool is not None:
        lines.append(
            f"- Fetal foundation model: `{fetal_tool.outputs.get('model_name', 'fetal foundation model')}` zero-shot inference enabled for this case"
        )
    fetalnet_tool = next(
        (result for result in specialist_results if result.task_id and result.task_id.startswith("fetalnet/")),
        None,
    )
    if fetalnet_tool is not None:
        lines.append("- Fetal biometric specialist: `FetalNet` measurement workflow enabled for this case")
    roboflow_tool = next(
        (result for result in specialist_results if result.task_id == "roboflow/fetal_brain_abnormality_classification"),
        None,
    )
    if roboflow_tool is not None:
        lines.append(
            "- Fetal brain specialist: `Roboflow` "
            + str(roboflow_tool.outputs.get("deployment_mode", "hosted"))
            + " abnormality classification"
        )

    lines.extend(["", "## Findings", ""])
    if completed_results:
        for result in completed_results:
            lines.append(f"- {_result_finding(result)}")
    else:
        lines.append("- No completed specialist model output is available for this case.")

    if pending_results:
        lines.extend(["", "## Additional Planned Tools", ""])
        for result in pending_results:
            lines.append(f"- `{result.tool_name}`: {result.summary}")

    lines.extend(["", "## Impression", ""])
    impression_lines = _build_impression(context, completed_results)
    if impression_lines:
        for index, line in enumerate(impression_lines, start=1):
            lines.append(f"{index}. {line}")
    else:
        lines.append("1. No clinically styled impression could be synthesized from the available outputs.")

    lines.extend(["", "## Limitations", ""])
    if quality_label == "low":
        lines.append("- Low image quality may reduce the reliability of automated routing and downstream predictions.")
    if not artifact_index:
        lines.append("- No trained checkpoints were found, so specialist inference could not be completed.")
    lines.append("- This is a research-use automated report and must not be treated as a clinical diagnosis.")
    return "\n".join(lines)


def build_technical_appendix(
    context: AnalysisContext,
    results: Iterable[ToolResult],
    artifact_index: Dict[str, str],
) -> str:
    results = list(results)
    lines: List[str] = [
        "# PULSE Technical Appendix",
        "",
        f"Request ID: `{context.request_id}`",
        "",
        "## Execution Plan",
        "",
    ]
    if context.plan:
        for step in context.plan:
            lines.append(f"- {step}")
    else:
        lines.append("- No specialist tasks were scheduled.")

    lines.extend(["", "## Model Outputs", ""])
    for result in results:
        lines.append(f"### `{result.tool_name}`")
        lines.append(f"- Status: `{result.status}`")
        lines.append(f"- Summary: {result.summary}")
        if result.confidence is not None:
            lines.append(f"- Confidence: `{result.confidence:.3f}`")
        if result.outputs:
            lines.append("- Outputs:")
            for key, value in result.outputs.items():
                if isinstance(value, float):
                    rendered = f"{value:.4f}"
                else:
                    rendered = str(value)
                lines.append(f"  - `{key}`: `{rendered}`")
        if result.artifacts:
            lines.append("- Artifacts:")
            for name, path in result.artifacts.items():
                lines.append(f"  - [{name}]({path})")
        if result.errors:
            lines.append(f"- Errors: `{' ; '.join(result.errors)}`")

    lines.extend(["", "## Available Checkpoints", ""])
    if artifact_index:
        for task_id, path in artifact_index.items():
            lines.append(f"- `{task_id}` -> `{path}`")
    else:
        lines.append("- No trained checkpoints were found under the configured model root.")
    return "\n".join(lines)
