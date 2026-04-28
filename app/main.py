from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pulse.runtime import PULSEInferenceService, RuntimeConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = Path(__file__).resolve().parent / "static"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _env_path(name: str, default: Path) -> str:
    return os.environ.get(name, str(default.resolve()))


def load_runtime_config_from_env() -> RuntimeConfig:
    return RuntimeConfig(
        data_root=_env_path("PULSE_DATA_ROOT", REPO_ROOT / "Datasets"),
        model_root=_env_path("PULSE_MODEL_ROOT", REPO_ROOT / "runs" / "pulse_retrain_new"),
        runtime_root=_env_path("PULSE_RUNTIME_ROOT", REPO_ROOT / "runs" / "pulse_runtime"),
        image_size=int(os.environ.get("PULSE_IMAGE_SIZE", "224")),
        device=os.environ.get("PULSE_DEVICE", "auto"),
        fetalclip_enabled=os.environ.get("PULSE_FETALCLIP_ENABLED", "1") not in {"0", "false", "False"},
        fetalclip_weights=os.environ.get("PULSE_FETALCLIP_WEIGHTS"),
        fetalclip_config=os.environ.get("PULSE_FETALCLIP_CONFIG"),
        fetalnet_enabled=os.environ.get("PULSE_FETALNET_ENABLED", "1") not in {"0", "false", "False"},
        fetalnet_repo_root=os.environ.get("PULSE_FETALNET_REPO_ROOT"),
        fetalnet_weights=os.environ.get("PULSE_FETALNET_WEIGHTS"),
        roboflow_brain_enabled=os.environ.get("PULSE_ROBOFLOW_BRAIN_ENABLED", "1") not in {"0", "false", "False"},
        roboflow_api_url=os.environ.get("PULSE_ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
        roboflow_api_key=os.environ.get("PULSE_ROBOFLOW_API_KEY"),
        roboflow_brain_model_id=os.environ.get(
            "PULSE_ROBOFLOW_BRAIN_MODEL_ID",
            "fetal-brain-abnormalities-ultrasound/1",
        ),
    )


def _safe_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_SUFFIXES else ".png"


def _is_image_upload(upload: UploadFile | None) -> bool:
    if upload is None:
        return False
    content_type = upload.content_type or ""
    suffix = Path(upload.filename or "").suffix.lower()
    return content_type.startswith("image/") or suffix in ALLOWED_SUFFIXES


async def _save_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    await upload.close()
    return destination


def _parse_task_hints(raw: str) -> List[str]:
    hints = []
    for item in raw.split(","):
        hint = item.strip().lower()
        if hint:
            hints.append(hint)
    return hints


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    runtime_config = config or load_runtime_config_from_env()
    service = PULSEInferenceService(runtime_config)

    app = FastAPI(
        title="PULSE Agent API",
        description="Upload ultrasound images, route them through PULSE specialist models, and receive a structured report.",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    app.state.service = service

    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    app.mount("/generated", StaticFiles(directory=service.runtime_root), name="generated")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/api/health")
    async def health() -> Dict:
        return service.health()

    @app.get("/api/tasks")
    async def tasks() -> Dict:
        inventory = service.task_inventory()
        domains = sorted({task["domain"] for task in inventory})
        return {
            "tasks": inventory,
            "domains": domains,
            "counts": {
                "total": len(inventory),
                "ready": sum(1 for task in inventory if task["status"] == "ready"),
                "checkpoints": sum(1 for task in inventory if task["checkpoint_available"]),
            },
        }

    @app.post("/api/analyze")
    async def analyze(
        image: UploadFile = File(...),
        prompt: str = Form(""),
        task_hints: str = Form(""),
        pixel_spacing_mm: str = Form(""),
        cdfi: UploadFile | None = File(None),
        elastic: UploadFile | None = File(None),
    ) -> Dict:
        if not _is_image_upload(image):
            raise HTTPException(status_code=400, detail="The primary upload must be an image file.")
        if cdfi is not None and cdfi.filename and not _is_image_upload(cdfi):
            raise HTTPException(status_code=400, detail="The CDFI modality upload must be an image file.")
        if elastic is not None and elastic.filename and not _is_image_upload(elastic):
            raise HTTPException(status_code=400, detail="The elastography upload must be an image file.")

        parsed_pixel_spacing = None
        if pixel_spacing_mm.strip():
            try:
                parsed_pixel_spacing = float(pixel_spacing_mm)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Pixel spacing must be a valid number in mm/pixel.") from exc
            if parsed_pixel_spacing <= 0:
                raise HTTPException(status_code=400, detail="Pixel spacing must be greater than zero.")

        request_id, input_root, _ = service.create_request_workspace()
        primary_path = await _save_upload(image, input_root / f"primary{_safe_suffix(image)}")

        extra_images: Dict[str, Path] = {}
        if cdfi is not None and cdfi.filename:
            extra_images["cdfi"] = await _save_upload(cdfi, input_root / f"cdfi{_safe_suffix(cdfi)}")
        if elastic is not None and elastic.filename:
            extra_images["elastic"] = await _save_upload(elastic, input_root / f"elastic{_safe_suffix(elastic)}")

        try:
            payload = service.analyze(
                primary_image=primary_path,
                prompt=prompt,
                extra_images=extra_images,
                task_hints=_parse_task_hints(task_hints),
                metadata={
                    "request_id": request_id,
                    "primary_filename": image.filename or primary_path.name,
                    "extra_modalities": sorted(extra_images),
                    "pixel_spacing_mm": parsed_pixel_spacing,
                },
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"PULSE analysis failed: {exc}") from exc

        return payload

    return app


app = create_app()
