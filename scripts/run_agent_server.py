#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PULSE agent web server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the FastAPI server.")
    parser.add_argument("--data-root", default="Datasets", help="Path to the dataset root directory.")
    parser.add_argument("--model-root", default="runs/pulse_retrain_new", help="Directory containing trained PULSE checkpoints.")
    parser.add_argument("--runtime-root", default="runs/pulse_runtime", help="Directory for uploaded cases and generated reports.")
    parser.add_argument("--image-size", type=int, default=224, help="Inference resize used for uploaded images.")
    parser.add_argument("--device", default="auto", help="Torch device for inference, for example `cpu`, `cuda`, or `auto`.")
    parser.add_argument("--fetalclip-weights", default="", help="Optional path to `FetalCLIP_weights.pt` for fetal zero-shot inference.")
    parser.add_argument("--fetalclip-config", default="", help="Optional path to `FetalCLIP_config.json`. Defaults to the vendored config if omitted.")
    parser.add_argument("--disable-fetalclip", action="store_true", help="Disable FetalCLIP even if weights are present.")
    parser.add_argument("--fetalnet-repo", default="", help="Optional path to a local FetalNet checkout. Defaults to `external/FetalNet` when present.")
    parser.add_argument("--fetalnet-weights", default="", help="Optional path to FetalNet weights. Defaults to `runs/fuvai_weights.pt` when present.")
    parser.add_argument("--disable-fetalnet", action="store_true", help="Disable FetalNet even if source and weights are present.")
    parser.add_argument("--roboflow-api-url", default="https://serverless.roboflow.com", help="Roboflow inference base URL. Set `http://127.0.0.1:9001` for a local Inference Server.")
    parser.add_argument("--roboflow-api-key", default="", help="Roboflow API key. Required for the hosted serverless endpoint and usually recommended for local model download.")
    parser.add_argument("--roboflow-brain-model-id", default="fetal-brain-abnormalities-ultrasound/1", help="Roboflow model id for fetal brain abnormality classification.")
    parser.add_argument("--disable-roboflow-brain", action="store_true", help="Disable the Roboflow fetal-brain specialist.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload during development.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["PULSE_DATA_ROOT"] = str((REPO_ROOT / args.data_root).resolve())
    os.environ["PULSE_MODEL_ROOT"] = str((REPO_ROOT / args.model_root).resolve())
    os.environ["PULSE_RUNTIME_ROOT"] = str((REPO_ROOT / args.runtime_root).resolve())
    os.environ["PULSE_IMAGE_SIZE"] = str(args.image_size)
    os.environ["PULSE_DEVICE"] = args.device
    os.environ["PULSE_FETALCLIP_ENABLED"] = "0" if args.disable_fetalclip else "1"
    os.environ["PULSE_FETALNET_ENABLED"] = "0" if args.disable_fetalnet else "1"
    os.environ["PULSE_ROBOFLOW_BRAIN_ENABLED"] = "0" if args.disable_roboflow_brain else "1"
    os.environ["PULSE_ROBOFLOW_API_URL"] = args.roboflow_api_url
    os.environ["PULSE_ROBOFLOW_BRAIN_MODEL_ID"] = args.roboflow_brain_model_id
    if args.fetalclip_weights:
        os.environ["PULSE_FETALCLIP_WEIGHTS"] = str(Path(args.fetalclip_weights).expanduser().resolve())
    if args.fetalclip_config:
        os.environ["PULSE_FETALCLIP_CONFIG"] = str(Path(args.fetalclip_config).expanduser().resolve())
    if args.fetalnet_repo:
        os.environ["PULSE_FETALNET_REPO_ROOT"] = str(Path(args.fetalnet_repo).expanduser().resolve())
    if args.fetalnet_weights:
        os.environ["PULSE_FETALNET_WEIGHTS"] = str(Path(args.fetalnet_weights).expanduser().resolve())
    if args.roboflow_api_key:
        os.environ["PULSE_ROBOFLOW_API_KEY"] = args.roboflow_api_key

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
