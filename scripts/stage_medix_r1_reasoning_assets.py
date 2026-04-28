#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_APP_REASONING_DIR = Path(
    "/Users/salma.hassan/Ultrasound_Project/ios/PULSEOnDevice/PULSEOnDevice/Resources/Reasoning"
)
DEFAULT_TEXT_MODEL = "MediX-R1-2B.Q4_K_S.gguf"
DEFAULT_MMPROJ = "mmproj-MediX-R1-2b-F16.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage MediX-R1 GGUF assets into the iOS app resources and write a local VQA manifest."
    )
    parser.add_argument(
        "--assets-dir",
        default="/Users/salma.hassan/Ultrasound_Project/external_models/medix_r1_2b_gguf",
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--mmproj-model", default=DEFAULT_MMPROJ)
    parser.add_argument("--app-reasoning-dir", default=str(DEFAULT_APP_REASONING_DIR))
    return parser.parse_args()


def copy_if_needed(source: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)


def infer_variant_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "2B" in stem:
        tail = stem.split("2B", 1)[1].lstrip(".-_")
        if tail:
            return f"2B {tail}"
    return stem


def main() -> int:
    args = parse_args()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    reasoning_dir = Path(args.app_reasoning_dir).expanduser().resolve()
    reasoning_dir.mkdir(parents=True, exist_ok=True)

    text_model = assets_dir / args.text_model
    mmproj_model = assets_dir / args.mmproj_model

    if not text_model.exists():
        raise SystemExit(f"Missing text model: {text_model}")
    if not mmproj_model.exists():
        raise SystemExit(f"Missing mmproj model: {mmproj_model}")

    text_destination = reasoning_dir / text_model.name
    mmproj_destination = reasoning_dir / mmproj_model.name

    copy_if_needed(text_model, text_destination)
    copy_if_needed(mmproj_model, mmproj_destination)

    manifest = {
        "provider": "medix_r1",
        "variant": infer_variant_from_filename(text_destination.name),
        "modality": "vision-language",
        "text_model_filename": text_destination.name,
        "mmproj_filename": mmproj_destination.name,
        "runtime_supported": True,
        "runtime_target": "llama.cpp + libmtmd iOS bridge",
        "note": "These assets stage MediX-R1 for on-device ultrasound VQA. If the VLM cannot initialize on the device, the app falls back to the grounded local composer.",
    }

    manifest_path = reasoning_dir / "local_vqa_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Staged text model: {text_destination}")
    print(f"Staged mmproj model: {mmproj_destination}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
