#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "MBZUAI/MediX-R1-2B-GGUF"
DEFAULT_TEXT_MODEL = "MediX-R1-2B-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-MediX-R1-2b-F16.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the MediX-R1 2B GGUF text model and mmproj files for mobile staging."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--output-dir",
        default="/Users/salma.hassan/Ultrasound_Project/external_models/medix_r1_2b_gguf",
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--mmproj-model", default=DEFAULT_MMPROJ)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        allow_patterns=[args.text_model, args.mmproj_model, "README.md"],
    )

    print(f"Downloaded MediX-R1 mobile assets to: {snapshot_path}")
    print(f"Text model: {output_dir / args.text_model}")
    print(f"MMProj: {output_dir / args.mmproj_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
