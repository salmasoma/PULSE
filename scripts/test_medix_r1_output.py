#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path("/Users/salma.hassan/Ultrasound_Project")
DEFAULT_BINARY = ROOT / "external/llmfarm_core.swift/llama.cpp/build-mtmd/bin/llama-mtmd-cli"
DEFAULT_REPO = "MBZUAI/MediX-R1-2B-GGUF"
DEFAULT_QUANT = "Q4_K_M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediX-R1 GGUF VQA via llama-mtmd-cli on a local image."
    )
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="Path to llama-mtmd-cli.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="Hugging Face repo id.")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="Quant name passed to --hf-repo.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Question to ask about the image.")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--offline", action="store_true", help="Use cached files only.")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--output-json", help="Optional JSON output path.")
    return parser.parse_args()


def ensure_path(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    return path


def extract_answer(stdout: str) -> str:
    lines = [line.rstrip() for line in stdout.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("image decoded"):
            tail = "\n".join(lines[index + 1 :]).strip()
            if tail:
                return tail.split("llama_perf_context_print:", 1)[0].strip()
    return stdout.strip()


def main() -> int:
    args = parse_args()
    binary = ensure_path(args.binary, "llama-mtmd-cli binary")
    image = ensure_path(args.image, "image")

    command = [
        str(binary),
        "--hf-repo",
        f"{args.repo_id}:{args.quant}",
        "--image",
        str(image),
        "-p",
        args.question,
        "-n",
        str(args.max_tokens),
        "--temp",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
    ]
    if args.offline:
        command.append("--offline")

    if args.print_command:
        print(" ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        raise SystemExit(
            f"MediX-R1 command failed with exit code {result.returncode}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )

    answer = extract_answer(stdout)
    payload = {
        "repo_id": args.repo_id,
        "quant": args.quant,
        "image": str(image),
        "question": args.question,
        "answer": answer,
        "stdout": stdout,
        "stderr": stderr,
        "command": command,
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved output to {output_path}")

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
