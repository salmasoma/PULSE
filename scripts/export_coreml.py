#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pulse.coreml_export import (  # noqa: E402
    EXCLUDED_BUNDLED_TASKS,
    convert_checkpoint_to_coreml,
    export_record_template,
    find_checkpoint_files,
    load_checkpoint_record,
    manifest_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PULSE checkpoints to Core ML for on-device iPhone inference.")
    parser.add_argument("--model-root", default="runs/pulse_retrain_new", help="Directory containing trained PULSE checkpoints.")
    parser.add_argument("--output-dir", default="exports/pulse_coreml", help="Directory where .mlpackage exports and manifest are written.")
    parser.add_argument(
        "--checkpoint-name",
        default="best_model.pt",
        help="Checkpoint filename to export, for example `best_model.pt` or `student_best.pt`.",
    )
    parser.add_argument("--task-id", action="append", default=[], help="Optional task id filter. Repeat to export multiple tasks.")
    parser.add_argument("--minimum-ios", type=int, default=16, help="Minimum iOS deployment target for exported Core ML models.")
    parser.add_argument(
        "--compute-precision",
        choices=["float16", "float32"],
        default="float16",
        help="Preferred Core ML compute precision.",
    )
    parser.add_argument("--manifest-name", default="pulse_coreml_manifest.json", help="Filename for the generated export manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect exportable checkpoints and write only the manifest.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when a checkpoint export fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = (REPO_ROOT / args.model_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_task_ids = set(args.task_id or [])
    exports = []
    failures = []
    checkpoint_files = find_checkpoint_files(model_root, checkpoint_name=args.checkpoint_name)

    if not checkpoint_files:
        raise SystemExit(f"No checkpoints were found under {model_root}")

    for checkpoint_path in checkpoint_files:
        record = load_checkpoint_record(checkpoint_path)
        if requested_task_ids and record.task.task_id not in requested_task_ids:
            continue
        if not requested_task_ids and record.task.task_id in EXCLUDED_BUNDLED_TASKS:
            continue

        print(f"==> {record.task.task_id}")
        if args.dry_run:
            export = export_record_template(record, image_size=int(record.config.get("image_size", 224)))
            export.reason = "dry_run"
            exports.append(export)
            continue

        try:
            export = convert_checkpoint_to_coreml(
                record,
                output_dir=output_dir,
                minimum_ios_version=args.minimum_ios,
                compute_precision=args.compute_precision,
            )
            exports.append(export)
            print(f"    wrote {Path(export.coreml_path).name}")
        except Exception as exc:  # pragma: no cover
            export = export_record_template(record, image_size=int(record.config.get("image_size", 224)))
            export.reason = str(exc)
            exports.append(export)
            failures.append((record.task.task_id, str(exc)))
            print(f"    failed: {exc}")
            if args.fail_fast:
                break

    payload = manifest_payload(model_root, exports)
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Models listed: {payload['model_count']}")
    print(f"Models exported: {payload['exported_count']}")
    if failures:
        print("Failures:")
        for task_id, reason in failures:
            print(f"- {task_id}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
