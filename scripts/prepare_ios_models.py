#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def compile_mlpackage(package_path: Path, output_dir: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="pulse_coremlcompile_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(package_path), str(tmpdir_path)],
            check=True,
        )
        compiled = next(tmpdir_path.glob("*.mlmodelc"), None)
        if compiled is None:
            raise RuntimeError(f"coremlcompiler did not produce an .mlmodelc for {package_path.name}")

        destination = output_dir / compiled.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(compiled), str(destination))
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile iOS Core ML packages for bundling.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing pulse_coreml_manifest.json and .mlpackage exports.",
    )
    parser.add_argument(
        "--keep-packages",
        action="store_true",
        help="Keep the original .mlpackage exports after successful compilation.",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    if not models_dir.is_dir():
        raise SystemExit(f"Models directory not found: {models_dir}")

    packages = sorted(models_dir.glob("*.mlpackage"))
    if not packages:
        raise SystemExit(f"No .mlpackage models found in {models_dir}")

    compiled = []
    for package in packages:
        compiled_path = compile_mlpackage(package, models_dir)
        compiled.append(compiled_path.name)

    if not args.keep_packages:
        for package in packages:
            shutil.rmtree(package)

    print(f"Compiled {len(compiled)} models into {models_dir}")
    for name in compiled:
        print(f"- {name}")


if __name__ == "__main__":
    main()
