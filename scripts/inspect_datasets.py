#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulse.agents import DatasetInspectorAgent


def main():
    parser = argparse.ArgumentParser(description="Inspect PULSE dataset coverage and discovered task specs.")
    parser.add_argument("--data-root", default="Datasets", help="Path to the dataset root directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inspector = DatasetInspectorAgent(Path(args.data_root).resolve(), seed=args.seed)
    tasks = inspector.inspect()
    print(inspector.to_markdown(tasks))


if __name__ == "__main__":
    main()
