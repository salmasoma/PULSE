# PULSE On-Device iPhone App

This directory contains the SwiftUI iPhone application for PULSE.

The repository tracks source code and lightweight manifests only. It does **not** track staged Core ML bundles, GGUF reasoning models, exported checkpoints, or generated Xcode project files.

## Included Here

- `project.yml`: XcodeGen specification
- `PULSEOnDevice/`: Swift and Objective-C++ app source
- `Vendor/PULSEMoondreamVendor/`: vendored iOS `llama.cpp` / `mtmd` runtime package
- lightweight resource manifests and staging instructions

## Not Included Here

- `*.mlmodelc` specialist bundles
- `*.gguf` text or multimodal projector weights
- training checkpoints
- generated `PULSEOnDevice.xcodeproj`
- local signing state

## App Responsibilities

- intake from camera, photo library, or files
- on-device domain routing
- specialist execution through bundled Core ML models
- structured report generation from specialist evidence
- optional local VQA through staged GGUF multimodal assets
- PDF export and local history

## Build Workflow

### 1. Install prerequisites

- Xcode
- XcodeGen
- Python environment with `coremltools` for model export

```bash
brew install xcodegen
```

### 2. Export and prepare Core ML bundles

```bash
python3 scripts/export_coreml.py --help
python3 scripts/export_fetal_specialists_coreml.py --help
python3 scripts/prepare_ios_models.py --help
```

Those scripts write compiled resources into:

- `PULSEOnDevice/Resources/Models/`

### 3. Stage the local VQA backend, if needed

The current manifest targets a MediX-R1 backend through the iOS `llama.cpp + libmtmd` bridge.

```bash
python3 scripts/download_medix_r1_mobile_assets.py --help
python3 scripts/stage_medix_r1_reasoning_assets.py --help
```

Those scripts stage assets into:

- `PULSEOnDevice/Resources/Reasoning/`

### 4. Generate and open the Xcode project

```bash
cd ios/PULSEOnDevice
xcodegen generate
open PULSEOnDevice.xcodeproj
```

## Resource Directories

- `PULSEOnDevice/Resources/Models/`: placeholder directory plus manifests for Core ML specialists
- `PULSEOnDevice/Resources/Reasoning/`: placeholder directory plus manifests for local VQA assets

See the `README.md` files inside those folders for the exact staging expectations.
