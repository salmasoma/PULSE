This folder is reserved for compiled Core ML specialist bundles and their manifests.

The repository tracks only lightweight metadata here:

- `pulse_coreml_manifest.json`
- `fetal_specialists_manifest.json`

It does **not** track the generated `*.mlmodelc` bundles.

Populate this folder locally with:

1. `scripts/export_coreml.py`
2. `scripts/export_fetal_specialists_coreml.py`
3. `scripts/prepare_ios_models.py`

Expected local contents after staging:

- one compiled bundle per specialist
- `pulse_coreml_manifest.json`
- `fetal_specialists_manifest.json`
