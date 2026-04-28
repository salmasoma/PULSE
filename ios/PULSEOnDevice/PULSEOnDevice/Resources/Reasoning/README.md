This folder is reserved for optional local VQA assets used by the iPhone app.

The repository tracks only lightweight manifests here. It does **not** track staged GGUF model weights or multimodal projector files.

Typical local contents after staging:

- `local_vqa_manifest.json`
- `*.gguf` text model
- `*.gguf` multimodal projector

Current staging path:

1. `scripts/download_medix_r1_mobile_assets.py`
2. `scripts/stage_medix_r1_reasoning_assets.py`

If no local VQA backend is staged, the app can still run its grounded structured reporting flow without the multimodal model.
