---
title: DeepShield
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# DeepShield

Competition-oriented Gradio demo for multimodal forgery detection, prepared as a Hugging Face Spaces upload package.

## Demo Scope

- Audio branch: real inference with `aasist` using `aasist/models/weights/AASIST.pth`
- Image branch: real inference with `SAFE-main` using `SAFE-main/checkpoint/checkpoint-best.pth`
- Failure handling: if a branch cannot load its model, the UI returns a clear fallback diagnostic instead of crashing the whole page

## Included Backends

- Audio backend: `AASIST`
- Image backend: `SAFE`

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Default URL:

- `http://127.0.0.1:7860`

## Files Required For Spaces

- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `aasist/config/AASIST.conf`
- `aasist/models/AASIST.py`
- `aasist/models/weights/AASIST.pth`
- `SAFE-main/models/resnet.py`
- `SAFE-main/checkpoint/checkpoint-best.pth`

## Deployment Notes

- Target platform: Hugging Face Spaces
- SDK: Gradio
- Recommended hardware: CPU Basic for the competition demo
- No Docker setup is required
- All runtime model paths in `app.py` are relative to the project root

## Current Demo Strategy

- Audio branch remains real inference
- Image branch remains real inference when the SAFE checkpoint loads successfully
- If a branch fails to load, the UI reports fallback diagnostics without bringing down the full page
