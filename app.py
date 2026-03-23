import importlib.util
import json
import math
import os
import shutil
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
import pywt
import soundfile as sf
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
AASIST_DIR = ROOT / "aasist"
SAFE_DIR = ROOT / "SAFE-main"

AASIST_CONF = AASIST_DIR / "config" / "AASIST.conf"
AASIST_MODEL = AASIST_DIR / "models" / "weights" / "AASIST.pth"
SAFE_CHECKPOINT = SAFE_DIR / "checkpoint" / "checkpoint-best.pth"

IMAGE_ARCH = "CLIP:ViT-L/14"
IMAGE_THRESHOLD = 0.5
AUDIO_THRESHOLD = 0.5

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
BATCH_CACHE_DIR = ROOT / ".batch_cache"


@dataclass
class DetectionResult:
    modality: str
    result_text: str
    confidence: float
    label: str
    diagnostics: Dict[str, Any]


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_package(package_name: str, package_dir: Path):
    init_file = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load package from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def _format_confidence(value: float) -> str:
    return f"{value:.4f}"


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return path.name


class AASISTAudioDetector:
    def __init__(self, conf_path: Path, model_path: Path):
        self.conf_path = conf_path
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_cfg = None

    def _load_checkpoint(self, model, ckpt_path: Path):
        checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("model."):
                key = key[len("model.") :]
            cleaned[key] = value

        model.load_state_dict(cleaned, strict=False)

    def _ensure_loaded(self):
        if self.model is not None:
            return
        if not self.conf_path.exists():
            raise FileNotFoundError(f"Missing config: {self.conf_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing audio weight: {self.model_path}")

        conf = json.loads(self.conf_path.read_text(encoding="utf-8"))
        self.model_cfg = conf["model_config"]
        module = _load_module_from_file(
            "deepshield_aasist_model",
            AASIST_DIR / "models" / "AASIST.py",
        )
        model = module.Model(self.model_cfg).to(self.device).eval()
        self._load_checkpoint(model, self.model_path)
        self.model = model

    @staticmethod
    def _load_audio_mono(path: str, target_sr: int = 16000) -> np.ndarray:
        audio, sr = sf.read(path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != target_sr:
            gcd = math.gcd(int(sr), int(target_sr))
            up = target_sr // gcd
            down = int(sr) // gcd
            audio = resample_poly(audio, up, down).astype(np.float32)
        return audio

    @staticmethod
    def _pad_repeat_or_trunc(audio: np.ndarray, max_len: int) -> np.ndarray:
        if audio.shape[0] >= max_len:
            return audio[:max_len]
        if audio.shape[0] == 0:
            return np.zeros((max_len,), dtype=np.float32)
        repeat = int(max_len / audio.shape[0]) + 1
        return np.tile(audio, repeat)[:max_len].astype(np.float32)

    def predict(self, audio_path: str) -> DetectionResult:
        self._ensure_loaded()
        nb_samp = int(self.model_cfg.get("nb_samp", 64600))
        audio = self._load_audio_mono(audio_path, target_sr=16000)
        duration_sec = float(audio.shape[0] / 16000.0) if audio.shape[0] else 0.0
        model_input = self._pad_repeat_or_trunc(audio, nb_samp)
        batch = torch.from_numpy(np.expand_dims(model_input, axis=0)).to(self.device)

        with torch.no_grad():
            _, logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        is_fake = real_prob < AUDIO_THRESHOLD
        confidence = fake_prob if is_fake else real_prob
        label = "FAKE" if is_fake else "REAL"
        result_text = "Audio is likely forged" if is_fake else "Audio appears authentic"
        return DetectionResult(
            modality="audio",
            result_text=result_text,
            confidence=confidence,
            label=label,
            diagnostics={
                "mode": "real",
                "device": str(self.device),
                "real_prob": round(real_prob, 6),
                "fake_prob": round(fake_prob, 6),
                "duration_sec": round(duration_sec, 3),
                "model_path": _relative_path(self.model_path),
            },
        )


class SAFEImageDetector:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.last_error = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def _ensure_loaded(self):
        if self.model is not None or self.last_error is not None:
            return
        if not self.checkpoint_path.exists():
            self.last_error = f"Missing SAFE checkpoint: {self.checkpoint_path}"
            return

        try:
            module = _load_module_from_file(
                "deepshield_safe_resnet",
                SAFE_DIR / "models" / "resnet.py",
            )
            model = module.resnet50(num_classes=2)
            checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            cleaned = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    key = key[len("module."):]
                cleaned[key] = value
            model.load_state_dict(cleaned, strict=False)
            model.eval().to(self.device)
            self.model = model
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            self.model = None

    def predict(self, image: Image.Image) -> DetectionResult:
        self._ensure_loaded()
        if self.model is None:
            return DetectionResult(
                modality="image",
                result_text="Image inference is currently running in MOCK mode.",
                confidence=0.5,
                label="MOCK",
                diagnostics={
                    "mode": "mock",
                    "reason": self.last_error or "Image model is unavailable.",
                    "required_files": [
                        _relative_path(self.checkpoint_path),
                    ],
                },
            )

        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hh = torch.from_numpy(
                pywt.dwt2(tensor.cpu().numpy(), "bior1.3", mode="symmetric")[1][2]
            ).to(self.device)
            logits = self.model(hh)
            prob_fake = float(torch.softmax(logits, dim=1)[0, 1].cpu().item())
        is_fake = prob_fake >= IMAGE_THRESHOLD
        confidence = prob_fake if is_fake else 1.0 - prob_fake
        label = "FAKE" if is_fake else "REAL"
        result_text = "Image is likely forged" if is_fake else "Image appears authentic"
        return DetectionResult(
            modality="image",
            result_text=result_text,
            confidence=confidence,
            label=label,
            diagnostics={
                "mode": "real",
                "device": str(self.device),
                "prob_fake": round(prob_fake, 6),
                "checkpoint_path": _relative_path(self.checkpoint_path),
                "backend": "SAFE",
            },
        )


audio_detector = AASISTAudioDetector(AASIST_CONF, AASIST_MODEL)
image_detector = SAFEImageDetector(SAFE_CHECKPOINT)


def detect_image_forgery(image) -> Dict[str, Any]:
    if image is None:
        return {
            "modality": "image",
            "result_text": "No image uploaded",
            "confidence": None,
            "diagnostics": {"mode": "skipped"},
        }
    result = image_detector.predict(image)
    return {
        "modality": result.modality,
        "result_text": result.result_text,
        "confidence": round(result.confidence, 6),
        "diagnostics": result.diagnostics,
    }


def detect_audio_forgery(audio_path) -> Dict[str, Any]:
    if not audio_path:
        return {
            "modality": "audio",
            "result_text": "No audio uploaded",
            "confidence": None,
            "diagnostics": {"mode": "skipped"},
        }
    result = audio_detector.predict(audio_path)
    return {
        "modality": result.modality,
        "result_text": result.result_text,
        "confidence": round(result.confidence, 6),
        "diagnostics": result.diagnostics,
    }


def run_detection(image, audio_path):
    if image is None and not audio_path:
        diagnostics = {
            "summary": ["No image or audio input was provided."],
            "status": "no_input",
        }
        return (
            "No inference executed. Please upload at least one image or audio sample.",
            "none",
            "0.0000",
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
        )

    image_result = detect_image_forgery(image)
    audio_result = detect_audio_forgery(audio_path)

    used_modalities = []
    details = []
    confidences = []
    diagnostics: Dict[str, Any] = {}

    if image is not None:
        used_modalities.append("image")
        details.append(f"Image: {image_result['result_text']}")
        if image_result["confidence"] is not None:
            confidences.append(float(image_result["confidence"]))
        diagnostics["image"] = image_result["diagnostics"]

    if audio_path:
        used_modalities.append("audio")
        details.append(f"Audio: {audio_result['result_text']}")
        if audio_result["confidence"] is not None:
            confidences.append(float(audio_result["confidence"]))
        diagnostics["audio"] = audio_result["diagnostics"]

    if image is not None and audio_path:
        image_risky = "forged" in image_result["result_text"].lower()
        audio_risky = "forged" in audio_result["result_text"].lower()
        if image_risky or audio_risky:
            summary = "At least one modality shows elevated forgery risk."
        else:
            summary = "No obvious forgery risk was detected across the submitted modalities."
    else:
        summary = details[0]

    modality_text = " + ".join(used_modalities)
    confidence = max(confidences) if confidences else 0.5
    diagnostics["summary"] = details

    return (
        summary,
        modality_text,
        _format_confidence(confidence),
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
    )


def get_model_status() -> str:
    return json.dumps(
        {
            "image_inference": {
                "module": "SAFE",
                "checkpoint": _relative_path(SAFE_CHECKPOINT),
                "runtime_mode": "real_or_mock_fallback",
            },
            "audio_inference": {
                "module": "AASIST",
                "weight": _relative_path(AASIST_MODEL),
                "runtime_mode": "real",
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def run_detection_for_ui(image, audio_path):
    summary, modality_text, confidence_text, diagnostics_text = run_detection(
        image, audio_path
    )
    return (
        summary,
        modality_text,
        confidence_text,
        get_model_status(),
        diagnostics_text,
    )


def _coerce_uploaded_path(file_obj) -> Path:
    if file_obj is None:
        raise ValueError("Missing uploaded file.")
    if isinstance(file_obj, str):
        return Path(file_obj)
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    raise TypeError(f"Unsupported upload object: {type(file_obj)}")


def _collect_batch_image_paths(batch_files):
    if not batch_files:
        return [], []

    BATCH_CACHE_DIR.mkdir(exist_ok=True)
    image_paths = []
    extracted_dirs = []

    for file_obj in batch_files:
        source_path = _coerce_uploaded_path(file_obj)
        suffix = source_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_paths.append(source_path)
            continue
        if suffix == ".zip":
            extract_dir = BATCH_CACHE_DIR / f"batch_{uuid.uuid4().hex}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            extracted_dirs.append(extract_dir)
            for path in sorted(extract_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(path)

    unique_paths = []
    seen = set()
    for path in image_paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths, extracted_dirs


def run_batch_image_detection(batch_files):
    image_paths, extracted_dirs = _collect_batch_image_paths(batch_files)
    if not image_paths:
        return [], [], "No valid images found in the uploaded files."

    gallery_items = []
    table_rows = []
    fake_count = 0
    score_sum = 0.0

    try:
        for image_path in image_paths:
            with Image.open(image_path) as img:
                pil_image = img.convert("RGB")
                result = image_detector.predict(pil_image)
                prob_fake = float(result.diagnostics.get("prob_fake", 0.5))
                score_sum += prob_fake
                if result.label == "FAKE":
                    fake_count += 1
                gallery_items.append((pil_image.copy(), f"{image_path.name} | {prob_fake:.4f}"))
                table_rows.append(
                    [
                        image_path.name,
                        result.label,
                        f"{prob_fake:.4f}",
                        result.diagnostics.get("mode", "unknown"),
                        result.result_text,
                    ]
                )
    finally:
        for extracted_dir in extracted_dirs:
            shutil.rmtree(extracted_dir, ignore_errors=True)

    avg_score = score_sum / max(1, len(image_paths))
    summary = (
        f"Processed {len(image_paths)} images. "
        f"Predicted forged: {fake_count}. "
        f"Average model score: {avg_score:.4f}."
    )
    return gallery_items, table_rows, summary


def build_demo():
    css = """
    :root {
        --ds-bg-0: #07111b;
        --ds-bg-1: #0b1724;
        --ds-bg-2: #102233;
        --ds-card: rgba(12, 24, 37, 0.88);
        --ds-card-alt: rgba(14, 28, 43, 0.78);
        --ds-border: rgba(145, 177, 205, 0.18);
        --ds-border-strong: rgba(145, 177, 205, 0.28);
        --ds-ink: #ecf3fa;
        --ds-muted: #9cb1c4;
        --ds-soft: #c4d5e4;
        --ds-accent: #5ea7d5;
        --ds-accent-2: #3f7fa8;
        --ds-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
    }
    .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(63, 127, 168, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(94, 167, 213, 0.10), transparent 24%),
            linear-gradient(180deg, var(--ds-bg-0) 0%, var(--ds-bg-1) 45%, #08121d 100%);
        color: var(--ds-ink);
        min-height: 100vh;
    }
    .ds-shell {
        max-width: 1240px;
        margin: 0 auto;
        padding: 20px 14px 42px 14px;
    }
    .ds-hero,
    .ds-card,
    .ds-footer,
    .ds-batch-shell {
        background: linear-gradient(180deg, rgba(13, 24, 38, 0.94) 0%, rgba(10, 20, 31, 0.90) 100%);
        border: 1px solid var(--ds-border);
        border-radius: 20px;
        box-shadow: var(--ds-shadow);
        backdrop-filter: blur(8px);
    }
    .ds-hero {
        padding: 34px 36px 28px 36px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .ds-hero::after {
        content: "";
        position: absolute;
        inset: auto -10% -55% auto;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(94, 167, 213, 0.14), transparent 62%);
        pointer-events: none;
    }
    .ds-kicker {
        font-size: 0.76rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ds-accent);
        margin-bottom: 12px;
        font-weight: 700;
    }
    .ds-hero h1 {
        margin: 0;
        font-size: 2.8rem;
        letter-spacing: 0.03em;
        color: #f4f8fc;
        line-height: 1.05;
    }
    .ds-hero h2 {
        margin: 10px 0 16px 0;
        font-size: 1.08rem;
        font-weight: 600;
        color: var(--ds-soft);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .ds-hero p {
        max-width: 880px;
        margin: 0;
        line-height: 1.8;
        color: var(--ds-muted);
        font-size: 1rem;
    }
    .ds-card,
    .ds-batch-shell {
        padding: 20px 22px;
        margin-bottom: 16px;
    }
    .ds-card h3,
    .ds-batch-shell h3 {
        margin: 0 0 10px 0;
        font-size: 1.02rem;
        color: #f0f6fb;
        letter-spacing: 0.02em;
    }
    .ds-card p,
    .ds-card li,
    .ds-batch-shell p {
        margin: 0;
        line-height: 1.75;
        color: var(--ds-muted);
    }
    .ds-mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
    }
    .ds-mini-card {
        background: linear-gradient(180deg, rgba(17, 32, 49, 0.86), rgba(13, 24, 38, 0.72));
        border: 1px solid var(--ds-border);
        border-radius: 16px;
        padding: 16px 16px 14px 16px;
    }
    .ds-mini-card h4 {
        margin: 0 0 8px 0;
        color: var(--ds-ink);
        font-size: 0.96rem;
    }
    .ds-mini-card p {
        color: var(--ds-muted);
        font-size: 0.94rem;
    }
    .ds-section-title {
        font-size: 0.82rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ds-accent);
        margin-bottom: 12px;
        font-weight: 700;
    }
    .ds-footer {
        padding: 18px 22px;
        margin-top: 8px;
    }
    .ds-footer p {
        margin: 0;
        line-height: 1.75;
        color: var(--ds-muted);
        font-size: 0.94rem;
    }
    .block-title {
        color: var(--ds-ink) !important;
    }
    .gr-button.primary,
    button.primary {
        background: linear-gradient(135deg, #20557f 0%, #2c6a97 100%) !important;
        border: 1px solid rgba(110, 164, 205, 0.38) !important;
        box-shadow: 0 10px 24px rgba(20, 77, 118, 0.28);
    }
    .gr-button.secondary,
    button.secondary {
        background: linear-gradient(135deg, rgba(25, 42, 61, 0.95), rgba(18, 34, 50, 0.92)) !important;
        border: 1px solid var(--ds-border) !important;
        color: var(--ds-ink) !important;
    }
    .gr-box,
    .gr-form,
    .gr-panel,
    .gr-group,
    .gradio-container .wrap,
    .gradio-container .form {
        border-color: var(--ds-border) !important;
    }
    .gr-textbox,
    .gr-dataframe,
    .gr-file,
    .gr-audio,
    .gr-image,
    .gr-gallery {
        border-radius: 16px !important;
    }
    .gr-textbox textarea,
    .gr-textbox input,
    .gr-dataframe,
    .gr-file,
    .gr-audio,
    .gr-image,
    .gr-gallery,
    .gr-box {
        background: rgba(11, 20, 31, 0.72) !important;
        color: var(--ds-ink) !important;
        border: 1px solid var(--ds-border) !important;
    }
    .gr-dataframe table {
        background: transparent !important;
    }
    .gr-dataframe th {
        background: rgba(19, 35, 53, 0.96) !important;
        color: var(--ds-soft) !important;
    }
    .gr-dataframe td {
        background: rgba(9, 18, 28, 0.82) !important;
        color: var(--ds-ink) !important;
    }
    .gr-markdown,
    .gr-html,
    label,
    .gradio-container .prose,
    .gradio-container .prose p {
        color: var(--ds-ink) !important;
    }
    @media (max-width: 900px) {
        .ds-mini-grid {
            grid-template-columns: 1fr;
        }
        .ds-hero {
            padding: 26px 22px 22px 22px;
        }
        .ds-hero h1 {
            font-size: 2.2rem;
        }
    }
    """
    with gr.Blocks(title="DeepShield", css=css) as demo:
        with gr.Column(elem_classes=["ds-shell"]):
            gr.HTML(
                """
                <section class="ds-hero">
                  <div class="ds-kicker">Research Demonstration Interface</div>
                  <h1>DeepShield</h1>
                  <h2>Multimodal Forgery Detection Platform</h2>
                  <p>
                    DeepShield is a research-oriented demonstration system for multimodal forgery analysis.
                    The platform integrates image and audio inference, unified result presentation, diagnostic reporting,
                    and batch scoring workflows for competition presentation and technical review.
                  </p>
                </section>
                """
            )
            gr.HTML("<div class=\"ds-section-title\">Method Overview</div>")
            gr.HTML(
                """
                <section class="ds-card">
                  <div class="ds-mini-grid">
                    <div class="ds-mini-card">
                      <h4>Audio Branch</h4>
                      <p>AASIST is used for audio anti-spoofing and provides real inference on uploaded speech samples.</p>
                    </div>
                    <div class="ds-mini-card">
                      <h4>Image Branch</h4>
                      <p>SAFE is used for synthetic image detection with preprocessing aligned to the local validated inference path.</p>
                    </div>
                    <div class="ds-mini-card">
                      <h4>Inference Engine</h4>
                      <p>The system returns modality-level outcomes, model score summaries, runtime status, and diagnostic information.</p>
                    </div>
                  </div>
                </section>
                """
            )
            gr.HTML("<div class=\"ds-section-title\">Input Interface</div>")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <section class="ds-card">
                          <h3>Image Input</h3>
                          <p>Upload a single image for direct image-branch inference.</p>
                        </section>
                        """
                    )
                    image_input = gr.Image(label="Image Input", type="pil", height=320)
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <section class="ds-card">
                          <h3>Audio Input</h3>
                          <p>Upload a single audio sample for audio-branch anti-spoofing analysis.</p>
                        </section>
                        """
                    )
                    audio_input = gr.Audio(label="Audio Input", type="filepath")
            run_button = gr.Button("Run Inference", variant="primary")
            gr.HTML("<div class=\"ds-section-title\">Inference Results</div>")
            with gr.Row(equal_height=True):
                image_result_text = gr.Textbox(label="Image Inference")
                audio_result_text = gr.Textbox(label="Audio Inference")
                confidence_text = gr.Textbox(label="Confidence Score")
            with gr.Row():
                summary_text = gr.Textbox(label="Multimodal Summary", lines=4)
            gr.HTML("<div class=\"ds-section-title\">System Diagnostics</div>")
            with gr.Row(equal_height=True):
                model_status_text = gr.Textbox(
                    label="Model Status",
                    lines=12,
                    value=get_model_status(),
                )
                diagnostics_text = gr.Textbox(
                    label="Diagnostic Information",
                    lines=12,
                )
            gr.HTML("<div class=\"ds-section-title\">Batch Evaluation Workspace</div>")
            gr.HTML(
                """
                <section class="ds-batch-shell">
                  <h3>Batch Image Review</h3>
                  <p>Upload multiple images or a zip package to review preview samples, model scores, and batch-level summary metrics in one workspace.</p>
                </section>
                """
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    batch_files_input = gr.Files(
                        label="Batch Input",
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".zip"],
                    )
                    batch_run_button = gr.Button("Run Batch Inference", variant="secondary")
                    batch_summary = gr.Textbox(label="Batch Summary", lines=4)
                with gr.Column(scale=8):
                    batch_gallery = gr.Gallery(
                        label="Batch Preview",
                        columns=4,
                        height=360,
                        object_fit="contain",
                    )
            batch_table = gr.Dataframe(
                headers=["Filename", "Prediction", "Model Score", "Model Status", "Comment"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                label="Batch Scoring Table",
            )
            gr.HTML(
                """
                <section class="ds-footer">
                  <p>
                    DeepShield is presented as a restrained academic security interface. Image detection scores should be interpreted as model outputs rather than calibrated probabilities, and performance may degrade on screenshots, interface captures, or other out-of-distribution inputs.
                  </p>
                </section>
                """
            )

        run_button.click(
            fn=run_detection_for_ui,
            inputs=[image_input, audio_input],
            outputs=[
                summary_text,
                image_result_text,
                confidence_text,
                model_status_text,
                diagnostics_text,
            ],
        ).then(
            fn=lambda image, audio: (
                detect_image_forgery(image)["result_text"] if image is not None else "No image input submitted.",
                detect_audio_forgery(audio)["result_text"] if audio else "No audio input submitted.",
            ),
            inputs=[image_input, audio_input],
            outputs=[image_result_text, audio_result_text],
        )
        batch_run_button.click(
            fn=run_batch_image_detection,
            inputs=[batch_files_input],
            outputs=[batch_gallery, batch_table, batch_summary],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
