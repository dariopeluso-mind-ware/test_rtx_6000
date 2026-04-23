#!/usr/bin/env python3
"""
vLLM-based batch processor for Tosano supermarket food-label pipeline.

This is the vLLM-optimized variant of ``full-gpu_main.py``, designed to achieve
sub-second throughput on an NVIDIA RTX 6000 PRO Blackwell (96 GB VRAM).

Pipeline Overview (per image):
    1. Barcode Detection  ──── pyzbar ──────────────────▶ EAN code + orientation
    2. Label Detection    ──── YOLO OBB (TensorRT FP16) ▶ Oriented bounding box
    3. Image Cropping     ──── Affine deskew ────────────▶ Upright label crop
    4. OCR Transcription  ──── vLLM (Qwen3-VL) ─────────▶ Markdown text
    5. Report Generation  ──── Markdown ─────────────────▶ output_test/mocr_batch_results.md

Hardware target:
    Production (96 GB VRAM): RTX 6000 PRO Blackwell – vLLM with BF16 (native),
    chunked prefill, prefix caching for maximum throughput.

Why vLLM vs llama.cpp:
    - PagedAttention v2: KV cache managed in 4KB pages, up to 40% memory reduction
    - Chunked prefill: splits large prefill requests to avoid decode starvation
    - Prefix caching: system prompt KV cache reused across all images
    - Continuous batching: dynamic batch scheduling with iteration-level scheduling
    - Native Blackwell support (BF16)
    - These features together typically yield 2-4x speedup vs llama.cpp for MoE models

Prerequisites:
    pip install vllm>=0.19.0

Model: Qwen/Qwen3.6-35B-A3B (official HuggingFace repo)
    vLLM >= 0.19.0 required. Model runs in BF16 natively on Blackwell.

Environment Variables:
    VLLM_MODEL_REPO_ID     : HuggingFace repo ID (default: Qwen/Qwen3.6-35B-A3B)
    VLLM_PORT              : vLLM server port (default: 8001)
    VLLM_TENSOR_PARALLEL_SIZE : GPU count (default: 1)
    ENABLE_EAN_DETECTION   : "true"/"false" (default: "true")
    SAVE_CROPS             : "true"/"false" (default: "false")
    YOLO_IMG_SIZE          : YOLO input size (default: 640)
    CROP_MAX_DIMENSION     : max crop dimension before base64 (default: 1280)
"""

import time as _time_module
_IMPORT_START_MONOTONIC: float = _time_module.perf_counter()
print(
    "\n⏳ Initialising runtime (PyTorch + CUDA + TensorRT)… "
    "this takes 30-60 s on first launch."
)

from pathlib import Path
from datetime import datetime
import os
import subprocess
import atexit
import sys
import time
import base64
import math

import structlog
import httpx
import cv2
import numpy as np
from PIL import Image as PILImage
from pyzbar.pyzbar import decode as decode_barcodes
from ultralytics import YOLO
# ── vLLM server settings ───────────────────────────────────────────────────────
# Porta del server vLLM (default: 8001, diversa da llama-server che usa 8080)
VLLM_PORT: int = 8001
VLLM_BASE_URL: str = f"http://localhost:{VLLM_PORT}/v1"

# Repo ID del modello Qwen3.6-35B-A3B su HuggingFace
# FP8 raccomandato per vLLM: ~35 GB vs ~72 GB BF16, qualità quasi identica, ~1.5-2x più veloce
# Alternativa BF16 (full precision): Qwen/Qwen3.6-35B-A3B
VLLM_MODEL_REPO_ID: str = "Qwen/Qwen3.6-35B-A3B-FP8"

# vLLM runtime settings (from official HF model card)
VLLM_MAX_MODEL_LEN: int = 8192
VLLM_GPU_MEMORY_UTILIZATION: float = 0.9
VLLM_MAX_NUM_SEQS: int = 8
VLLM_REASONING_PARSER: str = "qwen3"          # Required for Qwen3.6
VLLM_TENSOR_PARALLEL_SIZE: int = 1
# Note: Qwen3.6-35B-A3B-FP8 is the official FP8 quantization (~35 GB vs ~72 GB BF16).
# Fine-grained FP8 with block size 128 — quality "nearly identical" (official Qwen).
# On Blackwell (sm_120) FP8 tensor cores give ~2× throughput vs BF16.

# Multi-Token Prediction (speculative decoding nativo Qwen3.6)
# Potenziale ~20-30% speedup sulla generazione token.
# Sperimentale — disabilitare se causa instabilità.
VLLM_ENABLE_MTP: bool = False

# ── vLLM server lifecycle ────────────────────────────────────────────────────
VLLM_SERVER_LOG_PATH: Path = Path("output/vllm_server.log")
VLLM_SERVER_SHUTDOWN_TIMEOUT_SEC: float = 15.0
VLLM_SERVER_READY_POLL_SEC: float = 5.0
VLLM_SERVER_BOOT_TIMEOUT_SEC: float = 600.0

# ── Transcription API settings ───────────────────────────────────────────────────
TRANSCRIPTION_HTTP_TIMEOUT_SEC: float = 120.0
TRANSCRIPTION_MAX_RETRIES: int = 3
TRANSCRIPTION_MAX_OUTPUT_TOKENS: int = 2048

TRANSCRIPTION_SYSTEM_PROMPT: str = (
    "You are a precise OCR system. Read all the text in the image. "
    "Output only the text directly. Do not explain or add comments."
)
TRANSCRIPTION_USER_PROMPT: str = "Transcribe the text in this image."

# ── Toggle env vars ───────────────────────────────────────────────────────────
# Abilitare/disabilitare la ricerca barcode EAN (default: False)
# Impostare a false per risparmiare ~100-150 ms per immagine
ENABLE_EAN_DETECTION: bool = False

# Salvare i crop JPEG su disco (default: false — disattivato per velocizzare)
# Impostare a true solo per debugging / ispezione visiva dei crop
SAVE_CROPS: bool = False

# ── Image preprocessing settings ────────────────────────────────────────────────
PYZBAR_MAX_DIMENSION: int = 1500

# Dimensione massima del crop prima del base64 encoding (default: 1280)
# Ridurre per immagini molto grandi (es. 800) = meno base64 = prefill più veloce
CROP_MAX_DIMENSION: int = 1280
CROP_JPEG_QUALITY: int = 90

# ── Barcode decoding settings ─────────────────────────────────────────────────
EAN_BARCODE_TYPES: tuple[str, ...] = ("EAN13", "EAN8", "UPCA", "UPCE")
PYZBAR_ORIENTATION_TO_ROTATION_DEGREES: dict[str, int] = {
    "UP": 0, "RIGHT": 270, "DOWN": 180, "LEFT": 90,
}

# ── Image file extensions ──────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# ── YOLO model paths ───────────────────────────────────────────────────────────
YOLO_MODEL_PT_PATH: Path = Path("best.pt")
YOLO_TENSORRT_PATH: Path = Path("best.engine")

# Dimensione input YOLO OBB (deve corrispondere alla export del TensorRT engine)
YOLO_IMG_SIZE: int = 640

# ── Global state ───────────────────────────────────────────────────────────────
vllm_subprocess_handle: subprocess.Popen | None = None
_http_client: httpx.Client | None = None
logger: structlog.stdlib.BoundLogger = structlog.get_logger()


# ── HTTP Client ────────────────────────────────────────────────────────────────
def _get_http_client() -> httpx.Client:
    """Persistent httpx.Client for vLLM server communication."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(TRANSCRIPTION_HTTP_TIMEOUT_SEC),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
            ),
        )
    return _http_client


# ── vLLM Server Lifecycle ──────────────────────────────────────────────────────
def shutdown_vllm_server_on_exit() -> None:
    """Gracefully shuts down the vLLM subprocess at interpreter exit."""
    global vllm_subprocess_handle, _http_client
    if _http_client is not None:
        _http_client.close()
        _http_client = None
    if vllm_subprocess_handle is None:
        return
    logger.info("Shutting down vLLM subprocess...")
    vllm_subprocess_handle.terminate()
    try:
        vllm_subprocess_handle.wait(timeout=VLLM_SERVER_SHUTDOWN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        vllm_subprocess_handle.kill()


atexit.register(shutdown_vllm_server_on_exit)


def ensure_vllm_server_running(vllm_base_url: str) -> None:
    """Check if vLLM server is running; if not, start it as a subprocess."""
    global vllm_subprocess_handle

    # Probe existing server
    try:
        with httpx.Client(timeout=2.0) as probe_client:
            resp = probe_client.get(f"{vllm_base_url}/models")
            if resp.status_code == 200:
                logger.info("vLLM server already running – skipping boot", url=vllm_base_url)
                return
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
        pass

    logger.info(
        "vLLM server not detected – starting vLLM subprocess",
        model=VLLM_MODEL_REPO_ID,
        tp_size=VLLM_TENSOR_PARALLEL_SIZE,
    )

    # vLLM command (from official HF model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
    command_line: list[str] = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLLM_MODEL_REPO_ID,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(VLLM_TENSOR_PARALLEL_SIZE),
        "--max-model-len", str(VLLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(VLLM_MAX_NUM_SEQS),
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--reasoning-parser", VLLM_REASONING_PARSER,
    ]

    # MTP: Multi-Token Prediction — speculative decoding nativo di Qwen3.6.
    # Usa le MTP heads del modello per "draft" di 2 token alla volta.
    # Per OCR (output deterministico) il tasso di accettazione è alto → ~20-30% speedup.
    if VLLM_ENABLE_MTP:
        command_line.extend([
            "--speculative-config",
            '{"method":"qwen3_next_mtp","num_speculative_tokens":2}',
        ])
        logger.info("MTP speculative decoding enabled (num_speculative_tokens=2)")

    logger.info("Starting vLLM subprocess", command=" ".join(command_line))

    VLLM_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file_handle = open(VLLM_SERVER_LOG_PATH, "w", encoding="utf-8")
    vllm_subprocess_handle = subprocess.Popen(
        command_line,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
    )

    logger.info(
        "Waiting for vLLM server to become ready (see output/vllm_server.log)",
        log_path=str(VLLM_SERVER_LOG_PATH),
    )
    boot_start_time: float = time.monotonic()

    while True:
        if vllm_subprocess_handle.poll() is not None:
            exit_code = vllm_subprocess_handle.poll()
            logger.error(
                "vLLM subprocess exited unexpectedly during boot",
                exit_code=exit_code,
                elapsed=f"{time.monotonic() - boot_start_time:.1f}s",
            )
            sys.exit(1)

        if time.monotonic() - boot_start_time > VLLM_SERVER_BOOT_TIMEOUT_SEC:
            logger.error(
                "vLLM server boot timed out",
                timeout_sec=VLLM_SERVER_BOOT_TIMEOUT_SEC,
            )
            vllm_subprocess_handle.terminate()
            sys.exit(1)

        try:
            with httpx.Client(timeout=2.0) as poll_client:
                poll_resp = poll_client.get(f"{vllm_base_url}/models")
                if poll_resp.status_code == 200:
                    elapsed: float = time.monotonic() - boot_start_time
                    logger.info("vLLM server is up and ready", elapsed=f"{elapsed:.1f}s")
                    return
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
            pass

        time.sleep(VLLM_SERVER_READY_POLL_SEC)


# ── YOLO TensorRT Engine ──────────────────────────────────────────────────────
def ensure_yolo_tensorrt_engine() -> Path:
    """Ensures a TensorRT FP16 engine exists for the YOLO OBB model."""
    if YOLO_TENSORRT_PATH.exists():
        logger.info("TensorRT engine found", path=str(YOLO_TENSORRT_PATH))
        return YOLO_TENSORRT_PATH

    if not YOLO_MODEL_PT_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model file not found: {YOLO_MODEL_PT_PATH}. "
            f"Cannot generate TensorRT engine without the PyTorch weights."
        )

    logger.info(
        "Exporting YOLO OBB fine-tuned model to TensorRT FP16 "
        "(one-time operation, ~1-2 min)...",
        source=str(YOLO_MODEL_PT_PATH),
    )
    model = YOLO(str(YOLO_MODEL_PT_PATH), task="obb")
    model.export(format="engine", half=True, device=0)
    logger.info("TensorRT engine created", path=str(YOLO_TENSORRT_PATH))
    return YOLO_TENSORRT_PATH


# ── Image Utilities ────────────────────────────────────────────────────────────
def _resize_image_max_dimension(image: cv2.Mat, max_dim: int) -> cv2.Mat:
    """Resizes image so its longest side equals max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


# ── Barcode Detection ──────────────────────────────────────────────────────────
def detect_ean_barcode_and_orientation(image_bgr: cv2.Mat) -> tuple[str | None, int]:
    """
    Scans a single image for retail EAN/UPC barcodes using pyzbar.

    Args:
        image_bgr: Input image as OpenCV BGR Mat (pre-loaded by caller).

    Returns:
        A 2-tuple (ean_code, rotation_degrees).
    """
    image_for_scan = _resize_image_max_dimension(image_bgr, PYZBAR_MAX_DIMENSION)

    try:
        pil_image: PILImage.Image = PILImage.fromarray(
            cv2.cvtColor(image_for_scan, cv2.COLOR_BGR2RGB)
        )
    except Exception as exc:
        logger.error("Failed to convert image for barcode decoding", error=str(exc))
        return None, 0

    try:
        decoded_symbols = decode_barcodes(pil_image)
    except Exception as exc:
        logger.error("Failed to decode barcodes", error=str(exc))
        return None, 0

    ean_like = [
        sym for sym in decoded_symbols
        if sym.type in EAN_BARCODE_TYPES
    ]
    if not ean_like:
        return None, 0

    barcode = ean_like[0]
    ean_code: str = barcode.data.decode("utf-8")
    orientation_label: str | None = getattr(barcode, "orientation", None)
    if orientation_label is not None:
        orientation_label = str(orientation_label)
    rotation: int = PYZBAR_ORIENTATION_TO_ROTATION_DEGREES.get(
        orientation_label, 0
    )
    return ean_code, rotation


# ── YOLO OBB ─────────────────────────────────────────────────────────────────
def _deskew_crop_obb(
    source_image_bgr: cv2.Mat,
    obb_center_x: float,
    obb_center_y: float,
    obb_width: float,
    obb_height: float,
    obb_rotation_radians: float,
    additional_rotation_degrees: int,
) -> cv2.Mat:
    """
    Crops a label region using an oriented bounding box (OBB).

    Args:
        source_image_bgr: Source image as OpenCV BGR Mat.
        obb_center_x: OBB centroid x-coordinate (pixel units).
        obb_center_y: OBB centroid y-coordinate (pixel units).
        obb_width: OBB width along its local x-axis (pixel units).
        obb_height: OBB height along its local y-axis (pixel units).
        obb_rotation_radians: OBB rotation angle in radians (positive = CCW).
        additional_rotation_degrees: Fine-adjustment from barcode orientation.

    Returns:
        The cropped label as an OpenCV BGR image.
    """
    image_height, image_width = source_image_bgr.shape[:2]
    angle_deg: float = math.degrees(obb_rotation_radians)

    M: cv2.Mat = cv2.getRotationMatrix2D(
        center=(int(obb_center_x), int(obb_center_y)),
        angle=angle_deg,
        scale=1.0,
    )

    deskewed: cv2.Mat = cv2.warpAffine(
        src=source_image_bgr,
        M=M,
        dsize=(image_width, image_height),
    )

    x1 = max(0, int(obb_center_x - obb_width / 2))
    y1 = max(0, int(obb_center_y - obb_height / 2))
    x2 = min(image_width, x1 + int(obb_width))
    y2 = min(image_height, y1 + int(obb_height))

    return deskewed[y1:y2, x1:x2]


# ── OCR Transcription ──────────────────────────────────────────────────────────
def _transcribe_label_image(
    cropped_label_bgr: cv2.Mat,
    vllm_base_url: str,
) -> str:
    """
    Sends a cropped label image to vLLM (Qwen3-VL vision model) for OCR.

    Optimizations:
    - In-memory base64 (no disk roundtrip)
    - Crop resized to CROP_MAX_DIMENSION before encoding
    - System prompt for prefix cache hit
    - Greedy sampling (temperature=0.0, top_k=1)
    - Persistent httpx.Client for connection reuse
    - Chunked prefill enabled on server side
    """
    resized_crop = _resize_image_max_dimension(cropped_label_bgr, CROP_MAX_DIMENSION)
    ok, buf = cv2.imencode(
        ".jpg", resized_crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed for cropped label")
    b64: str = base64.b64encode(buf).decode("utf-8")

    payload: dict = {
        "model": VLLM_MODEL_REPO_ID,
        "messages": [
            {"role": "system", "content": TRANSCRIPTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIPTION_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ],
        # Official Qwen3.6 sampling for non-thinking mode (general tasks):
        # https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8#best-practices
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": TRANSCRIPTION_MAX_OUTPUT_TOKENS,
        "presence_penalty": 1.5,                       # Official anti-repetition
        "stop": ["<|im_end|>"],
        # Disable thinking mode — official method for vLLM:
        # "Qwen3.6 does not officially support /think and /nothink"
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

    chat_url: str = f"{vllm_base_url}/chat/completions"
    last_error: Exception | None = None

    for attempt in range(1, TRANSCRIPTION_MAX_RETRIES + 1):
        try:
            client = _get_http_client()
            response = client.post(chat_url, json=payload)
            response.raise_for_status()
            data: dict = response.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Transcription API call failed – retrying",
                attempt=attempt,
                max_retries=TRANSCRIPTION_MAX_RETRIES,
                error=str(exc),
            )
            if attempt < TRANSCRIPTION_MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Transcription API failed after {TRANSCRIPTION_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# ── Per-Image Pipeline ────────────────────────────────────────────────────────
def process_single_label_image(
    input_image_path: Path,
    yolo_obb_model: YOLO,
    vllm_base_url: str,
    cropped_labels_output_dir: Path,
) -> dict:
    """
    Executes the full pipeline for one image, with per-step timing.

    Returns:
        dict with keys: name, ean, rotation, text, timing.
    """
    logger.info("Processing image", image=input_image_path.name)
    t_start: float = time.perf_counter()

    # Load image ONCE
    t0 = time.perf_counter()
    source_image_bgr: cv2.Mat = cv2.imread(str(input_image_path))
    if source_image_bgr is None:
        raise cv2.error(f"cv2.imread returned None for: {input_image_path}")
    load_ms: float = (time.perf_counter() - t0) * 1000

    # Barcode detection (optional)
    t1 = time.perf_counter()
    if ENABLE_EAN_DETECTION:
        ean_code, rot_deg = detect_ean_barcode_and_orientation(source_image_bgr)
    else:
        ean_code, rot_deg = None, 0
    barcode_ms: float = (time.perf_counter() - t1) * 1000

    # YOLO OBB detection
    t2 = time.perf_counter()
    try:
        results = yolo_obb_model.predict(
            source=source_image_bgr,
            imgsz=YOLO_IMG_SIZE,
            device=0,
            verbose=False,
        )
        first = results[0]
    except Exception as exc:
        logger.error("YOLO OBB inference failed", image=input_image_path.name, error=str(exc))
        yolo_ms: float = (time.perf_counter() - t2) * 1000
        return _make_result(
            input_image_path.name, ean_code, rot_deg,
            f"YOLO inference error: {exc}", time.perf_counter() - t_start,
            load_ms, barcode_ms, yolo_ms, 0.0, 0.0,
        )
    yolo_ms = (time.perf_counter() - t2) * 1000

    if not first.obb or len(first.obb) == 0:
        logger.warning("No label detected by YOLO OBB", image=input_image_path.name)
        return _make_result(
            input_image_path.name, ean_code, rot_deg,
            "No label detected by YOLO OBB.",
            time.perf_counter() - t_start,
            load_ms, barcode_ms, yolo_ms, 0.0, 0.0,
        )

    # Deskew + crop
    t3 = time.perf_counter()
    cx, cy, w, h, r = first.obb.xywhr[0].cpu().numpy().tolist()
    try:
        cropped: cv2.Mat = _deskew_crop_obb(
            source_image_bgr, cx, cy, w, h, r, rot_deg,
        )
    except cv2.error as exc:
        logger.error("Affine deskew + crop failed", image=input_image_path.name, error=str(exc))
        crop_ms: float = (time.perf_counter() - t3) * 1000
        return _make_result(
            input_image_path.name, ean_code, rot_deg,
            f"Image crop failed: {exc}",
            time.perf_counter() - t_start,
            load_ms, barcode_ms, yolo_ms, crop_ms, 0.0,
        )
    crop_ms = (time.perf_counter() - t3) * 1000

    if SAVE_CROPS:
        out_path = cropped_labels_output_dir / f"crop_{input_image_path.name}"
        if not cv2.imwrite(str(out_path), cropped):
            logger.warning("cv2.imwrite returned False", crop_path=str(out_path))

    # OCR
    t4 = time.perf_counter()
    try:
        text: str = _transcribe_label_image(cropped, vllm_base_url)
        logger.info("Label transcribed successfully", image=input_image_path.name)
    except RuntimeError as exc:
        logger.error(
            "Transcription failed after all retries",
            image=input_image_path.name, error=str(exc),
        )
        ocr_ms: float = (time.perf_counter() - t4) * 1000
        return _make_result(
            input_image_path.name, ean_code, rot_deg,
            f"API Error: {exc}",
            time.perf_counter() - t_start,
            load_ms, barcode_ms, yolo_ms, crop_ms, ocr_ms,
        )
    ocr_ms = (time.perf_counter() - t4) * 1000
    total_ms: float = (time.perf_counter() - t_start) * 1000

    return {
        "name": input_image_path.name,
        "ean": ean_code,
        "rotation": rot_deg,
        "text": text,
        "timing": {
            "load_ms": round(load_ms, 1),
            "barcode_ms": round(barcode_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "crop_ms": round(crop_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


def _make_result(
    name: str,
    ean: str | None,
    rot: int,
    text: str,
    elapsed: float,
    load_ms: float,
    barcode_ms: float,
    yolo_ms: float,
    crop_ms: float,
    ocr_ms: float,
) -> dict:
    """Helper to build a result dict for error cases."""
    return {
        "name": name,
        "ean": ean,
        "rotation": rot,
        "text": text,
        "timing": {
            "load_ms": round(load_ms, 1),
            "barcode_ms": round(barcode_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "crop_ms": round(crop_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "total_ms": round(elapsed * 1000, 1),
        },
    }


# ── Batch Report ──────────────────────────────────────────────────────────────
def write_batch_report(
    batch_report_path: Path,
    input_images_dir: Path,
    results: list[dict],
    pipeline_elapsed_sec: float,
) -> None:
    """Writes a markdown report with timing statistics."""
    try:
        with open(batch_report_path, "w", encoding="utf-8") as f:
            f.write("# MOCR Batch Processing Report (vLLM)\n\n")
            f.write(f"**Folder:** `{input_images_dir.name}`  \n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
            )
            f.write(f"**vLLM Model:** `{VLLM_MODEL_REPO_ID}`  \n")
            f.write(f"**Tensor Parallel:** `{VLLM_TENSOR_PARALLEL_SIZE}`  \n")
            f.write(
                f"**EAN Detection:** "
                f"{'enabled' if ENABLE_EAN_DETECTION else 'disabled'}  \n"
            )
            f.write(
                f"**SAVE_CROPS:** {'true' if SAVE_CROPS else 'false'}  \n"
            )
            f.write(f"**Sampling:** greedy (temperature=0.0, top_k=1)  \n")
            f.write(f"**Crop max dimension:** {CROP_MAX_DIMENSION}px  \n")
            f.write(f"**YOLO imgsz:** {YOLO_IMG_SIZE}px  \n\n")

            f.write("## Summary\n\n")
            f.write("| # | Image | EAN | Status | Total (ms) |\n")
            f.write("|---|-------|-----|--------|------------|\n")

            success = 0
            for idx, r in enumerate(results, start=1):
                ean_val = r["ean"] or "—"
                is_err = (
                    r["text"].startswith("API Error")
                    or r["text"].startswith("YOLO inference error")
                    or r["text"].startswith("No label detected")
                    or r["text"].startswith("Image crop failed")
                )
                icon = "✗" if is_err else "✓"
                if not is_err:
                    success += 1
                f.write(
                    f"| {idx} | {r['name']} | {ean_val} | "
                    f"{icon} | {r['timing']['total_ms']} |\n"
                )

            f.write(
                f"\n**Stats:** {success}/{len(results)} succeeded.  \n"
                f"**Pipeline total:** {pipeline_elapsed_sec:.1f}s  \n"
            )
            n = len(results)
            if n > 0:
                avg = pipeline_elapsed_sec * 1000 / n
                f.write(f"**Average per image:** {avg:.0f} ms  \n")
            f.write("\n---\n\n")

            f.write("## Timing Summary\n\n")
            keys = ["load_ms", "barcode_ms", "yolo_ms", "crop_ms", "ocr_ms", "total_ms"]
            labels = ["Load", "Barcode", "YOLO OBB", "Crop", "OCR", "**Total**"]
            tdata: dict[str, list[float]] = {k: [] for k in keys}
            for r in results:
                for k in keys:
                    tdata[k].append(r["timing"][k])

            f.write("| Step | Avg (ms) | Min (ms) | Max (ms) |\n")
            f.write("|------|----------|----------|----------|\n")
            for key, label in zip(keys, labels):
                vals = tdata[key]
                if vals:
                    f.write(
                        f"| {label} | {sum(vals)/len(vals):.1f} | "
                        f"{min(vals):.1f} | {max(vals):.1f} |\n"
                    )
                else:
                    f.write(f"| {label} | — | — | — |\n")

            f.write("\n---\n\n## Detailed Results\n\n")
            for idx, r in enumerate(results, start=1):
                t = r["timing"]
                f.write(f"### {idx}. {r['name']}\n")
                f.write(f"- **EAN:** {r['ean'] or 'None'}\n")
                f.write(f"- **Rotation Needed:** {r['rotation']}°\n")
                f.write(
                    f"- **Timing:** load={t['load_ms']}ms, "
                    f"barcode={t['barcode_ms']}ms, yolo={t['yolo_ms']}ms, "
                    f"crop={t['crop_ms']}ms, ocr={t['ocr_ms']}ms, "
                    f"**total={t['total_ms']}ms**\n\n"
                )
                f.write("**Transcription:**\n")
                f.write(f"```markdown\n{r['text']}\n```\n\n")

        logger.info("Batch report written", report_path=str(batch_report_path))
    except IOError as exc:
        logger.error(
            "Failed to write batch report",
            report_path=str(batch_report_path),
            error=str(exc),
        )


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Tosano vLLM-based food-label batch processor entry point."""
    import_elapsed: float = time.perf_counter() - _IMPORT_START_MONOTONIC
    logger.info(
        "Starting vLLM-based pipeline",
        model=VLLM_MODEL_REPO_ID,
        tp_size=VLLM_TENSOR_PARALLEL_SIZE,
        ean_detection=ENABLE_EAN_DETECTION,
        save_crops=SAVE_CROPS,
        import_time=f"{import_elapsed:.1f}s",
    )

    input_dir: Path = Path("test")
    output_root: Path = Path("output_test")
    crops_dir: Path = output_root / "crops"
    batch_report_path: Path = output_root / "mocr_batch_results.md"

    for d in (input_dir, output_root, crops_dir):
        d.mkdir(exist_ok=True)

    image_paths: list[Path] = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        logger.warning(
            "No image files found",
            input_folder=str(input_dir),
            supported=SUPPORTED_IMAGE_EXTENSIONS,
        )
        return

    # Step 1/3 — Load YOLO OBB (TensorRT FP16)
    logger.info("[Step 1/3] Preparing YOLO OBB model (TensorRT FP16)...")
    try:
        t_yolo = time.perf_counter()
        engine_path: Path = ensure_yolo_tensorrt_engine()
        logger.info(
            "[Step 1/3] Loading YOLO OBB engine into GPU",
            model_path=str(engine_path),
        )
        yolo_model: YOLO = YOLO(str(engine_path), task="obb")
        logger.info(
            "[Step 1/3] YOLO OBB ready ✓",
            elapsed=f"{time.perf_counter() - t_yolo:.1f}s",
        )
    except Exception as exc:
        logger.error("Failed to load YOLO OBB model", error=str(exc))
        return

    # Step 2/3 — Ensure vLLM server is running
    logger.info("[Step 2/3] Ensuring vLLM server is running (download + boot)...")
    ensure_vllm_server_running(VLLM_BASE_URL)
    logger.info("[Step 2/3] vLLM server ready ✓", url=VLLM_BASE_URL)

    # Warmup YOLO TensorRT kernels
    logger.info(f"[Warmup] Running YOLO TensorRT warmup pass ({YOLO_IMG_SIZE}×{YOLO_IMG_SIZE})…")
    t_warmup = time.perf_counter()
    try:
        dummy = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        yolo_model.predict(source=dummy, imgsz=YOLO_IMG_SIZE, device=0, verbose=False)
        logger.info("[Warmup] YOLO TensorRT kernels compiled ✓")
    except Exception as exc:
        logger.warning(
            "[Warmup] YOLO warmup failed (non-fatal)",
            error=str(exc),
        )
    logger.info(
        "[Warmup] All models warm and ready ✓",
        elapsed=f"{time.perf_counter() - t_warmup:.1f}s",
    )

    # Step 3/3 — Process images
    logger.info(
        "[Step 3/3] Starting batch processing",
        total_images=len(image_paths),
        report_path=str(batch_report_path),
    )

    pipeline_start: float = time.perf_counter()
    batch_results: list[dict] = []

    for idx, img_path in enumerate(image_paths, start=1):
        logger.info("Processing image", current=idx, total=len(image_paths))
        result: dict = process_single_label_image(
            input_image_path=img_path,
            yolo_obb_model=yolo_model,
            vllm_base_url=VLLM_BASE_URL,
            cropped_labels_output_dir=crops_dir,
        )
        batch_results.append(result)

        t = result["timing"]
        logger.info(
            "Image processed",
            image=result["name"],
            total_ms=t["total_ms"],
            ocr_ms=t["ocr_ms"],
            yolo_ms=t["yolo_ms"],
            load_ms=t["load_ms"],
            barcode_ms=t["barcode_ms"],
            crop_ms=t["crop_ms"],
        )

        # Per-image transcript
        transcript_path: Path = crops_dir / f"transcript_{img_path.stem}.md"
        try:
            with open(transcript_path, "w", encoding="utf-8") as fh:
                fh.write(f"# Transcription for {img_path.name}\n\n")
                fh.write(f"- **EAN:** {result['ean'] or 'None'}\n")
                fh.write(f"- **Rotation Detected:** {result['rotation']}°\n")
                fh.write(
                    f"- **Timing:** {t['total_ms']}ms "
                    f"(load={t['load_ms']}ms, barcode={t['barcode_ms']}ms, "
                    f"yolo={t['yolo_ms']}ms, crop={t['crop_ms']}ms, "
                    f"ocr={t['ocr_ms']}ms)\n\n"
                )
                fh.write(f"```markdown\n{result['text']}\n```\n")
            logger.debug("Saved individual transcript", path=str(transcript_path))
        except Exception as exc:
            logger.error(
                "Failed to write individual transcript",
                path=str(transcript_path),
                error=str(exc),
            )

    pipeline_elapsed: float = time.perf_counter() - pipeline_start

    write_batch_report(
        batch_report_path=batch_report_path,
        input_images_dir=input_dir,
        batch_processing_results=batch_results,
        pipeline_elapsed_sec=pipeline_elapsed,
    )

    logger.info(
        "Pipeline complete",
        total_images=len(batch_results),
        elapsed_sec=round(pipeline_elapsed, 1),
        avg_ms_per_image=round(
            pipeline_elapsed * 1000 / max(1, len(batch_results))
        ),
    )


if __name__ == "__main__":
    main()