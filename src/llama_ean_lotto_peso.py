#!/usr/bin/env python3
"""
Full-GPU batch processor for Tosano supermarket food-label pipeline — V2.

This is the production-optimized variant of ``main.py``, designed to maximise
throughput on an NVIDIA RTX 6000 PRO Blackwell (96 GB VRAM).

V2 improvements over V1 (based on 183-image batch analysis):
    - --swa-full: fixes SWA + cache-prompt repetition loop (Qwen3.6 MoE bug)
    - --cache-type-k/v q8_0: quantized KV cache for ~50% VRAM savings + bandwidth
    - Context size 16384→4096: based on real token stats (P99=458 tokens)
    - presence_penalty=1.5: official Qwen3.6 anti-repetition (from model card)
    - Downscale before warpAffine: faster crop on large images
    - Async image pre-loading: eliminates RunPod network volume cold-cache I/O

Pipeline Overview (per image):
    1. Barcode Detection  ──── pyzbar ──────────────────▶ EAN code + orientation
    2. Label Detection    ──── YOLO OBB (TensorRT FP16) ▶ Oriented bounding box
    3. Image Cropping     ──── Affine deskew ────────────▶ Upright label crop
    4. OCR Transcription  ──── llama-server (Qwen3.6) ──▶ Markdown text
    5. Report Generation  ──── Markdown ─────────────────▶ output_test/mocr_batch_results.md

Hardware target:
    Production (96 GB VRAM): RTX 6000 PRO Blackwell – tutto in GPU, Flash Attention,
    thinking mode disabilitato (metodo ufficiale Unsloth/HuggingFace),
    TensorRT FP16 per YOLO, ottimizzato per MASSIMA VELOCITÀ.

    Per development (8 GB VRAM) usare ``src/main.py`` con --cpu-moe.

References:
    - https://unsloth.ai/docs/models/qwen3.6
    - https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
    - https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf

WARNING: Do NOT use CUDA 13.2 — known bug causes gibberish outputs.
         Use CUDA 12.8 or 13.0 (see Unsloth docs).
"""

# --------------------------------------------------------------------------------------------------
# Early startup message — printed before heavy imports so the terminal isn't silent
# during the ~30-60 s PyTorch + CUDA initialisation phase.
# We import time early (stdlib, instant) to capture the timestamp.
# --------------------------------------------------------------------------------------------------
import time as _time_module
_IMPORT_START_MONOTONIC: float = _time_module.perf_counter()
print("\n⏳ Initialising runtime (PyTorch + CUDA + TensorRT)… this takes 30-60 s on first launch.")

# --------------------------------------------------------------------------------------------------
# Stdlib imports
# --------------------------------------------------------------------------------------------------
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import os
import subprocess
import atexit
import sys
import time
import math

# SIMD-accelerated base64 (~4× faster than stdlib on large payloads).
# Falls back to stdlib if pybase64 is not installed.
try:
    import pybase64 as base64
except ImportError:
    import base64

# --------------------------------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------------------------------
import structlog
import httpx
import cv2                                     # OpenCV – used for affine deskew + crop
import numpy as np                             # NumPy – used for YOLO warmup dummy image
from PIL import Image as PILImage              # Pillow – used by pyzbar for barcode decoding
from pyzbar.pyzbar import decode as decode_barcodes
from ultralytics import YOLO
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------------------------------

# --- Paths ---
LLAMA_SERVER_BINARY_PATH: Path = Path("./llama.cpp/llama-server")
LLAMA_SERVER_LOG_PATH: Path = Path("output/llama_server.log")

# --- Qwen3.6 model identifiers (HuggingFace Hub) ---
QWEN_GGUF_REPO_ID: str = "unsloth/Qwen3.6-35B-A3B-GGUF"

# Modelli GGUF disponibili — impostabile via env var per test A/B:
#   UD-Q4_K_XL  (~22.4 GB) — Unsloth Dynamic 2.0, SOTA KL Divergence (raccomandato da Unsloth)
#   Q4_K_M      (~20 GB)   — standard, velocità baseline
# Entrambi hanno ~4 bit per peso → stessa velocità di decode (memory-bandwidth-bound).
# La differenza è nella qualità: UD-Q4_K_XL ha layout layer ottimizzato per preservare accuratezza.
# Modello GGUF da usare (default: UD-Q4_K_XL — Unsloth Dynamic 2.0, raccomandato)
# Alternativa: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
QWEN_GGUF_WEIGHTS_FILENAME: str = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
QWEN_VISION_PROJECTOR_FILENAME: str = "mmproj-F16.gguf"
QWEN_MODEL_API_NAME: str = "Qwen3.6"          # Value used in the chat-completion "model" field

# --- llama-server runtime settings ---
LLAMA_SERVER_PORT: int = 8080
LLAMA_SERVER_BASE_URL: str = f"http://localhost:{LLAMA_SERVER_PORT}/v1"
LLAMA_SERVER_CONTEXT_SIZE: int = 4096          # Based on real token stats: P99=3008 tokens (50+2500+458)
LLAMA_SERVER_CPU_THREADS: int = 8             # With full GPU offload, 8 threads for tokenizer/sampling
LLAMA_SERVER_SHUTDOWN_TIMEOUT_SEC: float = 10.0
LLAMA_SERVER_READY_POLL_SEC: float = 5.0       # Sleep between readiness probes during boot
LLAMA_SERVER_BOOT_TIMEOUT_SEC: float = 600.0   # 10 min: includes first-time GGUF download (~22 GB)
LLAMA_SERVER_BATCH_SIZE: int = 4096            # V3: bigger prefill chunks on RTX 6000 PRO (96 GB VRAM)
LLAMA_SERVER_UBATCH_SIZE: int = 4096           # V3: must be >= --image-max-tokens (560) for vision

# --- Transcription API settings ---
TRANSCRIPTION_HTTP_TIMEOUT_SEC: float = 120.0
TRANSCRIPTION_MAX_RETRIES: int = 3
TRANSCRIPTION_MAX_OUTPUT_TOKENS: int = 256     # JSON output only: ~30-50 tokens

# System prompt (cached in KV cache, reused for all images)
# V3: JSON extraction instead of full OCR – extract only EAN, lotto, peso_netto
TRANSCRIPTION_SYSTEM_PROMPT: str = (
    "You are a food label data extraction system. "
    "Extract from the image the following fields: EAN code (barcode number), "
    "lot number (codice lotto/lotto/lotto n), net weight in kg (peso netto). "
    "Return ONLY valid JSON with these exact keys: ean, lotto, peso_netto. "
    "Use null for missing fields. Normalize weight to float kilograms."
)

# User prompt (image-specific, cannot be cached)
TRANSCRIPTION_USER_PROMPT: str = (
    "Extract: EAN code, lotto number, peso netto (kg). Output JSON only."
)

# --- EAN detection toggle ---
# Set ENABLE_EAN_DETECTION=false to skip barcode search (saves ~5-10 ms per image)
# Abilitare/disabilitare la ricerca barcode EAN (default: False)
# Impostare a false per risparmiare ~100-150 ms per immagine
ENABLE_EAN_DETECTION: bool = False

# --- Crop disk-write toggle ---
# Set SAVE_CROPS=true to write cropped label JPEGs to disk (for debugging).
# Default false: skip disk I/O in production hot path, only base64 encode in memory.
# Salvare i crop JPEG su disco (default: false — disattivato per velocizzare)
# Impostare a true solo per debugging / ispezione visiva dei crop
SAVE_CROPS: bool = False

# --- Image preprocessing settings ---
# Downscale barcode images to this max dimension before pyzbar scan.
# EAN barcodes are large enough (100+ px wide) to survive downscale with full recall.
PYZBAR_MAX_DIMENSION: int = 1500

# Resize crop to this max dimension before base64 encoding.
# Smaller images = less base64 data = faster prefill in llama-server.
# Dimensione massima del crop prima del base64 encoding (default: 1280)
# Ridurre per immagini molto grandi (es. 800) = meno base64 = prefill più veloce
CROP_MAX_DIMENSION: int = 800                  # V3: smaller = fewer vision tokens + faster crop

# JPEG quality for in-memory encoding (0-100, higher = larger file + more quality)
# V3: 75 produces ~30-40% smaller base64 payloads, negligible quality loss for text
CROP_JPEG_QUALITY: int = 75

# --- Barcode decoding settings ---
# Barcode types that encode a standard retail EAN/UPC product code.
EAN_BARCODE_TYPES: tuple[str, ...] = ("EAN13", "EAN8", "UPCA", "UPCE")

# Mapping from pyzbar's cryptic "orientation" string (returned by the barcode decoder)
# to the degrees of counter-clockwise rotation required to make the barcode text readable.
# pyzbar encodes "RIGHT" → the barcode is rotated 90° CW relative to the image, so we need
# to rotate 270° CCW to bring it upright, and so on.
PYZBAR_ORIENTATION_TO_ROTATION_DEGREES: dict[str, int] = {
    "UP":    0,
    "RIGHT": 270,
    "DOWN":  180,
    "LEFT":  90,
}

# --- Image file extensions accepted as input ---
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# --- YOLO model paths ---
YOLO_MODEL_PT_PATH: Path = Path("best.pt")
YOLO_TENSORRT_PATH: Path = Path("best.engine")
# Dimensione input YOLO OBB (deve corrispondere alla export del TensorRT engine)
YOLO_IMG_SIZE: int = 640

# --------------------------------------------------------------------------------------------------
# Global state – shared across function calls without passing as arguments.
# Initialised once in main() and cleaned up at interpreter shutdown.
# --------------------------------------------------------------------------------------------------

# Handle to the llama-server subprocess started on demand.
# None when no subprocess was spawned (server was already running or not needed).
llama_server_subprocess_handle: subprocess.Popen | None = None

# Persistent HTTP client for llama-server (reused across all transcription requests).
# Created lazily in _get_http_client() after server is confirmed ready.
_http_client: httpx.Client | None = None

# --------------------------------------------------------------------------------------------------
# Module logger (structured via structlog)
# --------------------------------------------------------------------------------------------------
logger: structlog.stdlib.BoundLogger = structlog.get_logger()


# ==================================================================================================
# HTTP Client Management
# ==================================================================================================

def _get_http_client() -> httpx.Client:
    """
    Returns a persistent httpx.Client connected to llama-server.

    The client is created once and reused for all transcription requests,
    avoiding TCP connection overhead on each call.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(TRANSCRIPTION_HTTP_TIMEOUT_SEC),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _http_client


# ==================================================================================================
# llama-server Lifecycle Management
# ==================================================================================================

def shutdown_llama_server_on_exit() -> None:
    """
    Gracefully shuts down the llama-server subprocess registered at interpreter exit.

    Uses atexit to ensure cleanup runs even when the script is interrupted (Ctrl+C) or
    terminates unexpectedly. First attempts a clean shutdown with a 10 s timeout; falls
    back to SIGKILL if the process does not exit voluntarily.
    """
    global llama_server_subprocess_handle, _http_client

    # Close persistent HTTP client first.
    if _http_client is not None:
        _http_client.close()
        _http_client = None

    if llama_server_subprocess_handle is None:
        return

    logger.info("Shutting down llama-server subprocess...")
    llama_server_subprocess_handle.terminate()
    try:
        llama_server_subprocess_handle.wait(timeout=LLAMA_SERVER_SHUTDOWN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        # Process ignored SIGTERM – force-kill it.
        llama_server_subprocess_handle.kill()


# Register the cleanup handler so it runs automatically at shutdown.
atexit.register(shutdown_llama_server_on_exit)


def ensure_llama_server_running(llama_server_base_url: str) -> None:
    """
    Checks whether llama-server is already running at the given base URL, and if not,
    downloads the required Qwen3.6 GGUF weights (including the vision projector) from
    HuggingFace Hub and starts the server as a background subprocess.

    Full-GPU configuration:
        - All 256 MoE experts in GPU VRAM (no --cpu-moe)
        - Flash Attention enabled (--flash-attn)
        - Thinking mode disabled via --chat-template-kwargs '{"enable_thinking":false}'
          (official method from Unsloth docs)
        - Continuous batching enabled
        - Prompt caching enabled
        - Batch size for prefill acceleration

    Args:
        llama_server_base_url: Base URL of the llama-server OpenAI-compatible API
                               (e.g. "http://localhost:8080/v1").
    """
    global llama_server_subprocess_handle

    # ----------------------------------------------------------------------------------------------
    # Step 1 – Probe existing server
    # ----------------------------------------------------------------------------------------------
    try:
        with httpx.Client(timeout=2.0) as probe_client:
            response = probe_client.get(f"{llama_server_base_url}/models")
            if response.status_code == 200:
                logger.info(
                    "llama-server already running – skipping boot",
                    url=llama_server_base_url,
                )
                return
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
        # Server is not reachable – proceed with download + spawn.
        pass

    logger.info(
        "llama-server not detected on port – initiating download and boot sequence"
    )

    # ----------------------------------------------------------------------------------------------
    # Step 2 – Download Qwen3.6 GGUF weights from HuggingFace Hub
    # ----------------------------------------------------------------------------------------------
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.error(
            "huggingface_hub package not installed",
            hint="pip install huggingface_hub",
        )
        raise ImportError(
            "huggingface_hub is required to download Qwen3.6 weights"
        ) from exc

    logger.info(
        "Downloading Qwen3.6 GGUF model weights — this may take a while on first run "
        "(~22 GB for UD-Q4_K_XL)...",
        filename=QWEN_GGUF_WEIGHTS_FILENAME,
    )
    qwen_weights_path: Path = Path(
        hf_hub_download(
            repo_id=QWEN_GGUF_REPO_ID,
            filename=QWEN_GGUF_WEIGHTS_FILENAME,
        )
    )
    logger.info("GGUF weights downloaded", path=str(qwen_weights_path))

    logger.info("Downloading Qwen3.6 Vision Projector (mmproj)...")
    vision_projector_path: Path = Path(
        hf_hub_download(
            repo_id=QWEN_GGUF_REPO_ID,
            filename=QWEN_VISION_PROJECTOR_FILENAME,
        )
    )
    logger.info("Vision projector downloaded", path=str(vision_projector_path))

    # ----------------------------------------------------------------------------------------------
    # Step 3 – Build llama-server command line (FULL GPU – RTX 6000 PRO Blackwell)
    # ----------------------------------------------------------------------------------------------
    # References for flag choices:
    #   - Thinking disable: https://unsloth.ai/docs/models/qwen3.6#how-to-enable-or-disable-thinking
    #   - llama-server example: https://unsloth.ai/docs/models/qwen3.6#llama-server-serving-and-openais-completion-library
    #   - HuggingFace model card: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
    command_line: list[str] = [
        str(LLAMA_SERVER_BINARY_PATH),
        "-m",        str(qwen_weights_path),
        "--mmproj",  str(vision_projector_path),
        "--port",    str(LLAMA_SERVER_PORT),
        "-c",        str(LLAMA_SERVER_CONTEXT_SIZE),
        "--n-gpu-layers", "99",                    # All layers on GPU
        "-t",        str(LLAMA_SERVER_CPU_THREADS),
        "--flash-attn", "on",                      # Flash Attention for Blackwell (explicit value required by newer llama.cpp)
        "--swa-full",                              # V2: fix SWA + cache-prompt repetition loop on Qwen3.6 MoE
        "--cache-type-k", "q8_0",                  # V2: quantized KV cache keys (~50% VRAM savings)
        "--cache-type-v", "q8_0",                  # V2: quantized KV cache values (negligible quality loss for OCR)
        "--cont-batching",                         # Continuous batching
        "--cache-prompt",                          # Reuse KV cache for system prompt
        "--batch-size", str(LLAMA_SERVER_BATCH_SIZE),   # Prefill batch size
        "--ubatch-size", str(LLAMA_SERVER_UBATCH_SIZE), # Micro-batch size
        "--image-max-tokens", "560",               # V3: cap vision tokens (aggressive — validate OCR quality)
        "--image-min-tokens", "70",                # V3: minimum vision tokens for small images
        "--chat-template-kwargs",                  # Disable thinking mode (official method)
        '{"enable_thinking":false}',               # from Unsloth docs + HuggingFace model card
    ]

    logger.info("Starting llama-server subprocess (full-GPU mode)", command=" ".join(command_line))

    # Write server stdout+stderr to a log file so failures can be diagnosed.
    # The handle is deliberately left open – the subprocess inherits it on fork.
    LLAMA_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file_handle = open(LLAMA_SERVER_LOG_PATH, "w", encoding="utf-8")
    llama_server_subprocess_handle = subprocess.Popen(
        command_line,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
    )

    # ----------------------------------------------------------------------------------------------
    # Step 4 – Wait for the server to become ready (poll /models until 200)
    # ----------------------------------------------------------------------------------------------
    logger.info(
        "Waiting for llama-server to become ready (see output/llama_server.log for details)",
        log_path=str(LLAMA_SERVER_LOG_PATH),
    )
    boot_start_time: float = time.monotonic()

    while True:
        # Detect crash during boot.
        if llama_server_subprocess_handle.poll() is not None:
            exit_code = llama_server_subprocess_handle.poll()
            logger.error(
                "llama-server subprocess exited unexpectedly during boot",
                exit_code=exit_code,
                elapsed=f"{time.monotonic() - boot_start_time:.1f}s",
            )
            sys.exit(1)

        # Abort if boot is taking unreasonably long.
        if time.monotonic() - boot_start_time > LLAMA_SERVER_BOOT_TIMEOUT_SEC:
            logger.error(
                "llama-server boot timed out",
                timeout_sec=LLAMA_SERVER_BOOT_TIMEOUT_SEC,
            )
            llama_server_subprocess_handle.terminate()
            sys.exit(1)

        # Check readiness.
        try:
            with httpx.Client(timeout=2.0) as poll_client:
                poll_response = poll_client.get(f"{llama_server_base_url}/models")
                if poll_response.status_code == 200:
                    elapsed: float = time.monotonic() - boot_start_time
                    logger.info(
                        "llama-server is up and ready",
                        elapsed=f"{elapsed:.1f}s",
                    )
                    return
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
            # Show progress every ~10 seconds so the user knows we're still waiting.
            elapsed_so_far = time.monotonic() - boot_start_time
            if int(elapsed_so_far) % 10 < int(LLAMA_SERVER_READY_POLL_SEC) + 1:
                logger.info(
                    "Waiting for llama-server to load model into VRAM...",
                    elapsed=f"{elapsed_so_far:.0f}s",
                    timeout=f"{LLAMA_SERVER_BOOT_TIMEOUT_SEC}s",
                )

        # Not ready yet – sleep before polling again.
        time.sleep(LLAMA_SERVER_READY_POLL_SEC)


# ==================================================================================================
# YOLO TensorRT Engine Management
# ==================================================================================================

def ensure_yolo_tensorrt_engine() -> Path:
    """
    Ensures a TensorRT FP16 engine exists for the custom fine-tuned YOLO OBB model.

    On first run, exports ``best.pt`` to a TensorRT ``.engine`` file optimised for
    the current GPU (sm_120 Blackwell). This is a one-time operation (~1-2 minutes).
    Subsequent runs reuse the cached engine.

    Returns:
        Path to the TensorRT ``.engine`` file ready for GPU inference.

    Raises:
        FileNotFoundError: If ``best.pt`` does not exist.
        RuntimeError: If the TensorRT export fails.
    """
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


# ==================================================================================================
# Image Preprocessing Utilities
# ==================================================================================================

def _resize_image_max_dimension(image: cv2.Mat, max_dim: int) -> cv2.Mat:
    """
    Resizes an image so its longest side equals max_dim, preserving aspect ratio.

    Args:
        image: Source image as OpenCV BGR Mat.
        max_dim: Target maximum dimension (width or height).

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ==================================================================================================
# Barcode Detection (optimized: downscaled input)
# ==================================================================================================

def detect_ean_barcode_and_orientation(
    image_bgr: cv2.Mat,
) -> tuple[str | None, int]:
    """
    Scans a single image for retail EAN/UPC barcodes using pyzbar.

    The input image is downscaled to PYZBAR_MAX_DIMENSION pixels before scanning
    to reduce computation while preserving full barcode detection accuracy.

    pyzbar returns one or more decoded barcode symbols, each tagged with a cryptic
    "orientation" string that indicates how the barcode is rotated relative to the
    image's natural reading direction.  This function extracts the first matching
    EAN/UPC symbol and converts that orientation label to a counter-clockwise rotation
    in degrees (0, 90, 180, or 270).

    Args:
        image_bgr: Input image as OpenCV BGR Mat (pre-loaded by caller to avoid
                   redundant disk I/O).

    Returns:
        A 2-tuple ``(ean_code, rotation_degrees)``:
        - ``ean_code`` – the decoded barcode value as a string, or ``None`` if no
          EAN/UPC barcode was found.
        - ``rotation_degrees`` – additional CCW rotation needed (0/90/180/270),
          defaulting to 0 when orientation cannot be determined.

    Raises:
        No explicit exceptions – errors are logged and cause the function to return
        ``(None, 0)`` so processing can continue even when barcode detection fails.
    """
    # Downscale for faster pyzbar scan (barcode metrics are preserved at this size).
    image_for_scan = _resize_image_max_dimension(image_bgr, PYZBAR_MAX_DIMENSION)

    try:
        # pyzbar requires RGB PIL image.
        pil_image: PILImage.Image = PILImage.fromarray(
            cv2.cvtColor(image_for_scan, cv2.COLOR_BGR2RGB)
        )
    except Exception as exc:
        logger.error(
            "Failed to convert image for barcode decoding",
            error=str(exc),
        )
        return None, 0

    try:
        decoded_symbols = decode_barcodes(pil_image)
    except Exception as exc:
        logger.error(
            "Failed to decode barcodes",
            error=str(exc),
        )
        return None, 0

    # Keep only symbols that encode a standard retail product code (EAN-13, EAN-8, UPC-A, UPC-E).
    ean_like_barcode_symbols = [
        symbol for symbol in decoded_symbols
        if symbol.type in EAN_BARCODE_TYPES
    ]

    if not ean_like_barcode_symbols:
        return None, 0

    # Select the first (highest-confidence) barcode symbol.
    selected_barcode_symbol = ean_like_barcode_symbols[0]
    decoded_barcode_value: str = selected_barcode_symbol.data.decode("utf-8")

    # Convert pyzbar's orientation string to a rotation amount.
    pyzbar_orientation_label: str | None = getattr(selected_barcode_symbol, "orientation", None)
    if pyzbar_orientation_label is not None:
        pyzbar_orientation_label = str(pyzbar_orientation_label)

    # Look up the required rotation; default to 0 if the orientation is unknown.
    additional_rotation_degrees: int = (
        PYZBAR_ORIENTATION_TO_ROTATION_DEGREES.get(pyzbar_orientation_label, 0)
    )

    return decoded_barcode_value, additional_rotation_degrees


# ==================================================================================================
# YOLO OBB Label Detection & Affine Cropping
# ==================================================================================================

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
    Crops a label region from an image using an oriented bounding box (OBB).

    The OBB gives the centre (xc, yc), size (w, h), and rotation (r) of the label
    in the image's coordinate system.  Because OpenCV's slice operator works only on
    axis-aligned rectangles, this function first rotates the *entire* image around
    the box centre by the negative of ``obb_rotation_radians`` so that the OBB becomes
    axis-aligned; a plain NumPy slice then yields the correctly oriented crop.

    Args:
        source_image_bgr: Source image as OpenCV BGR Mat (pre-loaded by caller).
        obb_center_x: OBB centroid x-coordinate (pixel units).
        obb_center_y: OBB centroid y-coordinate (pixel units).
        obb_width: OBB width along its local x-axis (pixel units).
        obb_height: OBB height along its local y-axis (pixel units).
        obb_rotation_radians: OBB rotation angle in radians (positive = CCW).
        additional_rotation_degrees: Fine-adjustment rotation from barcode orientation
                                     (0 / 90 / 180 / 270 degrees). Currently unused
                                     (kept for future extension).

    Returns:
        The cropped label as an OpenCV BGR image (height × width × 3).

    Raises:
        cv2.error: If the rotation/crop parameters are invalid.
    """
    image_height, image_width = source_image_bgr.shape[:2]
    bbox_rotation_degrees: float = math.degrees(obb_rotation_radians)

    # V2: Downscale image before warpAffine if it exceeds MAX_WARP_DIM.
    # The final crop will be resized to CROP_MAX_DIMENSION (800px) anyway,
    # so doing warpAffine on a smaller image saves time on large photos.
    MAX_WARP_DIM: int = 2048
    scale_factor: float = 1.0
    if max(image_height, image_width) > MAX_WARP_DIM:
        scale_factor = MAX_WARP_DIM / max(image_height, image_width)

    # V3: Try GPU path (cv2.cuda) for warpAffine, fall back to CPU if unavailable.
    # On RTX 6000 PRO, GPU warpAffine on a 2048px image takes ~2-5ms vs ~50ms on CPU.
    _use_cuda: bool = (
        hasattr(cv2, "cuda")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )

    if _use_cuda:
        # --- GPU path: upload once → resize + warpAffine in VRAM → download result ---
        gpu_mat = cv2.cuda.GpuMat()
        gpu_mat.upload(source_image_bgr)

        if scale_factor < 1.0:
            new_w = int(image_width * scale_factor)
            new_h = int(image_height * scale_factor)
            gpu_mat = cv2.cuda.resize(gpu_mat, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image_height, image_width = new_h, new_w
            obb_center_x *= scale_factor
            obb_center_y *= scale_factor
            obb_width *= scale_factor
            obb_height *= scale_factor

        deskew_rotation_matrix = cv2.getRotationMatrix2D(
            center=(int(obb_center_x), int(obb_center_y)),
            angle=bbox_rotation_degrees,
            scale=1.0,
        )
        gpu_mat = cv2.cuda.warpAffine(
            gpu_mat, deskew_rotation_matrix, (image_width, image_height),
        )
        # Download deskewed image back to CPU for numpy slice
        deskewed_image_bgr = gpu_mat.download()
    else:
        # --- CPU fallback (original V2 path) ---
        if scale_factor < 1.0:
            source_image_bgr = cv2.resize(
                source_image_bgr,
                (int(image_width * scale_factor), int(image_height * scale_factor)),
                interpolation=cv2.INTER_AREA,
            )
            image_height, image_width = source_image_bgr.shape[:2]
            obb_center_x *= scale_factor
            obb_center_y *= scale_factor
            obb_width *= scale_factor
            obb_height *= scale_factor

        # Build a 2-D affine rotation matrix that rotates the image around (xc, yc) by
        # bbox_rotation_degrees degrees.  cv2.getRotationMatrix2D uses a positive angle
        # for CCW rotation – which is the correct direction to "undo" a CCW tilt.
        deskew_rotation_matrix = cv2.getRotationMatrix2D(
            center=(int(obb_center_x), int(obb_center_y)),
            angle=bbox_rotation_degrees,
            scale=1.0,
        )

        # Apply the rotation to the whole image.
        deskewed_image_bgr = cv2.warpAffine(
            src=source_image_bgr,
            M=deskew_rotation_matrix,
            dsize=(image_width, image_height),
        )

    # After deskewing, the OBB is now axis-aligned, so a plain NumPy slice centred on
    # the same (xc, yc) yields the upright crop.
    x1 = int(obb_center_x - obb_width / 2)
    y1 = int(obb_center_y - obb_height / 2)
    x2 = x1 + int(obb_width)
    y2 = y1 + int(obb_height)

    # Guard against out-of-bounds slice corners (can happen at image edges).
    x1_clipped = max(0, x1)
    y1_clipped = max(0, y1)
    x2_clipped = min(image_width, x2)
    y2_clipped = min(image_height, y2)

    cropped_label_bgr: cv2.Mat = deskewed_image_bgr[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    # V3: Resize crop to CROP_MAX_DIMENSION here (single resize point).
    # Avoids a redundant second resize inside _transcribe_label_image.
    cropped_label_bgr = _resize_image_max_dimension(cropped_label_bgr, CROP_MAX_DIMENSION)

    return cropped_label_bgr


# ==================================================================================================
# OCR Transcription via llama-server (optimized: greedy + system prompt + in-memory base64)
# ==================================================================================================

def _transcribe_label_image(
    cropped_label_bgr: cv2.Mat,
    llama_server_base_url: str,
) -> str:
    """
    Sends a cropped label image to llama-server (Qwen3.6 vision model) for OCR.

    Optimizations:
    - In-memory base64 encoding (no disk roundtrip)
    - Crop resized to CROP_MAX_DIMENSION before encoding (smaller = faster prefill)
    - System prompt for KV cache hit (fixed prompt reused for all images)
    - Greedy sampling (temperature=0.0, top_k=1) for deterministic OCR output
    - Persistent httpx.Client for connection reuse

    Args:
        cropped_label_bgr: Cropped label as OpenCV BGR Mat (pre-cropped by caller).
        llama_server_base_url: Base URL of the llama-server OpenAI-compatible API.

    Returns:
        The transcribed text as a plain string (markdown-style line breaks are preserved).

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    # Crop is already resized to CROP_MAX_DIMENSION inside _deskew_crop_obb.
    # In-memory JPEG encoding → base64 (no disk I/O).
    ok, buf = cv2.imencode('.jpg', cropped_label_bgr, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed for cropped label")
    base64_encoded_crop: str = base64.b64encode(buf).decode("utf-8")

    # Build the OpenAI-compatible chat-completion payload.
    # System prompt is cached in KV; user prompt is minimal.
    chat_completion_payload: dict = {
        "model": QWEN_MODEL_API_NAME,
        "messages": [
            {
                "role": "system",
                "content": TRANSCRIPTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": TRANSCRIPTION_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_crop}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,                       # Greedy: deterministic, fastest
        "max_tokens": TRANSCRIPTION_MAX_OUTPUT_TOKENS,
        "top_k": 1,                               # Greedy: no sampling overhead
        "presence_penalty": 1.5,                   # V2: official Qwen3.6 anti-repetition (from model card)
        "stop": ["<|im_end|>"],                   # Explicit stop token
    }

    chat_url: str = f"{llama_server_base_url}/chat/completions"
    last_error: Exception | None = None

    for attempt in range(1, TRANSCRIPTION_MAX_RETRIES + 1):
        try:
            client = _get_http_client()
            response = client.post(chat_url, json=chat_completion_payload)
            response.raise_for_status()
            chat_completion_response_json: dict = response.json()

            transcribed_text: str = str(
                chat_completion_response_json["choices"][0]["message"]["content"]
            )
            return transcribed_text

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Transcription API call failed – retrying",
                attempt=attempt,
                max_retries=TRANSCRIPTION_MAX_RETRIES,
                error=str(exc),
            )
            if attempt < TRANSCRIPTION_MAX_RETRIES:
                # Exponential back-off: 2 s, 4 s between retries.
                time.sleep(2 ** attempt)

    # All retries exhausted – propagate as a RuntimeError so the caller can record the failure.
    raise RuntimeError(
        f"Transcription API failed after {TRANSCRIPTION_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# ==================================================================================================
# Per-Image Pipeline (with per-step timing, optimized I/O)
# ==================================================================================================

def process_single_label_image(
    input_image_path: Path,
    yolo_obb_model: YOLO,
    llama_server_base_url: str,
    cropped_labels_output_dir: Path,
    preloaded_image_bgr: cv2.Mat | None = None,
) -> dict:
    """
    Executes the full label-extraction pipeline for a single input image,
    measuring elapsed time for each step.

    Optimizations vs original:
    - Single image load: cv2.imread once, reuse numpy array for all stages
    - In-memory base64: cv2.imencode + base64.b64encode (no disk roundtrip for crop)
    - Optional disk write of crop only if SAVE_CROPS=true
    - Barcode on downscaled image (PYZBAR_MAX_DIMENSION px)
    - YOLO with explicit imgsz=YOLO_IMG_SIZE matching engine export

    Stages (in order):
        1. Image load (once) — cv2.imread
        2. Barcode detection (EAN code + orientation) — optional via ENABLE_EAN_DETECTION
        3. YOLO OBB label detection (TensorRT FP16 GPU)
        4. Affine deskew + axis-aligned crop
        5. OCR transcription via llama-server (Qwen3.6 vision model, greedy sampling)
        6. Optional: save cropped label JPEG to disk (only if SAVE_CROPS=true)

    Args:
        input_image_path: Path to the input photograph.
        yolo_obb_model: Pre-loaded YOLO OBB model (TensorRT engine).
        llama_server_base_url: Base URL of the llama-server OpenAI-compatible API.
        cropped_labels_output_dir: Directory where cropped label JPEGs are written
                                   (only used if SAVE_CROPS=true).
        preloaded_image_bgr: V2: pre-loaded image from async prefetch thread.
                             If None, falls back to cv2.imread.

    Returns:
        A dict with keys: name, ean, rotation, text, timing.
    """
    logger.info("Processing image", image=input_image_path.name)
    t_image_start: float = time.perf_counter()

    # ----------------------------------------------------------------------------------------------
    # Stage 0 – Load image ONCE (shared across all stages)
    # V2: use pre-loaded image from async prefetch if available.
    # ----------------------------------------------------------------------------------------------
    t_load_start: float = time.perf_counter()
    if preloaded_image_bgr is not None:
        source_image_bgr = preloaded_image_bgr
    else:
        source_image_bgr = cv2.imread(str(input_image_path))
    if source_image_bgr is None:
        raise cv2.error(f"cv2.imread returned None for: {input_image_path}")
    t_load_ms: float = (time.perf_counter() - t_load_start) * 1000

    # ----------------------------------------------------------------------------------------------
    # Stage 1 – Barcode detection (optional, on downscaled image)
    # ----------------------------------------------------------------------------------------------
    t_barcode_start: float = time.perf_counter()

    if ENABLE_EAN_DETECTION:
        ean_code, additional_rotation_degrees = detect_ean_barcode_and_orientation(
            source_image_bgr
        )
    else:
        ean_code = None
        additional_rotation_degrees = 0

    t_barcode_ms: float = (time.perf_counter() - t_barcode_start) * 1000

    # ----------------------------------------------------------------------------------------------
    # Stage 2 – YOLO OBB label detection (TensorRT GPU, numpy array input)
    # ----------------------------------------------------------------------------------------------
    t_yolo_start: float = time.perf_counter()

    try:
        # Pass numpy array directly (avoids redundant disk read inside predict).
        # imgsz=YOLO_IMG_SIZE must match the TensorRT engine export size.
        yolo_results = yolo_obb_model.predict(
            source=source_image_bgr,
            imgsz=YOLO_IMG_SIZE,
            device=0,         # GPU inference via TensorRT
            verbose=False,
        )
        first_result = yolo_results[0]
    except Exception as exc:
        logger.error(
            "YOLO OBB inference failed",
            image=input_image_path.name,
            error=str(exc),
        )
        t_yolo_ms = (time.perf_counter() - t_yolo_start) * 1000
        return {
            "name": input_image_path.name,
            "ean": ean_code,
            "rotation": additional_rotation_degrees,
            "text": f"YOLO inference error: {exc}",
            "timing": {
                "load_ms": round(t_load_ms, 1),
                "barcode_ms": round(t_barcode_ms, 1),
                "yolo_ms": round(t_yolo_ms, 1),
                "crop_ms": 0.0,
                "ocr_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_image_start) * 1000, 1),
            },
        }

    t_yolo_ms: float = (time.perf_counter() - t_yolo_start) * 1000

    if not first_result.obb or len(first_result.obb) == 0:
        logger.warning("No label detected by YOLO OBB", image=input_image_path.name)
        return {
            "name": input_image_path.name,
            "ean": ean_code,
            "rotation": additional_rotation_degrees,
            "text": "No label detected by YOLO OBB.",
            "timing": {
                "load_ms": round(t_load_ms, 1),
                "barcode_ms": round(t_barcode_ms, 1),
                "yolo_ms": round(t_yolo_ms, 1),
                "crop_ms": 0.0,
                "ocr_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_image_start) * 1000, 1),
            },
        }

    # ----------------------------------------------------------------------------------------------
    # Stage 3 – Deskew + crop (reuses source_image_bgr from Stage 0)
    # ----------------------------------------------------------------------------------------------
    t_crop_start: float = time.perf_counter()

    # OBB tensor shape: (1, N, 6) where columns are [xcenter, ycenter, width, height, rotation, conf]
    # We extract the single detection (index 0) and unpack all five OBB parameters.
    (
        bbox_center_x,
        bbox_center_y,
        bbox_width,
        bbox_height,
        bbox_rotation_radians,
    ) = first_result.obb.xywhr[0].cpu().numpy().tolist()

    try:
        cropped_label_bgr: cv2.Mat = _deskew_crop_obb(
            source_image_bgr=source_image_bgr,
            obb_center_x=bbox_center_x,
            obb_center_y=bbox_center_y,
            obb_width=bbox_width,
            obb_height=bbox_height,
            obb_rotation_radians=bbox_rotation_radians,
            additional_rotation_degrees=additional_rotation_degrees,
        )
    except cv2.error as exc:
        logger.error(
            "Affine deskew + crop failed",
            image=input_image_path.name,
            error=str(exc),
        )
        t_crop_ms = (time.perf_counter() - t_crop_start) * 1000
        return {
            "name": input_image_path.name,
            "ean": ean_code,
            "rotation": additional_rotation_degrees,
            "text": f"Image crop failed: {exc}",
            "timing": {
                "load_ms": round(t_load_ms, 1),
                "barcode_ms": round(t_barcode_ms, 1),
                "yolo_ms": round(t_yolo_ms, 1),
                "crop_ms": round(t_crop_ms, 1),
                "ocr_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_image_start) * 1000, 1),
            },
        }

    # Write the cropped label to disk ONLY if SAVE_CROPS=true (debug mode).
    # In production, we skip disk I/O and encode in memory for base64.
    if SAVE_CROPS:
        cropped_label_output_path: Path = cropped_labels_output_dir / f"crop_{input_image_path.name}"
        write_success: bool = cv2.imwrite(str(cropped_label_output_path), cropped_label_bgr)
        if not write_success:
            logger.warning(
                "cv2.imwrite returned False – file may be corrupted",
                crop_path=str(cropped_label_output_path),
            )

    t_crop_ms: float = (time.perf_counter() - t_crop_start) * 1000

    # ----------------------------------------------------------------------------------------------
    # Stage 4 – OCR transcription (in-memory, no disk roundtrip)
    # ----------------------------------------------------------------------------------------------
    t_ocr_start: float = time.perf_counter()

    try:
        transcription_markdown: str = _transcribe_label_image(
            cropped_label_bgr, llama_server_base_url
        )
        logger.info(
            "Label transcribed successfully",
            image=input_image_path.name,
        )
    except RuntimeError as exc:
        logger.error(
            "Transcription failed after all retries",
            image=input_image_path.name,
            error=str(exc),
        )
        t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000
        return {
            "name": input_image_path.name,
            "ean": ean_code,
            "rotation": additional_rotation_degrees,
            "text": f"API Error: {exc}",
            "timing": {
                "load_ms": round(t_load_ms, 1),
                "barcode_ms": round(t_barcode_ms, 1),
                "yolo_ms": round(t_yolo_ms, 1),
                "crop_ms": round(t_crop_ms, 1),
                "ocr_ms": round(t_ocr_ms, 1),
                "total_ms": round((time.perf_counter() - t_image_start) * 1000, 1),
            },
        }

    t_ocr_ms: float = (time.perf_counter() - t_ocr_start) * 1000
    t_total_ms: float = (time.perf_counter() - t_image_start) * 1000

    return {
        "name": input_image_path.name,
        "ean": ean_code,
        "rotation": additional_rotation_degrees,
        "text": transcription_markdown,
        "timing": {
            "load_ms": round(t_load_ms, 1),
            "barcode_ms": round(t_barcode_ms, 1),
            "yolo_ms": round(t_yolo_ms, 1),
            "crop_ms": round(t_crop_ms, 1),
            "ocr_ms": round(t_ocr_ms, 1),
            "total_ms": round(t_total_ms, 1),
        },
    }


# ==================================================================================================
# Batch Report Generation (with timing)
# ==================================================================================================

def write_batch_report(
    batch_report_path: Path,
    input_images_dir: Path,
    batch_processing_results: list[dict],
    pipeline_elapsed_sec: float,
) -> None:
    """
    Writes a human-readable markdown report summarising the batch run,
    including per-step timing statistics.

    Args:
        batch_report_path: Path where the markdown report should be written.
        input_images_dir: Name of the input folder (used in the report header).
        batch_processing_results: List of result dicts (one per processed image).
        pipeline_elapsed_sec: Total wall-clock time for the full pipeline.
    """
    try:
        with open(batch_report_path, "w", encoding="utf-8") as f:
            # ---- Header --------------------------------------------------------------------------
            f.write("# MOCR Batch Processing Report (Full-GPU)\n\n")
            f.write(f"**Folder:** `{input_images_dir.name}`  \n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**GGUF Model:** `{QWEN_GGUF_WEIGHTS_FILENAME}`  \n")
            f.write(f"**EAN Detection:** {'enabled' if ENABLE_EAN_DETECTION else 'disabled'}  \n")
            f.write(f"**SAVE_CROPS:** {'true' if SAVE_CROPS else 'false'}  \n")
            f.write(f"**Sampling:** greedy (temperature=0.0, top_k=1)  \n")
            f.write(f"**Crop max dimension:** {CROP_MAX_DIMENSION}px  \n")
            f.write(f"**YOLO imgsz:** {YOLO_IMG_SIZE}px  \n\n")

            # ---- Summary table ------------------------------------------------------------------
            f.write("## Summary\n\n")
            f.write("| # | Image | EAN | Status | Total (ms) |\n")
            f.write("|---|-------|-----|--------|------------|\n")

            success_count: int = 0
            for idx, result in enumerate(batch_processing_results, start=1):
                ean_val: str = result["ean"] or "—"
                text: str = result["text"]
                is_error: bool = (
                    text.startswith("API Error")
                    or text.startswith("YOLO inference error")
                    or text.startswith("No label detected")
                    or text.startswith("Image crop failed")
                )
                status_icon: str = "✗" if is_error else "✓"
                if not is_error:
                    success_count += 1
                total_ms_str: str = str(result["timing"]["total_ms"])
                f.write(
                    f"| {idx} | {result['name']} | {ean_val} | {status_icon} | {total_ms_str} |\n"
                )

            f.write(
                f"\n**Stats:** {success_count}/{len(batch_processing_results)} succeeded.  \n"
            )
            f.write(f"**Pipeline total:** {pipeline_elapsed_sec:.1f}s  \n")
            n = len(batch_processing_results)
            if n > 0:
                avg_ms = pipeline_elapsed_sec * 1000 / n
                f.write(f"**Average per image (wall-clock):** {avg_ms:.0f} ms  \n")
            if success_count > 0:
                # Average of only successful inferences (excludes failed images)
                success_total_ms = sum(
                    r["timing"]["total_ms"] for r in batch_processing_results
                    if not (
                        r["text"].startswith("API Error")
                        or r["text"].startswith("YOLO inference error")
                        or r["text"].startswith("No label detected")
                        or r["text"].startswith("Image crop failed")
                    )
                )
                f.write(f"**Average per image (successful only):** {success_total_ms / success_count:.0f} ms  \n")
            f.write("\n---\n\n")

            # ---- Timing Summary -----------------------------------------------------------------
            f.write("## Timing Summary\n\n")

            # Collect timing arrays — only from SUCCESSFUL results so failed images
            # (with 0.0ms crop/ocr) don't pollute min/avg calculations.
            timing_keys = ["load_ms", "barcode_ms", "yolo_ms", "crop_ms", "ocr_ms", "total_ms"]
            timing_labels = ["Load", "Barcode", "YOLO OBB", "Crop", "OCR", "**Total**"]
            timing_data: dict[str, list[float]] = {k: [] for k in timing_keys}

            for result in batch_processing_results:
                text = result["text"]
                is_error = (
                    text.startswith("API Error")
                    or text.startswith("YOLO inference error")
                    or text.startswith("No label detected")
                    or text.startswith("Image crop failed")
                )
                if is_error:
                    continue  # Skip failed results in timing stats
                for k in timing_keys:
                    timing_data[k].append(result["timing"][k])

            f.write("| Step | Avg (ms) | Min (ms) | Max (ms) |\n")
            f.write("|------|----------|----------|----------|\n")

            for key, label in zip(timing_keys, timing_labels):
                values = timing_data[key]
                if values:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    f.write(
                        f"| {label} | {avg_val:.1f} | {min_val:.1f} | {max_val:.1f} |\n"
                    )
                else:
                    f.write(f"| {label} | — | — | — |\n")

            f.write("\n---\n\n")

            # ---- Per-image detailed results ----------------------------------------------------
            f.write("## Detailed Results\n\n")
            for idx, result in enumerate(batch_processing_results, start=1):
                t = result["timing"]
                f.write(f"### {idx}. {result['name']}\n")
                f.write(f"- **EAN:** {result['ean'] or 'None'}\n")
                f.write(f"- **Rotation Needed:** {result['rotation']}°\n")
                f.write(
                    f"- **Timing:** load={t['load_ms']}ms, barcode={t['barcode_ms']}ms, "
                    f"yolo={t['yolo_ms']}ms, crop={t['crop_ms']}ms, "
                    f"ocr={t['ocr_ms']}ms, **total={t['total_ms']}ms**\n"
                )
                f.write("\n**Transcription:**\n")
                f.write(f"```markdown\n{result['text']}\n```\n\n")

        logger.info("Batch report written", report_path=str(batch_report_path))

    except IOError as exc:
        logger.error(
            "Failed to write batch report",
            report_path=str(batch_report_path),
            error=str(exc),
        )


# ==================================================================================================
# Main Entry Point
# ==================================================================================================

def main() -> None:
    """
    Procedural entry point for the Tosano full-GPU food-label batch processor.

    Responsibilities:
        1. Locate all image files in the input directory.
        2. Export/load the YOLO OBB model as TensorRT FP16 engine.
        3. Ensure llama-server is running (downloading Qwen3.6 weights if necessary).
        4. Process each image through the full pipeline (with per-step timing).
        5. Save per-image transcripts (crops only if SAVE_CROPS=true).
        6. Write a summary markdown report with timing statistics.
    """
    import_elapsed_sec: float = time.perf_counter() - _IMPORT_START_MONOTONIC
    logger.info(
        "Starting full-GPU pipeline",
        gguf_model=QWEN_GGUF_WEIGHTS_FILENAME,
        ean_detection=ENABLE_EAN_DETECTION,
        save_crops=SAVE_CROPS,
        import_time=f"{import_elapsed_sec:.1f}s",
    )

    # ----------------------------------------------------------------------------------------------
    # Directory setup
    # ----------------------------------------------------------------------------------------------
    input_images_dir: Path = Path("etichette_esempio")
    output_root_dir: Path = Path("output_etichette_esempio")
    cropped_labels_dir: Path = output_root_dir / "crops"
    batch_report_path: Path = output_root_dir / "mocr_batch_results.md"

    for directory in (input_images_dir, output_root_dir, cropped_labels_dir):
        directory.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------------------------
    # Collect input images
    # ----------------------------------------------------------------------------------------------
    input_image_paths: list[Path] = sorted(
        file_path
        for file_path in input_images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not input_image_paths:
        logger.warning(
            "No image files found to process",
            input_folder=str(input_images_dir),
            supported_extensions=SUPPORTED_IMAGE_EXTENSIONS,
        )
        return

    # ----------------------------------------------------------------------------------------------
    # Step 1/3 — Load YOLO OBB model (TensorRT FP16 — exports from best.pt on first run)
    # ----------------------------------------------------------------------------------------------
    logger.info("[Step 1/3] Preparing YOLO OBB model (TensorRT FP16)...")
    try:
        yolo_start = time.perf_counter()
        tensorrt_engine_path: Path = ensure_yolo_tensorrt_engine()
        logger.info(
            "[Step 1/3] Loading YOLO OBB engine into GPU",
            model_path=str(tensorrt_engine_path),
        )
        yolo_obb_model: YOLO = YOLO(str(tensorrt_engine_path), task="obb")
        yolo_elapsed = time.perf_counter() - yolo_start
        logger.info("[Step 1/3] YOLO OBB ready ✓", elapsed=f"{yolo_elapsed:.1f}s")
    except Exception as exc:
        logger.error(
            "Failed to load YOLO OBB model",
            error=str(exc),
        )
        return

    # ----------------------------------------------------------------------------------------------
    # Step 2/3 — Ensure llama-server is running (auto-download + boot if missing)
    # ----------------------------------------------------------------------------------------------
    logger.info("[Step 2/3] Ensuring llama-server is running (download weights + boot)...")
    ensure_llama_server_running(LLAMA_SERVER_BASE_URL)
    logger.info("[Step 2/3] llama-server ready ✓", url=LLAMA_SERVER_BASE_URL)

    # ----------------------------------------------------------------------------------------------
    # Warmup — YOLO TensorRT kernel compilation with a representative image size.
    #
    # TensorRT JIT-compiles CUDA kernels on the first inference for the current GPU architecture.
    # We use a realistic image size (640×640) to avoid recompilation on first real image.
    # NOTE: llama-server performs an automatic warmup pass during boot — no extra action needed.
    # ----------------------------------------------------------------------------------------------
    logger.info("[Warmup] Running YOLO TensorRT warmup pass (640×640)…")
    warmup_start = time.perf_counter()

    try:
        # Use a realistic warmup size matching YOLO_IMG_SIZE.
        dummy_image: np.ndarray = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        yolo_obb_model.predict(source=dummy_image, imgsz=YOLO_IMG_SIZE, device=0, verbose=False)
        logger.info("[Warmup] YOLO TensorRT kernels compiled ✓")
    except Exception as exc:
        logger.warning(
            "[Warmup] YOLO warmup failed (non-fatal — first image may be slower)",
            error=str(exc),
        )

    warmup_elapsed = time.perf_counter() - warmup_start
    logger.info("[Warmup] All models warm and ready ✓", elapsed=f"{warmup_elapsed:.1f}s")

    # ----------------------------------------------------------------------------------------------
    # Step 3/3 — Process each image (with timing)
    # ----------------------------------------------------------------------------------------------
    logger.info(
        "[Step 3/3] Starting batch processing",
        total_images=len(input_image_paths),
        report_path=str(batch_report_path),
    )

    pipeline_start_time: float = time.perf_counter()
    batch_processing_results: list[dict] = []

    # V2: Async image pre-loading — eliminates RunPod network volume cold-cache I/O.
    # While the GPU processes image N (OCR ~2s), we pre-load image N+1 from disk
    # in a background thread, triggering the OS page cache.
    PREFETCH_AHEAD: int = 2  # Pre-load 2 images ahead

    def _prefetch_image(path: Path) -> cv2.Mat | None:
        """Pre-load image from disk in background thread."""
        return cv2.imread(str(path))

    with ThreadPoolExecutor(max_workers=PREFETCH_AHEAD, thread_name_prefix="img-prefetch") as prefetch_pool:
        # Submit initial prefetch jobs
        prefetch_futures: dict[int, Future] = {}
        for ahead_idx in range(min(PREFETCH_AHEAD, len(input_image_paths))):
            prefetch_futures[ahead_idx] = prefetch_pool.submit(
                _prefetch_image, input_image_paths[ahead_idx]
            )

        for image_index, input_image_path in enumerate(input_image_paths, start=1):
            logger.info("Processing image", current=image_index, total=len(input_image_paths))

            # Retrieve pre-loaded image (or None if prefetch didn't cover it)
            zero_idx: int = image_index - 1
            preloaded: cv2.Mat | None = None
            if zero_idx in prefetch_futures:
                try:
                    preloaded = prefetch_futures.pop(zero_idx).result(timeout=30.0)
                except Exception:
                    preloaded = None  # Fallback to synchronous load

            # Submit next prefetch job
            next_idx: int = zero_idx + PREFETCH_AHEAD
            if next_idx < len(input_image_paths):
                prefetch_futures[next_idx] = prefetch_pool.submit(
                    _prefetch_image, input_image_paths[next_idx]
                )

            result: dict = process_single_label_image(
                input_image_path=input_image_path,
                yolo_obb_model=yolo_obb_model,
                llama_server_base_url=LLAMA_SERVER_BASE_URL,
                cropped_labels_output_dir=cropped_labels_dir,
                preloaded_image_bgr=preloaded,
            )

            batch_processing_results.append(result)

            # Log timing for this image
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

            # -----------------------------------------------------------------------------------------
            # Write per-image transcript async (I/O doesn't block the hot loop)
            # -----------------------------------------------------------------------------------------
            def _write_transcript(
                path: Path, img_name: str, result_dict: dict, timing: dict
            ) -> None:
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(f"# Transcription for {img_name}\n\n")
                        fh.write(f"- **EAN:** {result_dict['ean'] or 'None'}\n")
                        fh.write(f"- **Rotation Detected:** {result_dict['rotation']}°\n")
                        fh.write(
                            f"- **Timing:** {timing['total_ms']}ms "
                            f"(load={timing['load_ms']}ms, barcode={timing['barcode_ms']}ms, "
                            f"yolo={timing['yolo_ms']}ms, crop={timing['crop_ms']}ms, "
                            f"ocr={timing['ocr_ms']}ms)\n\n"
                        )
                        fh.write(f"```markdown\n{result_dict['text']}\n```\n")
                except Exception as exc:
                    logger.error(
                        "Failed to write individual transcript",
                        transcript_path=str(path),
                        error=str(exc),
                    )

            individual_transcript_path: Path = (
                cropped_labels_dir / f"transcript_{input_image_path.stem}.md"
            )
            prefetch_pool.submit(
                _write_transcript, individual_transcript_path,
                input_image_path.name, result, t,
            )

    # ----------------------------------------------------------------------------------------------
    # Write the consolidated batch report
    # ----------------------------------------------------------------------------------------------
    pipeline_elapsed_sec: float = time.perf_counter() - pipeline_start_time

    write_batch_report(
        batch_report_path=batch_report_path,
        input_images_dir=input_images_dir,
        batch_processing_results=batch_processing_results,
        pipeline_elapsed_sec=pipeline_elapsed_sec,
    )

    logger.info(
        "Pipeline complete",
        total_images=len(batch_processing_results),
        elapsed_sec=round(pipeline_elapsed_sec, 1),
        avg_ms_per_image=round(pipeline_elapsed_sec * 1000 / max(1, len(batch_processing_results))),
    )


if __name__ == "__main__":
    main()