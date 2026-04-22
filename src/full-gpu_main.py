#!/usr/bin/env python3
"""
Full-GPU batch processor for Tosano supermarket food-label pipeline.

This is the production-optimized variant of ``main.py``, designed to maximise
throughput on an NVIDIA RTX 6000 PRO Blackwell (96 GB VRAM).

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

Optimizations vs main.py:
    - Removed --cpu-moe: all 256 MoE experts run in GPU VRAM
    - Disabled thinking mode: --chat-template-kwargs '{"enable_thinking":false}'
      (official method from https://unsloth.ai/docs/models/qwen3.6)
    - Flash Attention: --flash-attn for Blackwell tensor cores
    - Continuous batching: --cont-batching
    - Prompt caching: --cache-prompt (reuse system prompt KV cache)
    - YOLO OBB: TensorRT FP16 from custom fine-tuned best.pt (not ONNX CPU)
    - Sampling: official non-thinking mode parameters from HuggingFace model card
    - max_tokens: reduced from 8192 to 2048 (food labels never exceed ~1500 tokens)
    - Per-step timing in batch report
    - EAN detection: optional via ENABLE_EAN_DETECTION env var

References:
    - https://unsloth.ai/docs/models/qwen3.6
    - https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
    - https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf

WARNING: Do NOT use CUDA 13.2 — known bug causes gibberish outputs.
         Use CUDA 12.8 or 13.0 (see Unsloth docs).
"""

# --------------------------------------------------------------------------------------------------
# Stdlib imports
# --------------------------------------------------------------------------------------------------
from pathlib import Path
from datetime import datetime
import os
import subprocess
import atexit
import sys
import time
import base64
import math

# --------------------------------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------------------------------
import structlog
import httpx
import cv2                                     # OpenCV – used for affine deskew + crop
from PIL import Image as PILImage              # Pillow – used by pyzbar for barcode decoding
from pyzbar.pyzbar import decode as decode_barcodes
from ultralytics import YOLO
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------------------
# Load .env file (if present) — must happen before any os.environ.get() calls below.
# Variables set via shell export take precedence over .env values.
# --------------------------------------------------------------------------------------------------
load_dotenv()

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
QWEN_GGUF_WEIGHTS_FILENAME: str = os.environ.get(
    "QWEN_GGUF_FILE",
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
)
QWEN_VISION_PROJECTOR_FILENAME: str = "mmproj-F16.gguf"
QWEN_MODEL_API_NAME: str = "Qwen3.6"          # Value used in the chat-completion "model" field

# --- llama-server runtime settings ---
LLAMA_SERVER_PORT: int = 8080
LLAMA_SERVER_BASE_URL: str = f"http://localhost:{LLAMA_SERVER_PORT}/v1"
LLAMA_SERVER_CONTEXT_SIZE: int = 16384         # As recommended by Unsloth llama-server tutorial
LLAMA_SERVER_CPU_THREADS: int = 4              # With full GPU offload, fewer CPU threads needed
LLAMA_SERVER_SHUTDOWN_TIMEOUT_SEC: float = 10.0
LLAMA_SERVER_READY_POLL_SEC: float = 5.0       # Sleep between readiness probes during boot
LLAMA_SERVER_BOOT_TIMEOUT_SEC: float = 600.0   # 10 min: includes first-time GGUF download (~22 GB)

# --- Transcription API settings ---
TRANSCRIPTION_HTTP_TIMEOUT_SEC: float = 120.0
TRANSCRIPTION_MAX_RETRIES: int = 3
TRANSCRIPTION_MAX_OUTPUT_TOKENS: int = 2048    # Food labels never exceed ~1500 tokens
TRANSCRIPTION_PROMPT: str = (
    "Read all the text in this image. Output only the text directly. "
    "Do not explain or add comments."
)

# --- EAN detection toggle ---
# Set ENABLE_EAN_DETECTION=false to skip barcode search (saves ~5-10 ms per image)
ENABLE_EAN_DETECTION: bool = os.environ.get(
    "ENABLE_EAN_DETECTION", "true"
).lower() == "true"

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

# --------------------------------------------------------------------------------------------------
# Global state – shared across function calls without passing as arguments.
# Initialised once in main() and cleaned up at interpreter shutdown.
# --------------------------------------------------------------------------------------------------

# Handle to the llama-server subprocess started on demand.
# None when no subprocess was spawned (server was already running or not needed).
llama_server_subprocess_handle: subprocess.Popen | None = None

# --------------------------------------------------------------------------------------------------
# Module logger (structured via structlog)
# --------------------------------------------------------------------------------------------------
logger: structlog.stdlib.BoundLogger = structlog.get_logger()


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
    global llama_server_subprocess_handle
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
        "--flash-attn",                            # Flash Attention for Blackwell
        "--cont-batching",                         # Continuous batching
        "--cache-prompt",                          # Reuse KV cache for system prompt
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
            if int(elapsed_so_far) % 10 < LLAMA_SERVER_READY_POLL_SEC + 0.5:
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
# Barcode Detection
# ==================================================================================================

def detect_ean_barcode_and_orientation(
    image_path: Path,
) -> tuple[str | None, int]:
    """
    Scans a single image for retail EAN/UPC barcodes using pyzbar.

    pyzbar returns one or more decoded barcode symbols, each tagged with a cryptic
    "orientation" string that indicates how the barcode is rotated relative to the
    image's natural reading direction.  This function extracts the first matching
    EAN/UPC symbol and converts that orientation label to a counter-clockwise rotation
    in degrees (0, 90, 180, or 270).

    Args:
        image_path: Absolute or relative path to the input image file.

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
    try:
        pil_image: PILImage.Image = PILImage.open(image_path)
    except Exception as exc:
        logger.error(
            "Failed to open image for barcode decoding",
            file=str(image_path),
            error=str(exc),
        )
        return None, 0

    try:
        decoded_symbols = decode_barcodes(pil_image)
    except Exception as exc:
        logger.error(
            "Failed to decode barcodes in image",
            file=str(image_path),
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
    source_image_path: Path,
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

    After the deskew crop, an additional "fine" rotation from pyzbar's barcode
    orientation is applied when the label itself is physically upside-down or
    sideways relative to the reading direction.

    Args:
        source_image_path: Path to the source image on disk.
        obb_center_x: OBB centroid x-coordinate (pixel units).
        obb_center_y: OBB centroid y-coordinate (pixel units).
        obb_width: OBB width along its local x-axis (pixel units).
        obb_height: OBB height along its local y-axis (pixel units).
        obb_rotation_radians: OBB rotation angle in radians (positive = CCW).
        additional_rotation_degrees: Fine-adjustment rotation from barcode orientation
                                     (0 / 90 / 180 / 270 degrees).

    Returns:
        The cropped label as an OpenCV BGR image (height × width × 3).

    Raises:
        cv2.error: If the image cannot be read or the rotation/crop parameters are invalid.
    """
    source_image_bgr: cv2.Mat = cv2.imread(str(source_image_path))
    if source_image_bgr is None:
        raise cv2.error(f"cv2.imread returned None for: {source_image_path}")

    image_height, image_width = source_image_bgr.shape[:2]
    bbox_rotation_degrees: float = math.degrees(obb_rotation_radians)

    # Build a 2-D affine rotation matrix that rotates the image around (xc, yc) by
    # bbox_rotation_degrees degrees.  cv2.getRotationMatrix2D uses a positive angle
    # for CCW rotation – which is the correct direction to "undo" a CCW tilt.
    deskew_rotation_matrix: cv2.Mat = cv2.getRotationMatrix2D(
        center=(int(obb_center_x), int(obb_center_y)),
        angle=bbox_rotation_degrees,
        scale=1.0,
    )

    # Apply the rotation to the whole image.  The output size equals the input size;
    # pixels that fall outside are clipped silently, which is safe here because the
    # label region is guaranteed to be well inside the image bounds.
    deskewed_image_bgr: cv2.Mat = cv2.warpAffine(
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
    
    '''
    # Apply the additional fine-rotation derived from the barcode's physical orientation.
    # This handles the case where the label was photographed physically rotated (e.g. printed
    # upside-down on the package) even after the OBB deskew has corrected the camera tilt.
    if additional_rotation_degrees == 90:
        cropped_label_bgr = cv2.rotate(cropped_label_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif additional_rotation_degrees == 180:
        cropped_label_bgr = cv2.rotate(cropped_label_bgr, cv2.ROTATE_180)
    elif additional_rotation_degrees == 270:
        cropped_label_bgr = cv2.rotate(cropped_label_bgr, cv2.ROTATE_90_CLOCKWISE)
    '''

    return cropped_label_bgr


# ==================================================================================================
# OCR Transcription via llama-server
# ==================================================================================================

def _transcribe_label_image(
    cropped_label_path: Path,
    llama_server_base_url: str,
) -> str:
    """
    Sends a cropped label image to llama-server (Qwen3.6 vision model) for OCR.

    Uses the official non-thinking mode sampling parameters from the HuggingFace
    model card (Instruct mode for general tasks):
        temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5

    Args:
        cropped_label_path: Path to the cropped label JPEG on disk.
        llama_server_base_url: Base URL of the llama-server OpenAI-compatible API.

    Returns:
        The transcribed text as a plain string (markdown-style line breaks are preserved).

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    with open(cropped_label_path, "rb") as image_file_handle:
        base64_encoded_crop: str = base64.b64encode(image_file_handle.read()).decode("utf-8")

    # Build the OpenAI-compatible chat-completion payload.
    # Sampling parameters from official HuggingFace model card:
    # "Instruct (non-thinking) mode for general tasks:
    #  temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5"
    chat_completion_payload: dict = {
        "model": QWEN_MODEL_API_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": TRANSCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_crop}"},
                    },
                ],
            }
        ],
        "temperature": 0.7,                        # Official non-thinking mode param
        "max_tokens": TRANSCRIPTION_MAX_OUTPUT_TOKENS,
        "top_p": 0.8,                              # Official non-thinking mode param
        "presence_penalty": 1.5,                   # Official non-thinking mode param
    }

    chat_url: str = f"{llama_server_base_url}/chat/completions"
    last_error: Exception | None = None

    for attempt in range(1, TRANSCRIPTION_MAX_RETRIES + 1):
        try:
            models_endpoint_response = httpx.post(
                chat_url,
                json=chat_completion_payload,
                timeout=httpx.Timeout(TRANSCRIPTION_HTTP_TIMEOUT_SEC),
            )
            models_endpoint_response.raise_for_status()
            chat_completion_response_json: dict = models_endpoint_response.json()

            transcribed_text: str = str(
                chat_completion_response_json["choices"][0]["message"]["content"]
            )
            return transcribed_text

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Transcription API call failed – retrying",
                image=cropped_label_path.name,
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
# Per-Image Pipeline (with per-step timing)
# ==================================================================================================

def process_single_label_image(
    input_image_path: Path,
    yolo_obb_model: YOLO,
    llama_server_base_url: str,
    cropped_labels_output_dir: Path,
) -> dict:
    """
    Executes the full label-extraction pipeline for a single input image,
    measuring elapsed time for each step.

    Stages (in order):
        1. Barcode detection (EAN code + orientation) — optional via ENABLE_EAN_DETECTION.
        2. YOLO OBB label detection (TensorRT FP16 GPU).
        3. Affine deskew + axis-aligned crop.
        4. OCR transcription via llama-server (Qwen3.6 vision model, non-thinking mode).
        5. Save cropped label JPEG to disk.

    Args:
        input_image_path: Path to the input photograph.
        yolo_obb_model: Pre-loaded YOLO OBB model (TensorRT engine).
        llama_server_base_url: Base URL of the llama-server OpenAI-compatible API.
        cropped_labels_output_dir: Directory where cropped label JPEGs are written.

    Returns:
        A dict with keys: name, ean, rotation, text, timing.
    """
    logger.info("Processing image", image=input_image_path.name)
    t_image_start: float = time.perf_counter()

    # ----------------------------------------------------------------------------------------------
    # Stage 1 – Barcode detection (optional)
    # ----------------------------------------------------------------------------------------------
    t_barcode_start: float = time.perf_counter()

    if ENABLE_EAN_DETECTION:
        ean_code, additional_rotation_degrees = detect_ean_barcode_and_orientation(
            input_image_path
        )
    else:
        ean_code = None
        additional_rotation_degrees = 0

    t_barcode_ms: float = (time.perf_counter() - t_barcode_start) * 1000

    # ----------------------------------------------------------------------------------------------
    # Stage 2 – YOLO OBB label detection (TensorRT GPU)
    # ----------------------------------------------------------------------------------------------
    t_yolo_start: float = time.perf_counter()

    try:
        yolo_results = yolo_obb_model.predict(
            source=str(input_image_path),
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
                "barcode_ms": round(t_barcode_ms, 1),
                "yolo_ms": round(t_yolo_ms, 1),
                "crop_ms": 0.0,
                "ocr_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_image_start) * 1000, 1),
            },
        }

    # ----------------------------------------------------------------------------------------------
    # Stage 3 – Deskew + crop
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
            source_image_path=input_image_path,
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
                "barcode_ms": round(t_barcode_ms, 1),
                "yolo_ms": round(t_yolo_ms, 1),
                "crop_ms": round(t_crop_ms, 1),
                "ocr_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_image_start) * 1000, 1),
            },
        }

    # Write the cropped label to disk so it can be inspected / used as a reference.
    cropped_label_output_path: Path = cropped_labels_output_dir / f"crop_{input_image_path.name}"
    write_success: bool = cv2.imwrite(str(cropped_label_output_path), cropped_label_bgr)
    if not write_success:
        logger.warning(
            "cv2.imwrite returned False – file may be corrupted",
            crop_path=str(cropped_label_output_path),
        )

    t_crop_ms: float = (time.perf_counter() - t_crop_start) * 1000

    # ----------------------------------------------------------------------------------------------
    # Stage 4 – OCR transcription
    # ----------------------------------------------------------------------------------------------
    t_ocr_start: float = time.perf_counter()

    try:
        transcription_markdown: str = _transcribe_label_image(
            cropped_label_output_path, llama_server_base_url
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
            f.write(f"**Thinking Mode:** disabled (non-thinking instruct mode)  \n\n")

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
                f.write(f"**Average per image:** {avg_ms:.0f} ms  \n")
            f.write("\n---\n\n")

            # ---- Timing Summary -----------------------------------------------------------------
            f.write("## Timing Summary\n\n")

            # Collect timing arrays (only successful results)
            timing_keys = ["barcode_ms", "yolo_ms", "crop_ms", "ocr_ms", "total_ms"]
            timing_labels = ["Barcode", "YOLO OBB", "Crop", "OCR", "**Total**"]
            timing_data: dict[str, list[float]] = {k: [] for k in timing_keys}

            for result in batch_processing_results:
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
                    f"- **Timing:** barcode={t['barcode_ms']}ms, "
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
        5. Save per-image cropped labels and transcripts.
        6. Write a summary markdown report with timing statistics.
    """
    logger.info(
        "Starting full-GPU pipeline",
        gguf_model=QWEN_GGUF_WEIGHTS_FILENAME,
        ean_detection=ENABLE_EAN_DETECTION,
    )

    # ----------------------------------------------------------------------------------------------
    # Directory setup
    # ----------------------------------------------------------------------------------------------
    input_images_dir: Path = Path("test")
    output_root_dir: Path = Path("output_test")
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
    # Step 3/3 — Process each image (with timing)
    # ----------------------------------------------------------------------------------------------
    logger.info(
        "[Step 3/3] Starting batch processing",
        total_images=len(input_image_paths),
        report_path=str(batch_report_path),
    )

    pipeline_start_time: float = time.perf_counter()
    batch_processing_results: list[dict] = []

    for image_index, input_image_path in enumerate(input_image_paths, start=1):
        logger.info("Processing image", current=image_index, total=len(input_image_paths))

        result: dict = process_single_label_image(
            input_image_path=input_image_path,
            yolo_obb_model=yolo_obb_model,
            llama_server_base_url=LLAMA_SERVER_BASE_URL,
            cropped_labels_output_dir=cropped_labels_dir,
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
        )

        # --------------------------------------------------------------------------------------------
        # Write per-image transcript immediately (checkpointing so partial results survive crashes)
        # --------------------------------------------------------------------------------------------
        individual_transcript_path: Path = (
            cropped_labels_dir / f"transcript_{input_image_path.stem}.md"
        )
        try:
            with open(individual_transcript_path, "w", encoding="utf-8") as transcript_file_handle:
                transcript_file_handle.write(
                    f"# Transcription for {input_image_path.name}\n\n"
                )
                transcript_file_handle.write(f"- **EAN:** {result['ean'] or 'None'}\n")
                transcript_file_handle.write(
                    f"- **Rotation Detected:** {result['rotation']}°\n"
                )
                transcript_file_handle.write(
                    f"- **Timing:** {t['total_ms']}ms "
                    f"(barcode={t['barcode_ms']}ms, yolo={t['yolo_ms']}ms, "
                    f"crop={t['crop_ms']}ms, ocr={t['ocr_ms']}ms)\n\n"
                )
                transcript_file_handle.write(f"```markdown\n{result['text']}\n```\n")
            logger.debug(
                "Saved individual transcript",
                transcript_path=str(individual_transcript_path),
            )
        except Exception as exc:
            logger.error(
                "Failed to write individual transcript",
                transcript_path=str(individual_transcript_path),
                error=str(exc),
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
