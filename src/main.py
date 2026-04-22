#!/usr/bin/env python3
"""
Procedural batch processor for Tosano supermarket food-label pipeline.

Pipeline Overview (per image):
    1. Barcode Detection  ──── pyzbar ──────────────────▶ EAN code + orientation
    2. Label Detection    ──── YOLO OBB (best.onnx) ──▶ Oriented bounding box
    3. Image Cropping     ──── Affine deskew ──────────▶ Upright label crop
    4. OCR Transcription  ──── llama-server (Qwen3.6) ─▶ Markdown text
    5. Report Generation  ──── Markdown ───────────────▶ output_test/mocr_batch_results.md

Hardware target:
    Development (8 GB VRAM): llama-server --cpu-moe keeps MoE experts on CPU so the
    activation tensor fits in VRAM while still benefiting from GPU attention layers.
"""

# --------------------------------------------------------------------------------------------------
# Stdlib imports
# --------------------------------------------------------------------------------------------------
from pathlib import Path
from datetime import datetime
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
QWEN_GGUF_WEIGHTS_FILENAME: str = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
QWEN_VISION_PROJECTOR_FILENAME: str = "mmproj-F16.gguf"
QWEN_MODEL_API_NAME: str = "Qwen3.6"          # Value used in the chat-completion "model" field

# --- llama-server runtime settings ---
LLAMA_SERVER_PORT: int = 8080
LLAMA_SERVER_BASE_URL: str = f"http://localhost:{LLAMA_SERVER_PORT}/v1"
LLAMA_SERVER_CONTEXT_SIZE: int = 8192          # Larger context accommodates the full label image
LLAMA_SERVER_CPU_THREADS: int = 8             # Threads for on-CPU MoE expert computation
LLAMA_SERVER_SHUTDOWN_TIMEOUT_SEC: float = 10.0
LLAMA_SERVER_READY_POLL_SEC: float = 5.0      # Sleep between readiness probes during boot
LLAMA_SERVER_BOOT_TIMEOUT_SEC: float = 300.0  # Abort if the server hasn't responded in 5 min

# --- Transcription API settings ---
TRANSCRIPTION_HTTP_TIMEOUT_SEC: float = 120.0
TRANSCRIPTION_MAX_RETRIES: int = 3
TRANSCRIPTION_PROMPT: str = (
    "Read all the text in this image. Output only the text directly."
)

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

    Detection logic:
        A GET to {base_url}/models returning HTTP 200 means the server is already live.
        Any other response (connection error, timeout, non-200) is treated as "not running"
        and triggers automatic download + spawn.

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

    logger.info("Downloading Qwen3.6 GGUF model weights (may require auth for gated repo)...")
    qwen_weights_path: Path = Path(
        hf_hub_download(
            repo_id=QWEN_GGUF_REPO_ID,
            filename=QWEN_GGUF_WEIGHTS_FILENAME,
        )
    )

    logger.info("Downloading Qwen3.6 Vision Projector (mmproj)...")
    vision_projector_path: Path = Path(
        hf_hub_download(
            repo_id=QWEN_GGUF_REPO_ID,
            filename=QWEN_VISION_PROJECTOR_FILENAME,
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Step 3 – Build llama-server command line
    # ----------------------------------------------------------------------------------------------
    # --cpu-moe: Move MoE (Mixture of Experts) expert FFN layers to CPU RAM.
    #            This allows the activation tensors to remain in 8 GB VRAM while the
    #            bulk of the model parameters reside in host RAM, enabling the model
    #            to run on a single consumer GPU.
    # -c / --ctx-size: Increases context length to accommodate high-resolution images.
    # --no-cache-prompt: Disable KV-cache prompt reuse – every request gets fresh
    #                    inference, which is appropriate for one-shot image processing.
    command_line: list[str] = [
        str(LLAMA_SERVER_BINARY_PATH),
        "-m",        str(qwen_weights_path),
        "--mmproj",  str(vision_projector_path),
        "--port",    str(LLAMA_SERVER_PORT),
        "-c",        str(LLAMA_SERVER_CONTEXT_SIZE),
        "--n-gpu-layers", "99",
        "--cpu-moe",                             # Fit Qwen3.6-A3B in 8 GB VRAM
        "-t",        str(LLAMA_SERVER_CPU_THREADS),
        "--no-cache-prompt",
    ]

    logger.info("Starting llama-server subprocess", command=" ".join(command_line))

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

        # Abort if boot is taking unreasonably long (default 5 minutes).
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
            pass

        # Not ready yet – sleep before polling again.
        time.sleep(LLAMA_SERVER_READY_POLL_SEC)


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

    The image is base64-encoded into an OpenAI-compatible ``/chat/completions`` payload.
    Temperature is set to 0 for deterministic verbatim output; max_tokens is set high
    enough to accommodate long ingredient lists.

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
    # The "model" field must match the alias the llama-server was started with.
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
        "temperature": 0.0,
        "max_tokens": LLAMA_SERVER_CONTEXT_SIZE,
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
# Per-Image Pipeline
# ==================================================================================================

def process_single_label_image(
    input_image_path: Path,
    yolo_obb_model: YOLO,
    llama_server_base_url: str,
    cropped_labels_output_dir: Path,
) -> tuple[str | None, int, str]:
    """
    Executes the full label-extraction pipeline for a single input image.

    Stages (in order):
        1. Barcode detection (EAN code + orientation).
        2. YOLO OBB label detection (oriented bounding box).
        3. Affine deskew + axis-aligned crop.
        4. OCR transcription via llama-server (Qwen3.6 vision model).
        5. Save cropped label JPEG to disk.

    Args:
        input_image_path: Path to the input photograph.
        yolo_obb_model: Pre-loaded YOLO OBB model (best.onnx).
        llama_server_base_url: Base URL of the llama-server OpenAI-compatible API.
        cropped_labels_output_dir: Directory where cropped label JPEGs are written.

    Returns:
        A 3-tuple ``(ean_code, rotation_degrees, transcription_or_error_message)``:
        - ``ean_code`` – decoded barcode value, or ``None`` if not detected.
        - ``rotation_degrees`` – additional barcode-derived rotation, or 0.
        - ``transcription_or_error_message`` – the transcribed markdown text on success,
          or an error description string on any failure.
    """
    logger.info("Processing image", image=input_image_path.name)

    # ----------------------------------------------------------------------------------------------
    # Stage 1 – Barcode detection
    # ----------------------------------------------------------------------------------------------
    ean_code: str | None
    additional_rotation_degrees: int
    ean_code, additional_rotation_degrees = detect_ean_barcode_and_orientation(
        input_image_path
    )

    # ----------------------------------------------------------------------------------------------
    # Stage 2 – YOLO OBB label detection
    # ----------------------------------------------------------------------------------------------
    try:
        yolo_results = yolo_obb_model.predict(
            source=str(input_image_path),
            device="cpu",    # ONNX model runs on CPU in dev; GPU auto-selected in prod.
            verbose=False,
        )
        first_result = yolo_results[0]
    except Exception as exc:
        logger.error(
            "YOLO OBB inference failed",
            image=input_image_path.name,
            error=str(exc),
        )
        return ean_code, additional_rotation_degrees, f"YOLO inference error: {exc}"

    if not first_result.obb or len(first_result.obb) == 0:
        logger.warning("No label detected by YOLO OBB", image=input_image_path.name)
        return ean_code, additional_rotation_degrees, "No label detected by YOLO OBB."

    # ----------------------------------------------------------------------------------------------
    # Stage 3 – Deskew + crop
    # ----------------------------------------------------------------------------------------------
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
        return ean_code, additional_rotation_degrees, f"Image crop failed: {exc}"

    # Write the cropped label to disk so it can be inspected / used as a reference.
    cropped_label_output_path: Path = cropped_labels_output_dir / f"crop_{input_image_path.name}"
    write_success: bool = cv2.imwrite(str(cropped_label_output_path), cropped_label_bgr)
    if not write_success:
        logger.warning(
            "cv2.imwrite returned False – file may be corrupted",
            crop_path=str(cropped_label_output_path),
        )

    # ----------------------------------------------------------------------------------------------
    # Stage 4 – OCR transcription
    # ----------------------------------------------------------------------------------------------
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
        return ean_code, additional_rotation_degrees, f"API Error: {exc}"

    return ean_code, additional_rotation_degrees, transcription_markdown


# ==================================================================================================
# Batch Report Generation
# ==================================================================================================

def write_batch_report(
    batch_report_path: Path,
    input_images_dir: Path,
    batch_processing_results: list[dict],
) -> None:
    """
    Writes a human-readable markdown report summarising the batch run.

    The report includes:
        - A header with the input folder name and generation timestamp.
        - A summary table (index, filename, EAN, success/failure status).
        - Per-image detailed sections with the full transcription enclosed in a
          fenced code block.

    Args:
        batch_report_path: Path where the markdown report should be written.
        input_images_dir: Name of the input folder (used in the report header).
        batch_processing_results: List of result dictionaries, one per processed image.
    """
    try:
        with open(batch_report_path, "w", encoding="utf-8") as report_file_handle:
            # ---- Header --------------------------------------------------------------------------
            report_file_handle.write("# MOCR Batch Processing Report\n\n")
            report_file_handle.write(f"**Folder:** `{input_images_dir.name}`  \n")
            report_file_handle.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n"
            )

            # ---- Summary table ------------------------------------------------------------------
            report_file_handle.write("## Summary\n\n")
            report_file_handle.write("| # | Image | EAN | Status |\n")
            report_file_handle.write("|---|-------|-----|--------|\n")

            success_count: int = 0
            for image_index, result in enumerate(batch_processing_results, start=1):
                image_filename: str = result["name"]
                ean_value: str = result["ean"] or "—"

                # A result is considered an error when the transcription text begins with
                # a known error prefix (set by the pipeline helpers on failure).
                transcription_markdown: str = result["text"]
                is_error: bool = (
                    transcription_markdown.startswith("API Error")
                    or transcription_markdown.startswith("YOLO inference error")
                    or transcription_markdown.startswith("No label detected")
                    or transcription_markdown.startswith("Image crop failed")
                )
                status_icon: str = "✗" if is_error else "✓"
                if not is_error:
                    success_count += 1

                report_file_handle.write(
                    f"| {image_index} | {image_filename} | {ean_value} | {status_icon} |\n"
                )

            report_file_handle.write(f"\n**Stats:** {success_count}/{len(batch_processing_results)} succeeded.\n")
            report_file_handle.write("\n---\n\n")

            # ---- Per-image detailed results ----------------------------------------------------
            report_file_handle.write("## Detailed Results\n\n")
            for image_index, result in enumerate(batch_processing_results, start=1):
                report_file_handle.write(f"### {image_index}. {result['name']}\n")
                report_file_handle.write(f"- **EAN:** {result['ean'] or 'None'}\n")
                report_file_handle.write(f"- **Rotation Needed:** {result['rotation']}°\n")
                report_file_handle.write("\n**Transcription:**\n")
                report_file_handle.write(f"```markdown\n{result['text']}\n```\n\n")

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
    Procedural entry point for the Tosano food-label batch processor.

    Responsibilities:
        1. Locate all image files in the input directory.
        2. Load the YOLO OBB model (best.onnx).
        3. Ensure llama-server is running (downloading Qwen3.6 weights if necessary).
        4. Process each image through the full pipeline.
        5. Save per-image cropped labels and transcripts.
        6. Write a summary markdown report to output_test/.
    """
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
    # Load YOLO OBB model
    # ----------------------------------------------------------------------------------------------
    logger.info("Loading YOLO OBB model", model_path="best.onnx")
    try:
        yolo_obb_model: YOLO = YOLO("best.onnx", task="obb")
    except Exception as exc:
        logger.error("Failed to load YOLO OBB model", model_path="best.onnx", error=str(exc))
        return

    # ----------------------------------------------------------------------------------------------
    # Ensure llama-server is running (auto-download + boot if missing)
    # ----------------------------------------------------------------------------------------------
    ensure_llama_server_running(LLAMA_SERVER_BASE_URL)
    logger.info("llama-server connection configured", url=LLAMA_SERVER_BASE_URL)

    # ----------------------------------------------------------------------------------------------
    # Process each image
    # ----------------------------------------------------------------------------------------------
    logger.info(
        "Starting batch processing",
        total_images=len(input_image_paths),
        report_path=str(batch_report_path),
    )

    batch_processing_results: list[dict] = []

    for image_index, input_image_path in enumerate(input_image_paths, start=1):
        logger.info("Processing image", current=image_index, total=len(input_image_paths))

        ean_code: str | None
        rotation_degrees: int
        transcription_markdown: str
        ean_code, rotation_degrees, transcription_markdown = process_single_label_image(
            input_image_path=input_image_path,
            yolo_obb_model=yolo_obb_model,
            llama_server_base_url=LLAMA_SERVER_BASE_URL,
            cropped_labels_output_dir=cropped_labels_dir,
        )

        batch_processing_results.append({
            "name": input_image_path.name,
            "ean": ean_code,
            "rotation": rotation_degrees,
            "text": transcription_markdown,
        })

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
                transcript_file_handle.write(f"- **EAN:** {ean_code or 'None'}\n")
                transcript_file_handle.write(
                    f"- **Rotation Detected:** {rotation_degrees}°\n\n"
                )
                transcript_file_handle.write(f"```markdown\n{transcription_markdown}\n```\n")
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
    write_batch_report(
        batch_report_path=batch_report_path,
        input_images_dir=input_images_dir,
        batch_processing_results=batch_processing_results,
    )


if __name__ == "__main__":
    main()