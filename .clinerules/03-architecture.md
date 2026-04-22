# Architecture

## Module Structure

```
1_Riprezzamento-con-foto/
├── src/
│   └── main.py                    # Batch processor — procedural, synchronous
├── llama.cpp/                     # llama.cpp source + compiled llama-server binary
│   ├── build/bin/llama-server     # Built binary (compiled from source)
│   └── ...                        # llama.cpp repository files
├── scripts/                        # Data preparation + training scripts
│   ├── prepare_dataset_obb.py
│   ├── prepare_dataset.py
│   └── train_yolo26_obb_tosano.ipynb
├── best.onnx / best.pt            # YOLO OBB custom trained model
├── data/                          # Training data (zipped)
├── test/                          # Input images for batch processing
├── output_test/                   # Generated output (gitignored)
│   ├── crops/                     # Cropped label JPEGs + per-image transcripts
│   └── mocr_batch_results.md      # Batch summary report
├── etichette_esempio/             # Example images for reference
├── requirements.txt              # Python dependencies
└── README.md                      # Deployment guide
```

> No `backend/`, no `frontend/`, no FastAPI. Everything lives in `src/main.py`.

## Data Flow

```
1. Image files collected from ./test/ (batch)
2. Barcode scan (pyzbar) → EAN code + orientation for fine rotation
3. YOLO OBB inference → oriented bounding box (best.onnx, CPU device)
4. Affine deskew + axis-aligned crop (OpenCV) → upright label JPEG
5. Vision transcription (Qwen3.6-35B-A3B via llama-server) → markdown text
6. Per-image transcript written to output_test/crops/
7. Batch report written to output_test/mocr_batch_results.md
```

## Separation of Concerns

| Function | Responsibility | Rules |
|----------|----------------|-------|
| `ensure_llama_server_running` | Lifecycle management of llama-server subprocess (auto-download, boot, poll-readiness, atexit cleanup) | No image I/O |
| `detect_ean_barcode_and_orientation` | pyzbar scan → EAN + CCW rotation (0/90/180/270) | Pure decode, no inference |
| `_deskew_crop_obb` | OpenCV affine rotation + slice using OBB parameters | No HTTP calls |
| `_transcribe_label_image` | HTTP POST to llama-server `/v1/chat/completions` with base64 image | Sync httpx, retry loop |
| `process_single_label_image` | Pipeline sequencing for one image, returns 3-tuple | Coordination only |
| `write_batch_report` | Writes markdown summary report to disk | No inference |

## Runtime Constants

```python
# llama-server (Qwen3.6-35B-A3B) — runs on port 8080
LLAMA_SERVER_BASE_URL: str = "http://localhost:8080/v1"
LLAMA_SERVER_PORT: int = 8080
QWEN_GGUF_REPO_ID: str = "unsloth/Qwen3.6-35B-A3B-GGUF"
QWEN_GGUF_WEIGHTS_FILENAME: str = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
QWEN_VISION_PROJECTOR_FILENAME: str = "mmproj-F16.gguf"

# Transcription settings
TRANSCRIPTION_HTTP_TIMEOUT_SEC: float = 120.0
TRANSCRIPTION_MAX_RETRIES: int = 3
TRANSCRIPTION_PROMPT: str = "Read all the text in this image. Output only the text directly."

# Barcode settings
EAN_BARCODE_TYPES: tuple[str, ...] = ("EAN13", "EAN8", "UPCA", "UPCE")
PYZBAR_ORIENTATION_TO_ROTATION_DEGREES: dict[str, int] = {
    "UP": 0, "RIGHT": 270, "DOWN": 180, "LEFT": 90,
}
```

## llama-server CLI Flags

**Dev (8 GB VRAM — `--cpu-moe` keeps MoE experts on CPU RAM):**

```bash
./llama-server \
    -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --mmproj mmproj-F16.gguf \
    --port 8080 \
    -c 8192 \
    --n-gpu-layers 99 \
    --cpu-moe \
    -t 8 \
    --no-cache-prompt
```

**Prod (96 GB VRAM — full GPU, flash attention):**

```bash
./llama-server \
    -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --mmproj mmproj-F16.gguf \
    --port 8080 \
    -c 16384 \
    --n-gpu-layers 99 \
    --flash-attn \
    --temp 0.0 \
    --top-k 1 \
    --no-cache-prompt
```

## Output Format

The batch report `output_test/mocr_batch_results.md` contains:

- **Summary table**: per-image index, filename, EAN code, ✓/✗ success indicator
- **Detailed sections**: per image: EAN, rotation degrees, full transcription in fenced markdown block

Per-image checkpoint files are also written to `output_test/crops/transcript_{stem}.md`.

## VRAM Budget (8 GB dev)

| Component | VRAM |
|-----------|------|
| YOLO OBB ONNX | ~50 MB |
| Qwen3.6-35B-A3B MoE (activation tensors) | ~5–6 GB |
| MoE expert FFN layers | CPU RAM (via `--cpu-moe`) |
| Safety margin | ~450 MB |
| **Total** | **~6 GB** ✅ |

## VRAM Budget (96 GB prod)

| Component | VRAM |
|-----------|------|
| YOLO OBB | ~50 MB |
| Qwen3.6-35B-A3B Q4_K_M full | ~20–40 GB |
| **Total** | **~40 GB** ✅ |