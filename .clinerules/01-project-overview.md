# Project Overview

## Purpose

Automated food-label processing pipeline for the Tosano supermarket chain. The system
detects barcodes (orientation), isolates product labels from photographs using a custom
trained YOLO OBB model, deskews and crops the label region, and transcribes the text via a
vision-capable LLM (Qwen3.6-35B-A3B MoE) running on llama-server.

Output is a human-readable markdown report with per-image transcription and EAN code.

## Target Hardware

| Environment | GPU | VRAM | Notes |
|-------------|-----|------|-------|
| Development | RTX PRO 1000 Blackwell | 8 GB | `--cpu-moe` keeps MoE experts on CPU so activation tensors fit in VRAM |
| Production   | RTX 6000 PRO Blackwell | 96 GB | All layers on GPU, `--flash-attn`, larger context |

## Pipeline Architecture

```
Image → [pyzbar] → EAN code + orientation
       → [YOLO OBB] → Oriented bounding box
       → [affine deskew + crop] → Upright label crop
       → [Qwen3.6-35B-A3B vision] (llama-server) → Markdown transcription
       → [batch report] → output_test/mocr_batch_results.md
```

| Stage | Component | Notes |
|-------|-----------|-------|
| 1 | pyzbar (ZBar barcode scanner) | EAN/UPC detection + orientation (0/90/180/270°) |
| 2 | YOLO OBB (`best.onnx` / `best.pt`) | Custom trained, oriented bounding box detection |
| 3 | OpenCV affine deskew + crop | Rotate + slice to get upright label |
| 4 | Qwen3.6-35B-A3B Q4_K_M + mmproj | Vision MoE via llama-server (port 8080), `--cpu-moe` on dev |
| 5 | Report generation | Markdown per-image transcripts + batch summary |

**VRAM budget (8 GB dev with `--cpu-moe`)**:

| Component | VRAM |
|-----------|------|
| YOLO OBB ONNX | ~50 MB |
| Qwen3.6-35B-A3B MoE (activation tensors) | ~5–6 GB |
| MoE expert FFN layers | CPU RAM (via `--cpu-moe`) |
| Safety margin | ~450 MB |
| **Total** | **~6 GB** ✅ |

**VRAM budget (96 GB prod)**:

| Component | VRAM |
|-----------|------|
| YOLO OBB | ~50 MB |
| Qwen3.6-35B-A3B Q4_K_M full | ~20–40 GB |
| **Total** | **~40 GB** ✅ |

## Why Qwen3.6-35B-A3B?

- **MoE (Mixture of Experts)**: Only ~35B parameters active per token, rest stay on CPU
  RAM via `--cpu-moe` flag in dev.
- **Vision-native**: Multimodal with mmproj projector; no separate OCR model needed.
- **Single model, single server**: llama-server handles both cropping and transcription.
- **No Docker**: Runs natively on host GPU via llama.cpp with CUDA.

## Why Not dots.mocr?

dots.mocr was evaluated early but **replaced** by Qwen3.6 vision because:
- Single model pipeline is simpler to operate (no vLLM server on port 8090).
- Qwen3.6-35B-A3B provides strong OCR accuracy in Italian without a separate OCR stage.
- No Docker/vLLM dependency reduces deployment complexity.

## Output Schema

The pipeline outputs **markdown transcription text**, not structured JSON. The report
`output_test/mocr_batch_results.md` contains per-image:

- EAN code (or "—" if not detected)
- Rotation degrees detected from barcode orientation
- Full transcribed markdown text (verbatim)

A later stage could extract structured JSON from the markdown, but this is not currently
implemented.

## Key Constraints

- Dev (8 GB): Processing time is longer due to `--cpu-moe` overhead; acceptable for batch.
- Prod (96 GB): Full GPU execution; `--flash-attn on` + larger context for speed.
- llama-server auto-downloads GGUF weights from HuggingFace on first run.
- `libzbar0` system package required for pyzbar barcode scanning.