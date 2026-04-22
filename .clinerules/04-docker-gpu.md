# GPU & llama-server Configuration

## No Docker

llama-server (Qwen3.6-35B-A3B) runs **natively on host GPU** via llama.cpp with CUDA.
No Docker, no vLLM, no container overhead. Benefits:

- No Blackwell GPU (sm_120) compatibility issues with container runtimes
- No glibc version conflicts
- Simpler deployment
- Faster startup (no container overhead)
- Single binary, one process

## llama-server Configuration

llama-server is started **automatically** by `src/main.py` via
`ensure_llama_server_running()` if not already running on port 8080.
The binary is compiled from `llama.cpp/` with CUDA support.

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

## VRAM Allocation

```python
# Total budget: 8GB (dev), 96GB (production)
VRAM_BUDGET_DEV = {
    "yolo_obb": 0.05,           # ~50MB  (best.onnx ONNX)
    "qwen_moe_activation": 5.5,  # ~5–6GB (Qwen3.6-35B-A3B activation tensors)
    "safety": 0.45,              # ~450MB headroom
}
# Total: ~6GB ✅

VRAM_BUDGET_PROD = {
    "yolo_obb": 0.05,            # ~50MB
    "qwen_moe_full": 35.0,       # ~20–40GB (Q4_K_M full weights)
    "safety": 10.0,              # headroom
}
# Total: ~45GB ✅ (well under 96GB)
```

## GPU Flags for llama-server

| Flag | Dev | Prod | Purpose |
|------|-----|------|---------|
| `-m` | GGUF weights path | GGUF weights path | Model weights file |
| `--mmproj` | mmproj path | mmproj path | Vision projector |
| `--port` | 8080 | 8080 | HTTP API port |
| `-c` | 8192 | 16384 | Context size |
| `--n-gpu-layers` | 99 | 99 | Load all layers on GPU |
| `--cpu-moe` | ✅ | ❌ | MoE experts to CPU RAM (8 GB only) |
| `-t` | 8 | — | CPU threads for MoE computation |
| `--flash-attn` | ❌ | ✅ | Flash attention (Ampere+ Blackwell) |
| `--temp` | — | 0.0 | Deterministic output |
| `--top-k` | — | 1 | Greedy decoding |
| `--no-cache-prompt` | ✅ | ✅ | No KV-cache reuse for one-shot image |

## Model Selection

| OCR Model | VRAM | Notes |
|-----------|------|-------|
| Qwen3.6-35B-A3B Q4_K_M (MoE) | ~5–6 GB (dev) / ~20–40 GB (prod) | **Current** — vision-native, single model |

| YOLO Model | VRAM | Notes |
|-----------|------|-------|
| `best.onnx` (OBB, CPU inference) | ~50 MB | **Current** — custom trained |
| `best.pt` (OBB, PyTorch) | ~21 MB | Same model, PyTorch format |

## llama.cpp Build (CUDA)

Build llama-server with CUDA support for Blackwell GPUs (sm_120):

```bash
cd llama.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF

cmake --build build --config Release -j$(nproc) --target llama-server
cp build/bin/llama-server .
```

Requires CUDA Toolkit ≥ 12.8 for sm_120 support.