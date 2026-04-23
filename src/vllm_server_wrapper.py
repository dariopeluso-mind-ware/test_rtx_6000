#!/usr/bin/env python3
"""
vLLM server launcher — forza la piattaforma CUDA quando NVML non è disponibile.

In alcuni ambienti container (es. RunPod), la NVIDIA Management Library
(NVML / libnvidia-ml.so) non riesce a inizializzarsi, anche se il compute
CUDA funziona correttamente. Questo wrapper patcha la piattaforma vLLM
PRIMA che il server venga importato, bypassando la detection basata su NVML.

Catena del problema:
    1. vllm.platforms.__init__.cuda_platform_plugin() chiama pynvml.nvmlInit()
    2. nvmlInit() fallisce nel container → cuda_platform_plugin() ritorna None
    3. Nessuna piattaforma rilevata → DeviceConfig.__post_init__ lancia RuntimeError
    4. L'errore avviene DURANTE il parsing degli argomenti CLI (in default_factory),
       quindi --device cuda o VLLM_TARGET_DEVICE non possono aiutare.

Soluzione:
    Importiamo vllm.platforms.cuda DIRETTAMENTE — il modulo gestisce
    l'assenza di NVML con un fallback a NonNvmlCudaPlatform (usa torch.cuda.*).
    Poi settiamo _current_platform PRIMA che l'auto-detection venga invocata.

Usage (da vllm_main.py):
    python src/vllm_server_wrapper.py --model <repo> --port 8000 ...
"""
import sys
import os

# Assicura visibilità CUDA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── Step 1: Forza la piattaforma CUDA ────────────────────────────────────────
# Importa vllm.platforms (lazy — non triggera auto-detection)
import vllm.platforms  # noqa: E402

# Importa CudaPlatform da cuda.py:
#   - Il codice module-level in cuda.py prova pynvml.nvmlInit()
#   - Se fallisce, cade su NonNvmlCudaPlatform (usa torch.cuda.* invece di NVML)
#   - Questo è esattamente il fallback che ci serve
from vllm.platforms.cuda import CudaPlatform  # noqa: E402

# Setta la piattaforma PRIMA che qualsiasi import vLLM triggeri auto-detection
vllm.platforms._current_platform = CudaPlatform()

# ── Step 2: Avvia il server vLLM ─────────────────────────────────────────────
# sys.argv[0] è già impostato correttamente dal subprocess call,
# gli argomenti CLI (--model, --port, etc.) sono in sys.argv[1:]
import runpy  # noqa: E402
runpy.run_module(
    "vllm.entrypoints.openai.api_server",
    run_name="__main__",
    alter_sys=True,
)
