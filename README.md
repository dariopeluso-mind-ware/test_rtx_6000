# 1_Riprezzamento-con-foto — Deployment Guide

> Pipeline per l'estrazione automatica di etichette alimentari da fotografie:
> **YOLO OBB** (crop etichetta) → **Qwen3.6‑35B‑A3B** (OCR vision) → dati strutturati.

## Backend OCR disponibili

Questo progetto offre **due backend OCR** alternativi, entrambi compatibili con Qwen3.6-35B-A3B:

| Backend | Script | VRAM | Velocità | Note |
|---------|--------|------|----------|------|
| llama.cpp | `src/full-gpu_main.py` | 96 GB | ~1.3–2.0 s/img | Più semplice, no build extra |
| **vLLM** | `src/vllm_main.py` | 96 GB | **< 1 s/img** | Target principale |

Questo documento copre entrambi i backend. La Sezione 9 è dedicata a vLLM.

---

## Hardware di riferimento

| Ambiente | GPU | VRAM | Note |
|----------|-----|------|------|
| Development (locale) | RTX PRO 1000 Blackwell | 8 GB | Configurazione attuale |
| Cloud target | RTX 6000 PRO Blackwell | 96 GB | Deployment remoto via SSH |

Questo documento descrive come replicare l'ambiente sulla macchina cloud RTX 6000 PRO Blackwell.

---

## Indice

1. [Prerequisiti hardware](#1-prerequisiti-hardware)
2. [Pacchetti di sistema](#2-pacchetti-di-sistema)
3. [CUDA Toolkit 12.8 / 13.x](#3-cuda-toolkit-128--13x)
4. [Setup del progetto](#4-setup-del-progetto)
5. [Environment Python](#5-environment-python)
6. [Build di llama.cpp con CUDA](#6-build-di-llamacpp-con-cuda) *(solo per `full-gpu_main.py`)*
7. [Pre-download dei pesi GGUF](#7-pre-download-dei-pesi-gguf) *(solo per `full-gpu_main.py`)*
8. [Esecuzione Full-GPU llama.cpp (96 GB VRAM)](#8-esecuzione-full-gpu-llamacpp-96-gb-vram)
9. **[Esecuzione vLLM (target < 1 s/immagine)](#9-esecuzione-vllm-target-1-s-immagine)**
10. [Esecuzione della pipeline](#10-esecuzione-della-pipeline)
11. [Recupero dei risultati](#11-recupero-dei-risultati)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisiti hardware

Verificare che la GPU sia visibile e abbia il driver corretto:

```bash
nvidia-smi
# Output atteso: "NVIDIA RTX 6000 PRO Blackwell" con VRAM ≈ 96 384 MiB
# Driver version: ≥ 535 (meglio ≥ 550 per Blackwell)
```

**Nota su compute capability**: la RTX 6000 PRO Blackwell usa architettura **sm_120** (compute capability 12.0).
CUDA Toolkit deve essere **≥ 12.8** per supportare sm_120 nativamente.

Se `nvidia-smi` mostra un driver vecchio o non trova la GPU, installare i driver NVIDIA:

```bash
# Ubuntu 22.04 / 24.04
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-560  # o più recente
sudo reboot
```

---

## 2. Pacchetti di sistema

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    tmux \
    python3-venv \
    python3-pip \
    python3-dev \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    pkg-config \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    libncurses5-dev \
    libffi-dev \
    libssl-dev
```

**Nota**: `libzbar0` è necessario per `pyzbar` (decode EAN barcode).

Verifica dell'installazione:

```bash
dpkg -l | grep -E "libzbar|libgl|cmake|build-essential"
```

---

## 3. CUDA Toolkit 12.8 / 13.x

La RTX 6000 PRO Blackwell richiede CUDA **≥ 12.8**. La 13.x è consigliata per pieno supporto sm_120.

> ⚠️ **NON usare CUDA 13.2** — bug noto che causa output gibberish con modelli Qwen3.6.
> NVIDIA sta lavorando a un fix. Usare CUDA 12.8 o 13.0.
> (Fonte: [Unsloth docs](https://unsloth.ai/docs/models/qwen3.6))

### Installazione (Ubuntu 22.04 / 24.04)

```bash
# Scarica il repository CUDA NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Installa CUDA Toolkit 13.0 (include nvcc) — NON 13.2!
sudo apt install -y cuda-toolkit-13-0

# oppure CUDA 12.8 (più stabile, supporta comunque sm_120)
# sudo apt install -y cuda-toolkit-12-8
```

### Configurazione PATH e LD_LIBRARY_PATH

```bash
# Aggiungere a ~/.bashrc (o ~/.zshrc)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

```bash
source ~/.bashrc
```

### Verifica

```bash
nvcc --version
# Output atteso: Cuda compilation tools, release 13.0, V13.0.x
```

---

## 4. Setup del progetto

Clonare il repository sulla macchina remota:

```bash
# Scegli la directory di lavoro (es. /workspace su RunPod, ~ su server dedicati)
cd /workspace

git clone https://github.com/<org>/<repo>.git
cd <repo>
```

Clonare llama.cpp all'interno del progetto (solo per `full-gpu_main.py`; NON necessario per `vllm_main.py`):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
# Commit verificato: 1f30ac0ce
```

> **Per `vllm_main.py`**: llama.cpp non è necessario — vLLM gestisce tutto internamente.

Struttura dei file dopo il clone:

```txt
<repo>/
├── best.onnx              # YOLO OBB model (~38 MB)
├── best.pt               # YOLO PyTorch model (~21 MB) — usato per export TensorRT
├── best.engine            # YOLO TensorRT FP16 (generato al primo avvio)
├── src/
│   ├── main.py            # Script dev (8 GB VRAM, --cpu-moe, ONNX CPU)
│   ├── full-gpu_main.py  # Script llama.cpp (96 GB VRAM, full GPU, TensorRT)
│   └── vllm_main.py      # Script vLLM (96 GB VRAM, PagedAttention, < 1 s/img) ⭐
├── test/                  # Immagini di input per batch processing
├── llama.cpp/             # Solo per full-gpu_main.py (clonato separatamente)
├── requirements.txt       # Dipendenze Python
└── output/                # Output processing (generato)
```

---

## 5. Environment Python

```bash
# Dalla root del progetto (dove si trova requirements.txt)
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
```

### Dipendenze Python

```bash
pip install -r requirements.txt
```

Se `requirements.txt` non è disponibile:

```bash
pip install \
    ultralytics>=8.4.0 \
    httpx>=0.28.0 \
    pyzbar>=0.1.9 \
    pillow>=12.0.0 \
    structlog>=25.0.0 \
    huggingface-hub>=0.28.0 \
    opencv-python-headless>=4.10.0 \
    numpy>=1.24.0

# Per full-gpu_main.py (llama.cpp): pip install onnxruntime-gpu>=1.23.0
# Per vllm_main.py: pip install vllm>=0.19.0
```

**Note importanti**:

- **`opencv-python-headless`**: versione headless (senza GUI), necessaria su server headless. Non installare `opencv-python`.
- **`pyzbar`**: richiede `libzbar0` installato a livello sistema (cfr. Sezione 2).

### Verifica dell'installazione

```bash
python -c "import ultralytics; import httpx; import pyzbar; import PIL; import structlog; print('✓ Tutte le dipendenze importabili')"
```

---

## 6. Build di llama.cpp con CUDA *(solo per `full-gpu_main.py`)*

### 6a. Configurazione CMake

```bash
cd llama.cpp

echo $CUDA_HOME
# Deve mostrare: /usr/local/cuda-13.0

cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DLLAMA_SERVER_VERBOSE=OFF
```

### 6b. Compilazione

```bash
cmake --build build --config Release -j$(nproc) --target llama-server
```

Tempo stimato: **5–15 minuti** su una macchina cloud con 32+ core.

### 6c. Copia del binario

```bash
cp build/bin/llama-server .
ls -lh llama-server
```

---

## 7. Pre-download dei pesi GGUF *(solo per `full-gpu_main.py`)*

I pesi vengono scaricati automaticamente alla prima esecuzione tramite `huggingface_hub`.
Il modello `unsloth/Qwen3.6-35B-A3B-GGUF` è **pubblico** — non serve login né token HuggingFace.

Per evitare attese durante il primo run, pre-scaricarli manualmente:

```bash
# Download del modello principale (UD-Q4_K_XL — raccomandato)
hf download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

# Download del vision encoder (mmproj)
hf download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    mmproj-F16.gguf
```

**Dimensione stimata**: ~22 GB totali (modello + mmproj).

---

## 8. Esecuzione Full-GPU llama.cpp (96 GB VRAM)

Con 96 GB VRAM, usare `src/full-gpu_main.py` al posto di `src/main.py`.

### 8a. Differenze rispetto a `main.py`

| Feature | `main.py` (dev, 8 GB) | `full-gpu_main.py` (prod, 96 GB) |
|---------|------------------------|-----------------------------------|
| MoE experts | CPU (`--cpu-moe`) | GPU (tutti i 256 esperti in VRAM) |
| YOLO inference | ONNX su CPU | TensorRT FP16 su GPU |
| Thinking mode | Abilitato | Disabilitato (`--chat-template-kwargs`) |
| Flash Attention | No | Sì (`--flash-attn`) |
| Continuous batching | No | Sì (`--cont-batching`) |
| Prompt caching | Disabilitato | Abilitato (`--cache-prompt`) |
| Context size | 8192 | 16384 |
| max_tokens | 8192 | 2048 |
| Timing report | No | Sì (per-step, per-image, totale) |
| EAN detection | Sempre | Opzionale (env var) |
| GGUF model | Hardcoded Q4_K_M | Configurabile via env var |

### 8b. Ottimizzazioni applicate (`full-gpu_main.py`)

Le seguenti ottimizzazioni sono già integrate in `full-gpu_main.py`:

| Ottimizzazione | Impatto stimato |
|---------------|----------------|
| Immagine caricata 1 volta (non 3) | ~100 ms/immagine |
| Barcode pyzbar downscaled a 1500px | ~600 ms/immagine |
| YOLO TensorRT con imgsz=1024 (kernel ottimizzati) | ~400 ms/immagine |
| llama-server: `--batch-size 2048 --ubatch-size 512 -t 8` | ~15-25% OCR |
| Greedy sampling (temperature=0.0, top_k=1) | ~10-20% OCR |
| Persistent httpx.Client (connection reuse) | ~5-10% OCR |

**Stima totale con llama.cpp**: ~1.3–2.0 s/immagine (dopo warmup)

### 8c. Configurazione (`.env`)

```bash
cp .env.example .env
```

Contenuto di `.env`:

```ini
# Modello GGUF (default: UD-Q4_K_XL)
QWEN_GGUF_FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

# Abilitare/disabilitare ricerca EAN barcode (default: true)
ENABLE_EAN_DETECTION=true
```

### 8d. Esecuzione

```bash
source .venv/bin/activate
python3 src/full-gpu_main.py
```

**Nota**: `llama-server` viene avviato automaticamente se non è già in ascolto su port 8080.
Il primo avvio scaricherà il modello GGUF (~22 GB) e genererà l'engine TensorRT (~1-2 min).
Log disponibile in: `output/llama_server.log`.

---

## 9. Esecuzione vLLM (target < 1 s/immagine)

> **Questa è la configurazione raccomandata per il target < 1 secondo per immagine.**

### 9a. Perché vLLM

vLLM offre funzionalità non disponibili in llama.cpp che accelerano drasticamente l'OCR:

| Feature vLLM | Beneficio per OCR |
|--------------|-------------------|
| **PagedAttention v2** | KV cache in 4KB page — fino a 40% riduzione memoria |
| **Chunked Prefill** | Evita prefill/decode conflict in batch misti |
| **Prefix Caching** | System prompt KV cache riutilizzato per tutte le immagini |
| **Continuous Batching** | Schedulazione dinamica a livello di iterazione |
| **FP8 nativo Blackwell** | Tensor core nativi per BF16 (il modello gira così) |
| **Reasoning parser** | Supporto nativo per Qwen3.6 |

**Stima con vLLM**: **< 1 s/immagine** (dopo warmup, per immagini etichetta typicali)

### 9b. Installazione di vLLM

```bash
source .venv/bin/activate

# vLLM >= 0.19.0 richiesto per Qwen3.6-35B-A3B
pip install vllm>=0.19.0
```

**Requisiti aggiuntivi** (installati automaticamente da vLLM):
- PyTorch con supporto CUDA
- Triton compiler

**Nota**: vLLM gestisce autonomamente i pesi del modello — **nessun build manuale necessario**.

### 9c. Modello

Il modello è `Qwen/Qwen3.6-35B-A3B` (repo ufficiale HuggingFace).
vLLM scaricherà automaticamente i pesi alla prima esecuzione.

**Dimensione**: ~72 GB (pesi BF16 in safetensors)

Per pre-scaricare (consigliato):

```bash
# Con huggingface-cli
huggingface-cli download Qwen/Qwen3.6-35B-A3B
```

### 9d. Configurazione (`.env`)

```bash
# Aggiungere a .env (o creare .env con solo queste variabili)
VLLM_MODEL_REPO_ID=Qwen/Qwen3.6-35B-A3B
VLLM_PORT=8001
VLLM_TENSOR_PARALLEL_SIZE=1
```

**.env completo per vLLM**:

```ini
# ── vLLM settings ──────────────────────────────
VLLM_MODEL_REPO_ID=Qwen/Qwen3.6-35B-A3B
VLLM_PORT=8001
VLLM_TENSOR_PARALLEL_SIZE=1

# ── Pipeline settings ─────────────────────────
ENABLE_EAN_DETECTION=true
SAVE_CROPS=false
YOLO_IMG_SIZE=1024
CROP_MAX_DIMENSION=1280
```

### 9e. Comandi vLLM (dalla model card ufficiale)

Il comando vLLM dalla [model card ufficiale HuggingFace](https://huggingface.co/Qwen/Qwen3.6-35B-A3B):

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3
```

**Flag importanti**:
- `--reasoning-parser qwen3`: obbligatorio per Qwen3.6
- `--enable-chunked-prefill`: accelera il prefill delle immagini
- `--enable-prefix-caching`: sistema prompt riutilizzato nella KV cache
- `--gpu-memory-utilization 0.85`: lascia 15% VRAM per YOLO + OS

### 9f. Esecuzione di `vllm_main.py`

```bash
source .venv/bin/activate

# Esecuzione (vLLM viene avviato automaticamente se non già in ascolto)
python3 src/vllm_main.py
```

Il primo avvio:
1. Scaricherà il modello Qwen3.6-35B-A3B (~72 GB, varios minuti)
2. Avvierà vLLM server su port 8001
3. Caricherà il modello in VRAM
4. Compilerà i kernel TensorRT per YOLO

Log vLLM: `output/vllm_server.log`

### 9g. Verifica che vLLM sia pronto

```bash
# Check modelli disponibili
curl http://localhost:8001/v1/models

# Checkhealth
curl http://localhost:8001/health
```

### 9h. Output atteso

```
⏳ Initialising runtime (PyTorch + CUDA + TensorRT)… this takes 30-60 s on first launch.
...
2026-04-22 12:00:00 [info] [Step 1/3] YOLO OBB ready ✓
2026-04-22 12:00:05 [info] [Step 2/3] vLLM server ready ✓ url=http://localhost:8001/v1
2026-04-22 12:00:07 [info] [Warmup] All models warm and ready ✓
2026-04-22 12:00:07 [info] [Step 3/3] Starting batch processing
2026-04-22 12:00:08 [info] Image processed image=17.jpg total_ms=850.3 ocr_ms=420.1 ...
Pipeline complete avg_ms_per_image=870
```

**Target raggiunto**: ~850 ms/immagine con YOLO + OCR + I/O.

### 9i. Monitoraggio VRAM

```bash
watch -n 2 nvidia-smi
```

Allocazione tipica con vLLM su RTX 6000 PRO 96 GB:

| Componente | VRAM |
|-----------|------|
| Qwen3.6-35B-A3B BF16 | ~35 GB |
| YOLO OBB TensorRT | ~50 MB |
| KV cache vLLM | ~50 GB |
| Safety margin | ~10 GB |
| **Totale** | **~95 GB** ✅ |

---

## 10. Esecuzione della pipeline

### 10a. Scelta dello script

```bash
# ── TARGET < 1 s/immagine (RACCOMANDATO) ──
source .venv/bin/activate
pip install vllm>=0.19.0
python3 src/vllm_main.py

# ── llama.cpp (~1.3–2 s/immagine) ──
source .venv/bin/activate
python3 src/full-gpu_main.py
```

### 10b. Avvio in tmux (consigliato)

```bash
tmux new -s tosano
source .venv/bin/activate
python3 src/vllm_main.py   # oppure: python3 src/full-gpu_main.py
```

**Nota**: `vllm-server` viene avviato automaticamente se non è già in ascolto su port 8001 (vLLM) o 8080 (llama.cpp).

### 10c. Output atteso

```
output_test/
├── crops/
│   ├── crop_17.jpg          # etichetta croppata
│   ├── transcript_17.md      # trascrizione individuale (con timing)
│   └── ...
└── mocr_batch_results.md     # report con summary, timing statistics, dettagli
```

Il report `mocr_batch_results.md` include una sezione **Timing Summary** con avg/min/max per step (barcode, YOLO, crop, OCR) e tempo totale pipeline.

### 10d. Staccare e riattaccare tmux

```bash
# Staccare (Ctrl+B, poi D)
tmux attach -t tosano
tmux kill-session -t tosano
```

---

## 11. Recupero dei risultati

```bash
# Con rsync (dal laptop locale)
rsync -avz --progress \
    user@<cloud-host>:<path-to-project>/output_test/ \
    ./output_cloud/

# Solo il report riassuntivo
rsync -avz user@<cloud-host>:<path-to-project>/output_test/mocr_batch_results.md ./output_cloud/
```

---

## 12. Troubleshooting

### Errore: `nvcc fatal: Unsupported gpu architecture 'compute_120'`

**Causa**: CUDA Toolkit < 12.8.
**Soluzione**: Installare CUDA 13.0 (cfr. Sezione 3).

### Errore: `CUDA error: no kernel image is available for execution`

**Causa**: I kernel CUDA sono compilati per un'architettura GPU diversa.
**Soluzione**: Ricompilare llama.cpp con `CMAKE_CUDA_ARCHITECTURES=120`.

### Errore: `ImportError: Unable to find zbar shared library`

```bash
sudo apt install libzbar0 libzbar0-dev
pip uninstall pyzbar -y && pip install pyzbar==0.1.9
```

### Errore: Port 8001 (vLLM) o 8080 (llama-server) già occupato

```bash
# Trova e kill il processo
ps aux | grep -E "llama-server|vllm" | grep -v grep
sudo kill -9 <PID>

# Oppure kill tutti i processi
pkill -9 -f llama-server
pkill -9 -f vllm
```

### Errore: vLLM out of memory (OOM)

**Causa**: Troppi modelli o KV cache in VRAM.
**Soluzione**: Ridurre `--gpu-memory-utilization`:
```bash
# In .env:
VLLM_GPU_MEMORY_UTILIZATION=0.80
```

### Errore: vLLM non trova il modello

```bash
# Verifica che il modello sia scaricato
huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir ./model_cache
```

### Verifica finale della pipeline (sanity check)

```bash
# Verifica salute vLLM
curl http://localhost:8001/v1/models

# Verifica salute llama-server
curl http://localhost:8080/v1/models

# Verifica GPU visibile a Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## Checklist di deploy rapido (vLLM)

```bash
# === SUL SERVER CLOUD (RTX 6000 PRO Blackwell, 96 GB) ===

# 1. Pacchetti di sistema
sudo apt update && sudo apt install -y build-essential cmake git curl wget tmux \
    python3-venv python3-pip libzbar0 libgl1 ca-certificates pkg-config

# 2. CUDA Toolkit 13.0 (⚠️ NON 13.2 — causa output gibberish con Qwen3.6!)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-13-0
echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. Clone del progetto
git clone https://github.com/<org>/<repo>.git && cd <repo>

# 4. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install vllm>=0.19.0

# 5. Crea .env per vLLM
cat > .env << 'EOF'
VLLM_MODEL_REPO_ID=Qwen/Qwen3.6-35B-A3B
VLLM_PORT=8001
VLLM_TENSOR_PARALLEL_SIZE=1
ENABLE_EAN_DETECTION=true
SAVE_CROPS=false
YOLO_IMG_SIZE=1024
CROP_MAX_DIMENSION=1280
EOF

# 6. Esegui il pipeline vLLM! (target < 1 s/immagine)
tmux new -s tosano
source .venv/bin/activate
python3 src/vllm_main.py
```

---

*Documento generato per il deployment su RTX 6000 PRO Blackwell (sm_120, 96 GB VRAM).*
*Modello: `Qwen/Qwen3.6-35B-A3B` — vLLM >= 0.19.0 — HuggingFace model card: huggingface.co/Qwen/Qwen3.6-35B-A3B*