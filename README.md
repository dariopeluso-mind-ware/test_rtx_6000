# 1_Riprezzamento-con-foto — Deployment Guide

> Pipeline per l'estrazione automatica di etichette alimentari da fotografie:
> **YOLO OBB** (crop etichetta) → **Qwen3.6‑35B‑A3B** (OCR vision via llama‑server)
> → dati strutturati. Tempo totale di processing < 1 s su hardware production (32 GB VRAM).

## Hardware di riferimento

| Ambiente | GPU | VRAM | Note |
| ---------- | ----- | ------ | ------ |
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
6. [Build di llama.cpp con CUDA](#6-build-di-llamacpp-con-cuda)
7. [Pre-download dei pesi GGUF](#7-pre-download-dei-pesi-gguf)
8. [Esecuzione Full-GPU (96 GB VRAM)](#8-esecuzione-full-gpu-96-gb-vram)
9. [Esecuzione della pipeline](#9-esecuzione-della-pipeline)
10. [Recupero dei risultati](#10-recupero-dei-risultati)
11. [Troubleshooting](#11-troubleshooting)

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

# Oppure CUDA 12.8 (più stabile, supporta comunque sm_120)
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

# Verifica supporto sm_120
cat $CUDA_HOME/include/cuda_arch_sm.h 2>/dev/null | grep -c "sm_120" || echo "sm_120 supportato da nvcc 13.0"
```

Se `nvcc --version` mostra una versione inferiore a 12.8, verificare che `$CUDA_HOME/bin/nvcc` sia quello corretto:

```bash
which nvcc
/usr/local/cuda/bin/nvcc --version   # verificare che non sia un vecchio link
/usr/local/cuda-13.0/bin/nvcc --version  # deve mostrare 13.0
```

---

## 4. Setup del progetto

Clonare il repository sulla macchina remota:

```bash
# Scegli la directory di lavoro (es. /workspace su RunPod, ~ su server dedicati)
cd /workspace   # oppure: cd ~

git clone https://github.com/<org>/<repo>.git
cd <repo>       # entrare nella cartella del progetto
```

> In tutto il resto di questa guida, i comandi assumono che la working directory
> sia la root del progetto (la cartella in cui si trova `requirements.txt`).

Struttura dei file:

```txt
<repo>/
├── best.onnx              # YOLO OBB model (~38 MB)
├── best.pt               # YOLO PyTorch model (~21 MB) — usato per export TensorRT
├── best.engine            # YOLO TensorRT FP16 (generato al primo avvio di full-gpu_main.py)
├── src/
│   ├── main.py            # Script dev (8 GB VRAM, --cpu-moe, ONNX CPU)
│   └── full-gpu_main.py   # Script prod (96 GB VRAM, full GPU, TensorRT, thinking OFF)
├── test/                  # Immagini di input per batch processing
├── llama.cpp/             # Repository llama.cpp (source)
├── requirements.txt       # Dipendenze Python
└── output/                # Output processing (generato)
```

**Nota**: la directory `llama.cpp/` è un repository standalone
(`https://github.com/ggml-org/llama.cpp`, commit verificato `1f30ac0ce`).

Verificare che i file essenziali siano presenti:

```bash
ls -lh best.onnx best.pt src/main.py requirements.txt
# Tutti i file devono essere presenti
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
    ultralytics==8.4.37 \
    httpx==0.28.1 \
    pyzbar==0.1.9 \
    pillow==12.2.0 \
    structlog==25.5.0 \
    huggingface_hub==0.36.2 \
    onnxruntime-gpu==1.23.2 \
    opencv-python-headless==4.10.0.84 \
    numpy>=1.24.0 \
    pyzbar-utils==0.1.1
```

**Note importanti**:

- **`opencv-python-headless`**: versione headless (senza GUI), necessaria su server headless. Non installare `opencv-python`.
- **`onnxruntime-gpu`**: usa la GPU per inferenza ONNX. Se preferisci CPU-only, usa `onnxruntime`.
- **`pyzbar`**: richiede `libzbar0` installato a livello sistema (cfr. Sezione 2).

### Verifica dell'installazione

```bash
python -c "import ultralytics; import httpx; import pyzbar; import PIL; import structlog; print('✓ Tutte le dipendenze importabili')"
```

Se `pyzbar` dà errore "Unable to find zbar shared library", reinstallare:

```bash
sudo apt install libzbar0-dev
pip uninstall pyzbar -y && pip install pyzbar==0.1.9
```

---

## 6. Build di llama.cpp con CUDA

### 6a. Configurazione CMake

```bash
# Dalla root del progetto
cd llama.cpp

# Verifica che CUDA sia nel PATH
echo $CUDA_HOME
# Deve mostrare: /usr/local/cuda-13.0 (o /usr/local/cuda-12.8)

cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DLLAMA_SERVER_VERBOSE=OFF
```

**Significato dei flag**:

| Flag | Valore | Descrizione |
| ------ | -------- | ------------- |
| `GGML_CUDA=ON` | Abilita | Abilita il supporto CUDA per GGML |
| `CMAKE_CUDA_ARCHITECTURES=120` | sm_120 | Compila kernel per Blackwell (RTX 6000 PRO / RTX PRO 1000) |
| `CMAKE_BUILD_TYPE=Release` | Ottimizzato | Flag `-O3`, nessun debug symbol |
| `LLAMA_CURL=OFF` | Disabilitato | Disabilita download interno (usiamo huggingface_hub) |

### 6b. Compilazione

```bash
cmake --build build --config Release -j$(nproc) --target llama-server
```

Sul cloud con 96 GB VRAM e 32+ core CPU, usa parallelism massimo:

```bash
# Con 32 thread
cmake --build build --config Release -j32 --target llama-server
```

Tempo stimato: **5–15 minuti** su una macchina cloud con 32+ core.

### 6c. Copia del binario

```bash
cp build/bin/llama-server .
ls -lh llama-server
# Verifica: file eseguibile, dimensione plausibile (200–500 MB range linker)
```

### 6d. Test rapido del binario

```bash
./llama-server --help 2>&1 | head -20
# Deve mostrare usage info senza errori CUDA
```

---

## 7. Pre-download dei pesi GGUF

I pesi vengono scaricati automaticamente da `src/main.py` alla prima esecuzione tramite `huggingface_hub`.
Per evitare attese durante il primo run, pre-scaricarli:

```bash
# Accesso a HuggingFace (richiede account + token se il modello è gated)
huggingface-cli login
# Inserire HF_TOKEN (https://huggingface.co/settings/tokens)

# Oppure: esportare il token come variabile d'ambiente
export HF_TOKEN=hf_your_token_here

# Download del modello principale (Q4_K_M quantization)
huggingface-cli download \
    --token $HF_TOKEN \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# Download del vision encoder (mmproj)
huggingface-cli download \
    --token $HF_TOKEN \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    mmproj-F16.gguf
```

**Dimensione stimata**: ~20 GB totali (modello + mmproj).

**Nota**: se il modello è gated (richiede accettazione licenza su HF), accettare prima su <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF>.

### Cache location

I file vengono memorizzati nella cache HuggingFace:

```bash
ls ~/.cache/huggingface/hub/
# Cerca: models--unsloth--Qwen3.6-35B-A3B-GGUF
```

Il path usato da `hf_hub_download` è trasparente, `main.py` lo gestisce automaticamente.

---

## 8. Esecuzione Full-GPU (96 GB VRAM)

Con 96 GB VRAM, usare `src/full-gpu_main.py` al posto di `src/main.py`. Questo script è
ottimizzato per massima velocità e include tutte le configurazioni GPU automaticamente.

### 8a. Differenze rispetto a `main.py`

| Feature | `main.py` (dev, 8 GB) | `full-gpu_main.py` (prod, 96 GB) |
| --------- | ---------------------- | ---------------------------------- |
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

### 8b. Variabili d'ambiente

```bash
# Scegli quale quantizzazione usare (default: UD-Q4_K_XL)
export QWEN_GGUF_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"   # Unsloth Dynamic 2.0 (raccomandato)
# export QWEN_GGUF_FILE="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"  # Standard Q4_K_M (alternativa)

# Opzionale: disabilitare ricerca EAN barcode (risparmio ~5-10 ms/immagine)
# export ENABLE_EAN_DETECTION=false
```

### 8c. Disabilitare la ricerca barcode (EAN)

Per default, la pipeline cerca codici EAN/UPC in ogni immagine tramite `pyzbar`.
Se non ti interessa il codice EAN (es. stai solo facendo OCR delle etichette),
puoi disabilitare questa fase per risparmiare ~5-10 ms per immagine:

```bash
# Disabilita EAN barcode detection
export ENABLE_EAN_DETECTION=false
python3 src/full-gpu_main.py

# Oppure inline:
ENABLE_EAN_DETECTION=false python3 src/full-gpu_main.py
```

Quando disabilitato:

- Il campo `EAN` nel report sarà sempre `None`
- La rotazione barcode-based sarà sempre `0°`
- Lo step "Barcode" nel timing sarà ~0 ms

Per riabilitarlo, basta non impostare la variabile o impostarla a `true`:

```bash
export ENABLE_EAN_DETECTION=true   # o semplicemente non impostarla
```

### 8d. Note importanti

- Il modello GGUF viene scaricato automaticamente al primo avvio (~22 GB, potrebbe richiedere diversi minuti)
- L'engine TensorRT (`best.engine`) viene generato automaticamente al primo avvio da `best.pt` (~1-2 min)
- L'engine TensorRT è specifico per la GPU: se cambi GPU, cancella `best.engine` e riesegui
- Il thinking mode è disabilitato tramite `--chat-template-kwargs '{"enable_thinking":false}'`
  come da [documentazione ufficiale Unsloth](https://unsloth.ai/docs/models/qwen3.6#how-to-enable-or-disable-thinking)

### 8e. Verifica carico VRAM

Dopo aver avviato lo script, monitora la VRAM:

```bash
watch -n 2 nvidia-smi
# Verifica che la memoria allocata sia ~30-33 GB (modello + KV cache + YOLO)
# Deve stare comodamente nei 96 GB disponibili
```

---

## 9. Esecuzione della pipeline

### 9a. Avvio in tmux (consigliato)

```bash
# Dalla root del progetto

# Crea una nuova sessione tmux
tmux new -s tosano

# Attiva il venv
source .venv/bin/activate

# --- PRODUCTION (96 GB VRAM) ---
python3 src/full-gpu_main.py

# --- DEVELOPMENT (8 GB VRAM) ---
# python3 src/main.py
```

**Nota**: `llama-server` viene avviato automaticamente se non è già in ascolto su port 8080.
Il primo avvio scaricherà il modello GGUF (~22 GB) e genererà l'engine TensorRT (~1-2 min).
Log disponibile in: `output/llama_server.log`.

### 9a.1 Test A/B quantizzazione

```bash
# Test con UD-Q4_K_XL (default)
QWEN_GGUF_FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf python3 src/full-gpu_main.py

# Test con Q4_K_M standard
QWEN_GGUF_FILE=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf python3 src/full-gpu_main.py

# Confronta i timing nei report generati:
diff output_test/mocr_batch_results.md output_test_q4km/mocr_batch_results.md
```

### 9b. Monitoraggio in tempo reale

```bash
# In un altro pannello tmux, monitora i log
tail -f output/llama_server.log

# Oppure monitora GPU + processo
watch -n 1 "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader && echo '---' && ps aux | grep llama-server | grep -v grep"
```

### 9c. Output atteso

Dopo l'esecuzione:

```txt
output_test/
├── crops/
│   ├── crop_17.jpg          # etichetta croppata
│   ├── transcript_17.md      # trascrizione individuale (con timing)
│   └── ...
└── mocr_batch_results.md     # report con summary, timing statistics, dettagli
```

Il report `mocr_batch_results.md` include una sezione **Timing Summary** con avg/min/max
per step (barcode, YOLO, crop, OCR) e tempo totale pipeline.

### 9d. Staccare e riattaccare tmux

```bash
# Staccare (Ctrl+B, poi D)
# Per riattaccare:
tmux attach -t tosano

# Per killare la sessione:
tmux kill-session -t tosano
```

### 9e. Come servizio systemd (opzionale)

Creare `/etc/systemd/system/tosano-label.service`:

```ini
[Unit]
Description=Tosano Food Label Processing Pipeline
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=<path-to-project>
ExecStart=<path-to-project>/.venv/bin/python src/full-gpu_main.py
Restart=on-failure
RestartSec=30
StandardOutput=append:<path-to-project>/output/service.log
StandardError=append:<path-to-project>/output/service_error.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tosano-label
sudo systemctl start tosano-label
sudo systemctl status tosano-label
```

---

## 10. Recupero dei risultati

I risultati vengono salvati in `output_test/`. Per sincronizzarli sulla macchina locale:

```bash
# Con rsync (dal laptop locale) — adattare il percorso al proprio setup
rsync -avz --progress \
    user@<cloud-host>:<path-to-project>/output_test/ \
    ./output_cloud/

# Solo il report riassuntivo
rsync -avz user@<cloud-host>:<path-to-project>/output_test/mocr_batch_results.md ./output_cloud/
```

---

## 11. Troubleshooting

### Errore: `nvcc fatal: Unsupported gpu architecture 'compute_120'`

**Causa**: CUDA Toolkit < 12.8. Il nvcc usato è troppo vecchio.
**Soluzione**: Installare CUDA 13.0 (cfr. Sezione 3) e verificare che `$CUDA_HOME/bin/nvcc` punti alla versione corretta:

```bash
/usr/local/cuda-13.0/bin/nvcc --version  # deve mostrare 13.0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
```

### Errore: `CUDA error: no kernel image is available for execution`

**Causa**: I kernel CUDA sono compilati per un'architettura GPU diversa (es. sm_89 per RTX 4090).
**Soluzione**: Ricompilare llama.cpp con `CMAKE_CUDA_ARCHITECTURES=120`:

```bash
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
cp build/bin/llama-server .
```

### Errore: `ImportError: Unable to find zbar shared library`

**Causa**: `libzbar0` non installato.
**Soluzione**:

```bash
sudo apt install libzbar0 libzbar0-dev
pip uninstall pyzbar -y && pip install pyzbar==0.1.9
```

### Errore: `libGL.so.1: cannot open shared object file`

**Causa**: `libgl1` (o `libgl1` su Ubuntu 22.04) non installato (necessario per OpenCV).
**Soluzione**:

```bash
sudo apt install libgl1 libglib2.0-0
```

### Errore: Port 8080 già occupato (llama-server zombie)

**Causa**: Un processo `llama-server` precedente è rimasto attivo dopo un SSH disconnect.
**Soluzione**:

```bash
# Trova e kill il processo
ps aux | grep llama-server | grep -v grep
sudo kill -9 <PID>

# Oppure kill tutti i processi llama-server
pkill -9 -f llama-server
```

### Errore: `hf_hub_download` restituisce 401/403

**Causa**: Modello gated su HuggingFace (richiede accettazione licenza).
**Soluzione**:

1. Accedere a <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF> e accettare la licenza
2. Generare un HF_TOKEN su <https://huggingface.co/settings/tokens>
3. Eseguire `huggingface-cli login` sul server, oppure:

```bash
export HF_TOKEN=hf_your_token_here
python3 src/main.py
```

### Errore: `Segmentation fault` in llama-server

**Causa**: Spesso driver GPU + CUDA incompatibili, o kernel non compilati per sm_120.
**Soluzione**: Verificare driver NVIDIA recenti e ricompilare con CUDA 13.0:

```bash
nvidia-smi | head -5
# Verifica: driver ≥ 550 per Blackwell
sudo apt install nvidia-driver-560
```

### Errore: YOLO OBB non trova etichette

**Causa**: Il modello `best.onnx` funziona meglio con immagini chiare, illuminazione uniforme.
**Soluzione**: Verificare che le immagini in `etichette_esempio/` siano quelle corrette. Se necessario, usare il modello `.pt` al posto dell'`.onnx`:

```python
# In src/main.py, riga 286:
model = YOLO("best.onnx", task="obb")  # cambiare in:
model = YOLO("best.pt", task="obb")
```

### Verifica finale della pipeline (sanity check)

```bash
# Verifica salute llama-server
curl http://localhost:8080/v1/models

# Verifica risposta minima
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-35B", "messages": [{"role": "user", "content": "Ciao"}], "max_tokens": 10}'

# Verifica GPU visibile a Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## Checklist di deploy rapido

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

# 3. Git pull del progetto
# git pull (già configurato)

# 4. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Build llama.cpp (con supporto Flash Attention e Blackwell sm_120)
cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc) \
       --target llama-server llama-cli \
    && cp build/bin/llama-server . && cd ..

# 6. Esegui il pipeline full-GPU!
#    - Download GGUF (~22 GB) avviene automaticamente al primo avvio
#    - Export TensorRT (~1-2 min) avviene automaticamente al primo avvio
tmux new -s tosano
source .venv/bin/activate
python3 src/full-gpu_main.py

# 7. (Opzionale) Test A/B con quantizzazione diversa
# QWEN_GGUF_FILE=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf python3 src/full-gpu_main.py
```

---

*Documento generato per il deployment su RTX 6000 PRO Blackwell (sm_120, 96 GB VRAM).*
*Commit verificato di llama.cpp: `1f30ac0ce` — <https://github.com/ggml-org/llama.cpp/commit/1f30ac0ce>*
