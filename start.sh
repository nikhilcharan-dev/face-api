#!/usr/bin/env bash
# start.sh — Launch face-api on port 5000 using the first MIG GPU slice

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

mkdir -p logs models/trt_cache

# ── Resolve CUDA/cuDNN libraries ──────────────────────────────────────────────
# onnxruntime-gpu bundles its own cuDNN/CUDA libs inside the venv.
# Add them to LD_LIBRARY_PATH so the TRT provider can find libcudnn.so.9.
ORT_LIBS="$(python -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")/capi"
if [ -d "$ORT_LIBS" ]; then
    export LD_LIBRARY_PATH="${ORT_LIBS}:${LD_LIBRARY_PATH:-}"
    echo "[start] Added ort libs to LD_LIBRARY_PATH: $ORT_LIBS"
fi

# Also add system TRT libs if present
if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

# Pick the first MIG UUID
MIG_UUID=$(nvidia-smi -L 2>/dev/null | grep -oP 'MIG-[0-9a-f\-]+' | head -1 || true)

if [ -z "$MIG_UUID" ]; then
    echo "[start] No MIG instance found — falling back to full GPU / CPU"
    CUDA_DEVICE="0"
else
    echo "[start] Using MIG slice: $MIG_UUID"
    CUDA_DEVICE="$MIG_UUID"
fi

# TRT 10.x installed via apt is built for CUDA 13.2 — incompatible with CUDA 12.8.
# Keep USE_TENSORRT=0 until a cuda-12 build of TRT is installed.
# Override with: USE_TENSORRT=1 ./start.sh
USE_TENSORRT="${USE_TENSORRT:-0}"

echo "[start] INFERENCE_WORKERS=${INFERENCE_WORKERS:-4}  USE_TENSORRT=${USE_TENSORRT}"
echo "[start] Starting uvicorn on 0.0.0.0:5000 ..."

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
INFERENCE_WORKERS="${INFERENCE_WORKERS:-4}" \
USE_TENSORRT="${USE_TENSORRT:-1}" \
OMP_NUM_THREADS=1 \
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --log-level info
