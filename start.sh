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

# Pick the first MIG UUID
MIG_UUID=$(nvidia-smi -L 2>/dev/null | grep -oP 'MIG-[0-9a-f\-]+' | head -1 || true)

if [ -z "$MIG_UUID" ]; then
    echo "[start] No MIG instance found — falling back to full GPU / CPU"
    CUDA_DEVICE="0"
else
    echo "[start] Using MIG slice: $MIG_UUID"
    CUDA_DEVICE="$MIG_UUID"
fi

echo "[start] INFERENCE_WORKERS=${INFERENCE_WORKERS:-4}  USE_TENSORRT=${USE_TENSORRT:-1}"
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
