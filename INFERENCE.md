# Face Recognition Inference — Model, Queue & Performance

## Model: InsightFace AntelopeV2

### What it is

AntelopeV2 is a two-stage face recognition pipeline shipped as a bundle of ONNX models:

| Stage | Model | Purpose |
|---|---|---|
| Detection | SCRFD-10G | Locates face bounding boxes + landmarks |
| Recognition | ArcFace-R100 (ResNet-100) | Extracts 512-D identity embedding |

Both models run through **ONNX Runtime**, which means the same weights run on CPU or GPU without code changes — the execution provider switches automatically.

### Why AntelopeV2 over alternatives

| Criteria | AntelopeV2 | FaceNet | DeepFace | dlib |
|---|---|---|---|---|
| Accuracy (LFW) | 99.77% | 99.65% | 99.30% | 99.38% |
| Embedding dim | 512 | 128–512 | 128–2622 | 128 |
| GPU inference | ONNX/CUDA | TF/PyTorch | TF/PyTorch | CPU only |
| Detection bundled | Yes (SCRFD) | No | Yes | Yes |
| Deployment size | ~300 MB | ~90 MB | ~500 MB+ | ~100 MB |

**Key reasons this model was chosen:**

1. **ONNX native** — no PyTorch or TensorFlow runtime needed in the container. ONNX Runtime is lighter and has first-class CUDA/TensorRT support.
2. **SCRFD detection is fast** — SCRFD-10G runs at ~1 ms/image on GPU, far below MTCNN or RetinaFace at equivalent accuracy.
3. **ArcFace-R100 embeddings are robust** — trained with additive angular margin loss, which produces tightly clustered same-identity embeddings and wide inter-identity separation. The 512-D space makes cosine similarity both fast and reliable.
4. **Local model files** — models are volume-mounted from `./models/`, so no download happens at startup and air-gapped deployments work out of the box.

---

## How inference works

```
Mobile client
    │
    │  POST /api/v1/detect
    │  { image_base64, user_id, organization_id }
    ▼
FastAPI (uvicorn, single process)
    │
    ├─ Decodes request
    ├─ Creates ticket_id (UUID)
    ├─ Stores ticket:{id} = {status: "queued"} in Redis (TTL 300s)
    ├─ Pushes task payload to Redis list "face_tasks_queue" (RPUSH)
    └─ Returns { status: "queued", ticket_id, position }

Client polls GET /api/v1/ticket/{ticket_id}
    │
    │  (every 100–300 ms)
    ▼
Redis
    └─ Returns current ticket status

Background inference_worker (N workers, one per CPU core)
    │
    ├─ BLPOP "face_tasks_queue"  ← blocks until task arrives
    ├─ Sets ticket status = "processing"
    │
    ├─ [Cache] GET face:user_emb:{user_id} from Redis
    │       hit  → decode binary float32 array
    │       miss → find_one from MongoDB, cache for 300s
    │
    ├─ [GPU Thread] run_in_executor(ThreadPoolExecutor)
    │       ├─ base64 decode image bytes
    │       ├─ cv2.imdecode → BGR image
    │       ├─ SCRFD → detect faces → pick largest by bounding-box area
    │       ├─ ArcFace → 512-D L2-normalised embedding
    │       └─ cosine_similarity = dot(query_emb, db_emb)
    │
    ├─ success = similarity >= 0.6
    ├─ Writes result into Redis ticket (TTL 300s)
    └─ Broadcasts NEW_ENTRY to WebSocket dashboard
```

### Similarity threshold

The default threshold is **0.6** (cosine similarity on L2-normalised vectors).

- `< 0.5` — almost certainly different people
- `0.5–0.6` — ambiguous; potential false positive under poor lighting
- `>= 0.6` — confident match (tuned against LFW benchmark)
- `>= 0.8` — near-identical image (same session, good lighting)

The threshold can be changed by editing `THRESHOLD = 0.6` in `app.py`.

---

## Queue architecture

### Why a queue instead of direct inference

Face embedding takes 100–4000 ms depending on hardware. If the API called inference synchronously, every HTTP connection would hold open for that duration. Under concurrent load this exhausts the connection pool and stalls the entire server.

The Redis queue decouples HTTP from GPU:

```
HTTP tier (instant)          GPU tier (async)
─────────────────            ─────────────────
POST /detect → 202           Worker 1 ─┐
POST /detect → 202           Worker 2 ─┤─ face_tasks_queue (Redis LIST)
POST /detect → 202           Worker N ─┘
```

Each worker does `BLPOP face_tasks_queue` — a blocking pop that yields to the asyncio event loop while waiting, so it never wastes a thread.

### Concurrency model

```
uvicorn (1 process)
└── asyncio event loop
    ├── FastAPI request handlers  (non-blocking, instant response)
    ├── inference_worker × N      (one per CPU core; BLPOP → run_in_executor)
    └── ThreadPoolExecutor × N   (OS threads; each runs ONNX GPU inference)
```

`N = os.cpu_count()` by default, overridable with `INFERENCE_WORKERS` env var.

Why one uvicorn process instead of `--workers 4`:
- A single process = one model loaded into GPU VRAM once
- Multiple processes each load the model separately → wastes GPU memory
- On a DGX B200 (192 GB HBM3e), memory isn't the constraint, but a single large batch on one context outperforms four small batches on four contexts for the same total throughput

### Redis keys

| Key pattern | Type | TTL | Purpose |
|---|---|---|---|
| `face_tasks_queue` | LIST | — | Task queue (RPUSH / BLPOP) |
| `ticket:{uuid}` | STRING | 300s | Result polling |
| `face:user_emb:{user_id}` | STRING | 300s | Per-user embedding cache (binary) |
| `face:embeddings:{org_id}` | STRING | 300s | Per-org full embedding list cache |

---

## Performance — measured load test results

Test setup: image 125 KB JPEG, user `imran.khan@workping.live`, DGX B200 via proxy, 30 requests per concurrency level.

**Current deployment (CPU — `gpu_available: false`)**

| Concurrency | Throughput | p50 total | p95 total | p50 embedding | Match rate |
|---|---|---|---|---|---|
| 5 | 1.67 req/s | 3 415 ms | 4 693 ms | 2 870 ms | 30/30 |
| 10 | 2.12 req/s | 4 029 ms | 6 978 ms | 885 ms* | 30/30 |
| 20 | 1.45 req/s | 11 171 ms | 13 965 ms | 3 763 ms | 30/30 |
| 50 | 1.51 req/s | 12 047 ms | 19 414 ms | 3 641 ms | 30/30 |

\* Redis cache warmed mid-run at c=10 — subsequent requests skipped MongoDB fetch, making poll resolution faster.

**Expected after GPU fix (`onnxruntime-gpu` + CUDA base image)**

| Concurrency | Throughput (est.) | p50 total (est.) | p50 embedding (est.) |
|---|---|---|---|
| 5 | ~8 req/s | ~300 ms | ~150 ms |
| 10 | ~15 req/s | ~450 ms | ~150 ms |
| 20 | ~25 req/s | ~700 ms | ~150 ms |
| N (saturated GPU) | ~40–60 req/s | ~800 ms | ~150 ms |

Confidence was stable at **0.814** across all runs and all concurrency levels, confirming the similarity score is deterministic (same image → same embedding every time).

---

## CPU-only fallback

**Yes — the service works on CPU-only VMs with zero code changes.**

In `embedding.py`:

```python
def is_gpu_available():
    return "CUDAExecutionProvider" in ort.get_available_providers()

def load_face_app():
    providers = ["CPUExecutionProvider"]
    if is_gpu_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ...
    _app.prepare(ctx_id=0 if is_gpu_available() else -1, ...)
```

ONNX Runtime's `get_available_providers()` lists what the current environment supports. If CUDA is not present, `CUDAExecutionProvider` is absent and the model silently falls back to `CPUExecutionProvider`.

The health endpoint reports the current mode:

```
GET /api/v1/health
→ { "gpu_available": false, ... }   # CPU-only VM
→ { "gpu_available": true,  ... }   # GPU node
```

**Performance impact of CPU fallback:**

| | GPU (B200) | CPU (general VM) |
|---|---|---|
| Embedding time | ~150 ms | ~2 500–4 000 ms |
| Throughput | ~40–60 req/s | ~1–2 req/s |
| Accuracy | Identical | Identical |

The accuracy is identical because the same ONNX weights are used — only the execution hardware changes. CPU-only is suitable for low-traffic development or staging environments; production attendance workloads need a GPU node.

**What to check if GPU is expected but `gpu_available` is false:**

1. Container does not have GPU access — verify `docker run --gpus all` or the `deploy.resources` block in `docker-compose.yaml`
2. `onnxruntime` (CPU-only package) is installed instead of `onnxruntime-gpu` — check `pip list | grep onnxruntime`
3. NVIDIA driver not installed on the host or CUDA version mismatch — run `nvidia-smi` on the host; the driver version must be >= the CUDA version the image was built against (12.3 in the current Dockerfile)
4. NVIDIA Container Toolkit not installed — required for Docker to expose GPUs; install with `apt install nvidia-container-toolkit`
