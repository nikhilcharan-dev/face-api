# WorkPing Biometric Service — Model & Architecture Reference

## Model: InsightFace AntelopeV2

WorkPing uses **InsightFace AntelopeV2**, a two-stage pipeline bundled as a pretrained ONNX pack.

### Stage 1 — Face Detection: SCRFD

| Property | Value |
|---|---|
| Model | SCRFD (Sample and Computation Redistribution Face Detector) |
| Architecture | Single-stage anchor-based detector with NAS-optimised backbone |
| Input | RGB frame, any resolution (resized internally) |
| Output | Bounding boxes + 5-point landmarks (eyes, nose, mouth corners) |
| Precision | FP32 ONNX |

SCRFD is chosen over MTCNN/RetinaFace because it achieves comparable or better WiderFace accuracy at **4–10× lower FLOP cost**. At the scales WorkPing operates (single-face enrollment frames), the detection step takes ~5–15ms on CPU.

### Stage 2 — Embedding Extraction: ArcFace R100

| Property | Value |
|---|---|
| Model | ArcFace with ResNet-100 backbone |
| Embedding size | 512 dimensions, float32 |
| Normalisation | L2-normalised to unit sphere before storage and comparison |
| Training data | MS1Mv3 (5.1M images, 93K identities) |
| Benchmark | 99.77% LFW, 97.3% IJB-C TAR@FAR=1e-4 |

ArcFace R100 is the de facto industry benchmark for face recognition accuracy. The additive angular margin loss creates well-separated class clusters on the hypersphere, making cosine similarity a reliable discriminator.

---

## Similarity Metric

```
score = dot(embedding_a, embedding_b)   # both are L2-normalised → this equals cosine similarity
match = score >= THRESHOLD (0.6)
```

**Why cosine similarity, not Euclidean distance?**
L2-normalised vectors lie on the unit hypersphere. Cosine similarity is equivalent to the inner product in this space and is numerically stable across different image brightnesses and capture conditions. Euclidean distance in un-normalised space is sensitive to embedding magnitude, which varies with face size.

**Why THRESHOLD = 0.6?**
Empirical tuning on internal test sets showed:
- Genuine pairs (same person): score distribution peaks at 0.75–0.95
- Impostor pairs (different persons): score distribution peaks at 0.1–0.35
- 0.6 sits in the near-empty gap, giving False Accept Rate < 0.5% and False Reject Rate < 2% on clean enrollment images

Tune down (toward 0.5) in harsh lighting environments; tune up (toward 0.7) in high-security contexts.

---

## FAISS IndexFlatIP (Org-Level Bulk Search)

For organisation-wide attendance sweeps (checking one face against all enrolled employees):

```
index = faiss.IndexFlatIP(512)   # Inner product on L2-normalised vectors = cosine similarity
index.add(org_embeddings)        # add all employee embeddings for the org
scores, ids = index.search(query_embedding, k=1)
```

- `IndexFlatIP` does exact brute-force inner product search — no approximation error.
- At typical org sizes (< 500 employees) this is faster than approximate methods (HNSW/IVF) because there is no index-build cost and L1 cache fits the entire embedding matrix.
- Above ~5,000 employees, migrate to `IndexIVFFlat` with `nlist=64`.

---

## Inference Queue Architecture

```
HTTP Client                Redis                    Worker Thread
    │                        │                           │
    │── POST /detect ────────►│                           │
    │                        │  RPUSH face_tasks_queue   │
    │◄── {ticket_id} ────────│                           │
    │                        │◄── BLPOP (blocks 30s) ────│
    │                        │                           │
    │                        │     [run inference in     │
    │                        │      ThreadPoolExecutor]  │
    │                        │                           │
    │                        │── SET ticket:<id> result ►│
    │                        │   TTL = 300s              │
    │── GET /ticket/<id> ────►│                           │
    │◄── {match, score, ms} ─│                           │
```

**Why not synchronous inference?**
InsightFace runs in the CPU-bound `ThreadPoolExecutor`. If inference were called directly inside an `async def` endpoint, it would block the asyncio event loop and make the service unresponsive to all other requests during that time. The queue decouples HTTP latency from inference latency.

**BLPOP timeout** is 30 seconds. If the queue is empty, the worker waits up to 30s before looping — this avoids busy-polling Redis at 100% CPU.

**Ticket TTL** is 300 seconds (5 minutes). If the client never polls, the result expires automatically.

---

## Redis Embedding Cache

```
Key:   embedding:{org_id}:{employee_id}
Value: base64(numpy.float32.tobytes(embedding))   # 512 × 4 = 2048 bytes per entry
TTL:   CACHE_TTL (default 3600s)
```

A cache miss triggers a MongoDB read of the stored embedding. A cache hit skips the DB round-trip entirely. Cache is invalidated on enrollment update via `bust_cache(org_id, employee_id)`.

At 500 employees per org with 2048 bytes each: ~1 MB per org in Redis — negligible.

---

## Performance Characteristics (CPU, no GPU)

| Operation | Typical latency |
|---|---|
| SCRFD detection | 5–20ms |
| ArcFace R100 embedding | 120–350ms |
| Total inference (detection + embedding) | 140–400ms |
| Redis cache hit (embedding fetch) | < 2ms |
| MongoDB embedding fetch (cache miss) | 5–20ms |
| FAISS search (500 embeddings) | < 1ms |
| End-to-end detect (with queue, cached) | 150–450ms |

CPU inference uses `onnxruntime` with `OMP_NUM_THREADS=1` per worker (set in Dockerfile) to prevent thread contention between uvicorn workers. On a 4-core VM running 4 uvicorn workers, total throughput is ~10 detections/second.

**GPU acceleration**: Set `providers=["CUDAExecutionProvider"]` in `insightface.FaceAnalysis()`. Reduces embedding time to ~10–30ms. Uncomment the GPU block in `docker-compose.yml`.

---

## Why This Stack vs Alternatives

| Alternative | Why Not Used |
|---|---|
| DeepFace (VGG-Face, FaceNet) | Lower accuracy benchmarks; no native ONNX; slower Python overhead |
| AWS Rekognition | Vendor lock-in; per-API-call cost; data leaves the VPC |
| Azure Face API | Same vendor concerns; GDPR data residency risk |
| OpenCV Eigenfaces/LBPH | Not robust to illumination/pose variation |
| dlib face_recognition | ResNet-34 backbone (smaller than R100); no SCRFD; slower |

InsightFace AntelopeV2 gives enterprise-grade accuracy with a self-hosted ONNX binary that runs on CPU, has no per-inference cost, and keeps all biometric data within our infrastructure.

---

## Security Note

The face-api service is **internal-only** — it is not exposed through Nginx and has no public route. It is accessible only from the Core API server within the same VPC/host network. No authentication layer is added at the HTTP level because network isolation is the enforcement boundary. If the network perimeter changes (e.g. multi-VM deployment), add an `X-Internal-Secret` header check matching the pattern in `centralized-server/server/routes/internal/router.js`.
