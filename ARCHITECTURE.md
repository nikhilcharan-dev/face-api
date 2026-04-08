# System Architecture: Face Recognition API

This document outlines the technical design, component interactions, and performance optimizations of the Face Recognition API.

## Design Philosophy
- **Performance First**: Native GPU acceleration and efficient FAISS indexing.
- **Persistent-by-design**: Automated volume-mapped persistence for indices and models.
- **Workflow-centric**: Integrated "Dump-to-Process" queue for batch updates.
- **High Observability**: Real-time B&W dashboard with live WebSocket streaming.

## System Components

### 1. API Layer (FastAPI)
Acts as the orchestration layer, handling requests for:
- **Detection**: Real-time inference using InsightFace.
- **Management (CRUD)**: Direct modification of the FAISS vector space.
- **Monitoring**: Live metrics streaming via WebSockets.

### 2. Inference Engine (InsightFace)
- **Model**: `AntelopeV2` (SCRFD for detection, ArcFace for embeddings).
- **Optimization**: Singleton pattern deployment with ONNX Runtime. Automatic CUDA detection for GPU offloading.
- **Latency**: Targeted sub-200ms end-to-end processing for 1080p frames.

### 3. Vector Database (FAISS)
- **Strategy**: `IndexFlatIP` (Inner Product) for high-precision similarity search with normalized embeddings.
- **Persistence**: Hybrid storage using `.bin` (FAISS) and `.npy` (Labels) files, synchronized via atomic disk writes.

### 4. Real-time Bridge (WebSockets)
- **Broadcasting**: Asynchronous event emitter pattern to notify dashboard clients of every inference and index update.
- **State**: In-memory `deque` buffers the last 50 transactions to minimize database overhead.

## Data Workflow
1. **Manual Seeding**: User drops images into `image_dump/`. Seeder script extracts embeddings, updates the index, and moves successful images to `dumped/`.
2. **Real-time Recognition**: Mobile client sends base64 image. API detects, searches FAISS, records match metrics, and broadcasts to the Dashboard.
3. **Control**: Admin uses `/faiss/add` or `/faiss/delete` to refine the identity database in real-time.

## Performance Analytics
- **GPU**: 5-10x throughput increase over CPU.
- **Index Search**: Sub-1ms for up to 10k identities (flat index).
- **Cold Boot**: Optimized via Docker volume mounting for models, bypassing redundant downloads.
