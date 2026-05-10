import os
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Directory that contains the models/ folder — works for both Docker and venv
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRT_CACHE = os.path.join(APP_ROOT, "models", "trt_cache")

_app = None  # singleton

def _build_providers():
    """
    Return ordered provider list: TensorRT → CUDA → CPU.

    TensorRT compiles the ONNX graph for the exact GPU on first run (slow,
    ~2–5 min) then caches the engine — subsequent starts are instant.
    Disable with USE_TENSORRT=0 if you need faster restarts during dev.
    """
    available = ort.get_available_providers()
    use_trt   = os.environ.get("USE_TENSORRT", "1") != "0"

    if "TensorrtExecutionProvider" in available and use_trt:
        os.makedirs(TRT_CACHE, exist_ok=True)
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path":   TRT_CACHE,
            "trt_fp16_enable":         True,
            "trt_max_workspace_size":  4 * 1024 ** 3,
        }
        print("TensorRT + CUDA + CPU providers active (FP16 on)")
        return [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    if "CUDAExecutionProvider" in available:
        print("CUDA + CPU providers active")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("CPU-only provider active (no GPU detected)")
    return ["CPUExecutionProvider"]


def is_gpu_available():
    return "CUDAExecutionProvider" in ort.get_available_providers()


def load_face_app():
    global _app
    if _app is None:
        providers = _build_providers()
        print("Loading InsightFace antelopev2 model (local, no download)...")

        _app = FaceAnalysis(
            name="antelopev2",
            root=APP_ROOT,
            providers=providers,
        )

        ctx_id = 0 if is_gpu_available() else -1
        _app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        print(f"InsightFace antelopev2 ready  gpu={is_gpu_available()}  trt={'TensorrtExecutionProvider' in ort.get_available_providers()}")
    return _app


def get_face_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    app = load_face_app()  # ensures model is loaded

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected")

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    emb = face.normed_embedding.astype("float32")
    emb /= np.linalg.norm(emb)
    return emb