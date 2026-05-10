import os
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Directory that contains the models/ folder — works for both Docker and venv
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRT_CACHE = os.path.join(APP_ROOT, "models", "trt_cache")

_app = None             # singleton
_active_provider = "cpu"  # set after load_face_app() succeeds

def _trt_libs_present() -> bool:
    """
    Check libnvinfer is loadable AND compatible with the running CUDA driver.
    A mismatch (e.g. TRT built for CUDA 13.2 on a CUDA 12.8 driver) causes
    a segfault inside TRT's createInferRuntime — we probe with a subprocess
    so a crash there doesn't take down the main process.
    """
    import ctypes, subprocess, sys
    # Quick lib-load check first
    found = False
    for lib in ("libnvinfer.so.10", "libnvinfer.so.8"):
        try:
            ctypes.CDLL(lib)
            found = True
            break
        except OSError:
            pass
    if not found:
        return False

    # Validate TRT can actually init CUDA without crashing
    probe = (
        "import onnxruntime as ort; "
        "s = ort.InferenceSession.__new__(ort.InferenceSession); "
        "print('ok')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            timeout=10,
            capture_output=True,
            env={**os.environ, "ORT_TENSORRT_VALIDATION": "1"},
        )
        return result.returncode == 0
    except Exception:
        return False


def _build_providers():
    """
    Return ordered provider list: TensorRT → CUDA → CPU.
    TRT is only included when libnvinfer is actually present on the system.
    """
    available = ort.get_available_providers()
    use_trt   = (
        os.environ.get("USE_TENSORRT", "1") != "0"
        and "TensorrtExecutionProvider" in available
        and _trt_libs_present()
    )

    if use_trt:
        os.makedirs(TRT_CACHE, exist_ok=True)
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path":   TRT_CACHE,
            "trt_fp16_enable":         True,
            "trt_max_workspace_size":  4 * 1024 ** 3,
        }
        print("Provider: TensorRT + CUDA + CPU  (FP16 on)")
        return [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    if "CUDAExecutionProvider" in available:
        print("Provider: CUDA + CPU")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("Provider: CPU only")
    return ["CPUExecutionProvider"]


def is_gpu_available() -> bool:
    """True only when models actually loaded onto GPU (not just ort provider list)."""
    return _active_provider in ("cuda", "tensorrt")


def load_face_app():
    global _app, _active_provider
    if _app is None:
        providers = _build_providers()
        print("Loading InsightFace antelopev2 model (local, no download)...")

        _app = FaceAnalysis(
            name="antelopev2",
            root=APP_ROOT,
            providers=providers,
        )

        # ctx_id=0 → GPU, -1 → CPU
        available = ort.get_available_providers()
        on_gpu = "CUDAExecutionProvider" in available
        _app.prepare(ctx_id=0 if on_gpu else -1, det_size=(640, 640))

        # Detect which provider the first model actually loaded with
        first_session = next(iter(_app.models.values())).session
        loaded_providers = [p.lower() for p in first_session.get_providers()]
        if "tensorrtexecutionprovider" in loaded_providers:
            _active_provider = "tensorrt"
        elif "cudaexecutionprovider" in loaded_providers:
            _active_provider = "cuda"
        else:
            _active_provider = "cpu"

        print(f"InsightFace antelopev2 ready  provider={_active_provider}")
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