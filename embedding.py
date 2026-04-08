import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

_app = None  # singleton

def is_gpu_available():
    """Check if CUDAExecutionProvider is available in ONNX Runtime."""
    return "CUDAExecutionProvider" in ort.get_available_providers()

def load_face_app():
    global _app
    if _app is None:
        providers = ["CPUExecutionProvider"]
        if is_gpu_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("🚀 GPU detected! Using CUDAExecutionProvider")
        else:
            print("🔄 Using CPUExecutionProvider")

        print("🔄 Loading InsightFace antelopev2 model (local, no download)...")

        _app = FaceAnalysis(
            name="antelopev2",
            root="/app",
            providers=providers
        )

        # ctx_id=0 for GPU, -1 for CPU
        ctx_id = 0 if is_gpu_available() else -1
        _app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        print(f"✅ InsightFace antelopev2 loaded (GPU: {is_gpu_available()})")
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