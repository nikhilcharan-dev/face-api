import json
import numpy as np
import base64
import time
from datetime import datetime
from collections import deque
from pydantic import BaseModel
from typing import Optional, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from embedding import get_face_embedding_from_bytes, load_face_app, is_gpu_available
from db import get_embeddings_collection
from cache import embedding_key, cache_get, cache_set, cache_del, CACHE_TTL

# ---------------- CONFIG ----------------
DIM = 512
THRESHOLD = 0.6
START_TIME = time.time()
HISTORY_LIMIT = 50

# ---------------- MODELS ----------------
class EnrollRequest(BaseModel):
    image_base64: str
    employee_id: str
    organization_id: str

class DetectRequest(BaseModel):
    image_base64: str
    device_id: str
    location_id: Optional[str] = "main-entrance"
    organization_id: str

class ExtractEmbeddingRequest(BaseModel):
    image_base64: str

# ---------------- STATS ----------------
class StatsTracker:
    def __init__(self):
        self.total_processed = 0
        self.total_matches = 0
        self.total_latency = 0.0
        self.history = deque(maxlen=HISTORY_LIMIT)

    def add_entry(self, label, accuracy, time_ms, success):
        self.total_processed += 1
        if success:
            self.total_matches += 1
        self.total_latency += time_ms
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "label": label,
            "accuracy": round(accuracy, 3),
            "time_ms": round(time_ms, 2),
            "status": "MATCH" if success else "UNKNOWN",
        }
        self.history.appendleft(entry)
        return entry

    def get_analytics(self):
        avg = self.total_latency / self.total_processed if self.total_processed else 0
        rate = (self.total_matches / self.total_processed * 100) if self.total_processed else 0
        return {
            "total_processed": self.total_processed,
            "avg_latency": round(avg, 2),
            "match_rate": round(rate, 1),
            "uptime": int(time.time() - START_TIME),
        }

stats = StatsTracker()

# ---------------- WEBSOCKET ----------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, msg: dict):
        for ws in self.active:
            await ws.send_json(msg)

manager = ConnectionManager()

# ---------------- APP ----------------
app = FastAPI(title="Face Recognition API - v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Preloading InsightFace antelopev2 model...")
load_face_app()
print("InsightFace model ready")

# ---------------- HELPERS ----------------
async def load_embeddings(organization_id: str) -> list:
    """Return [{employee_id, embedding}] from Redis or MongoDB."""
    key = embedding_key(organization_id)
    cached = await cache_get(key)
    if cached:
        return json.loads(cached)

    col = get_embeddings_collection()
    docs = await col.find(
        {"organization_id": organization_id},
        {"_id": 0, "employee_id": 1, "embedding": 1},
    ).to_list(length=None)

    records = [{"employee_id": d["employee_id"], "embedding": d["embedding"]} for d in docs]
    await cache_set(key, json.dumps(records), CACHE_TTL)
    return records


async def bust_cache(organization_id: str):
    await cache_del(embedding_key(organization_id))


# ---------------- ROUTES ----------------

@app.get("/")
async def home():
    return {"message": "Face Recognition API running", "version": "v3"}


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "gpu_available": is_gpu_available(),
        "uptime_seconds": int(time.time() - START_TIME),
        "avg_latency_ms": stats.get_analytics()["avg_latency"],
    }


@app.post("/api/v1/enroll")
async def enroll(req: EnrollRequest):
    """
    Extract a 512-D embedding from the image and upsert it into MongoDB.
    Invalidates the org's Redis cache so the next detect picks up the change.
    """
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        emb = get_face_embedding_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Face extraction failed: {e}")

    col = get_embeddings_collection()
    await col.update_one(
        {"employee_id": req.employee_id},
        {"$set": {
            "employee_id": req.employee_id,
            "organization_id": req.organization_id,
            "embedding": emb.tolist(),
        }},
        upsert=True,
    )
    await bust_cache(req.organization_id)

    return {"success": True, "employee_id": req.employee_id}


@app.post("/api/v1/detect")
async def detect(req: DetectRequest):
    """
    Recognize a face against all enrolled embeddings for the given organisation.
    Embeddings are served from Redis (5 min TTL) with MongoDB as the source of truth.
    Similarity is computed via numpy dot-product on L2-normalised embeddings.
    """
    start_total = time.time()

    records = await load_embeddings(req.organization_id)

    if not records:
        return {
            "success": False, "person": None, "confidence": 0.0,
            "embedding_time_ms": 0.0, "search_time_ms": 0.0, "total_time_ms": 0.0,
            "attendance": {"status": "unknown", "is_late": False},
        }

    try:
        try:
            image_bytes = base64.b64decode(req.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        start_emb = time.time()
        query_emb = get_face_embedding_from_bytes(image_bytes)  # (512,) L2-normalised
        emb_time = (time.time() - start_emb) * 1000

        start_search = time.time()
        emb_matrix = np.array([r["embedding"] for r in records], dtype=np.float32)
        scores = emb_matrix @ query_emb  # cosine similarity
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        search_time = (time.time() - start_search) * 1000

        success = best_score >= THRESHOLD
        matched_id = records[best_idx]["employee_id"] if success else "Unknown"

        person = {"id": matched_id, "name": matched_id, "department": "Engineering", "avatar_url": None} if success else None
        total_time = (time.time() - start_total) * 1000

        entry = stats.add_entry(matched_id, best_score, total_time, success)
        await manager.broadcast({"type": "NEW_ENTRY", "data": entry, "analytics": stats.get_analytics()})

        return {
            "success": success,
            "person": person,
            "confidence": round(best_score, 3),
            "embedding_time_ms": round(emb_time, 2),
            "search_time_ms": round(search_time, 2),
            "total_time_ms": round(total_time, 2),
            "attendance": {
                "status": "checked_in" if success else "unknown",
                "check_in_time": datetime.now().isoformat() if success else None,
                "is_late": False,
            },
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/embeddings/{employee_id}")
async def delete_embedding(employee_id: str):
    """Remove a face embedding from MongoDB and invalidate the org's cache."""
    col = get_embeddings_collection()
    doc = await col.find_one_and_delete({"employee_id": employee_id})
    if doc:
        await bust_cache(doc["organization_id"])
    return {"success": True, "deleted_employee_id": employee_id}


@app.get("/api/v1/embeddings")
async def list_embeddings(organization_id: Optional[str] = None):
    """List enrolled employee IDs, optionally filtered by org."""
    col = get_embeddings_collection()
    query = {"organization_id": organization_id} if organization_id else {}
    docs = await col.find(query, {"_id": 0, "employee_id": 1, "organization_id": 1}).to_list(length=None)
    return {"total": len(docs), "employees": docs}


@app.post("/api/v1/extract-embedding")
async def extract_embedding(req: ExtractEmbeddingRequest):
    """Extract a 512-D embedding and return it without persisting. Useful for testing."""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        emb = get_face_embedding_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Face extraction failed: {e}")

    return {"success": True, "embedding": emb.tolist(), "dim": len(emb)}


# ---------------- WEBSOCKET ----------------

@app.websocket("/ws/stats")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "INITIAL_STATE",
            "history": list(stats.history),
            "analytics": stats.get_analytics(),
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------- DASHBOARD ----------------

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face API | Live Monitor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        * { margin:0; padding:0; box-sizing:border-box; font-family:'Inter', sans-serif; }
        body { background:#000; color:#fff; min-height:100vh; display:flex; flex-direction:column; padding:2rem; }
        header { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid #333; padding-bottom:1.5rem; margin-bottom:2rem; }
        .branding h1 { font-weight:800; font-size:2rem; letter-spacing:-0.05em; text-transform:uppercase; }
        .branding p { font-size:0.75rem; color:#666; letter-spacing:0.1em; margin-top:0.25rem; }
        .analytics { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:2rem; }
        .stat-item { display:flex; flex-direction:column; }
        .stat-label { font-size:0.65rem; color:#666; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; }
        .stat-value { font-size:1.5rem; font-weight:600; letter-spacing:-0.02em; }
        .stat-unit { font-size:0.75rem; color:#666; margin-left:0.25rem; }
        .card { border:1px solid #222; padding:1.5rem; }
        .card h2 { font-size:0.8rem; text-transform:uppercase; letter-spacing:0.2em; color:#444; margin-bottom:1.5rem; }
        table { width:100%; border-collapse:collapse; }
        th { font-size:0.65rem; text-transform:uppercase; color:#444; padding-bottom:1rem; border-bottom:1px solid #111; font-weight:400; }
        td { padding:1rem 0; font-size:0.85rem; border-bottom:1px solid #111; }
        .badge { font-size:0.6rem; padding:0.2rem 0.6rem; border:1px solid #fff; border-radius:99px; font-weight:600; }
        .badge-UNKNOWN { border-color:#333; color:#666; }
        .live { width:8px; height:8px; background:#fff; border-radius:50%; display:inline-block; margin-right:1rem; animation:pulse 2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        #conn { font-size:0.65rem; position:fixed; bottom:2rem; right:2rem; color:#444; text-transform:uppercase; }
    </style>
</head>
<body>
    <header>
        <div class="branding">
            <h1>Monitor</h1>
            <p>v3.0.0 / MongoDB-backed Inference Pipeline</p>
        </div>
        <div class="analytics">
            <div class="stat-item"><span class="stat-label">Processed</span><span class="stat-value" id="v-proc">0</span></div>
            <div class="stat-item"><span class="stat-label">Accuracy</span><span class="stat-value" id="v-acc">0<span class="stat-unit">%</span></span></div>
            <div class="stat-item"><span class="stat-label">Lat. Avg</span><span class="stat-value" id="v-lat">0<span class="stat-unit">ms</span></span></div>
        </div>
    </header>
    <div class="card">
        <h2><span class="live"></span>Activity Log</h2>
        <table>
            <thead><tr>
                <th style="width:15%">Time</th>
                <th style="width:35%">Identity</th>
                <th style="width:20%">Confidence</th>
                <th style="width:15%">Exec</th>
                <th style="width:15%;text-align:right">Result</th>
            </tr></thead>
            <tbody id="log"></tbody>
        </table>
    </div>
    <div id="conn">Disconnected</div>
    <script>
        const log = document.getElementById('log');
        const conn = document.getElementById('conn');
        function updateAnalytics(d) {
            document.getElementById('v-proc').textContent = d.total_processed;
            document.getElementById('v-acc').textContent = d.match_rate + '%';
            document.getElementById('v-lat').textContent = d.avg_latency + 'ms';
        }
        function createRow(e) {
            const tr = document.createElement('tr');
            [e.timestamp, e.label, String(e.accuracy), e.time_ms + 'ms'].forEach((t, i) => {
                const td = document.createElement('td');
                td.textContent = t;
                if (i === 1) td.style.fontWeight = '600';
                tr.appendChild(td);
            });
            const td = document.createElement('td');
            td.style.textAlign = 'right';
            const b = document.createElement('span');
            b.className = 'badge badge-' + e.status;
            b.textContent = e.status;
            td.appendChild(b);
            tr.appendChild(td);
            return tr;
        }
        function connect() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(proto + '//' + location.host + '/ws/stats');
            ws.onopen = () => { conn.textContent = 'Connected / Live'; conn.style.color = '#fff'; };
            ws.onmessage = (ev) => {
                const msg = JSON.parse(ev.data);
                if (msg.type === 'INITIAL_STATE') {
                    log.textContent = '';
                    msg.history.forEach(e => log.appendChild(createRow(e)));
                    updateAnalytics(msg.analytics);
                } else if (msg.type === 'NEW_ENTRY') {
                    log.prepend(createRow(msg.data));
                    if (log.children.length > 50) log.lastChild.remove();
                    updateAnalytics(msg.analytics);
                }
            };
            ws.onclose = () => { conn.textContent = 'Reconnecting...'; conn.style.color = '#444'; setTimeout(connect, 3000); };
        }
        connect();
    </script>
</body>
</html>
"""
