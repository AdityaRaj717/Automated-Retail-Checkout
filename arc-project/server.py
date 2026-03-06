"""
FastAPI Backend Server
======================
Serves the camera feed, detection API, and database operations
for the Next.js checkout frontend.

Usage:
    python server.py
"""

import os
import io
import json
import time
import threading
from collections import deque
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import database as db
from detector import ProductDetector
from depth_estimator import DepthEstimator

# ── Config ──────────────────────────────────────────────────────────────────
DROIDCAM_URL = "http://10.91.13.197:4747/video"


# ── Camera Manager ──────────────────────────────────────────────────────────
class CameraManager:
    """Thread-safe camera capture from DroidCam."""

    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self._thread = None

    def start(self):
        """Start the camera capture thread."""
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            print(f"WARNING: Cannot connect to DroidCam at {self.url}")
            print("The server will start, but camera features will not work.")
            print("Start DroidCam and restart the server.")
            return False

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"Camera connected: {self.url}")
        return True

    def _capture_loop(self):
        """Continuously capture frames in the background."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Rotate 180° — DroidCam feed is upside down
                frame = cv2.flip(frame, -1)
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        """Get the latest frame (thread-safe)."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        """Stop the camera."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()


# ── Globals ─────────────────────────────────────────────────────────────────
camera = CameraManager(DROIDCAM_URL)
detector = None
depth_est = None

PRODUCTS_FILE = os.path.join(os.path.dirname(__file__), "products.json")


def _load_products_json():
    """Load products.json for variant resolution."""
    with open(PRODUCTS_FILE, "r") as f:
        return json.load(f)


def _resolve_variant(product_info, slug, metrics):
    """Resolve size variant using depth/volume metrics."""
    catalog = _load_products_json()
    if slug not in catalog:
        return product_info

    entry = catalog[slug]
    variants = entry.get("variants")
    if not variants:
        return product_info

    volume = metrics.get("estimated_volume", 0)
    for v in variants:
        min_vol = v.get("min_volume", 0)
        max_vol = v.get("max_volume", float("inf"))
        if min_vol <= volume <= max_vol:
            return {
                **product_info,
                "name": v["name"],
                "price": v["price"],
                "variant_resolved": True,
            }

    return product_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global detector, depth_est
    print("=" * 50)
    print("  Retail Checkout System — API Server")
    print("=" * 50)
    print()

    # Initialize detector
    print("Loading product detector...")
    detector = ProductDetector()
    print("Detector ready!")

    # Initialize depth estimator
    print("Loading depth estimator...")
    depth_est = DepthEstimator()
    print("Depth estimator ready!")

    # Start camera
    print(f"Connecting to DroidCam: {DROIDCAM_URL}")
    camera.start()

    print()
    print("Server ready! API at http://localhost:8000")
    print("Docs at http://localhost:8000/docs")
    print()

    yield

    camera.stop()
    print("Server shut down.")


# ── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Checkout API",
    description="Camera-based product detection and checkout system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────────────
class TransactionItem(BaseModel):
    product_id: int
    quantity: int
    subtotal: float


class TransactionCreate(BaseModel):
    items: list[TransactionItem]


# ── Camera Endpoints ───────────────────────────────────────────────────────

@app.get("/video_feed")
def video_feed():
    """MJPEG stream from the camera."""
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
            time.sleep(0.033)  # ~30 fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _draw_detections(frame, detections):
    """Draw bounding boxes and labels on a frame."""
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        # Draw filled rectangle behind text
        text = f"{label} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Bounding box — green
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label background
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), (0, 255, 0), -1)

        # Label text — dark
        cv2.putText(frame, text, (x + 3, y - 6), font, font_scale, (0, 0, 0), thickness)

    return frame


@app.get("/video_feed_debug")
def video_feed_debug():
    """
    MJPEG stream with live bounding boxes drawn on detected products.
    Uses temporal smoothing: keeps a rolling window of recent detections
    and only shows products that appear consistently (>=50% of recent runs).
    """
    def generate():
        detection_history = deque(maxlen=6)  # ~3 seconds of history
        stable_detections = []
        last_detect_time = 0
        detect_interval = 1.0  # Run rembg detection every 1s (heavier than bg subtraction)

        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_detect_time >= detect_interval:
                try:
                    # Grab a few frames and average for this detection run
                    frames = [camera.get_frame() for _ in range(3)]
                    frames = [f for f in frames if f is not None]
                    if frames:
                        raw_dets = detector.capture_multi(frames)
                    else:
                        raw_dets = detector.capture(frame)

                    # Enrich with product names from DB
                    enriched = []
                    for det in raw_dets:
                        product = db.get_product_by_slug(det["label"])
                        name = product["name"] if product else det["label"]
                        enriched.append({
                            "slug": det["label"],
                            "label": name,
                            "confidence": det["confidence"],
                            "bbox": det["bbox"],
                        })
                    detection_history.append(enriched)

                    # ── Temporal consensus ──────────────────────────────
                    # Count how often each product appears across history
                    if len(detection_history) >= 3:
                        slug_counts = {}    # slug -> count of appearances
                        slug_bboxes = {}    # slug -> list of (bbox, conf, label)
                        for run in detection_history:
                            for det in run:
                                s = det["slug"]
                                slug_counts[s] = slug_counts.get(s, 0) + 1
                                if s not in slug_bboxes:
                                    slug_bboxes[s] = []
                                slug_bboxes[s].append((
                                    det["bbox"], det["confidence"], det["label"]
                                ))

                        # Keep only products in >= 50% of recent runs
                        threshold = len(detection_history) * 0.5
                        stable = []
                        for slug, count in slug_counts.items():
                            if count >= threshold:
                                entries = slug_bboxes[slug]
                                # Average bounding box across appearances
                                avg_x = int(sum(b[0] for b, _, _ in entries) / len(entries))
                                avg_y = int(sum(b[1] for b, _, _ in entries) / len(entries))
                                avg_w = int(sum(b[2] for b, _, _ in entries) / len(entries))
                                avg_h = int(sum(b[3] for b, _, _ in entries) / len(entries))
                                avg_conf = sum(c for _, c, _ in entries) / len(entries)
                                label = entries[-1][2]  # Latest name
                                stable.append({
                                    "label": label,
                                    "confidence": avg_conf,
                                    "bbox": (avg_x, avg_y, avg_w, avg_h),
                                })
                        stable_detections = stable
                    else:
                        # Not enough history yet, show raw
                        stable_detections = enriched

                except Exception:
                    pass  # Keep last known stable detections
                last_detect_time = now

            # Draw stable detections on the frame
            annotated = _draw_detections(frame.copy(), stable_detections)

            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
            time.sleep(0.033)  # ~30 fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/capture")
def capture():
    """
    Capture frame → detect products → estimate depth → resolve variants.
    Returns rich analytics: depth metrics, candidates, confirmation flags.
    """
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    detections = detector.capture(frame)

    # Run depth estimation
    depth_map = depth_est.estimate_depth(frame)
    depth_est.colorize_depth(depth_map)

    # Enrich detections with product info, depth metrics, and variants
    results = []
    for det in detections:
        product = db.get_product_by_slug(det["label"])
        if not product:
            continue

        # Depth metrics for this product
        metrics = depth_est.get_object_metrics(depth_map, det["bbox"])

        # Console log
        print(f"  [DEPTH] {product['name']}: depth={metrics['mean_depth']:.1f}, "
              f"volume={metrics['estimated_volume']:.0f}, "
              f"size={metrics['size_category']}")

        # Resolve size variant if applicable
        resolved_product = _resolve_variant(product, det["label"], metrics)

        # Build candidates list with product names
        candidates = []
        for c in det.get("candidates", []):
            c_product = db.get_product_by_slug(c["label"])
            candidates.append({
                "slug": c["label"],
                "name": c_product["name"] if c_product else c["label"],
                "confidence": c["confidence"],
                "price": c_product["price"] if c_product else 0,
            })

        results.append({
            "product": resolved_product,
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "depth_metrics": metrics,
            "candidates": candidates,
            "needs_confirmation": det.get("needs_confirmation", False),
        })

    return {
        "detections": results,
        "count": len(results),
        "has_depth_map": True,
    }


@app.get("/depth_map")
def get_depth_map():
    """Return the latest colorized depth map as JPEG."""
    colorized = depth_est.get_last_colorized() if depth_est else None
    if colorized is None:
        raise HTTPException(status_code=404, detail="No depth map available yet. Run /capture first.")

    _, jpeg = cv2.imencode('.jpg', colorized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(jpeg.tobytes()),
        media_type="image/jpeg",
    )


@app.get("/occlusion_map")
def get_occlusion_map():
    """Return the latest occlusion boundary map as JPEG."""
    occ = depth_est.get_last_occlusion() if depth_est else None
    if occ is None:
        raise HTTPException(status_code=404, detail="No occlusion map available yet. Run /capture first.")

    _, jpeg = cv2.imencode('.jpg', occ, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(jpeg.tobytes()),
        media_type="image/jpeg",
    )


@app.post("/set_background")
def set_background():
    """Capture and set the current frame as the background reference."""
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    detector.set_background(frame)
    return {"status": "ok", "message": "Background reference updated"}


@app.post("/reload")
def reload_products():
    """
    Hot-reload: rebuild embeddings from dataset, reload kNN classifier,
    and sync products.json into the database. No restart needed.
    """
    import build_embeddings

    print("\n" + "=" * 50)
    print("  HOT RELOAD — Rebuilding embeddings...")
    print("=" * 50)

    try:
        build_embeddings.build()
        detector.reload_embeddings()
        db.sync_products()
        return {"status": "ok", "message": "Embeddings rebuilt and classifier reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot")
def snapshot():
    """Get a single JPEG snapshot of the current frame."""
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(jpeg.tobytes()),
        media_type="image/jpeg",
    )


# ── Product Endpoints ──────────────────────────────────────────────────────

@app.get("/products")
def get_products():
    """Get all products from the catalog."""
    return db.get_all_products()


# ── Transaction Endpoints ──────────────────────────────────────────────────

@app.post("/transactions")
def create_transaction(txn: TransactionCreate):
    """Save a completed checkout transaction."""
    if not txn.items:
        raise HTTPException(status_code=400, detail="No items in transaction")

    items = [item.model_dump() for item in txn.items]
    txn_id = db.save_transaction(items)
    return {"transaction_id": txn_id, "status": "ok"}


@app.get("/transactions")
def get_transactions(limit: int = 20):
    """Get recent transactions."""
    return db.get_transactions(limit)


# ── Health Check ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "camera_connected": camera.latest_frame is not None,
        "detector_ready": detector is not None,
    }


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
