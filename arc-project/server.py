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
import time
import threading
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import database as db
from detector import ProductDetector

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global detector
    print("=" * 50)
    print("  Retail Checkout System — API Server")
    print("=" * 50)
    print()

    # Initialize detector
    print("Loading product detector...")
    detector = ProductDetector()
    print("Detector ready!")

    # Start camera
    print(f"Connecting to DroidCam: {DROIDCAM_URL}")
    camera.start()

    # Set initial background after a short delay
    print("Waiting for initial background frame...")
    time.sleep(1)
    bg_frame = camera.get_frame()
    if bg_frame is not None:
        detector.set_background(bg_frame)
        print("Background reference set!")
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


@app.post("/capture")
def capture():
    """
    Capture the current frame and run product detection.
    Returns detected products with confidence scores.
    """
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    detections = detector.capture(frame)

    # Enrich detections with product info from DB
    results = []
    for det in detections:
        product = db.get_product_by_slug(det["label"])
        if product:
            results.append({
                "product": product,
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            })

    return {"detections": results, "count": len(results)}


@app.post("/set_background")
def set_background():
    """Capture and set the current frame as the background reference."""
    frame = camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    detector.set_background(frame)
    return {"status": "ok", "message": "Background reference updated"}


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
