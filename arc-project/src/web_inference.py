"""
Web Inference Engine — The Integrated ARC Pipeline.

This is the main runtime script that connects:
    1. Vision Module:      Camera Frame → Depth Map → Object Masks
    2. Recognition Module:  Object Crop → Embedding → Nearest Product
    3. Logic Module:        Embedding + Volume → Final Product (with weight variants)
    4. Active Learning:     Low-confidence → Cashier Confirmation → Dataset Update

The system supports TWO operating modes:
    - METRIC MODE (default): Uses embedding similarity (requires metric_train.py first)
    - CLASSIFIER MODE (fallback): Uses the old classification head if no embeddings exist

Data Flow:
    Camera → rembg (background removal) → Contour Detection
                                       ↓
                        Depth Anything V2 (depth map)
                                       ↓
                     For each contour:
                        ├── Volume Estimation (height vs surface)
                        ├── Stacking Detection (depth discontinuity)
                        ├── Embedding Extraction (256-dim vector)
                        └── Logic Engine → Product + Price + Confidence
                                       ↓
                        Socket.io → React UI (Cart / Cashier Prompt)
"""

import torch
import cv2
import numpy as np
import socketio
import base64
import time
import os
import sys
import sqlite3
from rembg import remove, new_session
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_model import RetailAttnNet
from modules.depth_estimator import DepthEstimator
from modules.volume_estimator import VolumeEstimator
from modules.vector_db import EmbeddingStore
from modules.logic_engine import LogicEngine
from modules.active_learning import ActiveLearningManager

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = "http://localhost:3000"
MODEL_PATH = os.path.join(BASE_DIR, "retail_model.pth")
CLASS_FILE = os.path.join(BASE_DIR, "classes.txt")
DB_FILE = os.path.join(BASE_DIR, "products.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CAMERA_SOURCE = "http://192.168.1.43:4747/video"

CONFIDENCE_THRESHOLD = 50      # Legacy threshold (used in classifier mode only)
MIN_OBJECT_AREA = 500
ENABLE_DEPTH = True            # Set to False to disable depth estimation (faster)
# ---------------

# Global State
cart = []
product_db = {}
scan_trigger = False
sio = socketio.Client()


# --- DATABASE HELPERS ---
def load_product_db():
    global product_db
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT key_name, display_name, price FROM products")
        rows = c.fetchall()
        product_db = {r[0]: {'name': r[1], 'price': r[2]} for r in rows}
        conn.close()
        print(f"[INFO] Loaded {len(product_db)} products from Database.")
    except Exception as e:
        print(f"[ERROR] Database Error: {e}. Run setup_db.py first!")


# --- SOCKET HANDLERS ---
@sio.on('connect')
def on_connect():
    print("[INFO] Connected to Node.js Server")


@sio.on('execute_scan')
def on_execute_scan(data=None):
    global scan_trigger
    print("[CMD] Received Web Trigger -> Scanning...")
    scan_trigger = True


@sio.on('cart_action_trigger')
def on_cart_action(data):
    global cart
    action_type = data.get('type')
    item_id = data.get('id')

    print(f"[ACTION] {action_type} on item {item_id}")

    if action_type == 'clear':
        cart = []
    elif action_type == 'remove':
        cart = [item for item in cart if item['id'] != item_id]
    elif action_type == 'increment':
        for item in cart:
            if item['id'] == item_id:
                item['qty'] += 1
                item['total_price'] = item['unit_price'] * item['qty']
    elif action_type == 'decrement':
        for item in cart:
            if item['id'] == item_id:
                if item['qty'] > 1:
                    item['qty'] -= 1
                    item['total_price'] = item['unit_price'] * item['qty']

    sio.emit('cart_update', cart)


@sio.on('cashier_confirm')
def on_cashier_confirm(data):
    """Handle cashier confirmation for active learning."""
    item_id = data.get('item_id')
    confirmed = data.get('confirmed', False)
    corrected_name = data.get('corrected_name')

    print(f"[ACTIVE_LEARN] Cashier response: item={item_id}, "
          f"confirmed={confirmed}, correction={corrected_name}")

    if active_learning_mgr:
        active_learning_mgr.process_feedback(item_id, confirmed, corrected_name)


# --- ML SETUP ---
def load_system():
    """Load the complete vision pipeline."""
    # Load class names
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model (with embedding support)
    model = RetailAttnNet(num_classes=len(class_names), embedding_dim=256)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # Handle old model format that used 'fc' instead of 'embedding_head' + 'classifier'
    if "fc.weight" in state_dict and "embedding_head.0.weight" not in state_dict:
        print("[WARN] Old model format detected (fc layer). Loading with strict=False.")
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load background removal
    session = new_session()

    # Load depth estimator (optional — can be disabled for speed)
    depth_est = None
    if ENABLE_DEPTH:
        try:
            depth_est = DepthEstimator(device=str(device))
            print("[INFO] Depth estimation ENABLED (Depth Anything V2 Small)")
        except Exception as e:
            print(f"[WARN] Could not load depth estimator: {e}")
            print("[WARN] Depth estimation DISABLED. Volume features unavailable.")

    # Load volume estimator
    volume_est = VolumeEstimator()

    # Load embedding store
    embedding_store = EmbeddingStore(DB_FILE)
    use_metric = embedding_store.has_embeddings()

    if use_metric:
        print("[INFO] ✓ Metric mode: Using embedding similarity for recognition.")
    else:
        print("[INFO] ✗ Classifier mode: No embeddings found. Using classification head.")
        print("[INFO]   Run 'python metric_train.py' to enable metric mode.")

    # Load logic engine
    logic_engine = LogicEngine(db_path=DB_FILE)

    return model, class_names, device, session, depth_est, volume_est, embedding_store, logic_engine, use_metric


def main():
    global scan_trigger, cart, active_learning_mgr

    # Init
    load_product_db()
    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

    (model, classes, device, session,
     depth_est, volume_est, embedding_store,
     logic_engine, use_metric) = load_system()

    # Active learning manager
    active_learning_mgr = ActiveLearningManager(
        dataset_dir=DATASET_DIR,
        low_threshold=0.40,
        high_threshold=0.70
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    print("\n" + "=" * 50)
    print("  ARC SYSTEM v2.0 — ADVANCED VISION PIPELINE")
    print("  Mode: " + ("METRIC LEARNING" if use_metric else "CLASSIFIER (legacy)"))
    print("  Depth: " + ("ENABLED" if depth_est else "DISABLED"))
    print("=" * 50 + "\n")
    print("--- SYSTEM READY ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stream Video
        stream_frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64_image = base64.b64encode(buffer).decode('utf-8')
        sio.emit('processed_frame', {'image': b64_image})

        key = cv2.waitKey(1) & 0xFF

        # Detection Logic
        if scan_trigger or key == ord(' '):
            scan_trigger = False
            print("\n--- PROCESSING IMAGE ---")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # === VISION MODULE: Depth Estimation ===
            depth_map = None
            surface_depth = 0.0
            if depth_est is not None:
                try:
                    depth_map = depth_est.estimate_depth(pil_img)
                    depth_map = depth_est.resize_depth_to_match(
                        depth_map, frame.shape[:2]
                    )
                    surface_depth = volume_est.calibrate_surface(depth_map)
                    print(f"  [DEPTH] Surface baseline: {surface_depth:.3f}")
                except Exception as e:
                    print(f"  [DEPTH] Error: {e}")
                    depth_map = None

            # === VISION MODULE: Background Removal & Contours ===
            pil_no_bg = remove(pil_img, session=session)
            alpha_mask = np.array(pil_no_bg)[:, :, 3]
            contours, _ = cv2.findContours(
                alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            found_items_count = 0

            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_OBJECT_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Create object mask for this contour
                object_mask = np.zeros(alpha_mask.shape, dtype=np.uint8)
                cv2.drawContours(object_mask, [cnt], -1, 255, -1)

                # === VISION MODULE: Stacking Detection ===
                masks_to_process = [(x, y, w, h, object_mask)]

                if depth_map is not None:
                    stacking = volume_est.detect_stacking(depth_map, object_mask)
                    if stacking.is_stacked:
                        print(f"  [STACK] Detected {stacking.num_layers} stacked items!")
                        masks_to_process = []
                        for layer_mask in stacking.layer_masks:
                            layer_cnts, _ = cv2.findContours(
                                layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            for lc in layer_cnts:
                                lx, ly, lw, lh = cv2.boundingRect(lc)
                                masks_to_process.append((lx, ly, lw, lh, layer_mask))

                # Process each object/layer
                for ox, oy, ow, oh, omask in masks_to_process:
                    # Crop & pad to square (matches training)
                    item_crop = pil_no_bg.crop((ox, oy, ox + ow, oy + oh))
                    max_dim = max(ow, oh)
                    bg = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                    offset_x = (max_dim - ow) // 2
                    offset_y = (max_dim - oh) // 2
                    bg.paste(item_crop, (offset_x, offset_y),
                             mask=item_crop.split()[3] if item_crop.mode == 'RGBA' else None)

                    img_t = transform(bg).unsqueeze(0).to(device)

                    # === VISION MODULE: Volume Estimation ===
                    volume_result = None
                    if depth_map is not None:
                        volume_result = volume_est.estimate_volume(
                            depth_map, omask, surface_depth
                        )
                        print(f"  [VOL] Volume: {volume_result.total_volume:.1f}, "
                              f"Max Height: {volume_result.max_height:.3f}")

                    # === RECOGNITION MODULE ===
                    if use_metric:
                        # --- METRIC LEARNING MODE ---
                        with torch.no_grad():
                            embedding = model(img_t, return_embedding=True)
                        embedding_np = embedding.cpu().numpy().flatten()

                        result = logic_engine.identify_product(
                            embedding_np, volume_result, embedding_store
                        )

                        if result is None:
                            print(f"  - Rejected: No confident match found")
                            continue

                        key_name = result.key_name
                        display_name = result.display_name
                        unit_price = result.price
                        conf_val = result.confidence * 100

                        # Active Learning: Cashier confirmation
                        if result.needs_confirmation:
                            prompt_id = int(time.time() * 1000) + ox
                            active_learning_mgr.queue_feedback(
                                prompt_id, key_name, result.confidence,
                                item_crop, embedding_np
                            )
                            # Send prompt to UI
                            _, crop_buffer = cv2.imencode('.jpg',
                                cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR))
                            crop_b64 = base64.b64encode(crop_buffer).decode('utf-8')
                            sio.emit('cashier_prompt', {
                                'item_id': prompt_id,
                                'suggested_name': display_name,
                                'confidence': round(conf_val, 1),
                                'image_b64': crop_b64
                            })
                            print(f"  ? Asking cashier: '{display_name}' ({conf_val:.1f}%)")
                            continue  # Don't add to cart until confirmed

                        if result.variant_info:
                            print(f"  [VARIANT] Resolved: {result.variant_info}")

                    else:
                        # --- CLASSIFIER MODE (Legacy Fallback) ---
                        with torch.no_grad():
                            output = model(img_t)
                            probs = torch.nn.functional.softmax(output, dim=1)
                            conf, idx = torch.max(probs, 1)

                        conf_val = conf.item() * 100
                        key_name = classes[idx.item()]

                        if conf_val <= CONFIDENCE_THRESHOLD:
                            print(f"  - Rejected: {key_name} ({conf_val:.1f}%)")
                            continue

                        db_item = product_db.get(key_name, {'name': key_name, 'price': 0})
                        display_name = db_item['name']
                        unit_price = db_item['price']

                    # === ADD TO CART ===
                    existing_item = next(
                        (item for item in cart if item['key_name'] == key_name), None
                    )

                    if existing_item:
                        existing_item['qty'] += 1
                        existing_item['total_price'] = existing_item['qty'] * unit_price
                        print(f"  + Incremented: {display_name} ({conf_val:.1f}%)")
                    else:
                        new_item = {
                            'id': int(time.time() * 1000) + ox,
                            'key_name': key_name,
                            'display_name': display_name,
                            'unit_price': unit_price,
                            'qty': 1,
                            'total_price': unit_price
                        }
                        cart.append(new_item)
                        print(f"  + New: {display_name} ({conf_val:.1f}%)")

                    found_items_count += 1

            if found_items_count > 0:
                sio.emit('cart_update', cart)
                print(f"Cart sync sent. ({found_items_count} items)")
            else:
                print("No items found.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()


# Module-level reference for socket handler
active_learning_mgr = None

if __name__ == "__main__":
    main()
