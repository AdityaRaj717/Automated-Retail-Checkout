"""
Retail Checkout System — Live Camera Feed
==========================================
Connects to DroidCam, runs real-time product detection, and displays a
live checkout interface with product identification and billing.

Usage:
    python checkout.py

Controls:
    q       — Quit
    r       — Reset bill (clear all items)
    b       — Reset background model (if detection is noisy)
    s       — Save screenshot
    SPACE   — Freeze/unfreeze the bill (lock current detections)
"""

import os
import sys
import json
import time
import cv2
import numpy as np

from detector import ProductDetector

# ── Config ──────────────────────────────────────────────────────────────────
DROIDCAM_URL = "http://10.91.13.197:4747/video"
PRODUCTS_FILE = os.path.join(os.path.dirname(__file__), "products.json")
WINDOW_NAME = "Retail Checkout System"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

# Colors (BGR)
COLOR_BOX = (0, 255, 128)        # Green bounding boxes
COLOR_TEXT = (255, 255, 255)      # White text
COLOR_BG = (40, 40, 40)          # Dark background for overlays
COLOR_HEADER = (255, 165, 0)     # Orange header
COLOR_PRICE = (0, 200, 255)      # Yellow price text
COLOR_TOTAL = (0, 255, 255)      # Cyan total
COLOR_STATUS = (100, 100, 255)   # Red-ish status

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_product_catalog():
    """Load product names and prices from products.json."""
    with open(PRODUCTS_FILE, "r") as f:
        return json.load(f)


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw the four straight sides
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    # Draw the four corners
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_detection_box(frame, det, catalog):
    """Draw bounding box and label for a detection."""
    x, y, w, h = det["bbox"]
    label = det["label"]
    confidence = det["confidence"]

    product_info = catalog.get(label, {"name": label, "price": 0})
    display_name = product_info["name"]
    price = product_info["price"]

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2)

    # Label background
    text = f"{display_name} ({confidence:.0%})"
    price_text = f"Rs.{price}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    (pw, ph), _ = cv2.getTextSize(price_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    total_w = max(tw, pw) + 16

    # Draw label background above the box
    label_y = max(y - 50, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, label_y), (x + total_w, label_y + 48),
                  COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text
    cv2.putText(frame, text, (x + 8, label_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, price_text, (x + 8, label_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRICE, 1, cv2.LINE_AA)


def draw_bill_overlay(frame, bill_items, catalog, frozen=False):
    """Draw the billing panel on the right side of the frame."""
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 260
    panel_x = w - panel_w - 10
    panel_y = 10

    # Calculate panel height
    num_items = len(bill_items)
    panel_h = 80 + max(num_items, 1) * 30 + 40  # Header + items + total

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), COLOR_HEADER, 2)

    # Header
    header_text = "CHECKOUT BILL"
    if frozen:
        header_text += " [LOCKED]"
    cv2.putText(frame, header_text, (panel_x + 10, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HEADER, 2, cv2.LINE_AA)

    # Divider line
    cv2.line(frame, (panel_x + 10, panel_y + 38),
             (panel_x + panel_w - 10, panel_y + 38), COLOR_HEADER, 1)

    # Column headers
    y_offset = panel_y + 60
    cv2.putText(frame, "Item", (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Price", (panel_x + panel_w - 60, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    y_offset += 8

    # Items
    total = 0
    if not bill_items:
        y_offset += 25
        cv2.putText(frame, "No items detected", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    else:
        for label in sorted(bill_items.keys()):
            info = catalog.get(label, {"name": label, "price": 0})
            y_offset += 25
            # Truncate long names
            name = info["name"]
            if len(name) > 22:
                name = name[:20] + ".."
            cv2.putText(frame, name, (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
            price_str = f"Rs.{info['price']}"
            cv2.putText(frame, price_str, (panel_x + panel_w - 60, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PRICE, 1, cv2.LINE_AA)
            total += info["price"]

    # Total divider
    y_offset += 15
    cv2.line(frame, (panel_x + 10, y_offset),
             (panel_x + panel_w - 10, y_offset), COLOR_TEXT, 1)

    # Total
    y_offset += 25
    cv2.putText(frame, "TOTAL", (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TOTAL, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Rs.{total}", (panel_x + panel_w - 80, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TOTAL, 2, cv2.LINE_AA)


def draw_status_bar(frame, fps, bg_ready, frozen):
    """Draw status bar at the bottom of the frame."""
    h, w = frame.shape[:2]

    # Background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 30), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Status text
    status_parts = [f"FPS: {fps:.0f}"]

    if not bg_ready:
        status_parts.append("Building background model...")
    else:
        status_parts.append("READY")

    if frozen:
        status_parts.append("BILL LOCKED (SPACE to unlock)")

    status_text = " | ".join(status_parts)
    color = COLOR_STATUS if not bg_ready else (0, 200, 0)
    cv2.putText(frame, status_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Controls hint
    controls = "Q:Quit  R:Reset  B:BG Reset  S:Screenshot  SPACE:Lock"
    cv2.putText(frame, controls, (w - 420, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)


# ── Main Loop ───────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Retail Checkout System — MVP")
    print("=" * 50)
    print()
    print(f"Connecting to DroidCam: {DROIDCAM_URL}")
    print("Loading product detector...")

    # Load product catalog
    catalog = load_product_catalog()
    print(f"Loaded {len(catalog)} products from catalog")

    # Initialize detector
    detector = ProductDetector()
    print("Detector ready!")
    print()

    # Connect to camera
    cap = cv2.VideoCapture(DROIDCAM_URL)
    if not cap.isOpened():
        print(f"ERROR: Cannot connect to DroidCam at {DROIDCAM_URL}")
        print("Make sure DroidCam is running on your phone and the IP is correct.")
        sys.exit(1)

    print("Camera connected! Starting detection loop...")
    print()
    print("Controls:")
    print("  Q     — Quit")
    print("  R     — Reset bill")
    print("  B     — Reset background model")
    print("  S     — Save screenshot")
    print("  SPACE — Lock/unlock bill")
    print()

    # State
    bill_items = {}      # {label: detection_info}
    frozen = False        # Whether bill is locked
    prev_time = time.time()
    fps = 0.0

    # Create screenshots dir
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed, attempting to reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(DROIDCAM_URL)
            continue

        # Calculate FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - prev_time, 0.001))
        prev_time = current_time

        # Run detection (skip if bill is frozen)
        if not frozen:
            detections = detector.detect(frame)

            # Update bill with current detections
            if detector.bg_ready:
                bill_items = {}
                for det in detections:
                    bill_items[det["label"]] = det

        # Draw detections
        for label, det in bill_items.items():
            draw_detection_box(frame, det, catalog)

        # Draw UI overlays
        draw_bill_overlay(frame, bill_items, catalog, frozen)
        draw_status_bar(frame, fps, detector.bg_ready, frozen)

        # Display
        cv2.imshow(WINDOW_NAME, frame)

        # Handle input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            bill_items = {}
            frozen = False
            print("Bill reset!")
        elif key == ord("b"):
            detector.reset_background()
            bill_items = {}
            frozen = False
            print("Background model reset! Keep frame clear for ~2 seconds...")
        elif key == ord("s"):
            timestamp = int(time.time())
            path = os.path.join(SCREENSHOT_DIR, f"checkout_{timestamp}.jpg")
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")
        elif key == ord(" "):
            frozen = not frozen
            state = "LOCKED" if frozen else "UNLOCKED"
            print(f"Bill {state}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print final bill
    if bill_items:
        print()
        print("=" * 40)
        print("  FINAL BILL")
        print("=" * 40)
        total = 0
        for label in sorted(bill_items.keys()):
            info = catalog.get(label, {"name": label, "price": 0})
            print(f"  {info['name']:30s}  Rs.{info['price']}")
            total += info["price"]
        print("-" * 40)
        print(f"  {'TOTAL':30s}  Rs.{total}")
        print("=" * 40)


if __name__ == "__main__":
    main()
