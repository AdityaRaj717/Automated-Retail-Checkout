"""
Product Detector
================
Two-stage detection pipeline:
  1. Deep learning segmentation (rembg / U2-Net) to find objects in the frame
  2. Embedding-based classification (ResNet18 + kNN) to identify products

The segmentation stage uses U2-Net (via rembg) instead of background subtraction,
making it immune to camera jitter, lighting changes, and background noise.

Usage:
    from detector import ProductDetector
    detector = ProductDetector()
    detections = detector.detect(frame)
"""

import os
import pickle
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neighbors import KNeighborsClassifier

from rembg import new_session, remove

# ── Config ──────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
MIN_CONTOUR_AREA = 5000       # Minimum pixel area to consider as a product
MAX_CONTOUR_AREA = 80000      # Above this, try to split (likely merged products)
CONFIDENCE_THRESHOLD = 0.55   # Minimum confidence to accept a classification
KNN_NEIGHBORS = 5             # Number of neighbors for kNN


class ProductDetector:
    """Detects and classifies retail products in camera frames."""

    def __init__(self, embeddings_path: str = EMBEDDINGS_FILE):
        # ── Load embeddings ─────────────────────────────────────────────
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        self.product_names = data["product_names"]

        # ── Train kNN classifier on stored embeddings ───────────────────
        self.knn = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            metric="cosine",
            weights="distance",
        )
        self.knn.fit(data["embeddings"], data["labels"])

        # ── Feature extractor (same as build_embeddings.py) ─────────────
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_model = nn.Sequential(*list(model.children())[:-1])
        self.feature_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_model = self.feature_model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ── Rembg session (U2-Net, loaded once) ────────────────────────
        print("  Loading U2-Net segmentation model...")
        self.rembg_session = new_session("u2net")
        print("  U2-Net ready!")

        # ── Legacy background reference (kept for backward compat) ──────
        self._reference_bg = None

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract a 512-d embedding from a BGR OpenCV crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_model(tensor)  # (1, 512, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (1, 512)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy()

    def _segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Use rembg (U2-Net) to produce a foreground mask.
        Returns a binary mask (0/255) where 255 = foreground object.
        """
        # rembg expects a PIL image or numpy BGR
        # It returns an RGBA image with transparent background
        pil_in = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_out = remove(pil_in, session=self.rembg_session)

        # Extract alpha channel as the foreground mask
        alpha = np.array(pil_out)[:, :, 3]

        # Threshold the alpha channel (rembg can produce soft edges)
        _, mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def _split_contour(self, mask_roi, offset_x, offset_y):
        """
        Split a merged contour region using distance transform + watershed.
        Returns a list of (x, y, w, h) bounding boxes in frame coordinates.
        """
        dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        sure_bg = cv2.dilate(mask_roi, None, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)

        if num_labels <= 2:
            return None

        markers = markers + 1
        markers[unknown == 255] = 0

        roi_color = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        cv2.watershed(roi_color, markers)

        bboxes = []
        for label_id in range(2, num_labels + 1):
            label_mask = (markers == label_id).astype(np.uint8) * 255
            label_contours, _ = cv2.findContours(
                label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in label_contours:
                area = cv2.contourArea(c)
                if area >= MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    bboxes.append((x + offset_x, y + offset_y, w, h))

        return bboxes if len(bboxes) > 1 else None

    def _classify_contours(self, frame, mask, contours):
        """Classify each contour region and return detections."""
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # If contour is very large, try to split it
            if area > MAX_CONTOUR_AREA:
                roi = mask[y:y+h, x:x+w]
                split_bboxes = self._split_contour(roi, x, y)
                if split_bboxes:
                    for sx, sy, sw, sh in split_bboxes:
                        pad = 10
                        sx1 = max(0, sx - pad)
                        sy1 = max(0, sy - pad)
                        sx2 = min(frame.shape[1], sx + sw + pad)
                        sy2 = min(frame.shape[0], sy + sh + pad)
                        crop = frame[sy1:sy2, sx1:sx2]
                        if crop.size == 0:
                            continue
                        embedding = self._extract_embedding(crop)
                        probabilities = self.knn.predict_proba(embedding)[0]
                        best_idx = np.argmax(probabilities)
                        confidence = probabilities[best_idx]
                        label = self.knn.classes_[best_idx]
                        if confidence >= CONFIDENCE_THRESHOLD:
                            detections.append({
                                "label": label,
                                "confidence": float(confidence),
                                "bbox": (sx, sy, sw, sh),
                            })
                    continue

            # Normal single-product contour
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = self._extract_embedding(crop)
            probabilities = self.knn.predict_proba(embedding)[0]
            best_idx = np.argmax(probabilities)
            confidence = probabilities[best_idx]
            label = self.knn.classes_[best_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                detections.append({
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": (x, y, w, h),
                })

        # Deduplicate: keep highest confidence per label
        seen = {}
        for det in detections:
            lbl = det["label"]
            if lbl not in seen or det["confidence"] > seen[lbl]["confidence"]:
                seen[lbl] = det

        return list(seen.values())

    # ── Public API ──────────────────────────────────────────────────────

    def capture(self, frame: np.ndarray) -> list:
        """
        Detect and classify products using deep learning segmentation.
        Uses rembg (U2-Net) for foreground extraction — no background
        reference needed, immune to camera jitter.

        Returns:
            List of dicts with 'label', 'confidence', 'bbox' (x, y, w, h).
        """
        # Segment the frame using U2-Net
        mask = self._segment_frame(frame)

        # Find contours in the segmentation mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return self._classify_contours(frame, mask, contours)

    def capture_multi(self, frames: list) -> list:
        """
        Multi-frame detection: average multiple frames to suppress noise,
        then run segmentation + classification.
        """
        if len(frames) < 2:
            return self.capture(frames[0]) if frames else []

        averaged = np.mean(frames, axis=0).astype(np.uint8)
        return self.capture(averaged)

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get the raw segmentation mask for a frame (useful for debugging)."""
        return self._segment_frame(frame)

    # ── Legacy methods (kept for backward compatibility) ────────────────

    def set_background(self, frame: np.ndarray):
        """Legacy: store a reference background frame. Not needed with rembg."""
        self._reference_bg = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (31, 31), 0
        )

    def reset_background(self):
        """Legacy: reset background model. Not needed with rembg."""
        self._reference_bg = None
