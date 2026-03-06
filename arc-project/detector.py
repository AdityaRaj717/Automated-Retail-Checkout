"""
Product Detector
================
Hybrid detection pipeline for reliable retail product identification:
  1. Deep learning segmentation (rembg / ISNet) to find objects
  2. Sliding window scan (filtered by segmentation mask) as fallback
  3. Embedding-based classification (ResNet18 + kNN) with top-3 candidates
  4. Ambiguity detection — flags items needing cashier confirmation

Usage:
    from detector import ProductDetector
    detector = ProductDetector()
    detections = detector.capture(frame)
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
MIN_CONTOUR_AREA = 1500
MAX_CONTOUR_AREA = 80000
CONFIDENCE_THRESHOLD = 0.45   # Lower threshold — ambiguous items still returned
AMBIGUITY_BAND = (0.45, 0.70) # If best confidence falls here, flag as ambiguous
AMBIGUITY_GAP = 0.15          # If top-2 are within this gap, flag as ambiguous
KNN_NEIGHBORS = 5

# Sliding window config
TILE_SIZE = 224
TILE_STRIDE = 112
TILE_CONFIDENCE = 0.70
TILE_MASK_OVERLAP = 0.15      # Min fraction of tile that must be foreground


class ProductDetector:
    """Detects and classifies retail products in camera frames."""

    def __init__(self, embeddings_path: str = EMBEDDINGS_FILE):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        self.product_names = data["product_names"]
        self.embeddings_path = embeddings_path

        self.knn = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            metric="cosine",
            weights="distance",
        )
        self.knn.fit(data["embeddings"], data["labels"])

        # Feature extractor
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

        # Rembg (ISNet)
        print("  Loading ISNet segmentation model...")
        self.rembg_session = new_session("isnet-general-use")
        print("  ISNet ready!")

        self._reference_bg = None

    # ── Reload ──────────────────────────────────────────────────────────

    def reload_embeddings(self):
        """Hot-reload embeddings from disk and rebuild kNN."""
        print("  Reloading embeddings from disk...")
        with open(self.embeddings_path, "rb") as f:
            data = pickle.load(f)

        self.product_names = data["product_names"]
        self.knn = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            metric="cosine",
            weights="distance",
        )
        self.knn.fit(data["embeddings"], data["labels"])
        print(f"  Reloaded: {len(self.product_names)} products, "
              f"{data['embeddings'].shape[0]} embeddings")

    # ── Embedding Extraction ────────────────────────────────────────────

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract a 512-d embedding from a single BGR crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_model(tensor)
            features = features.squeeze(-1).squeeze(-1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy()

    def _extract_embeddings_batch(self, crops_bgr: list) -> np.ndarray:
        """Batch embedding extraction (GPU-accelerated)."""
        if not crops_bgr:
            return np.empty((0, 512))

        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil_img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.feature_model(batch)
            features = features.squeeze(-1).squeeze(-1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy()

    # ── Classification with Candidates ──────────────────────────────────

    def _classify_with_candidates(self, embedding: np.ndarray) -> dict:
        """
        Classify an embedding and return top-3 candidates + ambiguity flag.
        """
        probs = self.knn.predict_proba(embedding.reshape(1, -1))[0]
        sorted_idx = np.argsort(probs)[::-1]

        # Top-3 candidates
        candidates = []
        for i in range(min(3, len(sorted_idx))):
            idx = sorted_idx[i]
            candidates.append({
                "label": self.knn.classes_[idx],
                "confidence": float(probs[idx]),
            })

        best_conf = candidates[0]["confidence"]
        second_conf = candidates[1]["confidence"] if len(candidates) > 1 else 0

        # Determine if this needs cashier confirmation
        needs_confirmation = False
        if AMBIGUITY_BAND[0] <= best_conf <= AMBIGUITY_BAND[1]:
            needs_confirmation = True
        if len(candidates) > 1 and (best_conf - second_conf) < AMBIGUITY_GAP:
            needs_confirmation = True

        return {
            "label": candidates[0]["label"],
            "confidence": best_conf,
            "candidates": candidates,
            "needs_confirmation": needs_confirmation,
        }

    # ── Segmentation ────────────────────────────────────────────────────

    def _segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Use rembg (ISNet) to produce a foreground mask."""
        pil_in = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_out = remove(pil_in, session=self.rembg_session)

        alpha = np.array(pil_out)[:, :, 3]
        _, mask = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def _split_contour(self, mask_roi, offset_x, offset_y):
        """Split merged contours using distance transform + watershed."""
        dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return None
        _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        sure_bg = cv2.dilate(mask_roi, None, iterations=3)
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
                if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    bboxes.append((x + offset_x, y + offset_y, w, h))

        return bboxes if len(bboxes) > 1 else None

    # ── Segmentation-Based Detection ────────────────────────────────────

    def _detect_via_segmentation(self, frame: np.ndarray, mask: np.ndarray) -> list:
        """Stage 1: Find products using rembg segmentation."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        pad = 15

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Try splitting large contours
            if area > MAX_CONTOUR_AREA:
                roi = mask[y:y+h, x:x+w]
                split_bboxes = self._split_contour(roi, x, y)
                if split_bboxes:
                    crops = []
                    boxes = []
                    for sx, sy, sw, sh in split_bboxes:
                        sx1 = max(0, sx - pad)
                        sy1 = max(0, sy - pad)
                        sx2 = min(frame.shape[1], sx + sw + pad)
                        sy2 = min(frame.shape[0], sy + sh + pad)
                        crop = frame[sy1:sy2, sx1:sx2]
                        if crop.size > 0:
                            crops.append(crop)
                            boxes.append((sx1, sy1, sx2 - sx1, sy2 - sy1))

                    if crops:
                        embeddings = self._extract_embeddings_batch(crops)
                        for emb, bbox in zip(embeddings, boxes):
                            result = self._classify_with_candidates(emb)
                            if result["confidence"] >= CONFIDENCE_THRESHOLD:
                                result["bbox"] = bbox
                                detections.append(result)
                    continue

            # Single product contour
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = self._extract_embedding(crop)
            result = self._classify_with_candidates(embedding)

            if result["confidence"] >= CONFIDENCE_THRESHOLD:
                result["bbox"] = (x1, y1, x2 - x1, y2 - y1)
                detections.append(result)

        return detections

    # ── Sliding Window Detection (filtered by mask) ─────────────────────

    def _detect_via_sliding_window(self, frame: np.ndarray, mask: np.ndarray) -> list:
        """
        Stage 2: Scan frame with overlapping tiles.
        Only processes tiles that overlap with the foreground mask.
        """
        h, w = frame.shape[:2]
        crops = []
        positions = []

        for y in range(0, h - TILE_SIZE + 1, TILE_STRIDE):
            for x in range(0, w - TILE_SIZE + 1, TILE_STRIDE):
                # Check mask overlap — skip purely background tiles
                tile_mask = mask[y:y + TILE_SIZE, x:x + TILE_SIZE]
                fg_fraction = np.count_nonzero(tile_mask) / (TILE_SIZE * TILE_SIZE)

                if fg_fraction >= TILE_MASK_OVERLAP:
                    tile = frame[y:y + TILE_SIZE, x:x + TILE_SIZE]
                    crops.append(tile)
                    positions.append((x, y))

        if not crops:
            return []

        # Batch classify
        embeddings = self._extract_embeddings_batch(crops)

        detections = []
        for emb, (x, y) in zip(embeddings, positions):
            result = self._classify_with_candidates(emb)

            if result["confidence"] >= TILE_CONFIDENCE:
                result["bbox"] = (x, y, TILE_SIZE, TILE_SIZE)
                detections.append(result)

        # Keep highest confidence per label
        seen = {}
        for det in detections:
            lbl = det["label"]
            if lbl not in seen or det["confidence"] > seen[lbl]["confidence"]:
                seen[lbl] = det

        return list(seen.values())

    # ── Merge & Deduplicate ─────────────────────────────────────────────

    @staticmethod
    def _bbox_iou(box1, box2):
        """Compute IoU between two (x, y, w, h) bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def _merge_detections(self, seg_dets, sw_dets):
        """Merge segmentation + sliding window results."""
        merged = {det["label"]: det for det in seg_dets}

        for sw_det in sw_dets:
            label = sw_det["label"]
            if label in merged:
                continue
            overlaps = any(
                self._bbox_iou(sw_det["bbox"], seg_det["bbox"]) > 0.2
                for seg_det in seg_dets
            )
            if not overlaps:
                merged[label] = sw_det

        return list(merged.values())

    # ── Public API ──────────────────────────────────────────────────────

    def capture(self, frame: np.ndarray) -> list:
        """
        Detect products using hybrid approach:
          1. ISNet segmentation → contours → classify with candidates
          2. Sliding window (mask-filtered) → batch classify
          3. Merge and deduplicate

        Each detection includes:
          - label, confidence, bbox
          - candidates (top-3 with probabilities)
          - needs_confirmation (bool)
        """
        # Get segmentation mask
        mask = self._segment_frame(frame)

        # Stage 1: Segmentation-based
        seg_detections = self._detect_via_segmentation(frame, mask)

        # Stage 2: Sliding window (filtered by mask)
        sw_detections = self._detect_via_sliding_window(frame, mask)

        # Merge
        results = self._merge_detections(seg_detections, sw_detections)

        # Console log
        for det in results:
            flag = " ⚠ AMBIGUOUS" if det.get("needs_confirmation") else ""
            print(f"  [DETECT] {det['label']} — {det['confidence']:.0%}{flag}")

        return results

    def capture_multi(self, frames: list) -> list:
        """Multi-frame: average then detect."""
        if len(frames) < 2:
            return self.capture(frames[0]) if frames else []
        averaged = np.mean(frames, axis=0).astype(np.uint8)
        return self.capture(averaged)

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get raw segmentation mask (for debugging)."""
        return self._segment_frame(frame)

    # ── Legacy ──────────────────────────────────────────────────────────

    def set_background(self, frame: np.ndarray):
        self._reference_bg = frame

    def reset_background(self):
        self._reference_bg = None
