"""
Product Detector
================
Two-stage detection pipeline:
  1. Background subtraction (MOG2) to find objects in the camera frame
  2. Embedding-based classification (ResNet18 + kNN) to identify products

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

# ── Config ──────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
MIN_CONTOUR_AREA = 5000       # Minimum pixel area to consider as a product
CONFIDENCE_THRESHOLD = 0.55   # Minimum confidence to accept a classification
KNN_NEIGHBORS = 5             # Number of neighbors for kNN
MOG2_HISTORY = 200            # Frames of history for background model
MOG2_THRESHOLD = 50           # Threshold for background subtraction
LEARNING_RATE = 0.001         # How fast the background model adapts


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

        # ── Background subtractor ──────────────────────────────────────
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_THRESHOLD,
            detectShadows=True,
        )
        self.frame_count = 0
        self.bg_ready = False

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract a 512-d embedding from a BGR OpenCV crop."""
        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_model(tensor)  # (1, 512, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (1, 512)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy()

    def _find_contours(self, frame: np.ndarray):
        """Apply background subtraction and find product contours."""
        # Apply background subtraction
        lr = LEARNING_RATE if self.bg_ready else -1  # -1 = auto learning rate
        fg_mask = self.bg_subtractor.apply(frame, learningRate=lr)

        self.frame_count += 1
        if self.frame_count > 60:  # Need ~60 frames to build background model
            self.bg_ready = True

        # Remove shadows (MOG2 marks shadows as 127, foreground as 255)
        fg_mask = (fg_mask == 255).astype(np.uint8) * 255

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Dilate to merge nearby regions
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours, fg_mask

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect and classify products in a frame.

        Returns:
            List of dicts, each with keys:
            - 'label': product directory name (e.g. 'monaco')
            - 'confidence': float 0-1
            - 'bbox': (x, y, w, h) bounding box
        """
        if not self.bg_ready:
            # Still building background model — just feed frames
            self._find_contours(frame)
            return []

        contours, fg_mask = self._find_contours(frame)
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Add padding around the bounding box
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Classify the crop
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

        # Deduplicate: if multiple detections have the same label,
        # keep the one with highest confidence
        seen = {}
        for det in detections:
            lbl = det["label"]
            if lbl not in seen or det["confidence"] > seen[lbl]["confidence"]:
                seen[lbl] = det
        detections = list(seen.values())

        return detections

    def reset_background(self):
        """Reset the background model (call when scene changes significantly)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_THRESHOLD,
            detectShadows=True,
        )
        self.frame_count = 0
        self.bg_ready = False
