"""
Depth Estimator
===============
Wraps Depth Anything V2 (via HuggingFace Transformers) for monocular depth
estimation. Provides per-object depth metrics and colorized depth map output.

Usage:
    from depth_estimator import DepthEstimator
    estimator = DepthEstimator()
    depth_map = estimator.estimate_depth(frame)
    metrics = estimator.get_object_metrics(depth_map, bbox)
"""

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2."""

    def __init__(self):
        print("  Loading Depth Anything V2 (Small)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._last_depth_map = None
        self._last_colorized = None
        print(f"  Depth Anything V2 ready! (device: {self.device})")

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from a BGR frame.

        Returns:
            Depth map as float32 numpy array (same H×W as input).
            Higher values = farther from camera.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original resolution
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        self._last_depth_map = depth
        return depth

    def colorize_depth(self, depth_map: np.ndarray = None) -> np.ndarray:
        """
        Convert depth map to a colorized heatmap (BGR) for visualization.
        Uses the last estimated depth map if none provided.
        """
        if depth_map is None:
            depth_map = self._last_depth_map
        if depth_map is None:
            return None

        # Normalize to 0–255
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 0:
            normalized = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply colormap (INFERNO gives a nice hot-to-cold gradient)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        self._last_colorized = colorized
        return colorized

    def get_object_metrics(self, depth_map: np.ndarray, bbox: tuple) -> dict:
        """
        Compute depth metrics for a detected object region.

        Args:
            depth_map: full-frame depth map from estimate_depth()
            bbox: (x, y, w, h) bounding box

        Returns:
            dict with depth analytics for this object.
        """
        x, y, w, h = bbox
        h_frame, w_frame = depth_map.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "mean_depth": 0, "min_depth": 0, "max_depth": 0,
                "depth_range": 0, "bbox_area": w * h,
                "estimated_volume": 0, "size_category": "unknown",
            }

        mean_depth = float(np.mean(roi))
        min_depth = float(np.min(roi))
        max_depth = float(np.max(roi))
        depth_range = max_depth - min_depth
        bbox_area = w * h

        # Volume proxy: area × depth_range (relative, not calibrated)
        estimated_volume = bbox_area * depth_range

        # Size categories based on bbox area
        if bbox_area < 8000:
            size_category = "small"
        elif bbox_area < 30000:
            size_category = "medium"
        else:
            size_category = "large"

        return {
            "mean_depth": round(mean_depth, 2),
            "min_depth": round(min_depth, 2),
            "max_depth": round(max_depth, 2),
            "depth_range": round(depth_range, 2),
            "bbox_area": bbox_area,
            "estimated_volume": round(estimated_volume, 1),
            "size_category": size_category,
        }

    def get_last_colorized(self) -> np.ndarray:
        """Return the last colorized depth map, or None."""
        return self._last_colorized

    def generate_occlusion_map(self, frame: np.ndarray,
                                depth_map: np.ndarray = None) -> np.ndarray:
        """
        Generate a Screen-Space Ambient Occlusion (SSAO) map with:
          1. Cavity detection (Laplacian of depth → concavities)
          2. Contact shadows (proximity darkening near depth edges)
          3. Hemisphere occlusion (range-checked sampling)

        Result is grayscale: black = occluded (contact points), white = exposed.
        """
        if depth_map is None:
            depth_map = self._last_depth_map
        if depth_map is None:
            return None

        h, w = depth_map.shape
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min == 0:
            ao = np.ones((h, w), dtype=np.uint8) * 255
            self._last_occlusion = cv2.cvtColor(ao, cv2.COLOR_GRAY2BGR)
            return self._last_occlusion

        # Normalize depth to 0–1 (0 = closest, 1 = farthest)
        depth_norm = ((depth_map - d_min) / (d_max - d_min)).astype(np.float32)

        # Smooth slightly to reduce noise from DAv2
        depth_smooth = cv2.GaussianBlur(depth_norm, (3, 3), 0)

        # ── Layer 1: Cavity Detection (Laplacian) ──────────────────
        # Second derivative of depth: positive = concavity (contact points)
        laplacian = cv2.Laplacian(depth_smooth, cv2.CV_32F, ksize=5)
        # Concavities → positive laplacian → darker
        cavity = np.clip(laplacian * 8.0, 0, 1)

        # ── Layer 2: Contact Shadows (proximity to depth edges) ────
        # Find sharp depth edges (where objects meet the table)
        grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Strong edges = object-to-table or object-to-object borders
        edge_threshold = np.percentile(grad_mag, 85)
        edges = (grad_mag > edge_threshold).astype(np.uint8) * 255

        # Spread contact shadow from edges using distance transform
        # Invert: distance from nearest edge
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        # Contact shadow falloff: strong near edge, fading over ~20px
        contact_shadow_radius = 20.0
        contact = np.clip(1.0 - dist / contact_shadow_radius, 0, 1)
        contact = contact * 0.6  # Max darkness from contact

        # ── Layer 3: Hemisphere Occlusion (range-checked) ──────────
        ao_hemi = np.zeros((h, w), dtype=np.float32)
        radii = [5, 12, 22]
        n_dirs = 8
        range_check = 0.08  # Only count samples within this depth range

        for radius in radii:
            occ = np.zeros((h, w), dtype=np.float32)
            for i in range(n_dirs):
                angle = 2.0 * np.pi * i / n_dirs
                dx = int(round(radius * np.cos(angle)))
                dy = int(round(radius * np.sin(angle)))

                shifted = np.roll(np.roll(depth_smooth, dy, axis=0), dx, axis=1)
                diff = depth_smooth - shifted

                # Only count as occluder if it's closer AND within range
                # (prevents distant surfaces from causing occlusion)
                occluding = (diff > 0.005) & (diff < range_check)
                occ += occluding.astype(np.float32)

            ao_hemi += occ / n_dirs

        ao_hemi = ao_hemi / len(radii)

        # ── Combine all layers ─────────────────────────────────────
        # Cavity: strongest at concavities
        # Contact: darkens near depth edges
        # Hemisphere: general ambient occlusion
        combined = np.clip(cavity * 0.5 + contact + ao_hemi * 0.4, 0, 1)

        # Invert: occluded → dark, exposed → bright
        ao_result = 1.0 - combined

        # Contrast enhancement to make contacts pop
        ao_result = np.clip((ao_result - 0.3) * 1.8, 0, 1)

        # Convert to uint8
        ao_uint8 = (ao_result * 255).astype(np.uint8)

        # Smooth for clean appearance
        ao_uint8 = cv2.bilateralFilter(ao_uint8, 5, 40, 40)

        self._last_occlusion = cv2.cvtColor(ao_uint8, cv2.COLOR_GRAY2BGR)
        return self._last_occlusion

    def get_last_occlusion(self) -> np.ndarray:
        """Return the last generated occlusion map, or None."""
        return getattr(self, '_last_occlusion', None)


