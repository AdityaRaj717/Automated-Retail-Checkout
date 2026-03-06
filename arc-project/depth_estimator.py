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
        Generate a Screen-Space Ambient Occlusion (SSAO) map.

        For each pixel, samples surrounding points and checks how many are
        at a shallower depth (occluding). Result is a grayscale image:
        - Black = highly occluded (crevices, contact points, under objects)
        - White = fully exposed to ambient light

        Returns:
            Grayscale BGR image of the AO map.
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

        # SSAO: compare each pixel with surrounding samples
        # Use multiple radii for multi-scale AO
        ao = np.zeros((h, w), dtype=np.float32)
        sample_radii = [3, 7, 15, 25]
        num_directions = 8

        for radius in sample_radii:
            occlusion_count = np.zeros((h, w), dtype=np.float32)
            for i in range(num_directions):
                angle = 2.0 * np.pi * i / num_directions
                dx = int(round(radius * np.cos(angle)))
                dy = int(round(radius * np.sin(angle)))

                # Shift the depth map
                shifted = np.roll(np.roll(depth_norm, dy, axis=0), dx, axis=1)

                # Where shifted pixel is closer (smaller depth) → that sample occludes
                # We want a threshold-based comparison
                diff = depth_norm - shifted
                occlusion_count += (diff > 0.01).astype(np.float32)

            # Normalize: fraction of samples that occlude this pixel
            ao += occlusion_count / num_directions

        # Average across radii
        ao = ao / len(sample_radii)

        # ao is in [0, 1] where 1 = fully occluded
        # Invert: 0 = occluded (dark), 1 = exposed (bright)
        ao_inverted = 1.0 - ao

        # Apply contrast enhancement
        ao_inverted = np.clip(ao_inverted * 1.5 - 0.2, 0, 1)

        # Convert to uint8
        ao_uint8 = (ao_inverted * 255).astype(np.uint8)

        # Apply slight blur for smoother appearance
        ao_uint8 = cv2.GaussianBlur(ao_uint8, (5, 5), 0)

        # Convert to BGR for serving
        self._last_occlusion = cv2.cvtColor(ao_uint8, cv2.COLOR_GRAY2BGR)
        return self._last_occlusion

    def get_last_occlusion(self) -> np.ndarray:
        """Return the last generated occlusion map, or None."""
        return getattr(self, '_last_occlusion', None)

