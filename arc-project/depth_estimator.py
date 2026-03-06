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
        Screen-Space Ambient Occlusion from the raw depth map.

        Uses normal-weighted hemisphere sampling:
        - Computes surface normals from depth gradients
        - For each pixel, samples neighbors and checks if they rise
          above the local surface tangent plane (occlude the hemisphere)
        - Contact points (object base on table, stacked objects) get
          high occlusion because nearby surfaces block more of the hemisphere

        Result: grayscale image — black at contact/crevices, white on open surfaces.
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

        # Normalize depth to 0–1 (0 = near/closest, 1 = far)
        depth_norm = ((depth_map - d_min) / (d_max - d_min)).astype(np.float32)
        depth_smooth = cv2.GaussianBlur(depth_norm, (5, 5), 1.0)

        # ── Compute surface normals from depth gradients ───────────
        # dz/dx and dz/dy give the surface slope
        dzdx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3) * 0.5
        dzdy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3) * 0.5

        # Normal = (-dz/dx, -dz/dy, 1), normalized
        nx = -dzdx
        ny = -dzdy
        nz = np.ones_like(nx)
        norm_len = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

        # ── Horizon-based AO ──────────────────────────────────────
        # For each direction, march outward and find the max elevation
        # angle of any sample point above the tangent plane.
        # More "blocked" hemisphere = more occlusion.
        n_dirs = 12
        steps = [4, 8, 14, 22, 32]
        total_occlusion = np.zeros((h, w), dtype=np.float32)

        for d in range(n_dirs):
            angle = 2.0 * np.pi * d / n_dirs
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            max_horizon = np.full((h, w), -1.0, dtype=np.float32)

            for step in steps:
                dx = int(round(step * cos_a))
                dy = int(round(step * sin_a))

                if dx == 0 and dy == 0:
                    continue

                # Sample depth at offset
                shifted = np.roll(np.roll(depth_smooth, -dy, axis=0), -dx, axis=1)

                # How much does the sample "rise" above this pixel?
                # In depth-space: negative diff means sample is CLOSER (higher)
                depth_diff = depth_smooth - shifted  # positive if sample is closer

                # Tangent plane correction: project along normal
                # The tangent at this pixel slopes by (dzdx, dzdy)
                # Expected depth change along (dx, dy) direction
                tangent_offset = dzdx * dx + dzdy * dy
                # Elevation above tangent plane
                elevation = depth_diff - tangent_offset

                # Distance in screen space
                dist = np.sqrt(float(dx**2 + dy**2))

                # Horizon angle: elevation / distance
                horizon_angle = elevation / (dist + 1e-8)

                # Track maximum horizon angle in this direction
                max_horizon = np.maximum(max_horizon, horizon_angle)

            # Occlusion: how much of the hemisphere is blocked in this direction
            # Positive max_horizon = something above the horizon = occlusion
            dir_occ = np.clip(max_horizon * 15.0, 0, 1)
            total_occlusion += dir_occ

        # Average over all directions
        total_occlusion = total_occlusion / n_dirs

        # ── Cavity boost (Laplacian) ───────────────────────────────
        # Subtle extra darkening in true concavities
        laplacian = cv2.Laplacian(depth_smooth, cv2.CV_32F, ksize=5)
        cavity = np.clip(laplacian * 3.0, 0, 1) * 0.3

        # ── Combine ───────────────────────────────────────────────
        ao = np.clip(total_occlusion + cavity, 0, 1)

        # Invert: 0 = fully occluded (black), 1 = exposed (white)
        ao_bright = 1.0 - ao

        # Slight contrast
        ao_bright = np.clip(ao_bright * 1.2 - 0.1, 0, 1)

        ao_uint8 = (ao_bright * 255).astype(np.uint8)

        # Bilateral filter: smooth AO but preserve depth edges
        ao_uint8 = cv2.bilateralFilter(ao_uint8, 7, 50, 50)

        self._last_occlusion = cv2.cvtColor(ao_uint8, cv2.COLOR_GRAY2BGR)
        return self._last_occlusion

    def get_last_occlusion(self) -> np.ndarray:
        """Return the last generated occlusion map, or None."""
        return getattr(self, '_last_occlusion', None)



