"""
Depth Estimator Module — Monocular Depth Estimation using Depth Anything V2 Small.

Converts a 2D RGB image into a grayscale depth map where pixel intensity
represents relative distance from the camera.

Research Value:
    - Enables volumetric estimation (1kg vs 2kg differentiation)
    - Enables occlusion/stacking detection via depth discontinuities
    - Uses a transformer-based monocular depth model (no stereo camera needed)
"""

import numpy as np
from PIL import Image


class DepthEstimator:
    """
    Wraps the Depth Anything V2 Small model from Hugging Face.

    The model is loaded lazily on first call to avoid startup overhead
    when depth estimation is not needed.

    Usage:
        estimator = DepthEstimator(device="cpu")
        depth_map = estimator.estimate_depth(pil_image)
        # depth_map is a float32 numpy array (H, W) in range [0, 1]
        # 0 = furthest from camera, 1 = closest to camera
    """

    def __init__(self, device="cpu", model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        self.device = device
        self.model_name = model_name
        self._pipe = None  # Lazy-loaded

    def _load_model(self):
        """Lazily load the depth estimation pipeline on first use."""
        if self._pipe is not None:
            return

        print(f"[DEPTH] Loading Depth Anything V2 Small on '{self.device}'...")
        from transformers import pipeline

        self._pipe = pipeline(
            task="depth-estimation",
            model=self.model_name,
            device=self.device if self.device != "cpu" else -1,
        )
        print("[DEPTH] Model loaded successfully.")

    def estimate_depth(self, pil_image: Image.Image) -> np.ndarray:
        """
        Estimate depth from a single RGB image.

        Args:
            pil_image: PIL Image in RGB mode.

        Returns:
            depth_map: np.ndarray of shape (H, W), float32, range [0, 1].
                       Higher values = closer to camera.
        """
        self._load_model()

        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Run inference
        result = self._pipe(pil_image)

        # The pipeline returns a dict with 'depth' key containing a PIL Image
        depth_pil = result["depth"]
        depth_map = np.array(depth_pil, dtype=np.float32)

        # Normalize to [0, 1] range
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map

    def resize_depth_to_match(self, depth_map: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Resize a depth map to match a target (H, W) shape.
        Useful when the depth map and the original frame have different resolutions.

        Args:
            depth_map: depth array (H1, W1)
            target_shape: (H2, W2) to resize to

        Returns:
            Resized depth map as float32 numpy array.
        """
        import cv2
        target_h, target_w = target_shape[:2]
        resized = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32)
