"""
Volume Estimator Module — 3D Volumetric Calculation & Stacking Detection.

Uses depth maps to:
1. Estimate the relative volume of objects on a surface (the "1kg vs 2kg" fix)
2. Detect stacked/occluded objects via depth discontinuity analysis

Research Value:
    - Fixed-Height Calibration: uses the table surface as a zero baseline
    - 3D Point Cloud approximation: pixel heights summed within an object mask
    - Depth discontinuity analysis: gradient-based stacking detection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VolumeResult:
    """Result of volumetric estimation for a single object."""
    total_volume: float       # Sum of all pixel heights (relative units)
    max_height: float         # Maximum height above surface
    mean_height: float        # Average height above surface
    pixel_count: int          # Number of pixels in the object mask


@dataclass
class StackingResult:
    """Result of stacking detection within a single contour."""
    is_stacked: bool          # True if multiple layers detected
    num_layers: int           # Number of distinct depth layers
    layer_masks: List[np.ndarray] = field(default_factory=list)  # Binary masks for each layer


class VolumeEstimator:
    """
    Estimates object volume from depth maps and detects stacked items.

    Workflow:
        1. Calibrate the surface (table) depth from the frame
        2. For each detected object mask, compute volume relative to surface
        3. Check for depth discontinuities that indicate stacking

    Usage:
        estimator = VolumeEstimator()
        surface_depth = estimator.calibrate_surface(depth_map)
        volume = estimator.estimate_volume(depth_map, object_mask, surface_depth)
        stacking = estimator.detect_stacking(depth_map, object_mask)
    """

    def __init__(self,
                 surface_region_ratio: float = 0.15,
                 stacking_gradient_threshold: float = 0.08,
                 min_layer_pixels: int = 200):
        """
        Args:
            surface_region_ratio: Fraction of the bottom of the frame to use
                                  for surface depth calibration.
            stacking_gradient_threshold: Minimum depth gradient magnitude to
                                         consider as a stacking boundary.
            min_layer_pixels: Minimum pixel count for a valid depth layer.
        """
        self.surface_region_ratio = surface_region_ratio
        self.stacking_gradient_threshold = stacking_gradient_threshold
        self.min_layer_pixels = min_layer_pixels

    def calibrate_surface(self, depth_map: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Determine the baseline depth of the table/surface.

        Strategy:
            - If no mask provided, use the bottom `surface_region_ratio` of the frame
            - The table is assumed to be the most common depth value in that region
            - We use the median for robustness against objects at the edges

        Args:
            depth_map: (H, W) float32 depth array, range [0, 1]
            mask: Optional (H, W) binary mask of the surface region

        Returns:
            surface_depth: float, the baseline depth value of the table
        """
        if mask is not None:
            surface_pixels = depth_map[mask > 0]
        else:
            h = depth_map.shape[0]
            bottom_start = int(h * (1 - self.surface_region_ratio))
            surface_pixels = depth_map[bottom_start:, :].flatten()

        if len(surface_pixels) == 0:
            return 0.0

        return float(np.median(surface_pixels))

    def estimate_volume(self, depth_map: np.ndarray,
                        object_mask: np.ndarray,
                        surface_depth: float) -> VolumeResult:
        """
        Estimate the volume of an object by summing pixel heights above the surface.

        Each pixel's "height" is the difference between its depth value and the
        surface baseline. Objects closer to the camera have HIGHER depth values
        (in our normalized map), so: height = pixel_depth - surface_depth.

        Args:
            depth_map: (H, W) float32 depth array, range [0, 1]
            object_mask: (H, W) binary mask (uint8, 0 or 255)
            surface_depth: baseline depth from calibrate_surface()

        Returns:
            VolumeResult with computed metrics
        """
        # Ensure mask is boolean
        mask_bool = object_mask > 0

        # Get depth values within the object
        object_depths = depth_map[mask_bool]

        if len(object_depths) == 0:
            return VolumeResult(
                total_volume=0.0,
                max_height=0.0,
                mean_height=0.0,
                pixel_count=0
            )

        # Height = how much closer to camera than the surface
        # In our depth map: higher value = closer to camera
        heights = object_depths - surface_depth

        # Only consider positive heights (above the surface)
        heights = np.maximum(heights, 0.0)

        return VolumeResult(
            total_volume=float(np.sum(heights)),
            max_height=float(np.max(heights)),
            mean_height=float(np.mean(heights)),
            pixel_count=int(np.sum(mask_bool))
        )

    def detect_stacking(self, depth_map: np.ndarray,
                        object_mask: np.ndarray) -> StackingResult:
        """
        Detect if objects are stacked within a single contour by analyzing
        depth discontinuities.

        Method:
            1. Extract depth values within the object mask
            2. Compute Sobel gradients on the masked depth region
            3. Large gradients indicate physical boundaries between stacked items
            4. Use connected components on the non-boundary regions to count layers

        Args:
            depth_map: (H, W) float32 depth array
            object_mask: (H, W) binary mask (uint8, 0 or 255)

        Returns:
            StackingResult with stacking info and layer masks
        """
        import cv2

        mask_bool = object_mask > 0

        if np.sum(mask_bool) < self.min_layer_pixels:
            return StackingResult(is_stacked=False, num_layers=1, layer_masks=[object_mask])

        # Create a masked depth image (zero outside the object)
        masked_depth = np.zeros_like(depth_map)
        masked_depth[mask_bool] = depth_map[mask_bool]

        # Compute Sobel gradients (edges in depth = stacking boundaries)
        grad_x = cv2.Sobel(masked_depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masked_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold gradients to find boundaries
        boundary_mask = (gradient_magnitude > self.stacking_gradient_threshold).astype(np.uint8)

        # Only consider boundaries within the object
        boundary_mask = boundary_mask & mask_bool.astype(np.uint8)

        # The "non-boundary" region within the object = actual object layers
        interior_mask = mask_bool.astype(np.uint8) - boundary_mask
        interior_mask = np.clip(interior_mask, 0, 1).astype(np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        interior_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_OPEN, kernel)

        # Connected components to find separate layers
        num_labels, labels = cv2.connectedComponents(interior_mask)

        # Filter out small components (noise)
        layer_masks = []
        for label_id in range(1, num_labels):  # Skip background (0)
            layer = (labels == label_id).astype(np.uint8) * 255
            if np.sum(layer > 0) >= self.min_layer_pixels:
                layer_masks.append(layer)

        num_layers = max(1, len(layer_masks))

        # If no valid layers found, return the original mask as a single layer
        if len(layer_masks) == 0:
            layer_masks = [object_mask]

        return StackingResult(
            is_stacked=num_layers > 1,
            num_layers=num_layers,
            layer_masks=layer_masks
        )
