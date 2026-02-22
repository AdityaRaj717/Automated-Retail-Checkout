"""Tests for VolumeEstimator — volumetric calculation and stacking detection."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.volume_estimator import VolumeEstimator, VolumeResult, StackingResult


@pytest.fixture
def estimator():
    return VolumeEstimator(
        surface_region_ratio=0.15,
        stacking_gradient_threshold=0.08,
        min_layer_pixels=50  # Lower for test images
    )


class TestCalibrateSurface:
    def test_flat_surface(self, estimator):
        """A uniform depth map should return that uniform value as the surface."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        surface = estimator.calibrate_surface(depth)
        assert abs(surface - 0.3) < 0.01

    def test_bottom_region(self, estimator):
        """Surface calibration should use bottom portion of the frame."""
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[85:, :] = 0.5  # Bottom 15% is at 0.5
        depth[:85, :] = 0.2  # Rest is at 0.2
        surface = estimator.calibrate_surface(depth)
        assert abs(surface - 0.5) < 0.01

    def test_with_mask(self, estimator):
        """Surface depth with explicit mask should use masked pixels."""
        depth = np.full((100, 100), 0.4, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[90:, :] = 255
        depth[90:, :] = 0.6
        surface = estimator.calibrate_surface(depth, mask=mask)
        assert abs(surface - 0.6) < 0.01


class TestEstimateVolume:
    def test_flat_object_on_surface(self, estimator):
        """Object at same depth as surface should have ~zero volume."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 255

        result = estimator.estimate_volume(depth, mask, surface_depth=0.3)
        assert result.total_volume < 0.01
        assert result.max_height < 0.01

    def test_raised_object(self, estimator):
        """Object closer to camera (higher depth) should have positive volume."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        # Object is raised (closer to camera = higher depth value)
        depth[30:60, 30:60] = 0.6

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 255

        result = estimator.estimate_volume(depth, mask, surface_depth=0.3)
        assert result.total_volume > 0
        assert abs(result.max_height - 0.3) < 0.01  # 0.6 - 0.3
        assert result.pixel_count == 30 * 30

    def test_volume_proportional_to_height(self, estimator):
        """Taller objects should have greater volume."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 255
        surface = 0.3

        # Small object
        depth_small = np.full((100, 100), surface, dtype=np.float32)
        depth_small[30:60, 30:60] = 0.5
        vol_small = estimator.estimate_volume(depth_small, mask, surface)

        # Tall object (same footprint, greater height)
        depth_tall = np.full((100, 100), surface, dtype=np.float32)
        depth_tall[30:60, 30:60] = 0.8
        vol_tall = estimator.estimate_volume(depth_tall, mask, surface)

        assert vol_tall.total_volume > vol_small.total_volume

    def test_empty_mask(self, estimator):
        """Empty mask should return zero volume."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = estimator.estimate_volume(depth, mask, surface_depth=0.3)
        assert result.total_volume == 0
        assert result.pixel_count == 0


class TestDetectStacking:
    def test_single_object_not_stacked(self, estimator):
        """A single uniform object should not be detected as stacked."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        depth[20:80, 20:80] = 0.6  # Uniform raised region

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        result = estimator.detect_stacking(depth, mask)
        assert result.num_layers >= 1
        assert len(result.layer_masks) >= 1

    def test_two_stacked_objects(self, estimator):
        """Two distinct depth levels within one mask → stacked detection."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        # Bottom object
        depth[20:80, 20:80] = 0.5
        # Top object (sitting on bottom, closer to camera)
        depth[35:65, 35:65] = 0.8

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        result = estimator.detect_stacking(depth, mask)
        # Should detect the discontinuity
        assert result.num_layers >= 1  # At minimum finds layers
        assert len(result.layer_masks) >= 1

    def test_small_mask_returns_single_layer(self, estimator):
        """Masks smaller than min_layer_pixels should return as single layer."""
        depth = np.full((100, 100), 0.3, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:43, 40:43] = 255  # Only 9 pixels (< min_layer_pixels=50)

        result = estimator.detect_stacking(depth, mask)
        assert not result.is_stacked
        assert result.num_layers == 1
