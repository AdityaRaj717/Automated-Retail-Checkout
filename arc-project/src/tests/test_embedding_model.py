"""Tests for RetailAttnNet — embedding model dual-mode forward pass."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from custom_model import RetailAttnNet


@pytest.fixture
def model():
    return RetailAttnNet(num_classes=8, embedding_dim=256)


class TestRetailAttnNet:
    def test_classification_output_shape(self, model):
        """Classification mode should output (batch, num_classes)."""
        x = torch.randn(2, 3, 224, 224)
        output = model(x, return_embedding=False)
        assert output.shape == (2, 8)

    def test_embedding_output_shape(self, model):
        """Embedding mode should output (batch, embedding_dim)."""
        x = torch.randn(2, 3, 224, 224)
        output = model(x, return_embedding=True)
        assert output.shape == (2, 256)

    def test_embeddings_are_normalized(self, model):
        """Embeddings should be L2-normalized (norm ≈ 1.0)."""
        x = torch.randn(4, 3, 224, 224)
        embeddings = model(x, return_embedding=True)
        norms = torch.norm(embeddings, p=2, dim=1)

        for norm in norms:
            assert abs(norm.item() - 1.0) < 0.01, f"Embedding norm = {norm.item()}, expected ≈ 1.0"

    def test_default_is_classification(self, model):
        """Default forward should be classification mode."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 8)

    def test_different_embedding_dims(self):
        """Model should work with different embedding dimensions."""
        for dim in [64, 128, 512]:
            m = RetailAttnNet(num_classes=5, embedding_dim=dim)
            m.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                emb = m(x, return_embedding=True)
            assert emb.shape == (1, dim)

    def test_batch_consistency(self, model):
        """Same input should produce same embeddings."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            emb1 = model(x, return_embedding=True)
            emb2 = model(x, return_embedding=True)

        assert torch.allclose(emb1, emb2, atol=1e-6)

    def test_single_image(self, model):
        """Should work with batch size 1."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(x, return_embedding=True)
            cls = model(x, return_embedding=False)
        assert emb.shape == (1, 256)
        assert cls.shape == (1, 8)
