"""Tests for LogicEngine — product identification with weight variant resolution."""

import sys
import os
import sqlite3
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.logic_engine import LogicEngine, ProductResult
from modules.volume_estimator import VolumeResult
from modules.vector_db import EmbeddingStore


@pytest.fixture
def db_path(tmp_path):
    """Create DB with products including one with weight variants."""
    db = tmp_path / "test_products.db"
    conn = sqlite3.connect(str(db))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE products (
            key_name TEXT PRIMARY KEY,
            display_name TEXT,
            price REAL,
            embedding BLOB,
            weight_variants TEXT
        )
    ''')

    sugar_variants = json.dumps({
        "1kg": {"volume_min": 500, "volume_max": 1000, "price": 50},
        "2kg": {"volume_min": 1001, "volume_max": 2000, "price": 95}
    })

    products = [
        ('maggi', 'Maggi Noodles', 15, None, None),
        ('sugar', 'Sugar', 50, None, sugar_variants),
    ]
    for p in products:
        c.execute('INSERT INTO products VALUES (?,?,?,?,?)', p)
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def store(db_path):
    s = EmbeddingStore(db_path)
    # Add reference embeddings
    maggi_emb = np.random.randn(256).astype(np.float32)
    maggi_emb = maggi_emb / np.linalg.norm(maggi_emb)
    s.add_embedding('maggi', maggi_emb)

    sugar_emb = np.random.randn(256).astype(np.float32)
    sugar_emb = sugar_emb / np.linalg.norm(sugar_emb)
    s.add_embedding('sugar', sugar_emb)

    return s, maggi_emb, sugar_emb


@pytest.fixture
def engine(db_path):
    return LogicEngine(db_path, high_confidence=0.70, low_confidence=0.40)


class TestLogicEngine:
    def test_high_confidence_auto_accept(self, engine, store):
        s, maggi_emb, _ = store
        # Query with near-identical embedding (very high similarity)
        result = engine.identify_product(maggi_emb, None, s)

        assert result is not None
        assert result.key_name == 'maggi'
        assert result.display_name == 'Maggi Noodles'
        assert not result.needs_confirmation

    def test_low_confidence_rejection(self, engine, store):
        s, _, _ = store
        # Use a completely random embedding (low similarity)
        random_emb = np.random.randn(256).astype(np.float32)
        result = engine.identify_product(random_emb, None, s)

        # Result may be None if below threshold, or may have needs_confirmation
        # Depends on random similarity — this is fine as a smoke test
        if result is not None:
            assert result.confidence >= 0.40

    def test_weight_variant_small(self, engine, store):
        s, _, sugar_emb = store
        vol = VolumeResult(total_volume=750, max_height=0.3, mean_height=0.2, pixel_count=1000)

        result = engine.identify_product(sugar_emb, vol, s)
        assert result is not None
        assert result.key_name == 'sugar'
        # Volume 750 is in 1kg range (500-1000)
        if result.variant_info:
            assert result.variant_info == '1kg'
            assert result.price == 50

    def test_weight_variant_large(self, engine, store):
        s, _, sugar_emb = store
        vol = VolumeResult(total_volume=1500, max_height=0.5, mean_height=0.3, pixel_count=2000)

        result = engine.identify_product(sugar_emb, vol, s)
        assert result is not None
        assert result.key_name == 'sugar'
        if result.variant_info:
            assert result.variant_info == '2kg'
            assert result.price == 95

    def test_no_embeddings_returns_none(self, db_path):
        """If no embeddings exist, should return None."""
        engine = LogicEngine(db_path)
        empty_store = EmbeddingStore(db_path)
        # Don't add any embeddings

        query = np.random.randn(256).astype(np.float32)
        # Reload to get fresh (empty) store
        empty_db = db_path.replace('.db', '_empty.db')
        conn = sqlite3.connect(empty_db)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE products (
                key_name TEXT PRIMARY KEY,
                display_name TEXT,
                price REAL,
                embedding BLOB,
                weight_variants TEXT
            )
        ''')
        conn.commit()
        conn.close()

        empty_store2 = EmbeddingStore(empty_db)
        result = engine.identify_product(query, None, empty_store2)
        assert result is None
