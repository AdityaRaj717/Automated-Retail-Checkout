"""Tests for EmbeddingStore — vector database with cosine similarity."""

import sys
import os
import tempfile
import sqlite3
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.vector_db import EmbeddingStore, MatchResult


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary SQLite database with products table."""
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
    products = [
        ('maggi', 'Maggi Noodles', 15, None, None),
        ('monaco', 'Monaco Biscuits', 35, None, None),
        ('sugar_1kg', 'Sugar 1kg', 50, None, None),
    ]
    for p in products:
        c.execute('INSERT INTO products VALUES (?,?,?,?,?)', p)
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def store(db_path):
    return EmbeddingStore(db_path)


class TestEmbeddingStore:
    def test_initially_no_embeddings(self, store):
        assert not store.has_embeddings()
        assert len(store.get_all_keys()) == 0

    def test_add_and_retrieve(self, store):
        emb = np.random.randn(256).astype(np.float32)
        store.add_embedding('maggi', emb)

        assert store.has_embeddings()
        assert 'maggi' in store.get_all_keys()

    def test_find_nearest_identical(self, store):
        """Identical vectors should have similarity ≈ 1.0."""
        emb = np.random.randn(256).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        store.add_embedding('maggi', emb)

        results = store.find_nearest(emb, top_k=1)
        assert len(results) == 1
        assert results[0].key_name == 'maggi'
        assert abs(results[0].similarity - 1.0) < 0.01

    def test_find_nearest_ordering(self, store):
        """Closest embedding should be ranked first."""
        base = np.random.randn(256).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Similar to base
        similar = base + np.random.randn(256).astype(np.float32) * 0.1
        # Very different from base
        different = np.random.randn(256).astype(np.float32)

        store.add_embedding('maggi', similar)
        store.add_embedding('monaco', different)

        results = store.find_nearest(base, top_k=2)
        assert len(results) == 2
        assert results[0].key_name == 'maggi'
        assert results[0].similarity > results[1].similarity

    def test_top_k_limit(self, store):
        """Should return at most top_k results."""
        for i, name in enumerate(['maggi', 'monaco', 'sugar_1kg']):
            emb = np.random.randn(256).astype(np.float32)
            store.add_embedding(name, emb)

        results = store.find_nearest(np.random.randn(256), top_k=2)
        assert len(results) == 2

    def test_empty_query(self, store):
        """Zero vector query should return empty results."""
        emb = np.random.randn(256).astype(np.float32)
        store.add_embedding('maggi', emb)

        results = store.find_nearest(np.zeros(256), top_k=3)
        assert len(results) == 0

    def test_reload(self, store, db_path):
        """Reload should pick up new embeddings."""
        emb = np.random.randn(256).astype(np.float32)
        store.add_embedding('maggi', emb)

        # Simulate another process adding an embedding
        import pickle
        emb2 = np.random.randn(256).astype(np.float32)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("UPDATE products SET embedding = ? WHERE key_name = ?",
                  (pickle.dumps(emb2), 'monaco'))
        conn.commit()
        conn.close()

        store.reload()
        assert 'monaco' in store.get_all_keys()
