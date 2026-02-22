"""
Vector Database Module — Embedding Store with Cosine Similarity Lookup.

Stores product embeddings as BLOBs in SQLite and performs nearest-neighbor
lookup using cosine similarity. This enables "zero-shot" product registration:
a new product is added by saving ONE reference embedding, not retraining.

Research Value:
    - Replaces hard classification with similarity-based matching
    - New products added instantly without retraining
    - Supports hot-reloading when new embeddings are inserted
"""

import sqlite3
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of a nearest-neighbor embedding lookup."""
    key_name: str
    similarity: float  # Cosine similarity in [-1, 1], higher = more similar


class EmbeddingStore:
    """
    SQLite-backed vector database for product embeddings.

    Embeddings are stored as pickled numpy arrays in a BLOB column.
    Nearest-neighbor search uses cosine similarity computed in Python
    (fast enough for <1000 products; no external vector DB needed).

    Usage:
        store = EmbeddingStore("products.db")
        store.add_embedding("maggi_ketchup", embedding_vector)
        matches = store.find_nearest(query_embedding, top_k=3)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache = {}  # In-memory cache: {key_name: np.ndarray}
        self._ensure_schema()
        self._load_cache()

    def _ensure_schema(self):
        """Ensure the embedding column exists in the products table."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Check if embedding column exists
        c.execute("PRAGMA table_info(products)")
        columns = [row[1] for row in c.fetchall()]

        if 'embedding' not in columns:
            c.execute("ALTER TABLE products ADD COLUMN embedding BLOB")
            print("[VECTOR_DB] Added 'embedding' column to products table.")

        if 'weight_variants' not in columns:
            c.execute("ALTER TABLE products ADD COLUMN weight_variants TEXT")
            print("[VECTOR_DB] Added 'weight_variants' column to products table.")

        conn.commit()
        conn.close()

    def _load_cache(self):
        """Load all embeddings into memory for fast similarity search."""
        self._cache = {}
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT key_name, embedding FROM products WHERE embedding IS NOT NULL")

        for key_name, emb_blob in c.fetchall():
            try:
                embedding = pickle.loads(emb_blob)
                self._cache[key_name] = embedding.astype(np.float32)
            except Exception as e:
                print(f"[VECTOR_DB] Warning: Could not load embedding for '{key_name}': {e}")

        conn.close()
        print(f"[VECTOR_DB] Loaded {len(self._cache)} embeddings into cache.")

    def add_embedding(self, key_name: str, embedding: np.ndarray):
        """
        Store an embedding for a product. Overwrites if exists.

        Args:
            key_name: Product identifier (must match products table primary key).
            embedding: numpy array of shape (embedding_dim,), e.g. (256,).
        """
        embedding = embedding.astype(np.float32).flatten()
        emb_blob = pickle.dumps(embedding)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE products SET embedding = ? WHERE key_name = ?", (emb_blob, key_name))

        if c.rowcount == 0:
            print(f"[VECTOR_DB] Warning: No product found with key_name='{key_name}'. "
                  f"Embedding saved but product may not exist in DB.")

        conn.commit()
        conn.close()

        # Update cache
        self._cache[key_name] = embedding
        print(f"[VECTOR_DB] Stored embedding for '{key_name}' ({len(embedding)}-dim)")

    def find_nearest(self, query_embedding: np.ndarray, top_k: int = 3) -> List[MatchResult]:
        """
        Find the top-k most similar products by cosine similarity.

        Args:
            query_embedding: (embedding_dim,) numpy array.
            top_k: Number of results to return.

        Returns:
            List of MatchResult sorted by similarity (highest first).
            Empty list if no embeddings in the database.
        """
        if len(self._cache) == 0:
            return []

        query = query_embedding.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        query = query / query_norm

        similarities = []
        for key_name, stored_emb in self._cache.items():
            stored_norm = np.linalg.norm(stored_emb)
            if stored_norm < 1e-8:
                continue
            normalized = stored_emb / stored_norm
            sim = float(np.dot(query, normalized))
            similarities.append(MatchResult(key_name=key_name, similarity=sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x.similarity, reverse=True)

        return similarities[:top_k]

    def reload(self):
        """Hot-reload embeddings from the database (e.g., after new product added)."""
        self._load_cache()

    def get_all_keys(self) -> List[str]:
        """Return all product keys that have embeddings."""
        return list(self._cache.keys())

    def has_embeddings(self) -> bool:
        """Check if there are any stored embeddings."""
        return len(self._cache) > 0
