"""
Logic Engine Module — Product Decision & Weight Variant Resolution.

Combines recognition results (embedding similarity) with volumetric data
to make the final product identification decision.

Key Logic:
    - If product has weight variants AND volume data is available:
      Match volume to the correct variant (e.g., "Sugar 1kg" vs "Sugar 2kg")
    - If confidence is ambiguous (40-70%): flag for active learning / cashier review
    - Otherwise: return best match directly
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional, List
from modules.volume_estimator import VolumeResult
from modules.vector_db import EmbeddingStore, MatchResult


@dataclass
class ProductResult:
    """Final identification result for a single detected object."""
    key_name: str              # Product identifier
    display_name: str          # Human-readable name
    price: float               # Unit price
    confidence: float          # Similarity score (0-1 range)
    volume: Optional[float]    # Estimated volume (if depth was used)
    needs_confirmation: bool   # True if confidence is in the ambiguous zone
    variant_info: Optional[str] = None  # e.g., "2kg" if weight variant resolved


class LogicEngine:
    """
    Decision engine that combines embedding similarity + volume to identify products.

    Workflow:
        1. Find nearest product by embedding (cosine similarity)
        2. If product has weight_variants, resolve using volume data
        3. Apply confidence thresholds:
           - > 70%: auto-accept
           - 40-70%: flag for cashier confirmation (active learning)
           - < 40%: reject as unrecognized

    Usage:
        engine = LogicEngine(db_path="products.db")
        result = engine.identify_product(embedding, volume_result, embedding_store)
    """

    def __init__(self, db_path: str,
                 high_confidence: float = 0.70,
                 low_confidence: float = 0.40):
        """
        Args:
            db_path: Path to the SQLite products database
            high_confidence: Above this threshold → auto-accept
            low_confidence: Below this threshold → reject
        """
        self.db_path = db_path
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence

    def _get_product_info(self, key_name: str) -> dict:
        """Fetch product details from the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT display_name, price, weight_variants FROM products WHERE key_name = ?",
                  (key_name,))
        row = c.fetchone()
        conn.close()

        if row is None:
            return {'display_name': key_name, 'price': 0, 'weight_variants': None}

        weight_variants = None
        if row[2]:
            try:
                weight_variants = json.loads(row[2])
            except json.JSONDecodeError:
                weight_variants = None

        return {
            'display_name': row[0],
            'price': row[1],
            'weight_variants': weight_variants
        }

    def _resolve_weight_variant(self, product_info: dict,
                                 volume: float) -> Optional[dict]:
        """
        Given a product with weight_variants and a measured volume,
        return the matching variant.

        weight_variants format (JSON stored in DB):
        {
            "1kg": {"volume_min": 800, "volume_max": 1200, "price": 50},
            "2kg": {"volume_min": 1201, "volume_max": 2000, "price": 95}
        }
        """
        variants = product_info.get('weight_variants')
        if not variants:
            return None

        for variant_name, thresholds in variants.items():
            v_min = thresholds.get('volume_min', 0)
            v_max = thresholds.get('volume_max', float('inf'))

            if v_min <= volume <= v_max:
                return {
                    'variant_name': variant_name,
                    'price': thresholds.get('price', product_info['price']),
                    'display_name': f"{product_info['display_name']} ({variant_name})"
                }

        # No matching variant found — use default
        return None

    def identify_product(self, embedding,
                         volume_result: Optional[VolumeResult],
                         embedding_store: EmbeddingStore) -> Optional[ProductResult]:
        """
        Main decision function: embedding + volume → product identification.

        Args:
            embedding: numpy array (embedding_dim,) — the live object's embedding
            volume_result: VolumeResult from volume estimation (can be None if no depth)
            embedding_store: The vector database to search against

        Returns:
            ProductResult or None if confidence is below the rejection threshold
        """
        # Step 1: Find nearest product by embedding
        matches = embedding_store.find_nearest(embedding, top_k=3)

        if not matches:
            return None

        best_match = matches[0]
        confidence = best_match.similarity

        # Step 2: Reject if below low confidence threshold
        if confidence < self.low_confidence:
            return None

        # Step 3: Get product info from DB
        product_info = self._get_product_info(best_match.key_name)

        # Step 4: Resolve weight variant if volume data is available
        display_name = product_info['display_name']
        price = product_info['price']
        variant_info = None

        volume_val = volume_result.total_volume if volume_result else None

        if volume_val is not None and product_info.get('weight_variants'):
            variant = self._resolve_weight_variant(product_info, volume_val)
            if variant:
                display_name = variant['display_name']
                price = variant['price']
                variant_info = variant['variant_name']

        # Step 5: Determine if cashier confirmation is needed
        needs_confirmation = self.low_confidence <= confidence < self.high_confidence

        return ProductResult(
            key_name=best_match.key_name,
            display_name=display_name,
            price=price,
            confidence=confidence,
            volume=volume_val,
            needs_confirmation=needs_confirmation,
            variant_info=variant_info
        )
