"""
SQLite Database Module
======================
Manages the products catalog and transaction history.
Auto-seeds products from products.json on first run.
"""

import os
import json
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "checkout.db")
PRODUCTS_FILE = os.path.join(os.path.dirname(__file__), "products.json")


def get_connection():
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables and seed products if needed."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            image_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total REAL NOT NULL,
            item_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transaction_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            subtotal REAL NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES transactions(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
    """)

    # Seed products from products.json if table is empty
    count = cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if count == 0:
        with open(PRODUCTS_FILE, "r") as f:
            catalog = json.load(f)

        for slug, info in catalog.items():
            # Count images in processed dataset
            img_dir = os.path.join(os.path.dirname(__file__), "dataset", "processed", slug)
            img_count = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0

            cursor.execute(
                "INSERT INTO products (slug, name, price, image_count) VALUES (?, ?, ?, ?)",
                (slug, info["name"], info["price"], img_count),
            )
        print(f"Seeded {len(catalog)} products into database")

    conn.commit()
    conn.close()


def get_all_products():
    """Return all products as a list of dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM products ORDER BY name").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_product_by_slug(slug: str):
    """Fetch a single product by its slug."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM products WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_transaction(items: list) -> int:
    """
    Save a completed checkout transaction.

    Args:
        items: list of dicts with keys: product_id, quantity, subtotal

    Returns:
        transaction ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    total = sum(item["subtotal"] for item in items)
    item_count = sum(item["quantity"] for item in items)
    timestamp = datetime.now().isoformat()

    cursor.execute(
        "INSERT INTO transactions (timestamp, total, item_count) VALUES (?, ?, ?)",
        (timestamp, total, item_count),
    )
    txn_id = cursor.lastrowid

    for item in items:
        cursor.execute(
            "INSERT INTO transaction_items (transaction_id, product_id, quantity, subtotal) VALUES (?, ?, ?, ?)",
            (txn_id, item["product_id"], item["quantity"], item["subtotal"]),
        )

    conn.commit()
    conn.close()
    return txn_id


def get_transactions(limit: int = 20):
    """Get recent transactions with their items."""
    conn = get_connection()
    txns = conn.execute(
        "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()

    result = []
    for txn in txns:
        items = conn.execute("""
            SELECT ti.*, p.name, p.slug 
            FROM transaction_items ti
            JOIN products p ON ti.product_id = p.id
            WHERE ti.transaction_id = ?
        """, (txn["id"],)).fetchall()

        result.append({
            **dict(txn),
            "items": [dict(item) for item in items],
        })

    conn.close()
    return result


# Initialize on import
init_db()
