import sqlite3
import os
import json

DB_FILE = "products.db"

def init_db():
    # Connect (creates file if not exists)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create Table with new columns for embeddings and weight variants
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            key_name TEXT PRIMARY KEY,
            display_name TEXT,
            price REAL,
            embedding BLOB,
            weight_variants TEXT
        )
    ''')

    # Migrate existing tables: add new columns if they don't exist
    c.execute("PRAGMA table_info(products)")
    columns = [row[1] for row in c.fetchall()]

    if 'embedding' not in columns:
        c.execute("ALTER TABLE products ADD COLUMN embedding BLOB")
        print("[MIGRATE] Added 'embedding' column.")

    if 'weight_variants' not in columns:
        c.execute("ALTER TABLE products ADD COLUMN weight_variants TEXT")
        print("[MIGRATE] Added 'weight_variants' column.")

    # Initial Data (Your Prices)
    # Format: (Model_Class_Name, Human_Readable_Name, Price, Embedding, Weight_Variants)
    # Embedding starts as NULL — populated by metric_train.py after training
    products = [
        ('50-50_maska_chaska', '50-50 Maska Chaska', 30, None, None),
        ('aim_matchstick', 'Aim Matchbox', 2, None, None),
        ('farmley_panchmeva', 'Farmley Panchmeva', 375, None, None),
        ('hajmola', 'Hajmola (Regular)', 65, None, None),
        ('maggi_ketchup', 'Maggi Ketchup', 15, None, None),
        ('moms_magic', 'Moms Magic Biscuits', 10, None, None),
        ('monaco', 'Monaco Biscuits', 35, None, None),
        ('tic_tac_toe', 'Tic Tac Toe', 20, None, None),
    ]

    # Example: How to add a product with weight variants
    # sugar_variants = json.dumps({
    #     "1kg": {"volume_min": 800, "volume_max": 1200, "price": 50},
    #     "2kg": {"volume_min": 1201, "volume_max": 2000, "price": 95}
    # })
    # products.append(('sugar', 'Sugar', 50, None, sugar_variants))

    print("--- Seeding Database ---")
    for p in products:
        print(f"Adding: {p[1]}")
        c.execute('INSERT OR REPLACE INTO products VALUES (?,?,?,?,?)', p)

    conn.commit()
    conn.close()
    print(f"\n[SUCCESS] Database created at '{os.path.abspath(DB_FILE)}'")

if __name__ == '__main__':
    init_db()

