import sqlite3
import os

DB_FILE = "products.db"

def init_db():
    # Connect (creates file if not exists)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            key_name TEXT PRIMARY KEY,
            display_name TEXT,
            price REAL
        )
    ''')
    
    # Initial Data (Your Prices)
    # Format: (Model_Class_Name, Human_Readable_Name, Price)
    products = [
        ('50-50_maska_chaska', '50-50 Maska Chaska', 30),
        ('aim_matchstick', 'Aim Matchbox', 2),
        ('farmley_panchmeva', 'Farmley Panchmeva', 375),
        ('hajmola', 'Hajmola (Regular)', 65),
        ('maggi_ketchup', 'Maggi Ketchup', 15),
        ('moms_magic', 'Moms Magic Biscuits', 10),
        ('monaco', 'Monaco Biscuits', 35),
        ('tic_tac_toe', 'Tic Tac Toe', 20)
    ]
    
    print("--- Seeding Database ---")
    for p in products:
        print(f"Adding: {p[1]}")
        c.execute('INSERT OR REPLACE INTO products VALUES (?,?,?)', p)
    
    conn.commit()
    conn.close()
    print(f"\n[SUCCESS] Database created at '{os.path.abspath(DB_FILE)}'")

if __name__ == '__main__':
    init_db()
