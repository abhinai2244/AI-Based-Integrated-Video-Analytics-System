import sqlite3
import os

DB_PATH = 'security.db'
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        users = cursor.execute("SELECT * FROM users").fetchall()
        print(f"Users found: {len(users)}")
        for u in users:
            print(f"ID: {u['id']}, Username: {u['username']}, Role: {u['role']}")
    except Exception as e:
        print(f"Error: {e}")
    conn.close()
else:
    print("DB not found")
