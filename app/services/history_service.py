import os
import sqlite3
from datetime import datetime, timezone


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")


class HistoryService:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def add(self, filename: str, text: str) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO history (filename, text, created_at) VALUES (?, ?, ?)",
                (filename, text, now),
            )
            conn.commit()
            return {
                "id": cursor.lastrowid,
                "filename": filename,
                "text": text,
                "created_at": now,
            }

    def get_all(self) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, filename, text, created_at FROM history ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def delete(self, entry_id: int) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0
