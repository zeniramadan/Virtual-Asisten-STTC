"""
Penyimpanan riwayat percakapan (chat memory) pakai SQLite.

Kenapa SQLite dan bukan list di memori:
- Riwayat tetap ada walau skrip di-restart (mis. terminal ditutup, atau
  proses uvicorn di webhook WhatsApp crash/redeploy).
- Bisa dipakai lintas proses/request tanpa perlu simpan state global di
  memori Python (penting kalau nanti dipanggil dari app.py/webhook yang
  menangani banyak sesi/nomor sekaligus).
- Ringan, satu file, tidak butuh server DB terpisah.

Setiap baris = satu pesan (bukan satu turn), dikelompokkan per session_id
supaya beberapa percakapan (mis. beberapa nomor WhatsApp) tidak tercampur.
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager

import config

ChatHistory = list[dict[str, str]]


@contextmanager
def _connect():
    conn = sqlite3.connect(config.HISTORY_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Buat tabel kalau belum ada. Aman dipanggil berkali-kali."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session "
            "ON chat_history (session_id, id)"
        )
        conn.commit()


def load_history(session_id: str, max_turns: int | None = None) -> ChatHistory:
    """Ambil N pasang pesan terakhir untuk satu sesi, urut kronologis lama->baru."""
    max_turns = max_turns or config.MAX_HISTORY_TURNS
    limit = max_turns * 2

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def append_message(session_id: str, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def append_turn(session_id: str, question: str, answer: str) -> None:
    """Simpan sepasang pesan user+assistant sekaligus."""
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            [
                (session_id, "user", question),
                (session_id, "assistant", answer),
            ],
        )
        conn.commit()


def prune_old_messages(session_id: str, keep_turns: int | None = None) -> None:
    """Buang pesan lama di DB, sisakan cuma `keep_turns` pasang terakhir per
    sesi -- supaya tabel tidak membengkak selamanya untuk sesi yang panjang."""
    keep_turns = keep_turns or config.MAX_HISTORY_TURNS
    keep = keep_turns * 2

    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM chat_history
            WHERE session_id = ? AND id NOT IN (
                SELECT id FROM chat_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (session_id, session_id, keep),
        )
        conn.commit()


def reset_history(session_id: str) -> None:
    """Hapus semua riwayat satu sesi (mis. saat user ketik 'reset')."""
    with _connect() as conn:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
