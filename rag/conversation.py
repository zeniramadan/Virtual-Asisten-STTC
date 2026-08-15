"""
conversation.py
======================
Modul penyimpanan riwayat percakapan pakai SQLite, dengan AUTO-EXPIRE:
kalau seorang user tidak chat lagi selama lebih dari 1 jam, riwayat obrolannya
otomatis dianggap kadaluarsa dan dihapus saat dia chat lagi nanti.

Beda dengan versi sebelumnya (simpan di RAM/dict Python biasa), riwayat di sini
TERSIMPAN PERMANEN di file conversations.db -- jadi tidak hilang walau proses
(uvicorn / python telegram_bot.py) di-restart, kecuali memang sudah expired.

Tidak perlu install apa-apa tambahan -- sqlite3 sudah bawaan Python.
"""

import os
import sqlite3
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")

# Simpan file db di dalam folder database
DB_PATH = os.path.join(DATABASE_DIR, "conversations.db") 

EXPIRE_AFTER_SECONDS = 60 * 60   # 1 jam
MAX_HISTORY_TURNS = 5            # jumlah PASANGAN (user+assistant) terakhir yang disimpan per user

# SQLite dari Python standar tidak otomatis aman dipakai dari banyak thread sekaligus
# (uvicorn/FastAPI bisa proses beberapa request bersamaan) -- lock sederhana ini
# mencegah dua request nulis ke database di waktu yang sama secara bersamaan.
_lock = threading.Lock()


def _get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id)")
    return conn


def _cleanup_if_expired(conn, user_id: str) -> bool:
    """Kalau pesan TERAKHIR user ini sudah lebih dari 1 jam lalu, hapus semua riwayatnya."""
    row = conn.execute(
        "SELECT MAX(created_at) FROM messages WHERE user_id = ?", (user_id,)
    ).fetchone()
    last_active = row[0]

    if last_active is not None and (time.time() - last_active) > EXPIRE_AFTER_SECONDS:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
    return False


def get_history(user_id: str) -> list[dict]:
    """
    Ambil riwayat percakapan user ini, urut dari yang paling lama ke paling baru.
    Otomatis mengosongkan riwayat dulu kalau ternyata sudah expired (> 1 jam tidak aktif).
    """
    with _lock:
        conn = _get_connection()
        try:
            _cleanup_if_expired(conn, user_id)
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id ASC",
                (user_id,),
            ).fetchall()
            return [{"role": r, "content": c} for r, c in rows]
        finally:
            conn.close()


def add_message(user_id: str, role: str, content: str):
    """
    Simpan satu pesan baru ke riwayat user ini. Otomatis membuang pesan paling lama
    kalau riwayatnya sudah melebihi MAX_HISTORY_TURNS pasang.
    """
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )
            conn.commit()

            max_messages = MAX_HISTORY_TURNS * 2
            total = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
            ).fetchone()[0]

            if total > max_messages:
                excess = total - max_messages
                conn.execute(
                    """
                    DELETE FROM messages WHERE id IN (
                        SELECT id FROM messages WHERE user_id = ?
                        ORDER BY id ASC LIMIT ?
                    )
                    """,
                    (user_id, excess),
                )
                conn.commit()
        finally:
            conn.close()


def reset_history(user_id: str):
    """Hapus semua riwayat percakapan user ini secara manual (misal user ketik 'reset')."""
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()


def cleanup_all_expired() -> int:
    """
    Bersihkan SEMUA user yang riwayatnya sudah expired sekaligus.
    Ini OPSIONAL -- auto-expire per user sudah jalan otomatis tiap kali get_history()
    dipanggil. Fungsi ini cuma buat "beres-beres" database secara berkala (misal
    dijadwalkan jalan tiap beberapa jam) supaya file conversations.db tidak
    membengkak menyimpan riwayat user yang sudah lama tidak aktif.
    Return: jumlah baris pesan yang terhapus.
    """
    with _lock:
        conn = _get_connection()
        try:
            cutoff = time.time() - EXPIRE_AFTER_SECONDS
            cur = conn.execute(
                """
                DELETE FROM messages
                WHERE user_id IN (
                    SELECT user_id FROM messages
                    GROUP BY user_id
                    HAVING MAX(created_at) < ?
                )
                """,
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
