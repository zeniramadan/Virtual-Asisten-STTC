"""
telegram_bot.py
================
Versi TESTING Minci via Telegram Bot API — dipakai untuk uji coba cepat tanpa perlu
setup Cloudflare Tunnel / webhook publik, karena pakai LONG POLLING (bot yang aktif
"nanya terus" ke server Telegram, bukan Telegram yang kirim ke kita).

Token ditaruh di file .env untuk keamanan.

Cara dapat token bot:
    1. Buka Telegram, chat ke @BotFather
    2. Ketik /newbot, ikuti instruksinya (kasih nama & username bot)
    3. BotFather akan kasih token seperti: 123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    4. Copy token itu, paste di TELEGRAM_TOKEN dalam file .env

Cara jalankan:
    pip install requests python-dotenv
    python telegram_bot.py

Lalu buka chat ke bot kamu di Telegram dan mulai tanya-tanya.
"""

import os
import sys
import time
import logging
import requests
from dotenv import load_dotenv

# Load variabel dari file .env
load_dotenv()

# Supaya bisa import ask_minci dari folder rag
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag"))
from query import ask_minci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("minci-telegram")

# ====== AMBIL TOKEN BOT DARI .env ======
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# ==========================================

if not TELEGRAM_TOKEN:
    print("❌ TELEGRAM_TOKEN belum diisi di file .env!")
    print("   Buka file .env, tambahkan TELEGRAM_TOKEN=token_dari_BotFather")
    sys.exit(1)

API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def get_updates(offset=None, timeout=30):
    """Long polling: nanya ke server Telegram 'ada pesan baru gak?' setiap beberapa detik."""
    params = {"timeout": timeout}
    if offset:
        params["offset"] = offset

    resp = requests.get(f"{API_URL}/getUpdates", params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json().get("result", [])


def send_message(chat_id: int, text: str):
    """Kirim balasan teks ke user Telegram."""
    payload = {"chat_id": chat_id, "text": text}
    resp = requests.post(f"{API_URL}/sendMessage", json=payload)

    if resp.status_code != 200:
        logger.error(f"Gagal kirim pesan: {resp.status_code} - {resp.text}")


def send_typing_action(chat_id: int):
    """Kasih indikator 'sedang mengetik...' biar user tahu bot lagi proses (opsional, tapi enak buat UX)."""
    try:
        requests.post(f"{API_URL}/sendChatAction", json={"chat_id": chat_id, "action": "typing"})
    except Exception:
        pass


def main():
    logger.info("🤖 Bot Telegram Minci aktif. Menunggu pesan...")

    # Cek dulu token valid & tampilkan info bot
    me = requests.get(f"{API_URL}/getMe").json()
    if not me.get("ok"):
        logger.error(f"Token tidak valid atau ada masalah koneksi: {me}")
        sys.exit(1)

    bot_info = me["result"]
    logger.info(f"✅ Terhubung sebagai @{bot_info['username']} ({bot_info['first_name']})")
    logger.info("   Buka Telegram, cari bot ini, dan mulai chat!\n")

    offset = None

    while True:
        try:
            updates = get_updates(offset=offset)

            for update in updates:
                offset = update["update_id"] + 1

                message = update.get("message")
                if not message:
                    continue

                chat_id = message["chat"]["id"]
                text = message.get("text")

                if not text:
                    send_message(chat_id, "Maaf kak, Minci baru bisa balas pesan teks ya 🙏")
                    continue

                logger.info(f"Pesan dari {chat_id}: {text}")
                send_typing_action(chat_id)

                try:
                    # chat_id dipakai sebagai user_id, supaya riwayat obrolan tiap chat
                    # Telegram tersimpan terpisah (Minci ingat konteks per orang)
                    jawaban = ask_minci(text, user_id=str(chat_id))
                except Exception as e:
                    logger.error(f"Error saat generate jawaban: {e}")
                    jawaban = (
                        "Waduh, ada gangguan teknis nih kak 🙏 Coba tanya lagi sebentar ya, "
                        "atau hubungi bagian akademik langsung kalau mendesak."
                    )

                send_message(chat_id, jawaban)
                logger.info(f"Balasan terkirim ke {chat_id}")

        except requests.exceptions.ReadTimeout:
            # Wajar terjadi karena kita pakai long polling dengan timeout, lanjut loop lagi
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Gangguan koneksi ke Telegram: {e}")
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Bot dihentikan manual.")
            break


if __name__ == "__main__":
    main()