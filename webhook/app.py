"""
app.py
======
Webhook server untuk WhatsApp Business API (Meta Cloud API).
Menerima pesan masuk dari WhatsApp -> tanya ke Minci (RAG + model chat via Ollama,
lihat CHAT_MODEL di query.py) -> balas otomatis.

Jalankan:
    uvicorn app:app --host 0.0.0.0 --port 8000

Lalu expose ke internet pakai Cloudflare Tunnel (lihat SETUP_GUIDE.md):
    cloudflared tunnel --url http://localhost:8000
"""

import os
import sys
import time
import logging
import requests
from fastapi import FastAPI, Request, Response, BackgroundTasks
from dotenv import load_dotenv

# webhook/ dan rag/ adalah folder TERPISAH (sejajar), jadi perlu ditambahkan
# ke sys.path dulu supaya query.py di folder rag/ bisa diimport dari sini
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from query import ask_minci  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minci-webhook")

app = FastAPI(title="Minci WhatsApp Webhook")

# ====== KONFIGURASI dari .env ======
WA_VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN", "minci-verify-token")
WA_ACCESS_TOKEN = os.getenv("WA_ACCESS_TOKEN")          # token akses WhatsApp Cloud API
WA_PHONE_NUMBER_ID = os.getenv("WA_PHONE_NUMBER_ID")     # phone_number_id dari Meta App
WA_API_VERSION = os.getenv("WA_API_VERSION", "v26.0")
# ====================================

WA_SEND_URL = f"https://graph.facebook.com/{WA_API_VERSION}/{WA_PHONE_NUMBER_ID}/messages"

# query.py yang baru SELALU memanggil LLM untuk setiap pesan (tidak ada lagi
# jalur fallback cepat tanpa model) -- artinya proses di background sekarang
# konsisten lambat (~10-20 detik). Timeout ini adalah pengaman AGAR koneksi ke
# WhatsApp API tidak menahan thread background selamanya kalau Meta lagi lemot,
# bukan timeout untuk proses RAG-nya sendiri.
WA_SEND_TIMEOUT_SECONDS = 30

# ============================================================
# DEDUPLIKASI PESAN: cegah jawaban dobel kalau Meta retry webhook
# ============================================================
# WhatsApp Cloud API akan RETRY (kirim ulang) notifikasi webhook kalau server kita
# tidak segera balas 200 OK (biasanya timeout beberapa detik). Karena proses RAG
# (embedding + cari dokumen + generate jawaban LLM) bisa makan waktu 10-20+ detik
# di model 3B CPU, Meta seringkali sudah keburu retry SEBELUM kita selesai proses
# -- akibatnya pesan yang SAMA diproses berkali-kali, jawaban dikirim berkali-kali.
#
# Solusinya DUA LAPIS:
# 1. Balas 200 OK ke Meta SECEPATNYA (sebelum proses RAG), proses beneran jalan
#    di BACKGROUND (lihat BackgroundTasks di bawah) -- ini FIX UTAMA.
# 2. Simpan ID pesan yang sudah diproses, supaya walau Meta TETAP retry (datang
#    lagi notifikasi untuk pesan yang sama), kita skip -- tidak diproses ulang.
#    Ini jaring pengaman KEDUA, bukan solusi utama.
_processed_message_ids: dict[str, float] = {}
DEDUP_WINDOW_SECONDS = 300  # anggap ID pesan "sudah pernah diproses" selama 5 menit


def _is_duplicate(message_id: str) -> bool:
    now = time.time()
    # Beres-beres ID lama supaya dict tidak membengkak tanpa batas
    expired = [mid for mid, ts in _processed_message_ids.items() if now - ts > DEDUP_WINDOW_SECONDS]
    for mid in expired:
        del _processed_message_ids[mid]

    if message_id in _processed_message_ids:
        return True

    _processed_message_ids[message_id] = now
    return False


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Endpoint verifikasi yang dipanggil Meta saat pertama kali setup webhook
    di dashboard Meta for Developers.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WA_VERIFY_TOKEN:
        logger.info("Webhook berhasil diverifikasi oleh Meta.")
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Verifikasi webhook GAGAL — cek WA_VERIFY_TOKEN di .env vs di dashboard Meta.")
    return Response(content="Verification failed", status_code=403)


@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint yang menerima notifikasi pesan masuk dari WhatsApp.

    PENTING: endpoint ini WAJIB balas cepat (200 OK) -- proses RAG yang lambat
    (embedding, cari dokumen, generate jawaban LLM) dilakukan di BACKGROUND
    lewat BackgroundTasks, BUKAN sebelum return. Ini mencegah Meta menganggap
    webhook "gagal" dan melakukan retry, yang tadinya menyebabkan jawaban dobel.
    """
    body = await request.json()

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Kalau ini cuma notifikasi status (delivered/read), bukan pesan baru -> abaikan
        if "messages" not in value:
            return {"status": "ok"}

        message = value["messages"][0]
        message_id = message.get("id")
        from_number = message["from"]  # nomor WA pengirim
        msg_type = message.get("type")

        # Cek duplikasi SEBELUM diproses -- kalau ID ini sudah pernah masuk
        # (Meta retry), skip total, jangan proses/balas lagi.
        if message_id and _is_duplicate(message_id):
            logger.info(f"Pesan {message_id} dari {from_number} adalah DUPLIKAT (retry Meta), di-skip.")
            return {"status": "ok"}

        if msg_type != "text":
            reply_text = (
                "Maaf kak, Minci saat ini baru bisa balas pesan teks ya 🙏 "
                "Coba tulis pertanyaannya dalam bentuk teks ya."
            )
            background_tasks.add_task(send_whatsapp_message, from_number, reply_text)
        else:
            user_text = message["text"]["body"]
            logger.info(f"Pesan dari {from_number}: {user_text}")
            # Proses RAG (lambat) + kirim balasan dijadwalkan di BACKGROUND,
            # supaya endpoint ini bisa langsung return 200 OK ke Meta tanpa nunggu.
            background_tasks.add_task(process_and_reply, from_number, user_text)

    except (KeyError, IndexError) as e:
        logger.error(f"Format payload tidak dikenali: {e}")

    return {"status": "ok"}


def process_and_reply(from_number: str, user_text: str):
    """
    Fungsi yang jalan di BACKGROUND (setelah endpoint sudah balas 200 OK ke Meta).
    Di sinilah proses RAG yang lambat (ask_minci) benar-benar dijalankan.
    """
    try:
        reply_text = ask_minci(user_text)
        send_whatsapp_message(to=from_number, text=reply_text, user_text=user_text)
    except Exception as e:
        logger.error(f"Gagal memproses/membalas pesan dari {from_number}: {e}")
        # Percobaan kedua ini (kirim pesan error ke user) juga bisa gagal --
        # mis. kalau penyebab error di atas justru koneksi ke WhatsApp API
        # (timeout/down), maka percobaan kirim pesan error ini kemungkinan
        # besar akan gagal juga dengan sebab yang sama. Dibungkus try/except
        # terpisah supaya kegagalan ini TERCATAT DI LOG, bukan cuma diam --
        # tanpa ini, exception di sini akan jadi unhandled exception di dalam
        # background thread dan user tidak dapat balasan apapun tanpa jejak log.
        try:
            send_whatsapp_message(
                to=from_number,
                text="Waduh, ada gangguan teknis nih kak 🙏 Coba tanya lagi sebentar ya.",
            )
        except Exception as notify_error:
            logger.error(
                f"Gagal juga mengirim pesan error ke {from_number} "
                f"(kemungkinan WhatsApp API/koneksi bermasalah): {notify_error}"
            )


def send_whatsapp_message(to: str, text: str, user_text: str = ""):
    """Kirim balasan teks ke user via WhatsApp Cloud API."""
    headers = {
        "Authorization": f"Bearer {WA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    resp = requests.post(WA_SEND_URL, headers=headers, json=payload, timeout=WA_SEND_TIMEOUT_SECONDS)

    if resp.status_code != 200:
        logger.error(f"Gagal kirim pesan WA: {resp.status_code} - {resp.text}")
    else:
        logger.info(f"Balasan terkirim ke {to}")
        if user_text:
            logger.info(f"Pesan: {user_text}")
        logger.info(f"Jawaban: {text}")


@app.get("/")
async def health_check():
    return {"status": "Minci webhook aktif ✅"}