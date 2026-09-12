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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from main import ask as ask_minci

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minci-webhook")

app = FastAPI(title="Minci WhatsApp Webhook")

WA_VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN", "minci-verify-token")
WA_ACCESS_TOKEN = os.getenv("WA_ACCESS_TOKEN")
WA_PHONE_NUMBER_ID = os.getenv("WA_PHONE_NUMBER_ID")
WA_API_VERSION = os.getenv("WA_API_VERSION", "v26.0")

WA_SEND_URL = f"https://graph.facebook.com/{WA_API_VERSION}/{WA_PHONE_NUMBER_ID}/messages"

WA_SEND_TIMEOUT_SECONDS = 30

_processed_message_ids: dict[str, float] = {}
DEDUP_WINDOW_SECONDS = 300


def _is_duplicate(message_id: str) -> bool:
    now = time.time()
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

        if "messages" not in value:
            return {"status": "ok"}

        message = value["messages"][0]
        message_id = message.get("id")
        from_number = message["from"]
        msg_type = message.get("type")

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