"""
app.py
======
Webhook server untuk WhatsApp Business API (Meta Cloud API).
Menerima pesan masuk dari WhatsApp -> tanya ke Minci (RAG + model LoRA) -> balas otomatis.

Jalankan:
    uvicorn app:app --host 0.0.0.0 --port 8000

Lalu expose ke internet pakai Cloudflare Tunnel (lihat SETUP_GUIDE.md):
    cloudflared tunnel --url http://localhost:8000
"""

import os
import sys
import logging
import requests
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv

# Supaya bisa import ask_minci dari folder rag
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag"))
from query import ask_minci  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minci-webhook")

app = FastAPI(title="Minci WhatsApp Webhook")

# ====== KONFIGURASI dari .env ======
WA_VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN", "minci-verify-token")
WA_ACCESS_TOKEN = os.getenv("WA_ACCESS_TOKEN")          # token akses WhatsApp Cloud API
WA_PHONE_NUMBER_ID = os.getenv("WA_PHONE_NUMBER_ID")     # phone_number_id dari Meta App
WA_API_VERSION = os.getenv("WA_API_VERSION", "v20.0")
# ====================================

WA_SEND_URL = f"https://graph.facebook.com/{WA_API_VERSION}/{WA_PHONE_NUMBER_ID}/messages"


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
async def receive_message(request: Request):
    """
    Endpoint yang menerima notifikasi pesan masuk dari WhatsApp.
    """
    body = await request.json()
    logger.info(f"Payload masuk: {body}")

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Kalau ini cuma notifikasi status (delivered/read), bukan pesan baru -> abaikan
        if "messages" not in value:
            return {"status": "ok"}

        message = value["messages"][0]
        from_number = message["from"]  # nomor WA pengirim
        msg_type = message.get("type")

        if msg_type != "text":
            reply_text = (
                "Maaf kak, Minci saat ini baru bisa balas pesan teks ya 🙏 "
                "Coba tulis pertanyaannya dalam bentuk teks ya."
            )
        else:
            user_text = message["text"]["body"]
            logger.info(f"Pesan dari {from_number}: {user_text}")
            # from_number dipakai sebagai user_id, supaya riwayat obrolan tiap nomor WA
            # tersimpan terpisah (Minci ingat konteks per orang, bukan campur aduk)
            reply_text = ask_minci(user_text, user_id=from_number)

        send_whatsapp_message(to=from_number, text=reply_text)

    except (KeyError, IndexError) as e:
        logger.error(f"Format payload tidak dikenali: {e}")

    return {"status": "ok"}


def send_whatsapp_message(to: str, text: str):
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

    resp = requests.post(WA_SEND_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error(f"Gagal kirim pesan WA: {resp.status_code} - {resp.text}")
    else:
        logger.info(f"Balasan terkirim ke {to}")


@app.get("/")
async def health_check():
    return {"status": "Minci webhook aktif ✅"}