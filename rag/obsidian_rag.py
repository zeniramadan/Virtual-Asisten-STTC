"""
Skrip khusus untuk Chat Interaktif & Retrieval RAG.
Jalankan ini: python obsidian_rag.py
"""

from __future__ import annotations
import os
import re
import sys
import json

import chromadb
import ollama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "obsidian_vault"

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.1"

RETRIEVAL_K = 15          
FINAL_CONTEXT_K = 6       
MAX_DISTANCE = 0.60       
MIN_OVERLAP_IF_LONG_QUERY = 1  
DEBUG = True

# ============================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================

_STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan",
    "ini", "itu", "ada", "apa", "apakah", "bagaimana", "berapa",
    "kapan", "dimana", "mana", "saja", "aja", "adalah", "pada",
}

# ============================================================
# CHITCHAT DETECTION (basa-basi → skip RAG)
# ============================================================

CHITCHAT_PATH = os.path.join(BASE_DIR, "..", "dataset", "chitchat.json")


def _load_chitchat_phrases() -> list[str]:
    """Baca semua frasa chit-chat dari chitchat.json jadi satu list flat."""
    try:
        with open(CHITCHAT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    phrases = []
    for _kategori, daftar in data.items():
        phrases.extend(daftar)
    phrases.sort(key=len, reverse=True)
    return phrases


_CHITCHAT_PHRASES = _load_chitchat_phrases()

if DEBUG:
    print(f"[CHITCHAT] {len(_CHITCHAT_PHRASES)} frasa dimuat")


def is_chitchat(question: str) -> bool:
    """True kalau pertanyaan cuma basa-basi tanpa substansi akademik."""
    if not _CHITCHAT_PHRASES:
        return False
    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[!?.,]+$", "", q)
    for phrase in _CHITCHAT_PHRASES:
        if q == phrase:
            return True
        if q.startswith(phrase + " ") or q.startswith(phrase + ","):
            remainder = q[len(phrase):].strip(" ,.-")
            if len(remainder.split()) <= 3:
                return True
    return False


CHITCHAT_SYSTEM_PROMPT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Generasi Z, tapi tetap sopan.

TUGAS UTAMA:
Jawab sapaan, salam, ucapan terima kasih, atau obrolan ringan (chitchat) dari pengguna dengan SINGKAT (maksimal 2 kalimat) dan super natural!

ATURAN BALASAN SESUAI KONTEKS:
1. Jika pengguna MENYAPA (halo, hai, pagi, siang, sore, malam), balas sapaannya dengan ceria, lalu tawarkan bantuan seputar PMB, KRS, atau biaya.
2. Jika pengguna MENGUCAP SALAM (assalamualaikum), wajib balas "Waalaikumsalam kak!" lalu tawarkan bantuan.
3. Jika pengguna berterima kasih (makasih, thank you), balas dengan "Sama-sama kak! Senang bisa bantu."
4. Jika pengguna BERTANYA HAL LAIN (seperti "lagi apa?", "kamu siapa?", "mau nanya"), jawab sesuai pertanyaan ringan mereka dengan gaya santai Gen-Z, lalu arahkan kembali agar mereka bertanya tentang PMB, KRS, atau biaya.

KATA KUNCI LARANGAN KERAS:
- HARUS menggunakan kata "kak" atau "kakak"!
- DILARANG KERAS menggunakan kata "Kamu" atau "Anda" saat menyapa pengguna!
- JANGAN PERNAH memberikan jawaban template "Sama-sama" jika pengguna tidak sedang berterima kasih!
- JANGAN mengarang atau memberikan informasi akademik palsu di sini!
"""

# ============================================================
# RETRIEVAL HELPERS
# ============================================================

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NO_telemetry"] = "True"

def meaningful_tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }

def get_collection():
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_with_debug(question: str) -> list[dict]:
    print(f"\n{'='*50}")
    print(f"🔍 [DEBUG] Pertanyaan: \"{question}\"")
    
    q_tokens = meaningful_tokens(question)
    collection = get_collection()
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    results = collection.query(query_embeddings=[embedding], n_results=RETRIEVAL_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for idx, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        distance = float(distance)
        overlap = len(q_tokens & meaningful_tokens(document))
        
        passed_distance = distance <= MAX_DISTANCE
        passed_overlap = (len(q_tokens) < 2) or (overlap >= MIN_OVERLAP_IF_LONG_QUERY)
        is_valid = passed_distance and passed_overlap

        status = "✅ LOLOS" if is_valid else "❌ DIBUANG"
        title = metadata.get('title', '?')
        print(f"   [{idx+1}] Note: {title} | Jarak: {distance:.4f} | Overlap: {overlap} | Status: {status}")

        if not is_valid:
            continue

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "overlap": overlap,
        })

    candidates.sort(key=lambda c: (c["distance"], -c["overlap"]))
    final_chunks = candidates[:FINAL_CONTEXT_K]
    print(f"📌 [DEBUG] Total chunk terpilih untuk LLM: {len(final_chunks)}")
    print(f"{'='*50}\n")
    
    return final_chunks

# ============================================================
# LLM PROMPTS
# ============================================================

SYSTEM_PROMPT = """Kamu adalah asisten akademik yang menjawab pertanyaan tentang PMB, KRS, Jadwal dan Biaya..

CONTEXT di bawah ini SUDAH DIPASTIKAN BERISI DATA PANDUAN YANG RELEVAN dengan pertanyaan. Kamu WAJIB menjawab dari situ!

ATURAN JAWABAN:
- Jawab pertanyaan pengguna HANYA berdasarkan informasi faktual yang tertulis di dalam CONTEXT tersebut.
- Jika bertanya tentang "daftar", "pendaftaran", JAWAB dengan SYARAT PENDAFTARAN.

ATURAN WAJIB UNTUK SEMUA JAWABAN:
- HARUS menggunakan kata "kak" atau "kakak"! DILARANG menggunakan kata "Kamu".
- Gunakan bullet "-" untuk menampilkan data yang berbentuk daftar.
- JANGAN PERNAH menyebutkan kata teknis seperti "context", "metadata", "chunk", atau "RAG".
- JANGAN menyebut nomor bagian internal seperti "CHUNK 1", "CHUNK 5", atau "CHUNK 6". Langsung sebutkan informasi dan tanggalnya.
"""

# ============================================================
# CONTEXT AND CHAT API
# ============================================================

FALLBACK_TEXT = (
    "Maaf kak, informasi yang kakak tanyakan tidak ada di panduan kami, "
    "coba bertanya lebih spesifik, atau silakan kakak hubungi bagian Tata Usaha ya!"
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        title = c["metadata"].get("title", "?")
        path = c["metadata"].get("path", "?")
        parts.append(f"[Note: {title} ({path})]\n{c['document']}")
    return "\n\n---\n\n".join(parts)


def chat_with_model(system_prompt: str, user_prompt: str, **options) -> str:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=options,
    )
    return response.get("message", {}).get("content", "").strip()


def answer_chitchat(question: str) -> str:
    return chat_with_model(
        CHITCHAT_SYSTEM_PROMPT,
        question,
        temperature=0.4,
        num_predict=100,
    )


def answer_with_context(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    return chat_with_model(
        SYSTEM_PROMPT,
        f"CONTEXT:\n{context}\n\nPERTANYAAN:\n{question}",
        temperature=0.2,
    )


def ask(question: str) -> str:
    if is_chitchat(question):
        return answer_chitchat(question)

    chunks = retrieve_with_debug(question)

    if not chunks:
        return FALLBACK_TEXT

    return answer_with_context(question, chunks)

# ============================================================
# INTERACTIVE TERMINAL MODE
# ============================================================

def run_terminal_chat() -> None:
    print("\n" + "="*50)
    print("🤖 Minci Siap!")
    print("Ketik 'exit' atau 'quit' untuk keluar.")
    print("="*50 + "\n")

    while True:
        try:
            question = input("Kamu: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            
            print("Minci sedang menganalisis catatan...")
            answer = ask(question)
            print(f"Kamu: {question}")
            print(f"Minci: {answer}\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_terminal_chat()