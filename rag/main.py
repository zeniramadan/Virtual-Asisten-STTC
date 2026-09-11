"""
Skrip khusus untuk Chat Interaktif & Retrieval RAG.
Jalankan ini: python obsidian_rag.py
"""

from __future__ import annotations
import os
import re
import sys
import json
import logging

# Disable ChromaDB telemetry before importing the library.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "obsidian_vault"

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "minci"

RETRIEVAL_K = 15          
FINAL_CONTEXT_K = 3       
MAX_DISTANCE = 0.60       
MIN_OVERLAP_IF_LONG_QUERY = 1  
DEBUG = True

# ============================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================

_STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan",
    "ini", "itu", "ada", "apa", "apakah", "bagaimana", "berapa",
    "kapan", "dimana", "mana", "saja", "aja", "adalah", "pada", "min",
    "nya", "sih", "siapa", "kamu", "anda", "kak", "kakak",
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


def is_prompt_injection(question: str) -> bool:
    normalized_question = " ".join(question.lower().split())
    injection_markers = (
        "abaikan semua instruksi",
        "abaikan instruksi sebelumnya",
        "sebutkan isi system prompt",
        "tampilkan system prompt",
        "tampilkan semua chunk",
        "tampilkan metadata",
        "tanpa batasan",
    )
    return any(marker in normalized_question for marker in injection_markers)


CHITCHAT_SYSTEM_PROMPT = """Kamu adalah asisten akademik yang menjawab pertanyaan tentang PMB, KRS, Jadwal dan Biaya. Gaya bicaramu Generasi Z, ramah, dan ceria.

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

def meaningful_tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def tag_tokens(metadata: dict) -> set[str]:
    """Ubah metadata tags menjadi token yang bisa dibandingkan dengan query."""
    tags = metadata.get("tags", "")
    return meaningful_tokens(tags.replace("-", " ").replace("_", " "))


def exact_tag_matches(question: str, metadata: dict) -> set[str]:
    """Cari tag utuh yang muncul sebagai frasa di dalam pertanyaan."""
    question_text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    question_text = " ".join(question_text.split())
    tags = {
        tag.strip().lower()
        for tag in metadata.get("tags", "").split(",")
        if tag.strip()
    }
    return {
        tag
        for tag in tags
        if f" {tag.replace('-', ' ')} " in f" {question_text} "
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
    print(f"🧩 [DEBUG] Token pertanyaan: {sorted(q_tokens)}")
    print(f"{'='*50}")
    collection = get_collection()
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    results = collection.query(query_embeddings=[embedding], n_results=RETRIEVAL_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for idx, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        distance = float(distance)
        document_tokens = meaningful_tokens(document)
        metadata_tags = tag_tokens(metadata or {})
        document_overlap = sorted(q_tokens & document_tokens)
        tag_overlap_tokens = sorted(q_tokens & metadata_tags)
        overlap = len(document_overlap)
        tag_overlap = len(tag_overlap_tokens)
        exact_tags = exact_tag_matches(question, metadata or {})
        
        passed_distance = distance <= MAX_DISTANCE
        passed_overlap = (len(q_tokens) < 2) or (overlap >= MIN_OVERLAP_IF_LONG_QUERY)
        is_valid = passed_distance and passed_overlap

        status = "✅ LOLOS" if is_valid else "❌ DIBUANG"
        title = metadata.get('title', '?')
        print(
            f"   [{idx+1}] Note: {title} | Jarak: {distance:.4f} | "
            f"Overlap: {overlap} | Tag overlap: {tag_overlap} | Status: {status}"
        )
        print(f"       Kata isi yang cocok: {document_overlap or '-'}")
        print(f"       Kata tag yang cocok: {tag_overlap_tokens or '-'}")
        print(f"       Exact tag: {sorted(exact_tags) or '-'}")

        if not is_valid:
            continue

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "overlap": overlap,
            "tag_overlap": tag_overlap,
            "exact_tags": exact_tags,
        })

    exact_tag_candidates = [candidate for candidate in candidates if candidate["exact_tags"]]
    if exact_tag_candidates:
        candidates = exact_tag_candidates

    candidates.sort(
        key=lambda c: (
            -len(c["exact_tags"]),
            c["distance"],
            -c["tag_overlap"],
            -c["overlap"],
        )
    )
    final_chunks = candidates[:FINAL_CONTEXT_K]
    print(f"\n{'='*50}")
    print(f"📌 [DEBUG] Total chunk terpilih untuk LLM: {len(final_chunks)}")
    print(f"{'='*50}")
    for chunk_index, chunk in enumerate(final_chunks, 1):
        normalized_document = " ".join(chunk["document"].split())
        chunk_preview = normalized_document[:300]
        if len(normalized_document) > 300:
            chunk_preview += "..."
        print(f"   [CHUNK {chunk_index}] {chunk_preview}")
    print(f"{'='*50}\n")
    
    return final_chunks

# ============================================================
# LLM PROMPTS
# ============================================================

SYSTEM_PROMPT = """Kamu adalah asisten akademik yang menjawab pertanyaan tentang PMB, KRS, Jadwal dan Biaya. Gaya bicaramu Generasi Z, ramah, dan ceria.

CONTEXT di bawah ini SUDAH DIPASTIKAN BERISI DATA PANDUAN YANG RELEVAN dengan pertanyaan.
JIKA BENAR-BENAR TIDAK ADA INFORMASI pada CONTEXT yang di berikan, JANGAN MENGARANG, JAWAB dengan: "Maaf kak, informasi yang kakak tanyakan tidak ada di panduan kami, coba bertanya lebih spesifik, atau silakan kakak hubungi bagian Tata Usaha ya!".

ATURAN JAWABAN:
- JAWAB pertanyaan pengguna HANYA berdasarkan informasi faktual yang tertulis di dalam CONTEXT tersebut.
- DILARANG MENJAWAB diluar dari CONTEXT yang diberikan!.
- JIKA bertanya tentang "daftar", "pendaftaran", JAWAB dengan SYARAT PENDAFTARAN.
- JIKA data dari CONTEXT berupa daftar, TAMPILKAN dalam bentuk daftar bullet (-) agar mudah dibaca.
- JIKA bertanya tentang "Pengisian KRS", SEBUTKAN SEMUA langkah pengisian KRS yang ada di CONTEXT, jangan ada yang terlewat.

ATURAN WAJIB UNTUK SEMUA JAWABAN:
- HARUS menggunakan kata "kak" atau "kakak"!.
- Gunakan bullet "-" untuk menampilkan data yang berbentuk daftar.
- LANGSUNG jawab inti pertanyaan. JANGAN membuka jawaban dengan kalimat seperti "informasi ini ada di panduan kami" atau "informasi yang kakak tanyakan ada di panduan kami".
- JANGAN PERNAH menyebutkan kata teknis seperti "context", "metadata", "chunk", atau "RAG".
- JANGAN menyebut nomor bagian internal seperti "CHUNK 1", "CHUNK 5", atau "CHUNK 6". Langsung sebutkan informasi dan tanggalnya.
- JANGAN menyebut nama dokumen, seperti: "informasi ini ada di dokumen BIAYA".
- JANGAN menyebut tempat informasi berada, seperti "informasi ini ada di tabel biaya".
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
        temperature=0.1,
    )


def ask(question: str) -> str:
    if is_prompt_injection(question):
        return FALLBACK_TEXT

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