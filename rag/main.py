"""
Skrip chat interaktif & retrieval RAG.
Jalankan ini: python main.py
"""

from __future__ import annotations
import os
import re
import sys
import json
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama

import config
import history_store

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

BASE_DIR = config.BASE_DIR
DB_DIR = config.DB_DIR
COLLECTION_NAME = config.COLLECTION_NAME

EMBED_MODEL = config.EMBED_MODEL
CHAT_MODEL = config.CHAT_MODEL

TOP_K = config.TOP_K
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS
DEBUG = config.DEBUG

CHITCHAT_PATH = config.CHITCHAT_PATH


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
    return any(marker in normalized_question for marker in config.INJECTION_MARKERS)


CHITCHAT_SYSTEM_PROMPT = config.CHITCHAT_SYSTEM_PROMPT
CONDENSE_SYSTEM_PROMPT = config.CONDENSE_SYSTEM_PROMPT
SYSTEM_PROMPT = config.SYSTEM_PROMPT
FALLBACK_TEXT = config.FALLBACK_TEXT

# Satu riwayat percakapan = list of {"role": "user"|"assistant", "content": str}
ChatHistory = history_store.ChatHistory


def get_collection():
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def retrieve(question: str) -> list[dict]:
    """Ambil TOP_K chunk paling mirip lewat vector similarity -- POLOS,
    tanpa filter tambahan (distance threshold / token overlap / tag match).
    Sengaja disamakan dengan retriever di notebook (similarity_top_k=3):
    makin sedikit lapisan filter manual, makin kecil risiko chunk yang
    sebenarnya relevan malah kebuang sebelum sampai ke LLM."""
    collection = get_collection()
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    results = collection.query(query_embeddings=[embedding], n_results=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = [
        {"document": doc, "metadata": meta or {}, "distance": float(dist)}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    if DEBUG:
        print(f"\n🔍 [DEBUG] Pertanyaan: \"{question}\"")
        for i, c in enumerate(chunks, 1):
            title = c["metadata"].get("title", "?")
            preview = " ".join(c["document"].split())[:150]
            print(f"   [{i}] {title} | jarak={c['distance']:.4f} | {preview}...")
        print()

    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        title = c["metadata"].get("title", "?")
        path = c["metadata"].get("path", "?")
        parts.append(f"[Sumber: {title} ({path})]\n{c['document']}")
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


def format_history(history: ChatHistory | None, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Ubah riwayat percakapan jadi teks 'Pengguna: ... / Minci: ...' buat prompt."""
    if not history:
        return ""
    trimmed = history[-(max_turns * 2):]
    lines = []
    for turn in trimmed:
        speaker = "Pengguna" if turn.get("role") == "user" else "Minci"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines)


def condense_question(question: str, history: ChatHistory | None) -> str:
    """Tulis ulang pertanyaan lanjutan jadi pertanyaan mandiri berdasarkan
    riwayat percakapan, mirip CondensePlusContextChatEngine di LlamaIndex.
    Ini yang bikin retrieval tetap nyambung walau user cuma nanya
    'kalau yang itu gimana?' tanpa nyebut ulang topiknya."""
    if not history:
        return question

    history_text = format_history(history)
    prompt = f"RIWAYAT PERCAKAPAN:\n{history_text}\n\nPERTANYAAN LANJUTAN:\n{question}"
    condensed = chat_with_model(
        CONDENSE_SYSTEM_PROMPT,
        prompt,
        temperature=0.0,
        num_predict=120,
    )
    condensed = condensed.strip().strip('"').strip()

    if DEBUG:
        print(f"🔁 [DEBUG] Pertanyaan asli: \"{question}\"")
        print(f"🔁 [DEBUG] Pertanyaan mandiri: \"{condensed or question}\"")

    return condensed or question


def answer_chitchat(question: str, history: ChatHistory | None = None) -> str:
    history_block = f"RIWAYAT PERCAKAPAN SEBELUMNYA:\n{format_history(history)}\n\n" if history else ""
    return chat_with_model(
        CHITCHAT_SYSTEM_PROMPT,
        f"{history_block}PERTANYAAN:\n{question}",
        temperature=0.4,
        num_predict=100,
    )


def answer_with_context(question: str, chunks: list[dict], history: ChatHistory | None = None) -> str:
    context = build_context(chunks)
    history_block = f"RIWAYAT PERCAKAPAN SEBELUMNYA:\n{format_history(history)}\n\n" if history else ""
    return chat_with_model(
        SYSTEM_PROMPT,
        f"{history_block}CONTEXT:\n{context}\n\nPERTANYAAN:\n{question}",
        temperature=0.1,
    )


def ask(question: str, history: ChatHistory | None = None) -> str:
    """Jawab satu pertanyaan dengan mempertimbangkan riwayat percakapan
    (kalau ada). `history` dikelola di luar (mis. per-sesi di webhook/terminal),
    fungsi ini tidak mengubahnya, cuma membaca."""
    if is_prompt_injection(question):
        return FALLBACK_TEXT

    if is_chitchat(question):
        return answer_chitchat(question, history)

    standalone_question = condense_question(question, history)
    chunks = retrieve(standalone_question)

    if not chunks:
        return FALLBACK_TEXT

    return answer_with_context(question, chunks, history)


def run_terminal_chat(session_id: str = "terminal") -> None:
    history_store.init_db()

    print("\n" + "="*50)
    print("🤖 Minci Siap!")
    print("Ketik 'exit' atau 'quit' untuk keluar.")
    print("Ketik 'reset' untuk menghapus riwayat percakapan.")
    print("="*50 + "\n")

    history = history_store.load_history(session_id)
    if history:
        print(f"📜 Melanjutkan percakapan sebelumnya ({len(history) // 2} pertukaran terakhir tersimpan).\n")

    while True:
        try:
            question = input("Kamu: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question.lower() in {"reset", "/reset"}:
                history_store.reset_history(session_id)
                history = []
                print("🔄 Riwayat percakapan direset.\n")
                continue
            if not question:
                continue

            print("Minci sedang menganalisis catatan...")
            answer = ask(question, history)
            print(f"Kamu: {question}")
            print(f"Minci: {answer}\n")

            history_store.append_turn(session_id, question, answer)
            history_store.prune_old_messages(session_id)
            history = history_store.load_history(session_id)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_terminal_chat()