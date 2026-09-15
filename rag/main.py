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

RETRIEVAL_K = config.RETRIEVAL_K
FINAL_CONTEXT_K = config.FINAL_CONTEXT_K
MAX_DISTANCE = config.MAX_DISTANCE
MIN_OVERLAP_IF_LONG_QUERY = config.MIN_OVERLAP_IF_LONG_QUERY
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS
DEBUG = config.DEBUG

_STOPWORDS = config.STOPWORDS

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

# Satu riwayat percakapan = list of {"role": "user"|"assistant", "content": str}
ChatHistory = history_store.ChatHistory

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

SYSTEM_PROMPT = config.SYSTEM_PROMPT

FALLBACK_TEXT = config.FALLBACK_TEXT


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

    # Tulis ulang pertanyaan lanjutan jadi mandiri dulu SEBELUM retrieval,
    # supaya embedding search tetap kena topik yang benar.
    standalone_question = condense_question(question, history)

    chunks = retrieve_with_debug(standalone_question)

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