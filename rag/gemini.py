"""
Pure RAG query.py untuk Minci.
- Menggunakan Google Gemini API untuk LLM Chat (mengambil API key dari environment variable).
- ChromaDB & Embedding (bge-m3) tetap menggunakan Ollama secara lokal.
"""

from __future__ import annotations
import json
import os
import re
import time
from collections import Counter, defaultdict

import chromadb
import ollama
from google import genai
from google.genai import types
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", "webhook", ".env")
load_dotenv(dotenv_path=dotenv_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
COLLECTION_NAME = "minci_dokumen"

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "gemini-3.5-flash-lite"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

RETRIEVAL_K = 20
FINAL_CONTEXT_K = 8
MAX_DISTANCE = 0.60
ROUTED_MAX_DISTANCE = 0.85
DEBUG = True

# ============================================================
# CHROMADB
# ============================================================

_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

if DEBUG:
    print(
        f"[RAG] collection={COLLECTION_NAME} | "
        f"chunks={_collection.count()} | embedding={EMBED_MODEL} | "
        f"chat_model={CHAT_MODEL} (Gemini API)"
    )


# ============================================================
# NORMALISASI RINGAN
# ============================================================

_NUMBER_WORDS = {
    "nol": "0", "satu": "1", "dua": "2", "tiga": "3", "empat": "4",
    "lima": "5", "enam": "6", "tujuh": "7", "delapan": "8",
    "sembilan": "9", "sepuluh": "10",
}

_ABBREVIATION_ALIASES = (
    (r"\bp\s*\.?\s*m\s*\.?\s*b\s*\.?\b", "PMB penerimaan mahasiswa baru\n"),
    (r"\bk\s*\.?\s*r\s*\.?\s\s*\.?\b", "KRS kartu rencana studi\n"),
    (r"\bp\s*\.?\s*r\s*\.?\s*o\s*\.?\s*d\s*\.?\s*i\s*\.?\b", "PRODI program studi\n"),
    (r"\bu\s*\.??\s*k\s*\.??\s*m\s*\.??\b", "UKM unit kegiatan mahasiswa\n"),
    (r"\bk\s*\.??\s*p\s*\.??\s*r\s*\.??\s*s\s*\.??\b", "KPRS kartu perubahan rencana studi"),
)

_ADDRESS_TERMS = {"min", "minci", "kak", "kakak"}


def normalize_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATION_ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def normalize_query(question: str) -> str:
    q = str(question or "").strip()
    q = re.sub(r"\s+", " ", q)
    q = normalize_abbreviations(q)
    q = re.sub(r"\s+", " ", q).strip()

    q = " ".join(
        word for word in q.split()
        if word.lower().strip("!?.,") not in _ADDRESS_TERMS
    )

    q = re.sub(r"\bgelombang\s+i\b", "gelombang 1", q, flags=re.I)
    q = re.sub(r"\bgelombang\s+ii\b", "gelombang 2", q, flags=re.I)
    q = re.sub(r"\bgelombang\s+iii\b", "gelombang 3", q, flags=re.I)

    for word, number in _NUMBER_WORDS.items():
        q = re.sub(
            rf"\bgelombang\s+(?:ke[- ]?)?{word}\b",
            f"gelombang {number}",
            q,
            flags=re.I,
        )

    return q


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
- HARUS menggunakan kata "kak" atau "kakak" di setiap kalimat!
- DILARANG KERAS menggunakan kata "Kamu" atau "Anda" saat menyapa pengguna!
- JANGAN PERNAH memberikan jawaban template "Sama-sama" jika pengguna tidak sedang berterima kasih!
- JANGAN mengarang atau memberikan informasi akademik palsu di sini!
"""

# ============================================================
# ROUTING
# ============================================================

ROUTES = [
    ("BIAYA.docx", [
        "biaya", "berapa bayar", "berapa biaya", "nominal", "harga kuliah",
        "uang kuliah", "ukt", "pembayaran", "bayar", "cicilan", "cicil",
        "biaya pendaftaran", "biaya registrasi",
    ]),
    ("KALENDER.docx", [
        "jadwal", "tanggal", "tanggal pmb", "tanggal penerimaan", "kalender", "kapan", "gelombang",
        "pra ktmb", "ktmb", "hasil seleksi", "pengumuman", "seleksi",
        "perwalian", "herregistrasi", "kprs", "cuti kuliah", "uts", "uas",
    ]),
    ("KRS.docx", [
        "krs", "kartu rencana studi", "pengisian krs", "isi krs", "syarat krs",
        "mengisi krs", "cara krs", "tata cara krs", "prosedur krs", "syarat perwalian",
        "perwalian online", "rencana studi", "mata kuliah", "perwalian",
    ]),
    ("PMB.docx", [
        "pmb", "penerimaan mahasiswa baru", "mahasiswa baru",
        "calon mahasiswa", "pendaftaran", "mendaftar", "daftar kuliah",
        "syarat masuk", "syarat pendaftaran", "persyaratan masuk", "jalur masuk",
        "program studi", "prodi", "jurusan", "beasiswa", "ukm",
        "unit kegiatan mahasiswa", "profil kampus", "tentang kampus",
        "tentang stt cipasung",
    ]),
]


def _route_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_route(question: str) -> tuple[str | None, str]:
    q = _route_text(normalize_query(question))

    words = set(q.split())
    requirement_words = {"syarat", "persyaratan", "dokumen", "berkas"}
    krs_words = {"krs", "perwalian"}
    registration_words = {"pendaftaran", "mendaftar", "daftar", "pmb", "calon", "mahasiswa", "masuk"}

    if words & requirement_words and words & krs_words:
        return "KRS.docx", "prioritas syarat KRS/perwalian"

    if words & requirement_words and words & registration_words:
        return "PMB.docx", "prioritas syarat pendaftaran PMB"

    time_words = {"kapan", "tanggal", "jadwal"}
    if words & time_words:
        return "KALENDER.docx", "prioritas kata tanya waktu (kapan/tanggal/jadwal)"

    scores = Counter()
    matches = {}

    for source, keywords in ROUTES:
        for keyword in keywords:
            k = _route_text(keyword)
            if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", q):
                weight = max(1, len(k.split()))
                scores[source] += weight
                matches.setdefault(source, []).append(keyword)

    if not scores:
        return None, "tidak ada keyword routing"

    ranked = scores.most_common()
    best_source, best_score = ranked[0]

    if len(ranked) == 1:
        return best_source, f"keyword={matches[best_source]}"

    second_score = ranked[1][1]

    if best_score >= second_score + 2:
        return best_source, f"keyword={matches[best_source]}"

    if words & {"biaya", "bayar", "nominal", "ukt", "harga"}:
        return "BIAYA.docx", "prioritas biaya/pembayaran"

    if words & {"jadwal", "tanggal", "kapan", "kalender", "gelombang"}:
        return "KALENDER.docx", "prioritas jadwal/tanggal"

    if "krs" in words or "perwalian" in words:
        return "KRS.docx", "prioritas KRS/perwalian"

    if words & {"pmb", "pendaftaran", "prodi", "jurusan", "beasiswa"}:
        return "PMB.docx", "prioritas PMB/pendaftaran"

    return None, "routing ambigu -> retrieval global"


# ============================================================
# RETRIEVAL GATE
# ============================================================

_STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan",
    "ini", "itu", "ada", "apa", "apakah", "bagaimana", "berapa",
    "kapan", "dimana", "mana", "saja", "aja", "dong", "deh", "sih",
    "ya", "nih", "kak", "min", "minci", "tolong", "mohon", "bisa",
    "gak", "nggak", "enggak", "tidak", "tau", "tahu", "stt",
    "cipasung", "kampus", "informasi", "nya",
}

_REQUIREMENT_TERMS = {
    "syarat", "persyaratan", "dokumen", "berkas", "ketentuan",
}
_PROCEDURE_TERMS = {
    "cara", "tata cara", "langkah", "prosedur", "mengisi", "pengisian",
}


def meaningful_tokens(text: str) -> set[str]:
    text = normalize_abbreviations(text)
    return {
        x for x in re.findall(r"[a-z0-9]+", text.lower())
        if len(x) >= 3 and x not in _STOPWORDS
    }


def lexical_overlap(question: str, document: str) -> int:
    return len(meaningful_tokens(question) & meaningful_tokens(document))


def intent_match(question: str, document: str) -> int:
    question_text = _route_text(normalize_query(question))
    document_text = _route_text(document)

    requirement_query = bool(
        meaningful_tokens(question_text) & _REQUIREMENT_TERMS
    )
    procedure_query = bool(
        meaningful_tokens(question_text)
        & {"cara", "langkah", "prosedur", "mengisi", "pengisian"}
    )
    registration_query = bool(
        meaningful_tokens(question_text)
        & {"pendaftaran", "mendaftar", "daftar", "masuk"}
    )

    score = 0
    if requirement_query and any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", document_text)
        for term in _REQUIREMENT_TERMS
    ):
        score += 3
    if procedure_query and any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", document_text)
        for term in _PROCEDURE_TERMS
    ):
        score += 3
    if registration_query and "pendaftaran" in document_text:
        score += 2
    if registration_query and "persyaratan administrasi pendaftaran" in document_text:
        score += 6
    return score


def retrieve(question: str, route_source: str | None) -> list[dict]:
    embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=question,
    )["embedding"]

    kwargs = {
        "query_embeddings": [embedding],
        "n_results": RETRIEVAL_K,
    }

    if route_source:
        kwargs["where"] = {"source": route_source}

    results = _collection.query(**kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not documents:
        return []

    threshold = ROUTED_MAX_DISTANCE if route_source else MAX_DISTANCE
    q_tokens = meaningful_tokens(question)
    candidates = []

    if DEBUG:
        print(f"\n[RETRIEVE] where={route_source or '(global, semua dokumen)'} | threshold={threshold}")
        if not documents:
            print("[RETRIEVE] ChromaDB tidak mengembalikan dokumen apapun.")

    for doc_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if not document:
            continue

        distance = float(distance)
        overlap = lexical_overlap(question, document)
        intent = intent_match(question, document)
        source = (metadata or {}).get("source", "?")

        lolos_distance = distance <= threshold
        if DEBUG:
            status = "✅ lolos" if lolos_distance else "❌ DIBUANG (distance > threshold)"
            print(f"    distance={distance:.4f}  overlap={overlap}  intent={intent}  source={source}  -> {status}")
            print(f"       {document[:120].replace(chr(10), ' ')}...")

        if not lolos_distance:
            continue

        if route_source is None and len(q_tokens) >= 2 and overlap < 1:
            if DEBUG:
                print(f"       -> DIBUANG (overlap gate, tidak ada kata kunci konten sama; global search)")
            continue

        candidates.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata or {},
            "distance": distance,
            "overlap": overlap,
            "intent": intent,
        })

    candidates.sort(key=lambda x: (-x["intent"], x["distance"], -x["overlap"]))

    return candidates[:FINAL_CONTEXT_K]


# ============================================================
# CONTEXT
# ============================================================

def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "TIDAK ADA DATA PANDUAN YANG DITEMUKAN."
        
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "dokumen")
        parts.append(
            f"CHUNK {i}\n"
            f"Sumber internal: {source}\n"
            f"Isi:\n{chunk['document'].strip()}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# LLM PROMPT
# ============================================================

FALLBACK_TEXT = "Maaf kak, informasi yang kakak tanyakan tidak ada di panduan kami, coba bertanya lebih spesifik, atau silakan kakak hubungi bagian Tata Usaha ya!"

SYSTEM_PROMPT_WITH_CONTEXT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Generasi Z, tapi tetap sopan.

CONTEXT di bawah ini SUDAH DIPASTIKAN BERISI DATA PANDUAN YANG RELEVAN dengan pertanyaan. Kamu WAJIB menjawab dari situ -- JANGAN PERNAH bilang "tidak ditemukan" atau "tidak ada di panduan" untuk pertanyaan ini, karena datanya PASTI ada di CONTEXT.

ATURAN JAWABAN:
- Jawab pertanyaan pengguna HANYA berdasarkan informasi faktual yang tertulis di dalam CONTEXT tersebut.
- Jika pertanyaan meminta "syarat", berikan DAFTAR SYARAT saja dari context. Jangan jelaskan tata cara.
- Jika pertanyaan meminta "cara", berikan LANGKAH-LANGKAH saja dari context. Jangan berikan daftar syarat.
- JIKA pertanyaan meminta "syarat", "persyaratan" atau "pendaftaran", kamu WAJIB DAN HARUS MENULISKAN SEMUA DAFTAR SYARAT YANG ADA DI CONTEXT SECARA LENGKAP!
- Jika pertanyaan tidak spesifik mengenai jadwal PMB, cantumkan tanggal pendaftaran gelombang 1, 2, dan 3 yang tertera di context.
- Jika pertanyaan meminta "seleksi" berikan jadwal seleksi penerimaan mahasiswa baru yang ada di context.
- Jika pertanyaan meminta "biaya", "bayar", atau "nominal", berikan SEMUA INFORMASI BESERTA KETERANGANNYA.

ATURAN WAJIB UNTUK SEMUA JAWABAN:
- HARUS menggunakan kata "kak" atau "kakak"! DILARANG menggunakan kata "Kamu".
- Gunakan bullet "-" untuk menampilkan data yang berbentuk daftar.
- JANGAN PERNAH menyebutkan kata teknis seperti "context", "metadata", "chunk", atau "RAG".
- JANGAN menyebut nomor bagian internal seperti "CHUNK 1", "CHUNK 5", atau "CHUNK 6". Langsung sebutkan informasi dan tanggalnya.
"""

def build_user_prompt(question: str, context: str) -> str:
    return f"""CONTEXT:
{context}

PERTANYAAN:
{question}

Jawab langsung pertanyaan tersebut.
"""


# ============================================================
# CLEANING MINIMAL
# ============================================================

def clean_output(text: str) -> str:
    text = str(text or "").strip()

    text = re.sub(
        r"\s+(?:di|pada|dalam)\s+(?:CHUNK\s+\d+\s*(?:,|dan)?\s*)+",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bCHUNK\s+\d+\b", "", text, flags=re.I)
    text = re.sub(r"\s+([,:;.!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    text = re.sub(
        r"\[(?:Sumber|Konteks|Bagian|Sumber internal):[^\]]*\]\s*",
        "",
        text,
        flags=re.I,
    )

    for filename in ("PMB.docx", "KRS.docx", "BIAYA.docx", "KALENDER.docx"):
        text = text.replace(filename, "")
        
    text = text.replace("•", "-")
    text = text.replace("▪", "-")
    text = text.replace("⁃", "-")
    
    return text.strip()


# ============================================================
# RIWAYAT PERCAKAPAN (per user) -- auto-hapus setelah 1 jam
# ============================================================

HISTORY_TTL_SECONDS = 3600
MAX_HISTORY_TURNS = 6

_conversation_history: dict[str, list[dict]] = defaultdict(list)


def _prune_expired_history(now: float | None = None) -> None:
    now = now if now is not None else time.time()
    empty_users = []
    for user_id, messages in _conversation_history.items():
        fresh = [m for m in messages if now - m["ts"] <= HISTORY_TTL_SECONDS]
        if fresh:
            _conversation_history[user_id] = fresh
        else:
            empty_users.append(user_id)
    for user_id in empty_users:
        del _conversation_history[user_id]


def _remember(user_id: str, role: str, content: str) -> None:
    _conversation_history[user_id].append({"role": role, "content": content, "ts": time.time()})


_FOLLOWUP_HINT_WORDS = {
    "itu", "tadi", "tersebut", "lanjut", "terus", "trus", "kalau", "gimana",
    "berarti", "jadi", "nah", "terusan", "lah", "dong",
}


def _looks_like_followup(question: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return len(words) <= 4 or bool(words & _FOLLOWUP_HINT_WORDS)


def ask_minci(question: str, user_id: str = "default") -> str:
    question = normalize_query(question)

    _prune_expired_history()
    
    # Ambil riwayat percakapan untuk dikonversi ke format Google GenAI contents/history
    raw_history = _conversation_history.get(user_id, [])
    trimmed_history = raw_history[-(MAX_HISTORY_TURNS * 2):]

    if not question:
        return "Ada yang bisa Minci bantu, kak?"

    # --- Chitchat bypass via Gemini API ---
    if is_chitchat(question):
        if DEBUG:
            print(f"\n[CHITCHAT] '{question}' terdeteksi basa-basi -> skip RAG")
        try:
            # Membentuk history percakapan untuk client.chats.create / contents
            chat_contents = []
            for m in trimmed_history:
                role_mapped = "user" if m["role"] == "user" else "model"
                chat_contents.append(
                    types.Content(
                        role=role_mapped,
                        parts=[types.Part.from_text(text=m["content"])]
                    )
                )
            # Tambahkan pesan user saat ini
            chat_contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=question)]
                )
            )

            response = gemini_client.models.generate_content(
                model=CHAT_MODEL,
                contents=chat_contents,
                config=types.GenerateContentConfig(
                    system_instruction=CHITCHAT_SYSTEM_PROMPT,
                    temperature=0.4,
                    top_p=0.9,
                    max_output_tokens=100,
                ),
            )
            answer = clean_output(response.text or "")
        except Exception as exc:
            if DEBUG: print(f"[LLM] chitchat error (Gemini): {exc}")
            answer = "Halo kak! Ada yang bisa Minci bantu?"

        _remember(user_id, "user", question)
        _remember(user_id, "assistant", answer)
        return answer

    if _collection.count() == 0:
        if DEBUG: print("[RAG] collection kosong")

    retrieval_query = question
    if _looks_like_followup(question) and trimmed_history:
        previous_user_questions = [m["content"] for m in trimmed_history if m["role"] == "user"]
        if previous_user_questions:
            retrieval_query = f"{previous_user_questions[-1]} {question}"
            if DEBUG:
                print(f"[FOLLOWUP] Query retrieval digabung jadi: {retrieval_query!r}")

    route_source, route_reason = detect_route(retrieval_query)

    if DEBUG:
        print("\n" + "=" * 70)
        print(f"[QUERY] {question}")
        print(f"[ROUTE] {route_source or 'GLOBAL'}")
        print(f"[WHY]   {route_reason}")
        print("=" * 70)

    try:
        chunks = retrieve(retrieval_query, route_source=route_source)
    except Exception as exc:
        print(f"[RAG] retrieval error: {exc}")
        chunks = []

    if not chunks:
        if DEBUG:
            print("[RAG] Tidak ada chunk relevan -> fallback deterministik, LLM TIDAK dipanggil.")
        _remember(user_id, "user", question)
        _remember(user_id, "assistant", FALLBACK_TEXT)
        return FALLBACK_TEXT

    context = build_context(chunks)

    if DEBUG:
        print("\n" + "=" * 70)
        print("[CONTEXT YANG DIKIRIM KE GEMINI]")
        print("=" * 70)
        print(context)
        print("=" * 70)

    try:
        # Konversi riwayat lokal ke format `types.Content` untuk Gemini API
        chat_contents = []
        for m in trimmed_history:
            role_mapped = "user" if m["role"] == "user" else "model"
            chat_contents.append(
                types.Content(
                    role=role_mapped,
                    parts=[types.Part.from_text(text=m["content"])]
                )
            )
        
        # Tambahkan prompt utama yang berisi context dan pertanyaan saat ini
        final_prompt = build_user_prompt(question, context)
        chat_contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=final_prompt)]
            )
        )

        response = gemini_client.models.generate_content(
            model=CHAT_MODEL,
            contents=chat_contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_WITH_CONTEXT,
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
    except Exception as exc:
        print(f"[LLM] error (Gemini): {exc}")
        return "Maaf kak, sistem Minci sedang gangguan. Coba lagi nanti ya!"

    raw_answer = response.text or ""
    answer = clean_output(raw_answer)

    _remember(user_id, "user", question)
    _remember(user_id, "assistant", answer)

    return answer


# ============================================================
# TEST TERMINAL
# ============================================================

if __name__ == "__main__":
    print("\nMinci - Asisten Virtual Akademik STT Cipasung (Gemini API Powered)")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        q = input("Kamu: ").strip()

        if q.lower() in {"exit", "quit"}:
            break

        answer = ask_minci(q)
        print(f"\nKamu: {q}")
        print(f"Minci: {answer}\n")