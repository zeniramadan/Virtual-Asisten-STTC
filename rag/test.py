"""
Pure RAG query.py untuk Minci.
- Semua pertanyaan akademik melewati retrieval.
- Routing memilih source ChromaDB.
- Model SELALU dipanggil, bahkan ketika context kosong.
- Fallback sepenuhnya di-handle oleh System Prompt Llama 3.2.
"""

from __future__ import annotations
import os
import re
from collections import Counter

import chromadb
import ollama


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
COLLECTION_NAME = "minci_dokumen"

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.2"

RETRIEVAL_K = 20
FINAL_CONTEXT_K = 8
MAX_DISTANCE = 0.60
ROUTED_MAX_DISTANCE = 0.58
DEBUG = True

# Variabel FALLBACK manual dihapus karena sekarang diserahkan ke model


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
        f"chat_model={CHAT_MODEL}"
    )


# ============================================================
# NORMALISASI RINGAN
# ============================================================

_NUMBER_WORDS = {
    "nol": "0", "satu": "1", "dua": "2", "tiga": "3", "empat": "4",
    "lima": "5", "enam": "6", "tujuh": "7", "delapan": "8",
    "sembilan": "9", "sepuluh": "10",
}


def normalize_query(question: str) -> str:
    q = str(question or "").strip()
    q = re.sub(r"\s+", " ", q)

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
# ROUTING
# ============================================================

ROUTES = [
    ("BIAYA.docx", [
        "biaya", "berapa bayar", "berapa biaya", "nominal", "harga kuliah",
        "uang kuliah", "ukt", "pembayaran", "bayar", "cicilan", "cicil",
        "biaya pendaftaran", "biaya registrasi",
    ]),
    ("KALENDER.docx", [
        "jadwal", "tanggal", "kalender", "kapan", "gelombang",
        "pra ktmb", "ktmb", "hasil seleksi", "pengumuman", "seleksi",
        "perwalian", "herregistrasi", "kprs", "cuti kuliah", "uts", "uas",
    ]),
    ("KRS.docx", [
        "krs", "kartu rencana studi", "pengisian krs", "isi krs",
        "mengisi krs", "cara krs", "tata cara krs", "prosedur krs",
        "perwalian online", "rencana studi", "mata kuliah",
    ]),
    ("PMB.docx", [
        "pmb", "penerimaan mahasiswa baru", "mahasiswa baru",
        "calon mahasiswa", "pendaftaran", "mendaftar", "daftar kuliah",
        "syarat masuk", "persyaratan masuk", "jalur masuk",
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
    q = _route_text(question)
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

    words = set(q.split())

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
    "cipasung", "kampus", "informasi",
}


def meaningful_tokens(text: str) -> set[str]:
    return {
        x for x in re.findall(r"[a-z0-9]+", text.lower())
        if len(x) >= 3 and x not in _STOPWORDS
    }


def lexical_overlap(question: str, document: str) -> int:
    return len(meaningful_tokens(question) & meaningful_tokens(document))


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

    for doc_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if not document:
            continue

        distance = float(distance)
        overlap = lexical_overlap(question, document)

        if distance > threshold:
            continue

        if len(q_tokens) >= 2 and overlap < 1:
            continue

        candidates.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata or {},
            "distance": distance,
            "overlap": overlap,
        })

    candidates.sort(key=lambda x: (x["distance"], -x["overlap"]))

    return candidates[:FINAL_CONTEXT_K]


# ============================================================
# CONTEXT
# ============================================================

def build_context(chunks: list[dict]) -> str:
    # Jika tidak ada chunk (kosong), kembalikan string yang memberi tahu model
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

SYSTEM_PROMPT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung, yang membahas tentang PMB, KRS, dan biaya.

Jawab pertanyaan pengguna HANYA menggunakan informasi faktual yang ada di CONTEXT.

ATURAN WAJIB FALLBACK:
- Jika CONTEXT berisi "TIDAK ADA DATA PANDUAN YANG DITEMUKAN" atau informasi yang ditanyakan sama sekali TIDAK ADA di CONTEXT, kamu WAJIB menjawab PERSIS dengan kalimat ini:
  "Maaf kak, informasi yang kamu tanyakan tidak ada di panduan kami. Silakan hubungi bagian Tata Usaha ya!"
  (Jangan menambahkan penjelasan atau kalimat lain).
  PENGECUALIAN: Jika pertanyaan pengguna HANYA berisi sapaan/salam seperti "halo, selamat pagi/siang/sore/mala, assalamualaikum" atau basa-basi pendek, ABAIKAN aturan fallback ini dan gunakan aturan sapaan di bawah.

ATURAN LAINNYA:
- Jangan menggunakan pengetahuan dari luar CONTEXT, Jangan menebak, Jangan mengarang informasi.
- Jika context memiliki angka, tanggal, nama, syarat, biaya, atau aturan, pertahankan sesuai isi context.
- Jika context berupa daftar, tampilkan informasi relevan dengan jelas menggunakan bullet "-".
- Jangan menyebut nama file, metadata internal, skor similarity, routing, chunk, atau proses RAG.
- Jawab langsung dan natural dalam Bahasa Indonesia dengan gaya Gen-Z (ramah, sopan, gunakan kata "kak").
- Jika pengguna memberi sapaan "halo, selamat pagi/siang/sore/malam" jawab dengan "Halo kak!", jika salam "Assalamualaikum" jawab dengan "Waalaikumsalam kak!".
- Jika pertanyaan tidak spesifik (misal pola kalimat "ingin tanya"), jawab "Kakak bisa tanyakan lebih spesifik mengenai PMB, KRS, atau biaya ya!".
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
        r"\[(?:Sumber|Konteks|Bagian|Sumber internal):[^\]]*\]\s*",
        "",
        text,
        flags=re.I,
    )

    for filename in ("PMB.docx", "KRS.docx", "BIAYA.docx", "KALENDER.docx"):
        text = text.replace(filename, "")

    return text.strip()


# ============================================================
# API UTAMA
# ============================================================

def ask_minci(question: str) -> str:
    """
    Pure RAG:
        query -> routing -> retrieval -> gate -> context -> model

    Model SELALU dipanggil untuk menghasilkan jawaban.
    """
    question = normalize_query(question)

    # Tetap sediakan fallback error ringan jika pertanyaan benar-benar kosong
    if not question:
        return "Ada yang bisa Minci bantu, kak?"

    if _collection.count() == 0:
        if DEBUG: print("[RAG] collection kosong")
        # Biarkan model merespons dengan context kosong

    route_source, route_reason = detect_route(question)

    if DEBUG:
        print("\n" + "=" * 70)
        print(f"[QUERY] {question}")
        print(f"[ROUTE] {route_source or 'GLOBAL'}")
        print(f"[WHY]   {route_reason}")
        print("=" * 70)

    try:
        chunks = retrieve(question, route_source=route_source)
    except Exception as exc:
        if DEBUG: print(f"[RAG] retrieval error: {exc}")
        chunks = []

    # Blok 'if not chunks' manual dihapus, sehingga konteks kosong tetap dikirim ke model
    context = build_context(chunks)

    if DEBUG:
        print("\n" + "=" * 70)
        print("[CONTEXT YANG DIKIRIM KE MODEL]")
        print("=" * 70)
        print(context)
        print("=" * 70)

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 1024,
            },
        )
    except Exception as exc:
        if DEBUG: print(f"[LLM] error: {exc}")
        return "Maaf kak, sistem Minci sedang gangguan. Coba lagi nanti ya!"

    raw_answer = response.get("message", {}).get("content", "")
    answer = clean_output(raw_answer)

    return answer


# ============================================================
# TEST TERMINAL
# ============================================================

if __name__ == "__main__":
    print("Minci - Pure RAG Test")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        q = input("Kamu: ").strip()

        if q.lower() in {"exit", "quit"}:
            break

        answer = ask_minci(q)
        print(f"\nKamu: {q}")
        print(f"Minci: {answer}\n")