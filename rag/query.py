"""
Pure RAG query.py untuk Minci.
- Semua pertanyaan akademik melewati retrieval.
- Routing memilih source ChromaDB.
- Model SELALU dipanggil, bahkan ketika context kosong.
- Fallback sepenuhnya di-handle oleh System Prompt Llama 3.2.
"""

from __future__ import annotations
import json
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

_ABBREVIATION_ALIASES = (
    (r"\bp\s*\.?\s*m\s*\.?\s*b\s*\.?\b", "PMB penerimaan mahasiswa baru"),
    (r"\bk\s*\.?\s*r\s*\.?\s\s*\.?\b", "KRS kartu rencana studi"),
    (r"\bp\s*\.?\s*r\s*\.?\s*o\s*\.?\s*d\s*\.?\s*i\s*\.?\b", "PRODI program studi"),
    (r"\bu\s*\.??\s*k\s*\.??\s*m\s*\.??\b", "UKM unit kegiatan mahasiswa"),
    (r"\bk\s*\.??\s*p\s*\.??\s*r\s*\.??\s*s\s*\.??\b", "KPRS kartu perubahan rencana studi"),
)


def normalize_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATION_ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def normalize_query(question: str) -> str:
    q = str(question or "").strip()
    q = re.sub(r"\s+", " ", q)
    q = normalize_abbreviations(q)
    q = re.sub(r"\s+", " ", q).strip()

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


CHITCHAT_SYSTEM_PROMPT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi sopan.

Ini pesan basa-basi (sapaan/ucapan terima kasih/obrolan ringan), BUKAN pertanyaan akademik.
Balas SINGKAT (1-2 kalimat) dan natural sesuai basa-basinya.
- Jika sapaan ("halo", "selamat pagi"), balas sapaannya lalu tawarkan bantuan seputar PMB, KRS, atau biaya.
- Jika salam ("assalamualaikum"), balas "Waalaikumsalam kak!" lalu tawarkan bantuan.
- Jika ucapan terima kasih ("makasih"), balas "Sama-sama kak!" atau sejenisnya.
- Jika pertanyaan tidak spesifik ("mau nanya", "ingin bertanya"), jawab "Boleh kak! Silakan tanyakan lebih spesifik mengenai PMB, KRS, biaya, atau jadwal ya!"
- Gunakan kata "kak" atau "kakak", JANGAN GUNAKAN kata "Kamu".
JANGAN mengarang info akademik apapun di sini."""


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
    registration_words = {
        "pendaftaran", "mendaftar", "daftar", "pmb", "masuk",
        "calon", "mahasiswa",
    }

    if words & requirement_words and words & krs_words:
        return "KRS.docx", "prioritas syarat KRS/perwalian"

    if words & requirement_words and words & registration_words:
        return "PMB.docx", "prioritas syarat pendaftaran PMB"

    # PRIORITAS KATA TANYA WAKTU: "kapan"/"tanggal"/"jadwal" HARUS menang
    # duluan, sebelum scoring keyword biasa. Kenapa ini perlu: normalize_abbreviations()
    # mengubah "pmb" jadi "PMB penerimaan mahasiswa baru" -- akibatnya frasa panjang
    # ini ikut disisipkan ke teks query dan mendominasi skor Counter di bawah
    # (bobotnya = jumlah kata di frasa, jadi "penerimaan mahasiswa baru" dapat
    # bobot 3, sementara "kapan" cuma bobot 1). Tanpa aturan ini, pertanyaan
    # "kapan pmb dibuka" selalu di-route paksa ke PMB.docx dan KALENDER.docx
    # (tempat tanggal/jadwal sebenarnya disimpan) tidak pernah ikut dicari sama
    # sekali karena routing pakai hard where-filter.
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


def meaningful_tokens(text: str) -> set[str]:
    text = normalize_abbreviations(text)
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

SYSTEM_PROMPT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi sopan.

PENTING: Sebelum menjawab, tentukan apakah pertanyaan dari pengguna adalah pertanyaan AKADEMIK KAMPUS (PMB (Penerimaan Mahasiswa Baru), KRS, biaya, dsb) atau pertanyaan UMUM / BASA-BASI (seputar pengetahuan umum, AI, coding, sapaan, dsb).

1. JIKA PERTANYAAN AKADEMIK KAMPUS:
   - Jawab HANYA berdasarkan informasi faktual di CONTEXT.
   - JIKA informasi yang dicari TIDAK ADA di CONTEXT, kamu WAJIB menjawab PERSIS: "Maaf kak, informasi yang kamu tanyakan tidak ada di panduan kami. Silakan hubungi bagian Tata Usaha ya!" (Jangan tambahkan informasi lain).
   - JIKA pertanyaan tidak spesifik mengenai jadwal penerimaan mahasiswa baru (PMB), Cantumkan tanggal pendaftaran gelombang 1, 2, 3.
   
2. JIKA PERTANYAAN UMUM / BASA-BASI (Di luar urusan kampus):
   - JANGAN gunakan pesan "Maaf kak..." seperti di atas.
   - ABAIKAN CONTEXT sepenuhnya. Jawablah pertanyaan pengguna menggunakan pengetahuan umummu selayaknya AI yang pintar.
   - Jika pengguna hanya menyapa "halo", "selamat pagi/siang/sore/malam" balas sapaannya, jika salam "assalamualaikum" balas dengan "Waalaikum salam", lalu tawarkan bantuan seputar PMB, KRS, atau biaya.

ATURAN LAINNYA:
- Jika pertanyaan tidak spesifik (seperti "saya ingin bertanya", "min mau nanya", dsb), jawablah dengan: "Boleh kak! Silakan tanyakan lebih spesifik mengenai PMB, KRS, biaya, atau jadwal ya!"
- Jika menjawab dari context, pertahankan angka, tanggal, nama, syarat, atau biaya sesuai isi context.
- Gunakan bullet "-" untuk menampilkan data yang berbentuk daftar.
- DILARANG menyebut nama file, metadata internal, skor similarity, routing, chunk, atau proses RAG.
- GUNAKAN kata "kak" atau "kakak" disetiap kalimat, JANGAN GUNAKAN kata "Kamu".
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

    # --- Chitchat bypass: basa-basi langsung ke model TANPA RAG ---
    if is_chitchat(question):
        if DEBUG:
            print(f"\n[CHITCHAT] '{question}' terdeteksi basa-basi -> skip RAG")
        try:
            response = ollama.chat(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                options={"temperature": 0.3, "num_predict": 256},
            )
            return clean_output(response.get("message", {}).get("content", ""))
        except Exception as exc:
            if DEBUG: print(f"[LLM] chitchat error: {exc}")
            return "Halo kak! Ada yang bisa Minci bantu?"

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
    print("\nMinci - Asisten Virtual Akademik STT Cipasung")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        q = input("Kamu: ").strip()

        if q.lower() in {"exit", "quit"}:
            break

        answer = ask_minci(q)
        print(f"\nKamu: {q}")
        print(f"Minci: {answer}\n")