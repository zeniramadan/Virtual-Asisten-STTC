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
CHAT_MODEL = "qwen2.5:3b"

RETRIEVAL_K = 20
FINAL_CONTEXT_K = 8
MAX_DISTANCE = 0.60
# Sebelumnya 0.58 -- LEBIH KETAT dari MAX_DISTANCE global, padahal query yang
# sampai sini sudah lolos routing (artinya topik/dokumennya sudah "dijamin"
# benar oleh where-filter di ChromaDB). Ambang di sini seharusnya LEBIH
# LONGGAR, bukan lebih ketat -- distance gate di jalur routed cuma perlu jaga
# dari chunk yang benar-benar tidak nyambung SAMA SEKALI di dalam dokumen yang
# sama, bukan menyaring ketepatan topik (itu sudah tugas routing).
# Akibat nilai lama (0.58): query pendek/generik seperti "syarat pendaftaran"
# atau "pendaftaran" saja sering py py py punya distance ~0.6-0.7 ke chunk
# detail (daftar dokumen syarat) -- lolos MAX_DISTANCE global tapi kandas di
# sini, jadi context selalu kosong -> selalu fallback walau routing-nya benar.
ROUTED_MAX_DISTANCE = 0.85
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
    """Prioritaskan section dokumen yang sesuai dengan intent pertanyaan."""
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
            print("[RETRIEVE] ChromaDB tidak mengembalikan dokumen apapun (where-filter mungkin 0 hasil, atau collection kosong).")

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

        # Route sudah membatasi dokumen ke sumber yang relevan. Jangan buang
        # chunk teratas hanya karena bentuk katanya berbeda, misalnya
        # "syarat" vs "persyaratan" atau "daftar" vs "pendaftaran".
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

# Fallback DETERMINISTIK -- dipakai langsung oleh KODE (bukan diminta ke model)
# begitu chunks kosong. Model tidak pernah diberi "pilihan" untuk menjawab ini,
# jadi tidak ada lagi ruang bagi model buat salah menyimpulkan context kosong
# padahal isinya ada.

FALLBACK_TEXT = "Maaf kak, informasi yang kakak tanyakan tidak ada di panduan kami, coba bertanya lebih spesifik, atau silakan kakak hubungi bagian Tata Usaha ya!"

# System prompt ini HANYA dipakai ketika chunks SUDAH DIPASTIKAN ADA ISINYA oleh
# kode (lihat ask_minci). Makanya KONDISI 1 (context kosong) sengaja DIHAPUS
# dari sini -- model tidak perlu lagi menebak/mengecek apakah context kosong,
# karena kalau prompt ini yang dipakai, context SELALU ada isinya. Ini
# menghilangkan sumber kesalahan sebelumnya: model 3B yang kadang salah
# menyimpulkan context kosong padahal datanya ada persis di depannya.
#
# CATATAN PERUBAHAN PERILAKU: KONDISI 3 (jawab pertanyaan umum di luar kampus
# pakai pengetahuan umum model, mis. matematika/sejarah) ikut dihapus di sini.
# Sekarang pertanyaan di luar cakupan dokumen kampus akan selalu jatuh ke
# FALLBACK_TEXT (chunks kosong -> tidak pernah sampai ke LLM sama sekali),
# BUKAN dijawab pakai pengetahuan umum model seperti sebelumnya. Ini konsisten
# dengan semua perbaikan gate/routing yang sudah kita buat sebelumnya (supaya
# Minci tidak "mengarang" jawaban di luar dokumen resmi kampus). Kalau kamu
# justru MASIH mau Minci bisa jawab pertanyaan umum di luar kampus, kasih tau
# saya -- itu perlu jalur terpisah lagi (bukan sekadar taruh balik ke sini),
# karena kalau taruh di sini lagi, prompt ini jadi butuh model MENEBAK lagi
# kapan harus pakai pengetahuan umum vs kapan harus attach ke context -- balik
# ke masalah yang sama.
SYSTEM_PROMPT_WITH_CONTEXT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Generasi Z, tapi tetap sopan.

CONTEXT di bawah ini SUDAH DIPASTIKAN BERISI DATA PANDUAN YANG RELEVAN dengan pertanyaan. Kamu WAJIB menjawab dari situ -- JANGAN PERNAH bilang "tidak ditemukan" atau "tidak ada di panduan" untuk pertanyaan ini, karena datanya PASTI ada di CONTEXT.

ATURAN JAWABAN:
- Jawab pertanyaan pengguna HANYA berdasarkan informasi faktual yang tertulis di dalam CONTEXT tersebut.
- Jika pertanyaan meminta "syarat", berikan DAFTAR SYARAT saja dari context. Jangan jelaskan tata cara.
- Jika pertanyaan meminta "cara", berikan LANGKAH-LANGKAH saja dari context. Jangan berikan daftar syarat.
- JIKA pertanyaan meminta "syarat", "persyaratan" atau "pendaftaran", kamu WAJIB DAN HARUS MENULISKAN SEMUA DAFTAR SYARAT YANG ADA DI CONTEXT SECARA LENGKAP!
- Jika pertanyaan tidak spesifik mengenai jadwal PMB, cantumkan tanggal pendaftaran gelombang 1, 2, dan 3 yang tertera di context.
- Jika pertanyaan meminta "seleksi" berikan jadwal SELEKSI PENERIMAAN MAHASISWA BARU yang ada di context.
- Jika pertanyaan meminta "biaya", "bayar", atau "nominal", berikan SEMUA INFORMASI BESERTA KETERANGANNYA.

ATURAN WAJIB UNTUK SEMUA JAWABAN:
- HARUS menggunakan kata "kak" atau "kakak" di SETIAP kalimat! DILARANG menggunakan kata "Kamu".
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
        
    # 3. Ubah semua bullet poin fisik (• atau *) menjadi "-" tanpa merusak teks
    text = text.replace("•", "-")
    text = text.replace("▪", "-")
    text = text.replace("⁃", "-")
    
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
                options={
                    "temperature": 0.4,     # Naik sedikit ke 0.4 agar gaya Gen-Z nya lebih natural & tidak kaku
                    "top_p": 0.9,           # Membatasi pilihan kata agar tetap masuk akal
                    "num_predict": 100,     # Batasan respons chitchat pendek (maksimal ~100 token)
                },
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

    # --- Keputusan "ada data atau tidak" diambil di KODE, bukan diserahkan
    # ke model. Sebelumnya context kosong tetap dikirim ke LLM dengan harapan
    # dia "membaca" instruksi KONDISI 1 dan menyimpulkan sendiri -- tapi itu
    # juga berarti ketika context ADA ISINYA, model tetap harus "membuktikan
    # sendiri" bahwa ini bukan kasus KONDISI 1, dan model 3B kadang salah
    # simpul (lihat kasus "syarat pendaftaran" yang tetap fallback padahal
    # context-nya lengkap). Dengan cek eksplisit di sini, model HANYA PERNAH
    # melihat prompt yang isinya SUDAH DIPASTIKAN ada datanya, dan bahkan tidak
    # pernah dipanggil sama sekali kalau memang tidak ada apa-apa untuk dijawab.
    if not chunks:
        if DEBUG:
            print("[RAG] Tidak ada chunk relevan -> fallback deterministik, LLM TIDAK dipanggil.")
        return FALLBACK_TEXT

    context = build_context(chunks)

    if DEBUG:
        print("\n" + "=" * 70)
        print("[CONTEXT YANG DIKIRIM KE MODEL]")
        print("=" * 70)
        print(context)
        print("=" * 70)

    try:
        # PENTING: Gunakan format ini agar Ollama menyuntikkan template chat Llama 3.2 secara benar
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT_WITH_CONTEXT
                },
                {
                    "role": "user", 
                    "content": build_user_prompt(question, context)
                },
            ],
            options={
                "temperature": 0.1,    # Sudah benar (rendah agar konsisten)
                "num_predict": 1024,
                # Tambahkan parameter di bawah ini jika model masih suka tidak patuh:
                # "top_p": 0.9,
            },
        )
    except Exception as exc:
        if DEBUG: print(f"[LLM] error: {exc}")
        return "Maaf kak, sistem Minci sedang gangguan. Coba lagi nanti ya!"

    raw_answer = response.get("message", {}).get("content", "")
    if DEBUG:
        print("\n[RAW LLM OUTPUT sebelum clean_output]")
        print(raw_answer)
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