"""
query.py
========

Modul inti RAG Minci.

Alur:
1. Terima pertanyaan user.
2. Tentukan dokumen sumber berdasarkan kata kunci (routing).
3. Ambil konteks dari ChromaDB:
   - Kalau ada routing -> ambil SEMUA chunk milik dokumen itu
     (dokumennya kecil, jadi tidak perlu similarity ranking --
     ini menjamin tidak ada baris yang ketinggalan).
   - Kalau tidak ada routing -> semantic search ke semua dokumen,
     dibatasi TOP_K dan disaring pakai MAX_RELEVANT_DISTANCE supaya
     tidak kebawa chunk yang tidak relevan.
4. Kirim konteks + pertanyaan ke model Minci (Ollama).
5. Bersihkan sedikit formatting jawaban, lalu kembalikan.

ROUTING DOKUMEN:

    Jadwal / kalender / tanggal / agenda / jadwal perwalian
        -> KALENDER.docx

    Biaya / pembayaran / UKT
        -> BIAYA.docx

    PMB / pendaftaran / jurusan / persyaratan masuk
        -> PMB.docx

    KRS / pengisian KRS / syarat KRS / SKS / prosedur perwalian
        -> KRS.docx

    Pertanyaan lain
        -> semantic search semua dokumen

Catatan:
- LoRA/model Minci digunakan untuk gaya & perilaku jawaban.
- Fakta akademik diambil dari RAG (ChromaDB), model tidak boleh
  mengarang di luar konteks yang diberikan (lihat build_system_prompt).
"""

import os
import re
from typing import Optional

import ollama
import chromadb


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")

# Harus sama dengan ingest.py
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"
EMBED_QUERY_PREFIX = ""  # bge-m3 tidak butuh prefix

# Model LoRA yang dibuat lewat Ollama
CHAT_MODEL = "minci"

# Jumlah kandidat untuk semantic search TANPA routing (pertanyaan umum
# yang tidak jelas masuk dokumen mana).
TOP_K = 15

# Ambang jarak cosine untuk semantic search tanpa routing -- di atas ini
# dianggap tidak relevan dan dibuang. Tidak dipakai untuk pertanyaan yang
# sudah di-routing ke satu dokumen, karena di jalur itu semua chunk
# dokumennya diambil apa adanya (lihat retrieve_context_for_source).
MAX_RELEVANT_DISTANCE = 0.62

DEBUG = True


# ============================================================
# NAMA SUMBER DOKUMEN
# ============================================================

SOURCE_KALENDER = "KALENDER.docx"
SOURCE_KRS = "KRS.docx"
SOURCE_PMB = "PMB.docx"
SOURCE_BIAYA = "BIAYA.docx"


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

_count = _collection.count()

if _count == 0:
    print(f"WARNING: collection '{COLLECTION_NAME}' di {CHROMA_DB_DIR} KOSONG.")
    print("Jalankan terlebih dahulu: python ingest.py")
else:
    print(f"OK ChromaDB terbaca: {_count} chunk siap dipakai.")


# ============================================================
# NORMALISASI PERTANYAAN
# ============================================================

def normalize_question(question: str) -> str:
    """Lowercase + rapikan spasi ganda."""
    question = question.lower().strip()
    return re.sub(r"\s+", " ", question)


# ============================================================
# INTENT ROUTER
# ============================================================

def detect_route(question: str) -> Optional[str]:
    """
    Menentukan dokumen sumber berdasarkan kata kunci di pertanyaan.

    Return salah satu SOURCE_* atau None (-> semantic search semua dokumen).

    Prioritas: kalender > biaya > PMB > KRS.
    Kalender diletakkan paling awal karena kata "perwalian" juga muncul
    di KRS.docx -- untuk pertanyaan "jadwal perwalian" kita mau
    KALENDER.docx yang menang, bukan KRS.docx.
    """

    q = normalize_question(question)

    # --------------------------------------------------------
    # 1. KALENDER / JADWAL
    # --------------------------------------------------------

    calendar_keywords = [
        "jadwal", "kapan", "tanggal", "kalender", "agenda", "periode",
        "mulai", "dimulai", "berakhir", "selesai", "batas waktu", "pelaksanaan",
        "awal kuliah", "awal perkuliahan",
        "ujian tengah semester", "ujian akhir semester", "uts", "uas",
        "wisuda", "libur natal", "libur tahun baru", "pra ktmb", "ktmb",
        "herregistrasi", "perwalian",
    ]

    has_calendar_keyword = any(keyword in q for keyword in calendar_keywords)

    # "perwalian" itu ambigu: "jadwal perwalian" -> kalender,
    # tapi "syarat perwalian" / "cara perwalian" -> bukan kalender.
    if "perwalian" in q:
        temporal_keywords = ["jadwal", "kapan", "tanggal", "periode", "mulai", "pelaksanaan", "batas"]
        if any(keyword in q for keyword in temporal_keywords):
            return SOURCE_KALENDER

    if has_calendar_keyword:
        return SOURCE_KALENDER

    # --------------------------------------------------------
    # 2. BIAYA
    # --------------------------------------------------------

    biaya_keywords = [
        "biaya", "bayar", "pembayaran", "uang kuliah", "biaya kuliah",
        "ukt", "uang kuliah tunggal", "semester berapa bayar", "harga kuliah",
    ]

    if any(keyword in q for keyword in biaya_keywords):
        return SOURCE_BIAYA

    # --------------------------------------------------------
    # 3. PMB
    # --------------------------------------------------------

    pmb_keywords = [
        "pmb", "penerimaan mahasiswa baru", "mahasiswa baru", "pendaftaran",
        "mendaftar", "daftar kuliah", "daftar mahasiswa",
        "persyaratan pendaftaran", "syarat pendaftaran", "gelombang",
        "jurusan", "program studi", "prodi", "seleksi",
        "registrasi mahasiswa baru", "pendaftar",
    ]

    if any(keyword in q for keyword in pmb_keywords):
        return SOURCE_PMB

    # --------------------------------------------------------
    # 4. KRS
    # --------------------------------------------------------

    krs_keywords = [
        "krs", "kartu rencana studi", "pengisian krs", "isi krs", "mengisi krs",
        "krs online", "krs-an", "krsan", "sks", "sevima", "dosen wali",
        "dosen pembimbing akademik", "validasi krs", "mata kuliah",
        "ambil mata kuliah", "kartu perubahan rencana studi", "kprs",
        "cuti kuliah", "syarat krs", "syarat perwalian", "cara perwalian",
        "prosedur perwalian",
    ]

    if any(keyword in q for keyword in krs_keywords):
        return SOURCE_KRS

    return None


# ============================================================
# RETRIEVAL: DOKUMEN TER-ROUTE (ambil semua chunk-nya)
# ============================================================

def retrieve_context_for_source(source_name: str) -> str:
    """
    Ambil SEMUA chunk milik satu dokumen, urut sesuai posisi aslinya.

    Dokumen-dokumen ini (KALENDER/BIAYA/PMB/KRS) kecil -- belasan sampai
    puluhan chunk saja -- jadi tidak perlu similarity search atau filter
    jarak sama sekali. Similarity search hanya relevan kalau kita perlu
    memilih SEBAGIAN dari banyak kandidat; di sini kita justru mau semua
    baris dokumennya utuh, supaya pertanyaan seperti "rincian biayanya
    apa aja" selalu dapat konteks yang lengkap.
    """

    results = _collection.get(where={"source": source_name})

    documents = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []

    paired = sorted(
        zip(documents, metadatas),
        key=lambda pair: (pair[1] or {}).get("chunk_index", 0),
    )

    if DEBUG:
        print(f"\n[DEBUG] ROUTED RETRIEVAL -> {source_name}: {len(paired)} chunk")

    blocks = [
        f"[Sumber: {(meta or {}).get('source', source_name)}]\n{doc}"
        for doc, meta in paired
    ]

    return "\n\n---\n\n".join(blocks)


# ============================================================
# RETRIEVAL: SEMANTIC SEARCH SEMUA DOKUMEN (tanpa routing)
# ============================================================

def retrieve_context_semantic(question: str, top_k: int = TOP_K) -> str:
    """Semantic search ke semua dokumen, untuk pertanyaan tanpa route jelas."""

    query_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=f"{EMBED_QUERY_PREFIX}{question}",
    )["embedding"]

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if DEBUG:
        print(f"\n[DEBUG] SEMANTIC RETRIEVAL: {question!r}")

    blocks = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        source = (meta or {}).get("source", "dokumen")

        if dist > MAX_RELEVANT_DISTANCE:
            if DEBUG:
                print(f"  DIBUANG (distance={dist:.4f}) source={source}")
            continue

        if DEBUG:
            print(f"  DIPAKAI (distance={dist:.4f}) source={source}")

        blocks.append(f"[Sumber: {source}]\n{doc}")

    return "\n\n---\n\n".join(blocks)


# ============================================================
# RETRIEVAL: PINTU MASUK
# ============================================================

def retrieve_context(question: str, route: Optional[str]) -> str:
    if route:
        return retrieve_context_for_source(route)
    return retrieve_context_semantic(question)


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt() -> str:
    return """
Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung.

Kamu membantu mahasiswa dan calon mahasiswa mengenai:
- PMB
- KRS
- perwalian
- biaya kuliah
- kalender akademik
- informasi akademik lain yang tersedia dalam konteks.

Gaya bicaramu:
- santai
- ramah
- ceria ala Gen-Z
- sopan
- jelas
- tidak bertele-tele

ATURAN WAJIB:

1. Jawab HANYA berdasarkan konteks yang diberikan. Jangan mengarang informasi.

2. Jangan menggunakan pengetahuan di luar konteks.

3. Jika informasi yang ditanyakan tidak tersedia dalam konteks, jawab persis:
   "Maaf kak, informasi tersebut tidak ada di panduan. Silakan hubungi bagian Tata Usaha."

4. Jangan mencampur informasi dari topik yang berbeda.

5. Jangan mencampur data antara gelombang atau semester.

6. Jika pertanyaan meminta daftar/rincian (mis. daftar biaya, syarat, jadwal),
   tampilkan SEMUA item yang tersedia dalam konteks satu per satu.
   Jangan meringkas beberapa item jadi satu kalimat umum, dan jangan
   melewatkan item hanya karena nilainya bervariasi -- sebutkan variasinya.

7. Jangan menyebut nama file atau sumber dokumen seperti "KRS.docx",
   "PMB.docx", "KALENDER.docx", "BIAYA.docx", "dokumen", atau "file".

8. Jika user menyapa, balas salam terlebih dahulu.
   Jika user tidak menyapa, jangan membuka jawaban dengan sapaan yang tidak perlu.

9. Gunakan bullet point biasa dengan tanda "-". Jangan pakai heading markdown (##).

10. Jika ditanya daftar jurusan/prodi, sebutkan nama prodi langsung tanpa
    tambahan penjelasan kecuali diminta.

11. Jangan memberikan rekomendasi pribadi tentang pilihan beasiswa.

12. Jika pertanyaan tidak berhubungan dengan layanan akademik STT Cipasung,
    katakan bahwa Minci fokus membantu informasi akademik STT Cipasung.

13. Pertahankan angka, tanggal, satuan, nama kegiatan, dan detail lain
    persis sebagaimana tertulis dalam konteks.

Jawab singkat, jelas, natural, dan akurat.
"""


# ============================================================
# USER MESSAGE
# ============================================================

def build_user_message(question: str, context: str) -> str:
    return f"""
Konteks:

{context}

Pertanyaan user:

{question}
"""


# ============================================================
# BERSIHKAN MARKDOWN
# ============================================================

def clean_markdown(text: str) -> str:
    """Rapikan markdown yang tidak diinginkan dari jawaban model."""
    text = text.replace("**", "*")
    for marker in ("### ", "## ", "# "):
        text = text.replace(marker, "")
    text = text.replace("---\n", "").replace("\n---", "")
    return text.strip()


# ============================================================
# ASK MINCI
# ============================================================

def ask_minci(question: str) -> str:
    """Fungsi utama: routing -> retrieval -> generate jawaban."""

    question = question.strip()

    route = detect_route(question)

    if DEBUG:
        print("\n" + "=" * 70)
        print("[DEBUG] ROUTING")
        print(f"Pertanyaan : {question!r}")
        print(f"Route      : {route or 'SEMANTIC SEARCH SEMUA DOKUMEN'}")
        print("=" * 70)

    context = retrieve_context(question, route)

    if context.strip():
        user_content = build_user_message(question, context)
    else:
        user_content = f"Pertanyaan: {question}\n\nInfo: Tidak ada konteks relevan."

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    if DEBUG:
        print("[DEBUG] Memanggil model:", CHAT_MODEL)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        options={"num_predict": 2048},
    )

    answer = clean_markdown(response["message"]["content"])

    if DEBUG:
        print("\n[DEBUG] JAWABAN FINAL:")
        print(answer)
        print("=" * 70)

    return answer


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":
    print("Mode test RAG Minci")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        try:
            question = input("Kamu: ")
        except KeyboardInterrupt:
            print("\nKeluar.")
            break

        if question.strip().lower() in ("exit", "quit"):
            break

        try:
            answer = ask_minci(question)
            print(f"\nMinci: {answer}\n")
        except Exception as e:
            print("\nERROR:")
            print(e)
            print()