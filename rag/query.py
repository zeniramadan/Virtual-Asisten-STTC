"""
query.py
========
Modul inti RAG Minci: menerima pertanyaan user, mencari potongan dokumen paling
relevan dari ChromaDB (PMB.docx, KRS.docx, BIAYA.docx, KALENDER.docx), lalu
meminta model "minci" (hasil fine-tuning LoRA) di Ollama untuk menjawab dengan
gaya bahasanya sendiri, berdasarkan konteks tadi.

Dipanggil oleh webhook WhatsApp (webhook/app.py) & bot Telegram (telegram/tele.py),
atau bisa dites langsung: python query.py

=====================================================================
CATATAN DESAIN -- ini rewrite bersih, tapi semua bug yang sudah pernah
ditemukan & diperbaiki di iterasi-iterasi sebelumnya TETAP ditangani:
=====================================================================
1. Embedding pakai bge-m3 (bukan nomic-embed-text) -- jauh lebih akurat untuk
   Bahasa Indonesia. Tidak butuh prefix instruksi ("search_query:" dsb).
2. ChromaDB WAJIB pakai metrik cosine (bukan default L2) -- default L2 bikin
   ranking retrieval kacau/tidak konsisten.
3. Telemetry ChromaDB dimatikan (anonymized_telemetry=False) -- biar tidak
   spam "Failed to send telemetry event" di log.
4. Filter distance (MAX_RELEVANT_DISTANCE) -- buang chunk yang jelas tidak
   nyambung. Jangan set kelewat ketat (pernah 0.48 -> jawaban benar ikut
   kebuang kalau pertanyaan user agak typo/berantakan).
5. Filter dominasi dokumen sumber (SOURCE_DOMINANCE_MARGIN) -- cegah topik
   dari dokumen berbeda tercampur jadi satu jawaban (misal KRS nyasar pas
   nanya PMB).
6. RUTE TOPIK PAKSA (BARU) -- untuk topik yang HARUS selalu dijawab dari 1
   dokumen tertentu (misal semua pertanyaan jadwal/tanggal WAJIB dari
   KALENDER.docx), kita override hasil "dokumen top-1 by embedding" dengan
   dokumen yang sudah ditentukan, supaya konsisten -- tidak tergantung
   untung-untungan skor embedding.
7. Ekstraksi list item (buat fallback) HARUS ngerti 2 format: bullet "- item"
   (dari PMB/KRS) DAN baris hasil serialisasi tabel "Label: nilai, Label2:
   nilai2" (dari BIAYA/KALENDER yang sekarang tabel asli Word). Dulu cuma
   ngerti bullet "-", jadi baris tabel kelewat semua.
8. Ekstraksi list item WAJIB dibatasi ke SATU dokumen sumber saja (top_source
   / preferred_source) -- dulu ada bug nyata: syarat PMB kecampur baris KRS
   gara-gara diambil dari SELURUH context tanpa filter sumber.
9. Guardrail: JANGAN sebutkan nama file dokumen (PMB.docx, KALENDER.docx,
   dst) ke user -- user tidak perlu tahu urusan internal itu.
10. Guardrail: JANGAN bocorkan tag internal ([Sumber:], [Konteks:], [Bagian:])
    ke jawaban -- itu metadata internal buat sistem, bukan buat ditampilkan.
11. Guardrail: JANGAN markdown (**, ##) -- WhatsApp/Telegram tidak render itu
    dengan benar. Post-processing clean_markdown() jadi jaring pengaman kode.
12. Guardrail: JANGAN meringkas/menggabung daftar (biaya per semester, syarat,
    dst) jadi satu kalimat generik -- WAJIB sebutkan semua item satu-satu.
    Ada fallback deterministik di kode kalau model tetap gagal comply.
13. Kalimat pembuka fallback di-GENERATE (bukan template statis) lewat 1 LLM
    call kecil terpisah -- supaya tidak kedengaran template robotik.
14. Jangan campur nomor gelombang/semester antar baris berbeda (cek baris
    sumbernya PERSIS sebelum menjawab).
15. Bedakan jenis tanggal per kegiatan (Pendaftaran != Seleksi != Pengumuman
    != Registrasi) -- jangan ambil baris yang salah jenis.
16. JANGAN menambahkan saran/pengingat yang tidak diminta & tidak ada di
    konteks (anti-halusinasi "jangan lupa siapkan KRS" dkk).
17. TIDAK ada memori percakapan (conversation history) -- fitur ini sudah
    dilepas sebelumnya sesuai permintaan, setiap pertanyaan berdiri sendiri.
18. CHIT-CHAT EXCEPTION (BARU) -- pesan basa-basi (salam, sapaan, terima
    kasih, dll, dideteksi dari daftar frasa di chitchat.json) langsung
    dijawab model TANPA lewat retrieve_context()/RAG sama sekali. Hanya
    match kalau basa-basinya di AWAL kalimat dan sisanya tidak substansial
    -- supaya "halo, syarat daftar apa aja?" tetap masuk RAG.
"""

import os
import re
import json
import ollama
import chromadb

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
COLLECTION_NAME = "minci_dokumen"

CHITCHAT_PATH = os.path.join(BASE_DIR, "..", "dataset", "chitchat.json")

EMBED_MODEL = "bge-m3"
EMBED_QUERY_PREFIX = ""       # bge-m3 tidak butuh prefix instruksi khusus

CHAT_MODEL = "minci"

TOP_K = 10                     # jumlah kandidat chunk yang diambil dari ChromaDB
DEBUG = True                  # tampilkan proses retrieval & routing di terminal

MAX_RELEVANT_DISTANCE = 0.62  # ambang distance (cosine) -- di atas ini dianggap
                               # tidak nyambung & dibuang. JANGAN diturunkan terlalu
                               # jauh (pernah 0.48 -> jawaban benar ikut kebuang
                               # kalau pertanyaan user agak typo/berantakan).

SOURCE_DOMINANCE_MARGIN = 0.05  # chunk dari dokumen LAIN (beda dari dokumen topik
                               # utama) tetap dipakai KALAU distance-nya masih deket
                               # (selisih <= ini) sama chunk terbaik dari dokumen topik
                               # utama. Kalau lebih jauh, dibuang APAPUN angka distance-nya.

# --------------------------------------------------------------------
# RUTE TOPIK PAKSA: kata kunci -> nama file dokumen yang WAJIB dipakai.
# Kalau pertanyaan match salah satu grup ini, dan dokumen itu memang muncul
# di antara hasil retrieval (walau bukan rangking #1), dokumen itu dipaksa
# jadi "topik utama" -- TIDAK peduli dokumen mana yang skor embedding-nya
# paling dekat. Ini penting untuk konsistensi: semua pertanyaan jadwal/tanggal
# HARUS selalu dari KALENDER.docx, bukan kadang KALENDER kadang KRS/PMB
# tergantung untung-untungan skor embedding.
#
# Urutan penting: dicek dari atas ke bawah, yang pertama match dipakai.
# --------------------------------------------------------------------
PREFERRED_SOURCE_KEYWORDS: list[tuple[list[str], str]] = [
    (["jadwal", "tanggal", "kalender", "kapan", "gelombang"], "KALENDER.docx"),
    (["biaya", "bayar", "nominal", "harga", "ukt", "pembayaran", "cicil"], "BIAYA.docx"),
]

# ============================================================
# INISIALISASI CHROMADB
# ============================================================
_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)
# PENTING: metadata cosine ini harus SAMA PERSIS dengan yang dipakai ingest.py.
# Kalau collection sudah pernah dibuat dengan metrik lain, metadata di sini
# TIDAK mengubahnya -- WAJIB ingest ulang dari nol kalau ganti metrik.
_collection = _client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

_count = _collection.count()
if _count == 0:
    print(f"⚠️  PERINGATAN: collection '{COLLECTION_NAME}' di {CHROMA_DB_DIR} KOSONG (0 chunk).")
    print("   Jalankan dulu: python ingest.py (pastikan ada file .docx di folder documents/)")
else:
    print(f"✅ ChromaDB terbaca: {_count} chunk siap dipakai (dari {CHROMA_DB_DIR})")


# ============================================================
# BAGIAN 0: NORMALISASI QUERY (bahasa natural -> bentuk yang cocok dengan dokumen)
# ============================================================
# Query pengguna sering pakai kata seperti "gelombang satu" atau "gelombang dua"
# sedangkan dokumen menyimpan data sebagai "Gelombang 1", "Gelombang 2".
# Tanpa normalisasi, embedding bisa sedikit kurang cocok walau konteks sebenarnya
# sudah ada. Ini fix ringan tapi berdampak besar untuk pertanyaan seperti:
# "hasil seleksi gelombang satu kapan?"

_NUMBER_WORDS = {
    "nol": "0", "satu": "1", "dua": "2", "tiga": "3", "empat": "4",
    "lima": "5", "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9",
    "sepuluh": "10",
}


def normalize_query_text(question: str) -> str:
    """Normalisasi bentuk natural-language agar lebih cocok dengan entri dokumen."""
    q = question.strip()
    q = q.replace("Gelombang I", "Gelombang 1").replace("gelombang i", "gelombang 1")
    q = q.replace("Gelombang II", "Gelombang 2").replace("gelombang ii", "gelombang 2")
    q = q.replace("Gelombang III", "Gelombang 3").replace("gelombang iii", "gelombang 3")

    for word, number in _NUMBER_WORDS.items():
        q = re.sub(rf"\bgelombang\s+{word}\b", f"gelombang {number}", q, flags=re.IGNORECASE)
        q = re.sub(rf"\bgelombang\s+{word}\b", f"gelombang {number}", q, flags=re.IGNORECASE)

    return q


# ============================================================
# BAGIAN 0: CHIT-CHAT (basa-basi) -- pengecualian, TANPA RAG
# ============================================================
# Kalau pertanyaan user cuma basa-basi (salam, sapaan, ucapan terima
# kasih, dll), langsung dijawab oleh model TANPA lewat retrieve_context()
# sama sekali -- tidak ada embedding, tidak ada query ke ChromaDB. Selain
# lebih cepat, ini juga mencegah RAG "maksa" nyari konteks dokumen buat
# pertanyaan yang sebenarnya tidak butuh info akademik apapun.
#
# Daftar frasanya disimpan di file JSON terpisah (chitchat.json) supaya
# gampang ditambah/diedit tanpa utak-atik kode.
#
# PENTING: deteksinya HANYA match kalau basa-basinya ada di AWAL kalimat
# DAN sisa kalimat setelah itu memang tidak substansial (<= 3 kata, atau
# tidak ada kata tanya) -- supaya pesan seperti "halo min, syarat daftar
# apa aja?" TETAP masuk RAG (karena ada pertanyaan sungguhan di
# belakangnya), bukan ke jalur chit-chat.

def _load_chitchat_phrases() -> list[str]:
    """Baca semua frasa chit-chat dari chitchat.json jadi satu list flat."""
    try:
        with open(CHITCHAT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  PERINGATAN: {CHITCHAT_PATH} tidak ditemukan -- fitur chit-chat nonaktif.")
        return []
    except json.JSONDecodeError as e:
        print(f"⚠️  PERINGATAN: {CHITCHAT_PATH} isinya bukan JSON valid ({e}) -- fitur chit-chat nonaktif.")
        return []

    phrases = []
    for kategori, daftar in data.items():
        phrases.extend(daftar)

    # frasa yang lebih panjang dicek duluan, supaya "selamat pagi" match
    # duluan daripada cuma "pagi" (potongan dari frasa yang lebih panjang)
    phrases.sort(key=len, reverse=True)
    return phrases


_CHITCHAT_PHRASES = _load_chitchat_phrases()

if DEBUG:
    print(f"💬 Chit-chat: {len(_CHITCHAT_PHRASES)} frasa dimuat dari {CHITCHAT_PATH}")


def is_chitchat(question: str) -> bool:
    """
    True kalau pertanyaan terdeteksi cuma basa-basi di awal DAN tidak ada
    substansi pertanyaan sungguhan di belakangnya.
    """
    if not _CHITCHAT_PHRASES:
        return False

    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[!?.,]+$", "", q)  # buang tanda baca di ujung

    for phrase in _CHITCHAT_PHRASES:
        if q == phrase:
            return True
        if q.startswith(phrase + " ") or q.startswith(phrase + ","):
            remainder = q[len(phrase):].strip(" ,.-")
            # sisa kalimat pendek (<=3 kata) dianggap masih bagian dari
            # basa-basi (mis. "halo kak", "makasih banyak ya"), bukan
            # pertanyaan sungguhan
            if len(remainder.split()) <= 3:
                return True

    return False


def build_chitchat_system_prompt() -> str:
    """
    System prompt ringan khusus basa-basi -- sengaja jauh lebih pendek
    dari build_system_prompt() karena tidak butuh aturan seputar konteks
    dokumen/RAG sama sekali.
    """
    return """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi sopan.

Ini pesan basa-basi (sapaan/ucapan terima kasih/obrolan ringan), BUKAN pertanyaan akademik. Balas SINGKAT (1-2 kalimat) dan natural sesuai basa-basinya. Kalau relevan, tutup dengan menawarkan bantuan seputar PMB/KRS/biaya/jadwal akademik. JANGAN mengarang info akademik apapun di sini."""


# ============================================================
# BAGIAN 1: RETRIEVAL (cari dokumen relevan + filter)
# ============================================================

def _detect_preferred_source(question: str) -> str | None:
    """Cek apakah pertanyaan match salah satu rute topik paksa (lihat PREFERRED_SOURCE_KEYWORDS)."""
    q_lower = question.lower()
    for keywords, source_name in PREFERRED_SOURCE_KEYWORDS:
        if any(kw in q_lower for kw in keywords):
            return source_name
    return None


def retrieve_context(question: str, top_k: int = TOP_K) -> str:
    """Cari potongan dokumen paling relevan dengan pertanyaan user, sudah difilter."""
    query_embedding = ollama.embeddings(
        model=EMBED_MODEL, prompt=f"{EMBED_QUERY_PREFIX}{question}"
    )["embedding"]

    results = _collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        if DEBUG:
            print(f"\n🔍 [DEBUG] Retrieval untuk {question!r}: KOSONG, tidak ada chunk ditemukan.")
        return ""

    # --- Tentukan "dokumen topik utama" ---
    # Default: dokumen dari hasil top-1 (paling dekat secara embedding).
    # Tapi kalau pertanyaan match rute topik paksa DAN dokumen itu memang ada
    # di antara hasil retrieval, dokumen itu MENANG (override top-1 embedding).
    embedding_top_source = metadatas[0].get("source")
    embedding_best_distance = distances[0]

    preferred_source = _detect_preferred_source(question)
    top_source = embedding_top_source
    best_distance = embedding_best_distance
    routing_note = "dari top-1 embedding"

    if preferred_source:
        preferred_distances = [
            d for d, m in zip(distances, metadatas) if m.get("source") == preferred_source
        ]
        if preferred_distances:
            top_source = preferred_source
            best_distance = min(preferred_distances)
            routing_note = f"DIPAKSA rute topik ke '{preferred_source}' (override top-1 embedding: {embedding_top_source})"

    if DEBUG:
        print("\n" + "=" * 70)
        print(f"🔍 [DEBUG] Retrieval untuk: {question!r}")
        print(f"    Dokumen topik utama: {top_source}  ({routing_note})")
        print("=" * 70)

    context_blocks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        source = meta.get("source", "dokumen")
        dipakai = True
        alasan = "✅ DIPAKAI"

        if dist > MAX_RELEVANT_DISTANCE:
            dipakai = False
            alasan = "❌ DIBUANG (distance di atas ambang absolut)"
        elif source != top_source:
            if preferred_source:
                # Rute topik PAKSA aktif -> TIDAK ADA toleransi jarak sama sekali,
                # WAJIB persis dari dokumen yang dipaksa. Kalau pakai toleransi
                # margin di sini, dokumen lain yang kebetulan distance-nya LEBIH
                # KECIL dari dokumen yang dipaksa bisa lolos filter (bug yang
                # sempat kejadian) -- makanya di jalur rute paksa harus tegas.
                dipakai = False
                alasan = f"❌ DIBUANG (rute topik dipaksa ke '{top_source}', dokumen '{source}' tidak dipakai sama sekali)"
            elif dist > best_distance + SOURCE_DOMINANCE_MARGIN:
                dipakai = False
                alasan = f"❌ DIBUANG (dokumen '{source}' beda dari topik utama '{top_source}')"

        if DEBUG:
            print(f"(distance={dist:.4f}, sumber={source}) {alasan}")
            print(doc[:250], "..." if len(doc) > 250 else "")
            print()

        if dipakai:
            context_blocks.append(f"[Sumber: {source}]\n{doc}")

    if DEBUG:
        print("=" * 70 + "\n")

    return "\n\n---\n\n".join(context_blocks)


def get_top_source_from_context(context: str) -> str | None:
    """Ambil nama dokumen sumber dari blok PERTAMA di context (= yang paling relevan)."""
    if not context:
        return None
    first_block = context.split("\n\n---\n\n")[0]
    first_line = first_block.split("\n")[0] if first_block else ""
    if first_line.startswith("[Sumber:"):
        return first_line.replace("[Sumber:", "").replace("]", "").strip()
    return None


# ============================================================
# BAGIAN 2: EKSTRAKSI LIST ITEM (buat highlight & fallback)
# Harus ngerti 2 format konten: bullet "- item" (PMB/KRS) DAN baris hasil
# serialisasi tabel "Label: nilai, Label2: nilai2" (BIAYA/KALENDER).
# ============================================================

_TAG_PREFIX_RE = re.compile(r"^\[(Sumber|Bagian|Konteks):[^\]]*\]\s*")


def _strip_internal_tags(line: str) -> str:
    """Buang tag internal '[Sumber: ...]'/'[Bagian: ...]'/'[Konteks: ...]' dari depan baris."""
    return _TAG_PREFIX_RE.sub("", line).strip()


def _is_list_item_line(raw_line: str) -> bool:
    """
    Cek apakah baris ini "item list" yang layak ditampilkan -- entah bullet
    biasa ("- item") ATAU baris hasil serialisasi tabel ("Label: nilai, ...").
    """
    line = raw_line.strip()
    if not line or line.startswith("##"):
        return False

    content = _strip_internal_tags(line)
    if not content:
        return False

    if content.startswith("- "):
        return True

    # Baris tabel: minimal ada satu pola "Label: nilai" di dalamnya
    return ": " in content and len(content) > 3


def _format_list_item(raw_line: str) -> str:
    """Bersihkan tag internal dari satu baris item, pastikan tampil rapi sebagai bullet."""
    content = _strip_internal_tags(raw_line.strip())
    if content.startswith("- "):
        return content
    return f"- {content}"


def extract_items_from_source(context: str, source: str | None) -> list[str]:
    """
    Ambil semua "item list" (bullet ATAU baris tabel) dari 'context', dibatasi
    HANYA dari blok dokumen yang sumbernya SAMA PERSIS dengan 'source'.

    Kenapa dibatasi per-dokumen: context bisa berisi campuran blok dari BEBERAPA
    dokumen sekaligus. Tanpa filter ini, daftar hasil bisa kecampur 2 topik
    berbeda (misal syarat PMB kecampur baris prosedur KRS).
    """
    if not context:
        return []

    blocks = context.split("\n\n---\n\n") if source else [context]
    items = []
    for block in blocks:
        lines = block.split("\n")
        if not lines:
            continue
        if source:
            if not lines[0].startswith("[Sumber:"):
                continue
            block_source = lines[0].replace("[Sumber:", "").replace("]", "").strip()
            if block_source != source:
                continue
            content_lines = lines[1:]
        else:
            content_lines = lines

        for line in content_lines:
            if _is_list_item_line(line):
                items.append(_format_list_item(line))

    return items


def find_highlighted_lines(question: str, context: str) -> str:
    """
    Cari baris yang paling cocok secara HARFIAH (keyword sederhana, deterministik,
    bukan nebak-nebak kayak LLM) dengan tipe informasi spesifik yang ditanya, lalu
    tampilkan sebagai "petunjuk" terpisah di depan konteks. Terbukti dari testing,
    model 3B masih suka salah ambil baris meski sudah ada instruksi eksplisit.
    """
    q_lower = question.lower()

    keyword_groups = [
        ("pengumuman hasil seleksi", ["pengumuman"]),
        ("jadwal/tanggal kegiatan", ["jadwal", "tanggal", "kalender", "kapan"]),
        ("seleksi/tes", ["seleksi", "tes", "ujian"]),
        ("pendaftaran PMB", ["pendaftaran", "buka", "dibuka", "penerimaan", "gelombang", "syarat", "persyaratan", "administrasi"]),
        ("syarat pendaftaran", ["syarat", "persyaratan", "dokumen", "berkas"]),
        ("jurusan/prodi", ["jurusan", "prodi", "program studi"]),
        ("biaya/pembayaran", ["biaya", "pembayaran", "bayar", "ukt", "nominal"]),
        ("prosedur KRS/perwalian", ["krs", "perwalian", "sevima", "dosen wali", "herregistrasi"]),
    ]

    matched_label, matched_keywords = None, None
    for label, kws in keyword_groups:
        if any(kw in q_lower for kw in kws):
            matched_label, matched_keywords = label, kws
            break

    if not matched_keywords:
        return ""

    matching_lines = [
        _format_list_item(line) for line in context.split("\n")
        if any(kw in line.lower() for kw in matched_keywords) and _is_list_item_line(line)
    ]

    if not matching_lines:
        return ""

    daftar = "\n".join(matching_lines[:15])
    return (
        f"\n\n🎯 PETUNJUK FOKUS: Pertanyaan ini soal '{matched_label}'. "
        f"Baris paling relevan dari konteks:\n{daftar}\n"
        f"(Gunakan baris di atas sebagai acuan utama jawabanmu -- JANGAN pakai baris lain "
        f"yang jenis informasinya beda.)"
    )


# ============================================================
# BAGIAN 3: PROMPT (system prompt + user message)
# ============================================================

def build_system_prompt() -> str:
    return """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi sopan dan tidak berlebihan.

ATURAN WAJIB:
1. Jawab HANYA dari konteks yang diberikan. DILARANG mengarang, menambah tanggal/info yang tidak tertulis, atau pakai pengetahuan lain. Jika info TIDAK ADA di konteks, jawab PERSIS: "Maaf kak, informasi tersebut tidak ada di panduan. Silakan hubungi bagian Tata Usaha."
2. JANGAN campur data antar gelombang/semester atau salah sebut jenis kegiatan (Pendaftaran ≠ Seleksi ≠ Pengumuman). Ambil baris yang PERSIS sesuai.
3. JANGAN pakai format markdown (##, **, penomoran 1/2/3). Pakai bullet "-" untuk daftar.
4. SANGAT PENTING -- JANGAN PERNAH menyebutkan nama file dokumen (seperti "PMB.docx", dll) dan JANGAN PERNAH menampilkan tag internal seperti "[Sumber: ...]" ke jawaban.
5. Jika konteks berupa daftar/list (syarat, biaya, jadwal, dll), tulis dalam bentuk bullet point, JANGAN diringkas jadi paragraf. TULISKAN SEMUA item yang ada di konteks dengan lengkap.
6. JANGAN tambah penutup/saran/pengingat apapun yang tidak ada di konteks dan tidak diminta user. Jawab PERSIS yang ditanya saja.

Jawab dengan jelas, ceria, tidak bertele-tele, dan tidak ambigu."""


def build_user_message(question: str, context: str) -> str:
    msg = f"""Konteks:
{context}

Pertanyaan:
{question}"""

    q_lower = question.lower()
    if any(kw in q_lower for kw in ["syarat", "persyaratan", "pendaftaran"]):
        msg += "\n\nINSTRUKSI WAJIB: Kalau ada daftar item di konteks (bullet atau baris tabel), tampilkan SEMUA item itu sebagai bullet point terpisah. JANGAN ringkas jadi paragraf."
    elif any(kw in q_lower for kw in ["biaya", "pembayaran", "bayar", "kuliah", "ukt"]):
        msg += "\n\nINSTRUKSI WAJIB: Tampilkan SEMUA item biaya sebagai daftar bullet point terpisah, dengan nominal PERSIS seperti di konteks. JANGAN gabung atau ringkas."
    elif any(kw in q_lower for kw in ["jadwal", "tanggal", "kalender", "kegiatan"]):
        msg += "\n\nINSTRUKSI WAJIB: Tampilkan SEMUA tanggal/kegiatan yang relevan sebagai daftar bullet point terpisah, jangan cuma sebut sebagian."

    return msg


# ============================================================
# BAGIAN 4: POST-PROCESSING (pembersihan jawaban akhir)
# ============================================================

def clean_markdown(text: str) -> str:
    """
    Jaring pengaman KODE (bukan cuma andalkan model "nurut" instruksi) buat
    buang simbol markdown yang kadang masih kebablasan ditulis model.
    """
    text = text.replace("**", "*")
    text = text.replace("## ", "").replace("### ", "").replace("# ", "")
    text = text.replace("---\n", "").replace("\n---", "")
    return text.strip()


def strip_leaked_internal_tags(text: str) -> str:
    """
    Jaring pengaman KODE buat kasus model kebablasan nampilin tag internal
    ([Sumber: ...], [Bagian: ...], [Konteks: ...]) atau nama file dokumen
    langsung ke jawaban, walau sudah dilarang di guardrail #11 & #12.
    """
    text = _TAG_PREFIX_RE.sub("", text)
    # Buang juga kalau tag itu nyempil di TENGAH baris, bukan cuma di awal
    text = re.sub(r"\[(Sumber|Bagian|Konteks):[^\]]*\]\s*", "", text)
    # Buang penyebutan nama file dokumen kalau kebablasan disebut
    for fname in ["PMB.docx", "KRS.docx", "BIAYA.docx", "KALENDER.docx"]:
        text = text.replace(fname, "").replace(fname.replace(".docx", ""), "")
    return text.strip()


def generate_intro_sentence(question: str, item_count: int, topic_label: str) -> str:
    """
    Generate SATU/DUA kalimat pembuka yang natural (gaya bicara Minci) buat
    mengantar daftar bullet point, TANPA menyebutkan isi list-nya satu per
    satu -- isi list-nya sudah dijamin lengkap secara terpisah oleh kode.

    Dipanggil HANYA saat jawaban utama gagal menghasilkan bullet point sama
    sekali -- tidak menambah biaya/waktu di jalur normal yang sudah benar.
    """
    prompt = f"""Kamu adalah Minci, asisten virtual akademik STT Cipasung. Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi sopan.

Pertanyaan user: "{question}"

Tulis kalimat pembuka SINGKAT (1-2 kalimat, BUKAN daftar/list) untuk mengantar jawaban berupa {topic_label} yang berisi {item_count} item. JANGAN sebutkan isi item-nya satu per satu -- daftar itemnya akan ditampilkan TERPISAH setelah kalimat pembukamu. Kalau pertanyaan user diawali sapaan, balas sapaannya dulu di kalimat pembuka ini. JANGAN sebutkan nama file dokumen apapun.

PENTING: Jawab HANYA dengan kalimat pembukanya saja. JANGAN bullet point, JANGAN tanda kutip, JANGAN penjelasan lain."""

    try:
        response = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        intro = response["message"]["content"].strip().strip('"')
        intro = clean_markdown(intro)
        intro = strip_leaked_internal_tags(intro)
        intro = "\n".join(
            line for line in intro.split("\n") if not line.strip().startswith("-")
        ).strip()
        return intro if intro else f"Berikut {topic_label}-nya, kak:"
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Gagal generate kalimat pembuka ({e}), pakai fallback template")
        return f"Berikut {topic_label}-nya, kak:"


# ============================================================
# BAGIAN 5: FUNGSI UTAMA
# ============================================================

_LIST_QUESTION_KEYWORDS = [
    "syarat", "persyaratan", "biaya", "pembayaran", "apa saja", "apa aja",
    "jadwal", "tanggal", "kegiatan", "kalender",
]


def ask_minci(question: str) -> str:
    """
    Fungsi utama: retrieval + generation. Setiap pertanyaan diproses berdiri
    sendiri (TIDAK ada riwayat/memori percakapan -- fitur ini sudah dilepas).
    """
    question = normalize_query_text(question)

    # --- Pengecualian chit-chat: skip RAG sama sekali kalau basa-basi ---
    if is_chitchat(question):
        if DEBUG:
            print(f"\n💬 [DEBUG] '{question!r}' terdeteksi CHIT-CHAT -> skip RAG, langsung ke model.")

        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": build_chitchat_system_prompt()},
                {"role": "user", "content": question},
            ],
            options={"num_predict": 256},
        )
        answer = clean_markdown(response["message"]["content"])
        answer = strip_leaked_internal_tags(answer)
        return answer

    context = retrieve_context(question)

    if not context.strip():
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"Pertanyaan: {question}\n\nInfo: Tidak ada konteks relevan."},
        ]
    else:
        hint = find_highlighted_lines(question, context)
        full_context = hint + "\n\n" + context if hint else context
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_message(question, full_context)},
        ]

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        options={"num_predict": 2048},
    )

    answer = response["message"]["content"]
    answer = clean_markdown(answer)
    answer = strip_leaked_internal_tags(answer)

    # --- Fallback deterministik: kalau model gagal bikin bullet sama sekali,
    #     tapi konteksnya jelas-jelas punya >= 4 item list, kode yang susun
    #     daftarnya sendiri (dijamin lengkap), model cuma dipakai buat intro. ---
    q_lower = question.lower()
    is_list_question = any(kw in q_lower for kw in _LIST_QUESTION_KEYWORDS)

    if is_list_question and context:
        answer_bullets = [line.strip() for line in answer.split("\n") if line.strip().startswith("-")]

        top_source = get_top_source_from_context(context)
        context_items = extract_items_from_source(context, top_source)

        if len(answer_bullets) == 0 and len(context_items) >= 4:
            if "biaya" in q_lower or "bayar" in q_lower:
                topic_label = "daftar biaya"
            elif "jadwal" in q_lower or "tanggal" in q_lower or "kalender" in q_lower:
                topic_label = "jadwal kegiatan"
            else:
                topic_label = "daftar persyaratan"

            intro = generate_intro_sentence(question, len(context_items[:15]), topic_label)
            answer = intro + "\n" + "\n".join(context_items[:15])
            answer = clean_markdown(answer)
            answer = strip_leaked_internal_tags(answer)

    return answer


if __name__ == "__main__":
    print("💬 Mode test RAG Minci (ketik 'exit' untuk keluar)\n")
    while True:
        q = input("Kamu: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        jawaban = ask_minci(q)
        print(f"\nMinci: {jawaban}\n")