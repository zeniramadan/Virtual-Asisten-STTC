"""
query.py
============
Modul inti RAG: menerima pertanyaan user, mencari potongan dokumen paling relevan
dari ChromaDB (fakta dari PMB/KRS), lalu meminta model "minci" (hasil fine-tuning LoRA)
di Ollama untuk menjawab dengan gaya bahasanya sendiri, berdasarkan konteks tadi.

Modul ini dipanggil oleh webhook WhatsApp (lihat webhook/app.py) maupun bot Telegram
(lihat telegram/telegram_bot.py), tapi juga bisa dites langsung lewat terminal:
python rag_query.py
"""

import os
import ollama
import chromadb

# ====== KONFIGURASI ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Tambahkan "database" di tengah path-nya
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"        # HARUS SAMA PERSIS dengan yang dipakai rag_ingest.py
EMBED_QUERY_PREFIX = ""       # bge-m3 tidak butuh prefix (beda dengan nomic-embed-text dulu)
CHAT_MODEL = "minci"          # nama model hasil `ollama create minci -f Modelfile`
TOP_K = 8                     # jumlah chunk paling relevan yang diambil. Dinaikkan dari 4 -> 8
                               # karena skor similarity di dokumen ini rentangnya sempit/mirip-mirip,
                               # jadi butuh lebih banyak kandidat biar chunk yang benar ikut kebawa
DEBUG = True                  # set True biar konteks yang diambil ditampilkan di terminal

# Ambang batas distance (cosine) -- chunk yang jaraknya LEBIH BESAR dari ini dianggap
# TIDAK relevan dan DIBUANG dari konteks, tidak dikirim ke model sama sekali.
#
# Angka ini didapat dari observasi empiris debug log: chunk yang BENAR-BENAR relevan
# selalu ada di rentang distance ~0.36-0.54, sedangkan chunk yang topiknya beda
# (misal KRS nyasar pas nanya PMB) selalu >0.60. Tanpa filter ini, TOP_K=8 sering
# menarik campuran topik yang bikin model salah "mencampur" info (misal jadwal PMB
# dicampur aturan KRS jadi satu jawaban yang ngawur).
MAX_RELEVANT_DISTANCE = 0.62  # dinaikkan dari 0.48 -> 0.62. Ambang 0.48 terbukti TERLALU
                               # KETAT: jawaban yang BENAR bisa punya distance sampai ~0.58
                               # kalau pertanyaan user agak berantakan/typo (wajar terjadi di
                               # chat WhatsApp beneran, mis. "gelombang 3 kapan di bukan min?").
                               # Filter dominasi dokumen sumber (SOURCE_DOMINANCE_MARGIN di
                               # bawah) sudah jadi pertahanan UTAMA buat cegah topik lintas-
                               # dokumen ke campur, jadi ambang absolut ini cukup buat nolak
                               # yang JELAS tidak nyambung saja (biasanya di atas ~0.65-0.70).
SOURCE_DOMINANCE_MARGIN = 0.03  # toleransi: chunk dari dokumen LAIN (beda dari topik utama
                               # hasil top-1) tetap dipakai KALAU distance-nya masih deket
                               # banget (selisih <= 0.03) sama chunk terbaik. Kalau lebih
                               # jauh dari itu, dianggap topik beda dan dibuang APAPUN
                               # angka distance-nya (lebih tegas dari sekadar ambang absolut).
# ==========================

_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)
# PENTING: metadata cosine ini harus SAMA PERSIS dengan yang dipakai rag_ingest.py.
# Kalau collection sudah pernah dibuat sebelumnya dengan metrik lain, metadata di
# sini TIDAK akan mengubahnya -- makanya kalau ganti metrik, WAJIB jalankan ulang
# rag_ingest.py (yang selalu hapus & buat ulang collection dari nol).
_collection = _client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

_count = _collection.count()
if _count == 0:
    print(f"⚠️  PERINGATAN: collection '{COLLECTION_NAME}' di {CHROMA_DB_DIR} KOSONG (0 chunk).")
    print("   Jalankan dulu: python rag_ingest.py (dan pastikan ada file .docx di folder documents/)")
else:
    print(f"✅ ChromaDB terbaca: {_count} chunk siap dipakai (dari {CHROMA_DB_DIR})")


def find_highlighted_lines(question: str, context: str) -> str:
    """
    Cari baris-baris di 'context' yang paling cocok secara HARFIAH dengan tipe
    informasi spesifik yang ditanya (pengumuman / seleksi / registrasi / biaya, dll),
    lalu tampilkan sebagai "petunjuk" terpisah di depan konteks.

    KENAPA INI PERLU (bukan cuma andalkan instruksi teks ke model):
    Terbukti dari testing, model 3B masih suka salah ambil baris meski sudah ada
    instruksi eksplisit -- misal ditanya "pengumuman" tapi malah jawab pakai tanggal
    baris "Seleksi Penerimaan Mahasiswa Baru". Pencarian keyword sederhana di sini
    bersifat DETERMINISTIK (bukan nebak-nebak kayak LLM), jadi lebih bisa diandalkan
    untuk highlight baris yang tepat sebelum model mulai menjawab.
    """
    q_lower = question.lower()

    keyword_groups = [
        ("pengumuman", ["pengumuman"]),
        ("seleksi/tes", ["seleksi", "tes", "ujian"]),
        ("pendaftaran", ["pendaftaran", "buka", "dibuka", "penerimaan", "gelombang"]),
        ("jurusan/prodi", ["jurusan", "prodi", "program studi"]),
        ("biaya", ["biaya"]),
    ]

    matched_label, matched_keywords = None, None
    for label, kws in keyword_groups:
        if any(kw in q_lower for kw in kws):
            matched_label, matched_keywords = label, kws
            break

    if not matched_keywords:
        return ""

    matching_lines = [
        line.strip() for line in context.split("\n")
        if any(kw in line.lower() for kw in matched_keywords) and line.strip()
    ]

    if not matching_lines:
        return ""

    daftar = "\n".join(f"- {l}" for l in matching_lines)
    return (
        f"\n\n🎯 PETUNJUK FOKUS: Pertanyaan ini soal '{matched_label}'. "
        f"Baris paling relevan dari konteks:\n{daftar}\n"
        f"(Gunakan baris di atas sebagai acuan utama jawabanmu -- JANGAN pakai baris lain "
        f"yang jenis informasinya beda.)"
    )


def retrieve_context(question: str, top_k: int = TOP_K):
    """Cari potongan dokumen paling relevan dengan pertanyaan user."""
    # Prefix instruksi (kosong untuk bge-m3, disiapkan di sini kalau nanti ganti
    # model lain yang butuh prefix seperti nomic-embed-text dulu).
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=f"{EMBED_QUERY_PREFIX}{question}")["embedding"]

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Dokumen sumber dari hasil PALING relevan (top-1) dianggap "topik utama" pertanyaan
    # ini. Chunk dari dokumen LAIN dibuang sepenuhnya -- ini lebih tegas dibanding cuma
    # mengandalkan angka distance, karena terbukti distance relevan vs tidak relevan
    # bisa tumpang tindih (KRS yang tidak nyambung kadang skornya masih "lumayan dekat").
    # Trade-off: kalau nanti ada pertanyaan yang MEMANG butuh info dari 2 dokumen
    # sekaligus, ini bisa kepotong -- tapi untuk sekarang manfaatnya (mencegah topik
    # ke campur) lebih besar dari risikonya.
    top_source = metadatas[0].get("source") if metadatas else None
    best_distance = distances[0] if distances else None

    if DEBUG:
        print("\n" + "=" * 60)
        print(f"🔍 [DEBUG] Hasil retrieval untuk: {question!r}")
        print(f"    Dokumen topik utama (dari top-1): {top_source}")
        print("=" * 60)
        if not documents:
            print("(KOSONG — tidak ada chunk yang ditemukan sama sekali!)")
        for doc, meta, dist in zip(documents, metadatas, distances):
            source = meta.get("source")
            if dist > MAX_RELEVANT_DISTANCE:
                keterangan = "❌ DIBUANG (distance di atas ambang absolut)"
            elif top_source and source != top_source and dist > best_distance + SOURCE_DOMINANCE_MARGIN:
                keterangan = f"❌ DIBUANG (beda dokumen dari topik utama: {source} != {top_source})"
            else:
                keterangan = "✅ DIPAKAI"
            print(f"(distance={dist:.4f}, sumber={source}) {keterangan}")
            print(doc[:300], "..." if len(doc) > 300 else "")
            print()
        print("=" * 60 + "\n")

    context_blocks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        source = meta.get("source", "dokumen")
        if dist > MAX_RELEVANT_DISTANCE:
            continue
        if top_source and source != top_source and dist > best_distance + SOURCE_DOMINANCE_MARGIN:
            continue
        context_blocks.append(f"[Sumber: {source}]\n{doc}")

    return "\n\n---\n\n".join(context_blocks)


def build_system_prompt() -> str:
    return """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Kamu membantu mahasiswa dan calon mahasiswa terkait PMB (Penerimaan Mahasiswa Baru) dan KRS (Kartu Rencana Studi). Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi sopan dan tidak berlebihan.

ATURAN WAJIB:
1. Jawab HANYA dari konteks yang diberikan. kamu DILARANG mengarang, menambah tanggal/info yang tidak tertulis, atau pakai pengetahuan lain.
2. Jika info TIDAK ADA di konteks, jawab PERSIS: "Maaf kak, informasi tersebut tidak ada di panduan. Silakan hubungi bagian Tata Usaha."
3. Konteks bisa campur beberapa gelombang/semester/topik. Cek TIAP baris, cari yang cocok dengan pertanyaan (gelombang 1 = Gelombang I). Jangan menyerah hanya karena ada baris lain yang beda topik.
4. JANGAN campur data antar gelombang/semester. Nomor yang kamu sebut HARUS SAMA dengan nomor di baris sumber. Jangan labeli Gelombang II sebagai Gelombang I.
6. Jika pertanyaan tidak spesifik gelombang atau semester mana, kamu sebutkan SEMUA yang ada di konteks dengan LENGKAP.
7. Bedakan jenis tanggal: "Seleksi" ≠ "Pengumuman" ≠ "Registrasi". Ambil baris yang PERSIS sesuai jenis info yang ditanya.
8. Jika konteks campur topik beda (PMB + KRS), fokus HANYA pada topik yang ditanya. Abaikan topik lain.
9. JANGAN tambah penutup/saran/pengingat yang tidak ada di konteks dan tidak ditanya. Jawab PERSIS yang ditanya saja.
10. Sesuaikan pembuka: jika user menyapa, kamu balas sapaan + "Ada yang bisa Minci bantu, kak?". Jika tidak menyapa, kamu JANGAN menyapa duluan.
11. JANGAN membuat format markdown seperti ## dan lain-lain. PAKAI bullet point biasa (misal "-") untuk daftar. JANGAN pakai nomor urut 1, 2, 3, dan seterusnya.
12. Jika ditanya daftar jurusan/prodi, kamu sebutkan LANGSUNG namanya (misal: Informatika, Teknik Industri). Jangan jelaskan prospek/detail kecuali ditanya spesifik.
13. Jika ditanya tentang sesuatu konteks yang mempunyai list/daftar bullet/angka, kamu tulis dalam bentuk daftar (bullet points) biar mudah dibaca. Jangan tulis panjang lebar dalam paragraf. TULISKAN SEMUA item yang ada di konteks dengan LENGKAP, jangan pilih-pilih.
14. Jika ditanya tentang beasiswa, JANGAN sarankan memilih beasiswa tertentu, karena itu bukan keputusan pribadi.
15. Jika ditanya tentang syarat KRS/Perwalian, kamu TULISKAN SEMUA syarat yang ada di konteks DENGAN JELAS DAN LENGKAP. Jangan pilih-pilih.
16. Jika ditanya tentang biaya, TULISKAN SEMUA biaya yang ada di konteks DENGAN JELAS DAN LENGKAP dimana berisi biaya pendaftaran, biaya awal, semua list biaya per semester, biaya wisuda dan biaya KP.
19. Jika diberi salam "Selamat pagi/siang/sore/malam" kamu balas "Selamat pagi/siang/sore/malam, kak. Ada yang bisa Minci bantu?" sesuai salamnya.

Jawab singkat, jelas, ceria, tidak bertele-tele dan tidak ambigu."""


def build_user_message(question: str, context: str) -> str:
    return f"""Konteks:
{context}

Pertanyaan:
{question}"""

def clean_markdown(text: str) -> str:
    """
    Post-processing terakhir sebelum jawaban dikirim ke user: buang simbol markdown
    yang kadang masih kebablasan ditulis model walau sudah dilarang di guardrail
    (lihat aturan #11 di build_system_prompt). Ini jaring pengaman KODE, bukan
    cuma andalkan model "nurut" instruksi -- lebih pasti hasilnya.
    """
    # Bold markdown "**teks**" -> "teks" (WhatsApp/Telegram tidak render ** jadi tebal,
    # yang muncul ke user malah tanda bintang mentah yang aneh)
    text = text.replace("**", "")
 
    # Buang heading markdown "## " di awal baris (jadi teks biasa)
    text = text.replace("## ", "").replace("### ", "").replace("# ", "")
 
    return text.strip()


def ask_minci(question: str) -> str:
    """
    Fungsi utama: retrieval + generation. Dipanggil dari webhook WhatsApp / bot Telegram.
    Setiap pertanyaan diproses berdiri sendiri (TIDAK ada riwayat/memori percakapan
    -- fitur ini sudah dilepas sesuai permintaan).
    """
    context = retrieve_context(question)

    if not context.strip():
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"Pertanyaan: {question}\n\nInfo: Tidak ada konteks relevan."}
        ]
    else:
        hint = find_highlighted_lines(question, context)
        full_context = hint + "\n\n" + context if hint else context
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_message(question, full_context)}
        ]

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
    )
    
    answer = response["message"]["content"]
    answer = clean_markdown(answer)

    return answer


if __name__ == "__main__":
    print("💬 Mode test RAG Minci (ketik 'exit' untuk keluar)\n")
    while True:
        q = input("Kamu: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        jawaban = ask_minci(q)
        print(f"\nMinci: {jawaban}\n")