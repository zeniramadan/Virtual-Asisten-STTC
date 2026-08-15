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
from conversation import get_history, add_message, reset_history

# ====== KONFIGURASI ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Tambahkan "database" di tengah path-nya
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"        # HARUS SAMA PERSIS dengan yang dipakai rag_ingest.py
EMBED_QUERY_PREFIX = ""       # bge-m3 tidak butuh prefix (beda dengan nomic-embed-text dulu)
CHAT_MODEL = "minci"          # nama model hasil `ollama create minci -f Modelfile`
TOP_K = 8                     # jumlah chunk paling relevan yang diambil (dinaikkan dari 4 -> 8
                               # karena skor similarity di dokumen ini rentangnya sempit/mirip-mirip,
                               # jadi butuh lebih banyak kandidat biar chunk yang benar ikut kebawa)
DEBUG = False                  # set True biar konteks yang diambil ditampilkan di terminal

# Riwayat percakapan (memori chat per user) ditangani di modul terpisah
# conversation_store.py -- disimpan permanen di SQLite dengan auto-expire 1 jam.
# Lihat get_history() / add_message() / reset_history() yang diimport di atas.
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

    if DEBUG:
        print("\n" + "=" * 60)
        print(f"🔍 [DEBUG] Hasil retrieval untuk: {question!r}")
        print("=" * 60)
        if not documents:
            print("(KOSONG — tidak ada chunk yang ditemukan sama sekali!)")
        for doc, meta, dist in zip(documents, metadatas, distances):
            print(f"(distance={dist:.4f}, sumber={meta.get('source')})")
            print(doc[:300], "..." if len(doc) > 300 else "")
            print()
        print("=" * 60 + "\n")

    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "dokumen")
        context_blocks.append(f"[Sumber: {source}]\n{doc}")

    return "\n\n---\n\n".join(context_blocks)


def build_prompt(question: str, context: str) -> str:
    return f"""Peran: Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung. Kamu membantu mahasiswa dan calon mahasiswa baru terkait informasi PMB (Penerimaan Mahasiswa Baru) dan KRS (Kartu Rencana Studi). Gaya bicaramu santai, ramah, ceria, dan luwes ala anak muda (gen-z), tapi tetap sopan dan tidak berlebihan.

Konteks Dokumen (Retrieved Context):
{context}

Pertanyaan Pengguna:
{question}

Aturan Mutlak (Guardrails):
1. Kamu HANYA DIPERBOLEHKAN menjawab berdasarkan "Konteks Dokumen" di atas. DILARANG KERAS mengarang jawaban, menambahkan tanggal/informasi yang tidak tertulis eksplisit di konteks, atau memakai pengetahuan di luar "Konteks Dokumen".
2. "Konteks Dokumen" di atas bisa berisi CAMPURAN beberapa gelombang/semester/topik sekaligus dalam satu daftar. Periksa SETIAP baris satu per satu untuk menemukan bagian yang cocok dengan yang ditanyakan (misalnya "gelombang 1" sama artinya dengan "Gelombang I") SEBELUM menyimpulkan informasinya tidak ada. Jangan langsung menyerah hanya karena ada baris LAIN di konteks yang membahas gelombang/semester berbeda.
3. Jika setelah diperiksa dengan teliti informasi yang ditanyakan TIDAK ADA / TIDAK RELEVAN di "Konteks Dokumen", kamu WAJIB menjawab persis dengan kalimat: "Maaf kak, informasi tersebut tidak ada di panduan. Silakan hubungi bagian Tata Usaha."
4. SANGAT PENTING -- JANGAN PERNAH mencampur data antar gelombang/semester yang berbeda. Nomor gelombang/semester yang kamu SEBUTKAN di jawaban HARUS SAMA PERSIS dengan nomor gelombang/semester yang tertulis di baris konteks tempat kamu mengambil data itu -- JANGAN labeli data Gelombang II sebagai Gelombang I hanya karena itu yang ditanya user.
5. Di "Konteks Dokumen", penomoran gelombang/semester kadang ditulis 3 bentuk sekaligus dipisah garis miring (misal "Gelombang I / 1 / satu") -- ini HANYA format teknis internal untuk membantu pencarian, BUKAN untuk ditiru mentah-mentah ke jawaban. Saat menjawab, PILIH SATU bentuk saja yang paling natural (contoh: cukup "Gelombang 1" atau "Gelombang I").
6. Jika pertanyaan soal jadwal/tanggal PMB TIDAK spesifik menyebut gelombang berapa (atau pertanyaan sejenis), jawab dengan menyebutkan SEMUA gelombang yang ada di konteks, jangan pilih salah satu.
7. Sesuaikan KATA PEMBUKA dengan jenis pertanyaan: JANGAN pakai awalan "Bisa dong"/"Bisa banget" kecuali user memang bertanya "apakah bisa/boleh". Kalau user menyapa (misal "halo kak"), balas dengan sapaan natural ("Halo kak!", "Siap kak!") lalu tambahkan kalimat "Ada yang bisa Minci bantu, kak?" di akhir. Kalau user tidak menyapa, langsung jawab intinya.
8. Gunakan bahasa Indonesia santai, ceria, dan ramah ala Gen-Z sesuai Peran di atas -- tapi tetap singkat, jelas, rapi, dan tidak bertele-tele.

Berdasarkan Peran dan Aturan Mutlak di atas, berikan jawabanmu:"""


def ask_minci(question: str, user_id: str = "default") -> str:
    """
    Fungsi utama: retrieval + generation, SEKARANG dengan memori percakapan.
    Dipanggil dari webhook WhatsApp / bot Telegram -- WAJIB kasih user_id yang unik
    per pengguna (nomor WA / chat_id Telegram), supaya riwayat obrolan tiap orang
    tidak tercampur satu sama lain.
    """
    context = retrieve_context(question)

    if not context.strip():
        prompt = (
            f"Peran: Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung.\n\n"
            f"Pertanyaan Pengguna:\n{question}\n\n"
            "Kamu tidak menemukan satupun dokumen relevan untuk pertanyaan ini. "
            "Jawab persis dengan kalimat: \"Maaf kak, informasi tersebut tidak ada di panduan. "
            "Silakan hubungi bagian Tata Usaha.\""
        )
    else:
        prompt = build_prompt(question, context)

    # Riwayat sebelumnya (pertanyaan & jawaban ASLI, bukan versi lengkap dengan instruksi+konteks)
    # + pesan giliran ini (yang lengkap dengan instruksi & konteks RAG)
    history = get_history(user_id)
    messages = history + [{"role": "user", "content": prompt}]

    if DEBUG and history:
        print(f"[DEBUG] Memakai {len(history)} pesan riwayat untuk user_id={user_id!r}")

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
    )
    answer = response["message"]["content"]

    # Simpan versi RINGKAS (pertanyaan asli + jawaban), BUKAN prompt lengkap yang penuh
    # instruksi & konteks dokumen -- supaya riwayat tetap ringkas dan tidak boros token
    # di giliran-giliran chat berikutnya.
    add_message(user_id, "user", question)
    add_message(user_id, "assistant", answer)

    return answer


if __name__ == "__main__":
    print("💬 Mode test RAG Minci (ketik 'exit' untuk keluar, 'reset' untuk hapus riwayat)\n")
    test_user_id = "terminal-test"
    while True:
        q = input("Kamu: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        if q.strip().lower() == "reset":
            reset_history(test_user_id)
            print("(riwayat percakapan dihapus)\n")
            continue
        jawaban = ask_minci(q, user_id=test_user_id)
        print(f"\nMinci: {jawaban}\n")