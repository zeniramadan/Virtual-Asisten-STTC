"""
rag_query.py
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
# Path absolut berdasarkan lokasi file ini, SAMA PERSIS dengan rag_ingest.py,
# supaya keduanya selalu baca/tulis folder chroma_db yang sama.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"        # HARUS SAMA PERSIS dengan yang dipakai rag_ingest.py
EMBED_QUERY_PREFIX = ""       # bge-m3 tidak butuh prefix (beda dengan nomic-embed-text dulu)
CHAT_MODEL = "minci"          # nama model hasil `ollama create minci -f Modelfile`
TOP_K = 4                     # jumlah chunk paling relevan yang diambil (dinaikkan dari 4 -> 8
                               # karena skor similarity di dokumen ini rentangnya sempit/mirip-mirip,
                               # jadi butuh lebih banyak kandidat biar chunk yang benar ikut kebawa)
DEBUG = True                  # set True biar konteks yang diambil ditampilkan di terminal
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
    return f"""Konteks dokumen resmi STT Cipasung:
{context}

Pertanyaan: {question}

INSTRUKSI WAJIB:
1. Jawab HANYA menggunakan informasi dari konteks di atas.
2. JANGAN menambahkan informasi, tanggal, atau jadwal palsu yang tidak ada di konteks.
3. Jika konteks tidak memiliki jawabannya, katakan: "Maaf kak, informasi tersebut tidak ada di panduan. Silakan hubungi bagian Tata Usaha."
4.Sesuaikan KATA PEMBUKA dengan jenis pertanyaan. 
   - Jika user TIDAK bertanya "apakah bisa/boleh", JANGAN gunakan awalan "Bisa dong", "Bisa banget", atau sejenisnya. 
   - Gunakan sapaan natural seperti "Halo kak!", "Siap kak!" JIKA user menyapa, JIKA TIDAK langsung jawab intinya.
   - Jika user menyapa seperti "halo kak", dan lain-lain, maka setelah balasan, tambahkan kalimat "Ada yang bisa Minci bantu, kak?".
5. Gunakan gaya bahasa santai, ceria dan ramah ala Gen-Z.
6. Jika user bertanya tentang jadwal, tanggal PMB secara tidak spesifik gelombang berapa, jawab dengan menyebutkan semua gelombang yang ada di dokumen, jangan pilih salah satu. ini juga berlaku pada pertanyaan yang sejenis.
7. Konteks di atas bisa berisi CAMPURAN beberapa gelombang/semester/topik sekaligus dalam satu daftar. Periksa SETIAP baris satu per satu untuk menemukan bagian yang cocok dengan gelombang/semester spesifik yang ditanyakan (misalnya "gelombang 1" sama artinya dengan "Gelombang I"), SEBELUM menyimpulkan informasinya tidak ada. Jangan langsung bilang "tidak ada di panduan" hanya karena ada baris LAIN di konteks yang membahas gelombang/semester berbeda dari yang ditanyakan.
8. SANGAT PENTING -- JANGAN PERNAH mencampur data antar gelombang/semester yang berbeda. Sebelum menjawab, cek ulang: nomor gelombang/semester yang kamu SEBUTKAN di jawaban HARUS SAMA PERSIS dengan nomor gelombang/semester yang tertulis di baris konteks tempat kamu mengambil tanggal/angka itu. Kalau baris konteks yang kamu temukan itu tertulis "Gelombang II", maka jawabanmu juga harus bilang "Gelombang II" -- JANGAN labeli sebagai "Gelombang I" hanya karena itu urutan pertama di konteks atau karena itu yang ditanya user. Kalau kamu tidak menemukan baris yang nomor gelombangnya PERSIS sama dengan yang ditanya user, itu artinya informasinya TIDAK ADA -- ikuti instruksi nomor 3, JANGAN mengarang atau menukar dengan gelombang lain.
"""


def ask_minci(question: str) -> str:
    """Fungsi utama: retrieval + generation. Ini yang dipanggil dari webhook WhatsApp / bot Telegram."""
    context = retrieve_context(question)

    if not context.strip():
        prompt = (
            f"Pertanyaan: {question}\n\n"
            "Kamu tidak menemukan dokumen relevan untuk pertanyaan ini. "
            "Jawab dengan jujur bahwa kamu belum punya info itu dan sarankan "
            "hubungi bagian akademik STT Cipasung langsung."
        )
    else:
        prompt = build_prompt(question, context)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    print("💬 Mode test RAG Minci (ketik 'exit' untuk keluar)\n")
    while True:
        q = input("Kamu: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        jawaban = ask_minci(q)
        print(f"\nMinci: {jawaban}\n")