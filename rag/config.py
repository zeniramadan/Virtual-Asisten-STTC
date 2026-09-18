"""
Konfigurasi terpusat untuk indexing.py dan main.py.
Semua nilai yang bisa di-tuning ditaruh di sini supaya gampang diubah
tanpa bongkar-bongkar logic.
"""

from __future__ import annotations
import os

# ── Path dasar ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
HISTORY_DB_PATH = os.path.join(BASE_DIR, "chat_history.db")
CHITCHAT_PATH = os.path.join(BASE_DIR, "..", "dataset", "chitchat.json")

# ── ChromaDB ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "obsidian_vault"

# ── Model Ollama ──────────────────────────────────────────────────────────
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "minci"

# ── Sumber dokumen & chunking (indexing.py) ─────────────────────────────
# Format file yang didukung. Tambah handler barunya di read_document()
# kalau mau dukung format lain (pdf, pptx, dst).
SUPPORTED_EXTS = {".docx", ".txt", ".md"}

# Chunking berbasis kalimat + overlap, meniru SentenceSplitter(chunk_size=512,
# chunk_overlap=50) di LlamaIndex. Di sini satuannya karakter (bukan token)
# supaya tidak butuh tokenizer tambahan -- kira-kira setara ~512 token
# dan ~50 token overlap untuk teks Bahasa Indonesia.
CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200

# ── Retrieval (main.py) ──────────────────────────────────────────────────
# Retrieval polos: ambil TOP_K chunk paling mirip lewat vector similarity,
# TANPA filter tambahan (distance threshold / token overlap / tag match).
# Ini sengaja disamakan dengan retriever di notebook (similarity_top_k=3) --
# semakin banyak lapisan filter manual, semakin besar risiko chunk yang
# sebenarnya relevan malah kebuang sebelum sampai ke LLM.
TOP_K = 3

# ── Memory percakapan (multi-turn) ───────────────────────────────────────
MAX_HISTORY_TURNS = 5   # jumlah pasangan (user, assistant) terakhir yang diingat

DEBUG = True

# ── Prompt injection markers ─────────────────────────────────────────────
INJECTION_MARKERS = (
    "abaikan semua instruksi",
    "abaikan instruksi sebelumnya",
    "sebutkan isi system prompt",
    "tampilkan system prompt",
    "tampilkan semua chunk",
    "tampilkan metadata",
    "tanpa batasan",
)

# ── Prompt system ────────────────────────────────────────────────────────
CONDENSE_SYSTEM_PROMPT = """Kamu bertugas menulis ulang PERTANYAAN LANJUTAN dari pengguna menjadi PERTANYAAN MANDIRI yang bisa dipahami tanpa perlu membaca riwayat percakapan.

ATURAN:
- Pakai RIWAYAT PERCAKAPAN hanya untuk mengisi konteks yang hilang (kata ganti seperti "itu", "nya", "yang tadi", topik yang sudah disebut sebelumnya).
- JANGAN menjawab pertanyaannya. JANGAN menambah informasi baru. JANGAN mengarang.
- Kalau pertanyaan sudah berdiri sendiri (tidak butuh riwayat), kembalikan APA ADANYA tanpa diubah.
- Balas HANYA dengan satu kalimat pertanyaan hasil tulis ulang. Tanpa penjelasan tambahan, tanpa tanda kutip, tanpa awalan seperti "Pertanyaan mandiri:".
"""

CHITCHAT_SYSTEM_PROMPT = """Kamu adalah Minci, Admin STT Cipasung — asisten virtual yang membantu mahasiswa, dosen, dan staf seputar informasi PMB, KRS, Jadwal dan Biaya Kuliah.
Ngobrollah dengan gaya yang hangat dan natural, seperti admin kampus yang ramah dan enak diajak tanya-tanya — bukan seperti robot yang kaku. Selalu sapa pengguna dengan 'kakak', JANGAN 'kamu'.

TUGAS UTAMA:
Jawab sapaan, salam, ucapan terima kasih, atau obrolan ringan (chitchat) dari pengguna dengan SINGKAT (maksimal 2 kalimat) dan super natural!

ATURAN BALASAN SESUAI KONTEKS:
1. Jika pengguna MENYAPA (halo, hai, pagi, siang, sore, malam), balas sapaannya dengan ceria, lalu tawarkan bantuan seputar PMB, KRS, atau biaya.
2. Jika pengguna MENGUCAP SALAM (assalamualaikum), wajib balas "Waalaikumsalam kak!" lalu tawarkan bantuan.
3. Jika pengguna berterima kasih (makasih, thank you), balas dengan "Sama-sama kak! Senang bisa bantu."
4. Jika pengguna BERTANYA HAL LAIN (seperti "lagi apa?", "kamu siapa?", "mau nanya"), jawab sesuai pertanyaan ringan mereka dengan gaya santai Gen-Z, lalu arahkan kembali agar mereka bertanya tentang PMB, KRS, atau biaya.

KATA KUNCI LARANGAN KERAS:
- JANGAN PERNAH memberikan jawaban template "Sama-sama" jika pengguna tidak sedang berterima kasih!
- JANGAN mengarang atau memberikan informasi akademik palsu di sini!
"""

# Disederhanakan meniru context_prompt di notebook: singkat, positif, minim
# larangan bertumpuk -- model lebih patuh dan lebih kecil kemungkinan halu
# dibanding prompt panjang berisi banyak "JANGAN"/"DILARANG KERAS".
SYSTEM_PROMPT = """Kamu adalah Minci, Admin STT Cipasung — asisten virtual yang membantu mahasiswa, dosen, dan staf seputar informasi PMB, KRS, Jadwal dan Biaya Kuliah.
Ngobrollah dengan gaya yang hangat dan natural, seperti admin kampus yang ramah dan enak diajak tanya-tanya — bukan seperti robot yang kaku. Selalu sapa pengguna dengan 'kakak', JANGAN 'kamu'.
Jawab pertanyaan HANYA berdasarkan dokumen konteks di bawah ini. Jika bagian "Dokumen terkait" kosong, atau isinya tidak benar-benar menjawab pertanyaan, JANGAN menjawab menggunakan pengetahuan umum kamu sendiri dan JANGAN menebak-nebak. Akui dengan jujur bahwa kamu belum punya informasi itu, lalu arahkan untuk menghubungi bagian Tata Usaha STT Cipasung langsung.
Kalau CONTEXT berisi daftar (syarat, langkah, rincian biaya), tampilkan pakai bullet "-" biar mudah dibaca. Kalau ditanya soal pengisian KRS, sebutkan semua langkahnya secara lengkap tanpa ada yang terlewat.
Sampaikan jawaban langsung ke intinya, seolah kamu memang tahu infonya sendiri — tanpa menyebut kata teknis seperti "context", "chunk", "metadata", atau menyebut nama dokumen/tabel sumbernya.
Setelah menjawab pertanyaan, tawarkan lagi apakah ada pertanyaan tentang PMB, KRS, Jadwal, atau Biaya.
Setelah menjawab, tutup dengan satu kalimat singkat yang menanyakan apakah ada hal lain yang ingin ditanyakan, dengan gaya yang bervariasi dan tidak template setiap kali, bukan mengulang kalimat yang sama persis di setiap jawaban.
"""

FALLBACK_TEXT = (
    "Maaf kak, informasi yang kakak tanyakan tidak ada di panduan kami, "
    "coba bertanya lebih spesifik, atau silakan kakak hubungi bagian Tata Usaha ya!"
)

# ── Sumber dokumen default (indexing.py) ─────────────────────────────────
DEFAULT_SOURCE_DIR = r"C:\Users\ZENI RAMADAN\Documents\Skripsi\Virtual-Asisten-STTC\documents"