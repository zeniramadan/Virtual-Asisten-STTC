# 🤖 Minci — Virtual Asisten Akademik STT Cipasung

Minci adalah virtual assistant akademik untuk membantu mahasiswa dan calon mahasiswa STT Cipasung mendapatkan informasi seputar:

- 🎓 Penerimaan Mahasiswa Baru (PMB)
- 📅 Kalender akademik dan jadwal kegiatan
- 📝 Kartu Rencana Studi (KRS) dan perwalian
- 💰 Biaya kuliah dan beasiswa
- 🏫 Profil, program studi, UKM, serta kontak kampus

Minci menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**. Informasi faktual diambil dari dokumen resmi kampus melalui ChromaDB, kemudian dirangkum oleh model bahasa lokal melalui Ollama.

> **Status proyek:** prototipe akademik lokal dengan integrasi WhatsApp Cloud API.

---

## ✨ Fitur Utama

| Fitur                     | Keterangan                                                        |
| ------------------------- | ----------------------------------------------------------------- |
| 🔎 Semantic retrieval     | Mencari potongan dokumen relevan menggunakan embedding `bge-m3`.  |
| 🗂️ Document-aware routing | Mengarahkan pertanyaan ke dokumen PMB, KRS, kalender, atau biaya. |
| 🧠 Local LLM              | Membuat jawaban menggunakan model Ollama secara lokal.            |
| 💬 Chitchat               | Menangani sapaan, salam, dan percakapan ringan.                   |
| 📱 WhatsApp webhook       | Menerima dan membalas pesan melalui Meta WhatsApp Cloud API.      |
| 🛡️ Duplicate protection   | Mencegah balasan ganda ketika Meta mengirim ulang webhook.        |
| 🧪 Debug-friendly         | Menampilkan route, alasan routing, dan context retrieval.         |

---

## 🧩 Arsitektur

```text
Pengguna WhatsApp
        │
        ▼
Meta WhatsApp Cloud API
        │ HTTPS webhook
        ▼
Public tunnel
        │
        ▼
FastAPI: webhook/app.py
        │
        ▼
rag/query.py
   ┌────┴─────┐
   ▼          ▼
ChromaDB   Ollama
   │          │
   ▼          ▼
Dokumen    Embedding + LLM
kampus
```

### Alur Pertanyaan

1. Pesan diterima dari terminal atau WhatsApp.
2. Query dinormalisasi, termasuk singkatan PMB, KRS, dan PRODI.
3. Sistem menentukan sumber dokumen yang relevan.
4. Ollama membuat embedding query.
5. ChromaDB mengambil chunk dokumen.
6. Context dan pertanyaan dikirim ke model chat.
7. Jawaban dibersihkan sebelum ditampilkan atau dikirim ke WhatsApp.

---

## 🛠️ Teknologi dan Library

### Backend dan API

- `Python 3.10+`
- `FastAPI` — server webhook HTTP.
- `Uvicorn` — ASGI server.
- `Requests` — client WhatsApp Cloud API.
- `python-dotenv` — konfigurasi environment.

### RAG dan Dokumen

- `ChromaDB` — vector database lokal.
- `Ollama` — embedding dan model bahasa lokal.
- `bge-m3` — model embedding multilingual.
- `python-docx` — membaca paragraf dan tabel Word.

### Integrasi

- Meta WhatsApp Cloud API.
- Cloudflare Tunnel atau tunnel HTTPS lain untuk development.

Versi dependency tersedia di [webhook/requirements.txt](webhook/requirements.txt).

---

## 📁 Struktur Repository

```text
Virtual-Asisten-STTC/
├── dataset/
│   ├── chitchat.json              # Frasa percakapan ringan
│   ├── dataset.json               # Dataset contoh jawaban
│   ├── dataset_minci.json         # Dataset tambahan Minci
│   └── dataset_training.json      # Dataset training jika tersedia
├── documents/
│   ├── BIAYA.docx                 # Data biaya kuliah
│   ├── KALENDER.docx              # Jadwal akademik dan PMB
│   ├── KRS.docx                   # Panduan KRS dan perwalian
│   └── PMB.docx                   # Syarat dan informasi PMB
├── llm/
│   ├── Modelfile                  # Konfigurasi model Ollama
│   └── llama-3.2-3b-instruct.Q4_K_M.gguf
├── rag/
│   ├── ingest.py                  # Membaca DOCX dan membangun indeks
│   ├── query.py                   # Routing, retrieval, dan jawaban
│   ├── test.py                    # Sandbox pengujian RAG
│   └── database/
│       └── chroma_db/             # Database vector lokal
├── training/
│   └── model_training.ipynb       # Notebook training/fine-tuning
├── webhook/
│   ├── app.py                     # FastAPI WhatsApp webhook
│   ├── env.example                # Template environment variable
│   └── requirements.txt            # Dependency Python
├── README.md
└── SETUP_GUIDE.md
```

> Folder `rag/database/chroma_db/` merupakan hasil generate. Jangan mengedit database secara manual.

---

## ✅ Prasyarat

Pastikan sudah tersedia:

- Python 3.10 atau lebih baru.
- Ollama.
- Model embedding `bge-m3`.
- Model chat yang sesuai dengan `CHAT_MODEL` di `rag/query.py`.
- Dokumen `.docx` di folder `documents/`.
- Akun Meta Developer jika ingin memakai WhatsApp.

Instal Ollama dari [ollama.com/download](https://ollama.com/download).

---

## 🚀 Instalasi Lokal

### 1. Clone repository

```powershell
git clone <URL-REPOSITORY>
cd Virtual-Asisten-STTC
```

### 2. Buat virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell memblokir aktivasi script:

```powershell
.\.venv\Scripts\python.exe -m pip install -r webhook\requirements.txt
```

### 3. Install dependency

```powershell
python -m pip install --upgrade pip
pip install -r webhook\requirements.txt
```

### 4. Siapkan Ollama

```powershell
ollama pull bge-m3
ollama list
```

Konfigurasi model berada di `rag/query.py`:

```python
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "bge-m3"
```

Jika menggunakan GGUF lokal dari folder `llm/`:

```powershell
cd llm
ollama create minci -f Modelfile
ollama run minci
cd ..
```

Jika model dibuat dengan nama `minci`, ubah `CHAT_MODEL` di `rag/query.py` menjadi `minci`.

---

## 🗃️ Menyiapkan dan Mengindeks Dokumen

Letakkan dokumen resmi `.docx` di folder `documents/`:

- `PMB.docx`
- `KRS.docx`
- `KALENDER.docx`
- `BIAYA.docx`

Jalankan ingestion dari folder `rag`:

```powershell
cd rag
python ingest.py
```

Script akan:

1. Membaca paragraf dan tabel Word.
2. Mengenali heading dan konteks section.
3. Menormalkan angka Romawi seperti `Gelombang II`.
4. Memecah dokumen menjadi chunk.
5. Membuat embedding dengan `bge-m3`.
6. Menyimpan hasil ke `rag/database/chroma_db/`.

Jalankan ulang ingestion setiap kali dokumen berubah, kemudian mulai proses Python baru sebelum menguji query.

---

## 🧪 Menjalankan dan Menguji RAG

### Mode interaktif

```powershell
cd rag
python query.py
```

Contoh pertanyaan:

```text
Kapan pendaftaran PMB gelombang 1 dibuka?
Syarat daftar PMB apa saja?
Cara mengisi KRS online bagaimana?
UKT per semester berapa?
Jadwal KTMB 2026 kapan?
```

Ketik `exit` untuk keluar.

### Pengujian satu pertanyaan

```powershell
cd rag
python -c "import query; print(query.ask_minci('Syarat daftar PMB apa saja?'))"
```

`rag/test.py` dapat digunakan sebagai sandbox untuk menguji perubahan routing dan retrieval secara terpisah dari pipeline produksi.

### Debug retrieval

Aktifkan:

```python
DEBUG = True
```

Log akan menampilkan query yang dinormalisasi, route, alasan routing, dan context yang dikirim ke model.

---

## 📱 Integrasi WhatsApp

### 1. Buat aplikasi Meta

1. Buka [Meta for Developers](https://developers.facebook.com/).
2. Buat aplikasi bertipe Business.
3. Tambahkan produk WhatsApp.
4. Catat access token, phone number ID, dan nomor testing.

### 2. Buat file environment

```powershell
cd webhook
Copy-Item env.example .env
```

Isi `.env`:

```env
WA_VERIFY_TOKEN=ganti-dengan-token-verifikasi
WA_ACCESS_TOKEN=token-meta-whatsapp
WA_PHONE_NUMBER_ID=phone-number-id
WA_API_VERSION=v26.0
```

Jangan commit `.env` karena berisi credential.

### 3. Jalankan FastAPI

```powershell
cd webhook
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```text
http://localhost:8000/
```

Respons yang diharapkan:

```json
{ "status": "Minci webhook aktif ✅" }
```

### 4. Expose endpoint dengan HTTPS

Untuk development, gunakan Cloudflare Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Gunakan URL publik yang dihasilkan sebagai callback URL Meta:

```text
https://<URL-TUNNEL>/webhook
```

Verification token di Meta harus sama dengan `WA_VERIFY_TOKEN` pada `.env`. Setelah tersimpan, subscribe ke field `messages`.

> Quick tunnel cocok untuk testing. URL dapat berubah ketika tunnel dimulai ulang.

---

## 🔐 Catatan Keamanan

- Jangan commit `.env`, access token, atau credential WhatsApp.
- Gunakan permanent access token untuk production.
- Batasi logging payload WhatsApp jika berisi data pribadi.
- Tambahkan rate limiting sebelum deployment publik.
- Gunakan HTTPS stabil untuk production.

---

## 🧯 Troubleshooting

### `ModuleNotFoundError`

Pastikan environment aktif dan dependency terpasang:

```powershell
pip install -r webhook\requirements.txt
```

### Chroma menampilkan `0 chunks`

Jalankan ingestion dari folder `rag`:

```powershell
cd rag
python ingest.py
```

Pastikan output ingestion menunjukkan chunk tersimpan di `rag/database/chroma_db/`. Mulai proses Python baru setelah ingestion selesai.

### Jawaban fallback padahal data ada

Aktifkan `DEBUG = True`, lalu periksa:

1. Apakah route menuju dokumen yang benar?
2. Apakah chunk relevan masuk ke context?
3. Apakah distance melewati threshold?
4. Apakah model chat yang dipanggil sesuai dengan model Ollama?

### WhatsApp tidak membalas

Periksa server Uvicorn, URL tunnel, verification token, access token, phone number ID, log webhook, Ollama, dan model chat.

### Pesan WhatsApp dibalas dua kali

`app.py` memiliki deduplikasi berdasarkan message ID. Pastikan webhook mengembalikan HTTP `200 OK` dengan cepat dan proses RAG berjalan di background task.

---

## 📚 Dokumentasi Internal

- [Setup guide](SETUP_GUIDE.md)
- [RAG ingestion](rag/ingest.py)
- [RAG query pipeline](rag/query.py)
- [WhatsApp webhook](webhook/app.py)
- [Ollama Modelfile](llm/Modelfile)
- [Dataset chitchat](dataset/chitchat.json)

---

## 🛣️ Pengembangan Berikutnya

- Menambahkan evaluasi otomatis untuk seluruh pertanyaan uji RAG.
- Menyatukan logic eksperimen di `rag/test.py` dengan pipeline produksi secara terkontrol.
- Menambahkan reranker untuk retrieval yang lebih presisi.
- Menambahkan test untuk routing, normalisasi query, dan webhook verification.
- Menggunakan named tunnel dan permanent token untuk deployment production.

---

## 📄 Lisensi

Belum ditentukan. Tambahkan file lisensi sebelum proyek didistribusikan secara publik.
