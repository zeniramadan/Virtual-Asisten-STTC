# 📘 Panduan Lengkap: Minci — Virtual Asisten Akademik STT Cipasung

Panduan ini mengasumsikan kamu mulai dari nol. Ikuti urut dari atas ke bawah.

**Arsitektur singkat:**

```
WhatsApp User
     │
     ▼
Meta WhatsApp Cloud API
     │  (webhook HTTPS)
     ▼
Cloudflare Tunnel  ──►  Laptop kamu (localhost:8000)
                             │
                             ▼
                     FastAPI webhook (app.py)
                             │
                             ▼
                     query.py (RAG)
                       │            │
                       ▼            ▼
                  ChromaDB      Ollama (model "minci"
                (dokumen PMB/    hasil fine-tuning LoRA)
                    KRS)
```

- **Fakta** (jadwal, syarat, biaya, dsb) datang dari dokumen Word lewat RAG.
- **Gaya bahasa** (santai-tapi-sopan ala gen-z) datang dari model yang sudah di-fine-tune LoRA.
- Keduanya digabung di `query.py`: konteks dari dokumen disuntikkan ke prompt, lalu dijawab pakai model bergaya Minci.

---

## Bagian 1 — Fine-tuning LoRA di Google Colab

### 1.1 Siapkan/perluas dataset

File `dataset/dataset_training.json` sudah berisi 20 contoh gaya bahasa Minci. Ini **cukup untuk mulai**, tapi makin banyak contoh (idealnya 50–150+) makin konsisten gayanya. Kamu tinggal tambah entri baru dengan format yang sama:

```json
{
  "instruction": "pertanyaan atau statement dari user",
  "input": "",
  "output": "jawaban Minci dengan gaya santai-sopan"
}
```

Tidak perlu isi fakta PMB/KRS yang detail di sini — itu tugasnya RAG. Dataset ini cukup fokus ke **gaya bicara**: sapaan, cara merespons keluhan, cara minta klarifikasi, cara menutup obrolan, dll.

### 1.2 Jalankan training di Colab

1. Buka [Google Colab](https://colab.research.google.com/), lalu upload file `colab/train_lora_colab.ipynb`.
2. Runtime → Change runtime type → pilih **T4 GPU** → Save.
3. Runtime → Run all.
4. Saat diminta upload dataset, upload `dataset/dataset_training.json` (atau versi kamu yang sudah diperluas).
5. Tunggu proses training selesai (biasanya beberapa menit untuk dataset kecil).
6. Di cell terakhir, file `.gguf` akan otomatis ter-download ke laptop kamu (cek folder Downloads).

---

## Bagian 2 — Install & setup Ollama di laptop

### 2.1 Install Ollama

Download dan install dari https://ollama.com/download (tersedia untuk Windows/Linux/Mac).

Verifikasi instalasi:

```bash
ollama --version
```

### 2.2 Pull model embedding untuk RAG

```bash
ollama pull bge-m3
```

Model ini ringan (~1.2 GB), dipakai untuk mengubah teks jadi vektor saat pencarian dokumen.

### 2.3 Buat model "minci" dari hasil fine-tuning

1. Pindahkan file `.gguf` hasil download dari Colab ke folder `ollama/` di project ini.
2. Rename file tersebut jadi `llama-3.2-3b-instruct.Q4_K_M.gguf` (atau edit baris `FROM` di `ollama/Modelfile` supaya sesuai nama file kamu).
3. Masuk ke folder `ollama/` lalu jalankan:

```bash
cd minci-project/ollama
ollama create minci -f Modelfile
```

4. Test modelnya langsung di terminal:

```bash
ollama run minci
>>> Min, gimana cara daftar PMB?
```

Kalau jawabannya sudah kerasa santai-tapi-sopan, fine-tuning berhasil ✅

---

## Bagian 3 — Setup RAG (dokumen PMB & KRS)

### 3.1 Siapkan Python environment

```bash
cd minci-project/webhook
python -m venv venv
.\venv\Scripts\activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Taruh dokumen Word

Masukkan semua file `.docx` PMB dan KRS ke folder `minci-project/documents/`.

> 💡 Tips: kalau dokumen kamu masih format PDF/gambar, convert dulu ke `.docx`, atau kalau isinya tabel-tabel kompleks, cek dulu apakah tabelnya terbaca rapi (script `ingest.py` sudah menghandle isi tabel, bukan cuma paragraf).

### 3.3 Jalankan ingestion

```bash
cd minci-project/rag
python ingest.py
```

Ini akan membaca semua `.docx`, memecahnya jadi potongan teks, dan menyimpannya sebagai vector database lokal di folder `rag/chroma_db/`.

**Jalankan ulang script ini setiap kali dokumen PMB/KRS berubah atau bertambah.**

### 3.4 Test RAG saja (tanpa WhatsApp dulu)

```bash
python query.py
```

Coba tanya-tanya lewat terminal untuk memastikan jawabannya akurat berdasarkan dokumen sebelum lanjut ke integrasi WhatsApp.

---

## Bagian 4 — Setup WhatsApp Business API (Meta)

### 4.1 Buat App di Meta for Developers

1. Buka https://developers.facebook.com/ → **My Apps** → **Create App**.
2. Pilih tipe **Business**.
3. Di dashboard App, tambahkan produk **WhatsApp**.

### 4.2 Ambil kredensial testing

Di menu **WhatsApp → API Setup**, kamu akan melihat:

- **Temporary access token** (berlaku 24 jam, cukup untuk testing awal)
- **Phone number ID**
- Nomor test WhatsApp yang disediakan Meta

Untuk produksi nanti (bukan cuma testing), kamu perlu bikin **System User** dengan **Permanent Token** di Business Settings — tapi untuk mulai, temporary token dulu tidak apa-apa.

### 4.3 Isi file `.env`

```bash
cd minci-project/webhook
copy .env.example .env
```

Edit `.env`, isi:

- `WA_VERIFY_TOKEN` → bebas kamu tentukan sendiri (contoh: `minci-verify-123`), nanti dipakai lagi di step 4.5
- `WA_ACCESS_TOKEN` → dari dashboard Meta
- `WA_PHONE_NUMBER_ID` → dari dashboard Meta

### 4.4 Jalankan webhook server

```bash
cd minci-project/webhook
uvicorn app:app --host 0.0.0.0 --port 8000
```

Cek di browser: `http://localhost:8000` harus muncul `{"status": "Minci webhook aktif ✅"}`.

---

## Bagian 5 — Expose webhook ke internet dengan Cloudflare Tunnel

Meta mewajibkan webhook URL berupa **HTTPS publik**, makanya kita pakai Cloudflare Tunnel supaya laptop lokal bisa diakses dari internet tanpa perlu domain/hosting.

### 5.1 Install cloudflared

- **Windows**: download installer dari https://github.com/cloudflare/cloudflared/releases
- **Mac**: `brew install cloudflare/cloudflare/cloudflared`
- **Linux (Debian/Ubuntu)**:

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

### 5.2 Jalankan tunnel cepat (quick tunnel, untuk testing)

Pastikan `app.py` (Bagian 4.4) sudah jalan di terminal terpisah, lalu di terminal baru:

```bash
cloudflared tunnel --url http://localhost:8000
```

Kamu akan dapat URL publik seperti:

```
https://random-words-abcd.trycloudflare.com
```

URL ini yang akan kamu pakai sebagai webhook URL. **Catatan:** quick tunnel ini URL-nya berubah tiap kali di-restart — cocok untuk testing, tapi untuk produksi lihat catatan di Bagian 6.

### 5.3 Daftarkan webhook URL ke Meta

1. Di dashboard Meta → **WhatsApp → Configuration**.
2. Klik **Edit** pada Webhook.
3. **Callback URL**: `https://random-words-abcd.trycloudflare.com/webhook`
4. **Verify Token**: isi sama persis dengan `WA_VERIFY_TOKEN` di file `.env` kamu.
5. Klik **Verify and Save** — kalau berhasil, artinya endpoint `GET /webhook` di `app.py` sudah benar merespons challenge dari Meta.
6. Subscribe ke field **messages** (centang webhook fields → messages).

---

## Bagian 6 — Testing end-to-end

1. Kirim pesan WhatsApp ke nomor test dari Meta (nomor ini ada di dashboard **API Setup**, kamu harus daftarkan nomor HP kamu dulu sebagai tester di **API Setup → To**).
2. Tulis pertanyaan seputar PMB/KRS, misalnya: *"Min, biaya pendaftaran PMB berapa ya?"*
3. Cek terminal `uvicorn` — harus muncul log pesan masuk.
4. Dalam beberapa detik, balasan dari Minci harus masuk ke WhatsApp kamu.

Kalau tidak ada balasan, cek urutan ini:

- [ ] `ollama serve` aktif (biasanya otomatis jalan setelah install)?
- [ ] `ollama list` menampilkan model `minci`?
- [ ] `uvicorn app:app` masih jalan tanpa error?
- [ ] `cloudflared` masih jalan dan URL belum berubah?
- [ ] Log di terminal `uvicorn` menunjukkan payload masuk dari Meta?
- [ ] `.env` sudah diisi dengan token & phone_number_id yang benar?

---

## Bagian 7 — Untuk produksi (opsional, kalau nanti mau lanjut lebih serius)

Beberapa hal yang worth dipikirkan setelah versi testing jalan lancar:

1. **Named Tunnel (bukan quick tunnel)** — supaya URL webhook permanen dan tidak berubah tiap restart. Butuh akun Cloudflare + domain (bisa domain gratis/murah), setup dengan `cloudflared tunnel create` dan `cloudflared tunnel route dns`.
2. **Permanent Access Token** — bikin System User di Meta Business Settings supaya token tidak expired tiap 24 jam.
3. **Auto-start service** — jadikan `ollama serve`, `uvicorn`, dan `cloudflared` berjalan sebagai service (systemd di Linux / Task Scheduler di Windows) supaya otomatis nyala kalau laptop restart.
4. **Rate limiting & logging** — supaya kalau ada spam pesan, laptop tidak kewalahan (mengingat resource terbatas).
5. **Evaluasi kualitas RAG** — coba beberapa pertanyaan edge-case (pertanyaan di luar dokumen, pertanyaan ambigu) untuk pastikan Minci tidak halusinasi.

---

## Ringkasan urutan menjalankan (setelah semua setup selesai)

Setiap mau menyalakan Minci, jalankan 3 terminal terpisah:

```bash
# Terminal 1 — pastikan Ollama jalan (biasanya sudah auto-start)
ollama serve

# Terminal 2 — webhook server
cd minci-project/webhook
.\venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 3 — tunnel publik
cloudflared tunnel --url http://localhost:8000
```

Kalau pakai quick tunnel, ingat URL-nya berubah tiap restart — jadi Callback URL di dashboard Meta perlu di-update ulang tiap kali. Ini alasan utama kenapa untuk pemakaian jangka panjang disarankan pakai **Named Tunnel** (Bagian 7.1).
