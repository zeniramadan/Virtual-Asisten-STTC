# Testing — Minci RAG

Folder ini berisi 3 tahap pengujian sistem Minci: retrieval quality, generation
quality (faithfulness), dan guardrail/stress test.

## Struktur

```
testing/
├── ground_truth.json      # dataset query -> dokumen sumber yang benar (tahap 1 & 2)
├── stress_cases.json      # dataset query ekstrem/adversarial (tahap 3)
├── eval_retrieval.py      # Tahap 1: Hit Rate@k, Precision@k
├── eval_faithfulness.py   # Tahap 2: Faithfulness (LLM-as-judge, reference-free)
├── eval_guardrail.py      # Tahap 3: stress test / guardrail
├── run_all.py             # jalankan ketiga tahap sekaligus, simpan hasil
└── results/                # dibuat otomatis oleh run_all.py, hasil per-run (JSON)
```

## Setup

1. Taruh folder `testing/` ini **sejajar** dengan file utama RAG kamu (mis.
   `backup.py`), bukan di dalamnya. Contoh:

   ```
   project/
   ├── backup.py
   └── testing/
       ├── eval_retrieval.py
       └── ...
   ```
2. Buka tiap file `eval_*.py`, cek baris:

   ```python
   MODULE_NAME = "backup"
   ```

   Ganti `"backup"` sesuai nama file utama RAG kamu yang sebenarnya (misalnya
   kalau file kamu bernama `query.py`, ganti jadi `MODULE_NAME = "query"`).
3. Pastikan Ollama sudah jalan dan model (`llama3.1`, `bge-m3`) sudah ter-pull,
   sama seperti kebutuhan `backup.py` sendiri. Ini tetap dipakai untuk
   menghasilkan JAWABAN dari sistem Minci (yang sedang diuji).
4. **Tahap 2 (`eval_faithfulness.py`) pakai Gemini API sebagai JUDGE** (penilai
   faithfulness) via SDK resmi `google-genai` — sama seperti yang kamu pakai di
   `gemini.py`. Jawaban tetap dari sistem lokal kamu (Ollama), tapi penilainya
   sengaja model independen di luar sistem yang diuji. Pastikan:

   ```bash
   pip install google-genai
   export GEMINI_API_KEY="xxxxx"
   ```

   Kalau kamu simpan `GEMINI_API_KEY` di file `.env` (seperti pola
   `load_dotenv()` di `gemini.py`), pastikan file itu sudah ter-load ke
   environment sebelum menjalankan skrip di `testing/` — bisa dengan
   menambahkan `load_dotenv(...)` yang sama persis di baris atas
   `eval_faithfulness.py`, atau cukup jalankan lewat shell yang env-nya sudah
   di-`export`.

   Model judge default: `gemini-flash-latest` (alias resmi Google ke versi
   Gemini Flash GA terbaru) — bisa diganti lewat `--judge-model`, mis.
   `--judge-model gemini-3.5-flash-lite` kalau mau samain dengan `CHAT_MODEL`
   di `gemini.py`, atau versi lain yang di-pin.

## Menjalankan

Jalankan satu-satu untuk debugging cepat:

```bash
python testing/eval_retrieval.py --k 3
python testing/eval_faithfulness.py
python testing/eval_guardrail.py
```

Atau jalankan semuanya sekaligus (hasil otomatis tersimpan di `testing/results/<timestamp>/`):

```bash
python testing/run_all.py
```

## Menambah / mengubah dataset uji

- **`ground_truth.json`**: tambah query baru + `expected_source` yang benar
  (`PMB.docx` / `KRS.docx` / `BIAYA.docx` / `KALENDER.docx`). Sengaja sudah
  diselipkan variasi bahasa gaul/singkat (mis. "kalo mau daftar sttc perlu
  siapin apa aja") supaya bug seperti kata "daftar" yang tidak ke-route bisa
  ketahuan otomatis lewat pengujian, bukan lewat laporan user.
- **`stress_cases.json`**: tambah query baru dengan `category` dan `expected`
  (`"fallback"` untuk query di luar konteks/ambigu, `"menolak"` untuk prompt
  injection/jailbreak).

## Cara baca hasil

- **Hit Rate@k** turun → cek kolom `route_source` di output tiap kasus yang
  gagal: kalau `route_source` salah/None padahal seharusnya ter-route, akar
  masalah ada di `ROUTES`/`detect_route()`, bukan di ChromaDB.
- **Precision@k** rendah tapi Hit Rate tinggi → dokumen benar ketemu, tapi
  bercampur dengan chunk lain yang kurang relevan di posisi atas; cek scoring
  `intent_match()` / urutan sort di `retrieve()`.
- **Faithfulness** turun → baca field `claims` di hasil detail untuk lihat
  klaim spesifik yang `UNSUPPORTED`, lalu cek apakah system prompt atau chunk
  yang diambil yang bermasalah.
- **Guardrail pass rate** turun → baca `answer` pada kasus yang `❌ GAGAL` di
  `eval_guardrail.py`, cek apakah sistem membocorkan istilah internal atau
  menuruti instruksi manipulatif.

Simpan hasil tiap run (folder `results/<timestamp>/`) supaya bisa dibandingkan
sebelum/sesudah perubahan pada `ROUTES`, threshold distance, atau system prompt.
