# Testing — obsidian_rag.py

Folder ini adalah versi testing 3-tahap (Retrieval Quality, Generation
Quality/Faithfulness, Guardrail) yang disesuaikan khusus untuk
`obsidian_rag.py` — beda dari testing untuk `backup.py`, karena
`obsidian_rag.py` tidak pakai routing (`detect_route`) dan metadata
chunk-nya pakai `title`/`path`/`tags`, bukan `source`.

## Struktur

```
testing_obsidian/
├── eval_retrieval.py      # Tahap 1: Hit Rate@k, Precision@k
├── eval_faithfulness.py   # Tahap 2: Faithfulness, judge = Gemini API
├── eval_guardrail.py      # Tahap 3: stress test / guardrail
├── run_all.py             # jalankan ketiga tahap sekaligus
└── results/                # dibuat otomatis, hasil per-run (JSON)
```

## Setup

File `ground_truth.json` dan `stress_cases.json` dibaca dari folder
`dataset/` di root project.

1. Taruh folder `testing_obsidian/` **sejajar** dengan `obsidian_rag.py`:

   ```
   project/
   ├── obsidian_rag.py
   └── testing_obsidian/
       ├── eval_retrieval.py
       └── ...
   ```

2. **PENTING — sesuaikan `dataset/ground_truth.json`.** Isinya sekarang cuma
   contoh/placeholder (judul note seperti "Syarat Pendaftaran Umum", "Cara
   Mengisi KRS", dst). Ganti `expected_title` di tiap baris supaya persis
   sama dengan judul note (`metadata['title']`) yang sebenarnya ada di
   vault Obsidian kamu setelah proses ingest — kalau tidak persis sama,
   Hit Rate bakal selalu 0% bukan karena retrieval-nya jelek, tapi karena
   judulnya tidak cocok.

3. Pastikan Ollama sudah jalan (dipakai `obsidian_rag.ask()` untuk
   menghasilkan JAWABAN yang diuji — itu tetap backend lokal, tidak diubah).

4. **Judge di Tahap 2 pakai Gemini API**, via SDK resmi `google-genai`,
   baca `GEMINI_API_KEY` dari environment:

   ```bash
   pip install google-genai
   export GEMINI_API_KEY="xxxxx"
   ```

   Model judge default: `gemini-2.5-flash` (nama versi eksplisit, bukan
   alias `-latest` yang kadang ditolak API). Ganti lewat `--judge-model`
   kalau perlu, mis. `--judge-model gemini-2.5-flash-lite`.

   **Kena limit free tier (`429 RESOURCE_EXHAUSTED`)?** Script sudah
   otomatis retry (baca `retryDelay` dari error Gemini) dan punya jeda
   preventif `13` detik antar-request (`--request-delay` untuk mengubah).

## Menjalankan

```bash
python testing_obsidian/eval_retrieval.py --k 3
python testing_obsidian/eval_faithfulness.py
python testing_obsidian/eval_guardrail.py

# atau semua sekaligus:
python testing_obsidian/run_all.py
```

## Cara baca hasil

- **Hit Rate@k rendah** → cek dulu apakah `expected_title` di
  `dataset/ground_truth.json` sudah persis sama dengan judul note asli di vault
  (lihat poin 2 di Setup) sebelum menyimpulkan retrieval-nya bermasalah.
- **Precision@k rendah tapi Hit Rate tinggi** → note yang benar ketemu,
  tapi tercampur note lain yang kurang relevan di posisi atas; cek urutan
  sort di `retrieve_with_debug()` (`exact_tags` → `distance` → `tag_overlap`
  → `overlap`).
- **Faithfulness turun** → baca field `claims` di hasil detail untuk lihat
  klaim spesifik yang `UNSUPPORTED`.
- **Guardrail pass rate turun** → baca `answer` pada kasus `❌ GAGAL` di
  output `eval_guardrail.py`.

Simpan hasil tiap run (`results/<timestamp>/`) supaya bisa dibandingkan
sebelum/sesudah perubahan pada struktur vault, prompt, atau ambang jarak.
