"""
check_chroma.py
================
Script diagnostik: cek apakah ChromaDB benar-benar berisi chunk dari dokumen kamu,
dan coba query manual buat lihat chunk apa yang kereturn untuk pertanyaan tertentu.

Jalankan dari dalam folder rag/:
    python check_chroma.py
"""

import os
import chromadb
import ollama

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"
EMBED_QUERY_PREFIX = ""

print(f"📂 Membaca ChromaDB dari: {CHROMA_DB_DIR}")
print(f"   Folder ada? {os.path.exists(CHROMA_DB_DIR)}\n")

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    print(f"❌ Collection '{COLLECTION_NAME}' TIDAK DITEMUKAN. Error: {e}")
    print("   -> Artinya rag_ingest.py belum pernah berhasil jalan dari folder ini,")
    print("      atau ChromaDB-nya kebentuk di lokasi lain.")
    exit()

total = collection.count()
print(f"✅ Collection ditemukan. Total chunk tersimpan: {total}\n")

if total == 0:
    print("⚠️  Collection ADA tapi KOSONG (0 chunk).")
    print("   -> Artinya rag_ingest.py jalan tapi tidak nemu file .docx,")
    print("      atau dokumennya kosong / gagal dibaca.")
    exit()

# Ringkasan: dokumen apa saja yang ke-ingest, dan berapa chunk masing-masing
all_meta = collection.get(include=["metadatas"])["metadatas"]
from collections import Counter
source_counts = Counter(m.get("source", "?") for m in all_meta)

print("📚 Ringkasan dokumen yang ke-ingest:")
for source, count in source_counts.items():
    print(f"   - {source}: {count} chunk")
print()

# Tampilkan beberapa contoh chunk yang tersimpan
print("📄 Contoh 3 chunk pertama yang tersimpan:")
sample = collection.peek(limit=3)
for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
    print(f"\n--- Chunk {i+1} (sumber: {meta.get('source')}) ---")
    print(doc[:300], "..." if len(doc) > 300 else "")

# Coba query manual
print("\n\n🔍 Test query manual")
question = input("Masukkan pertanyaan test (contoh: 'biaya pendaftaran PMB'): ")

query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=f"{EMBED_QUERY_PREFIX}{question}")["embedding"]
results = collection.query(query_embeddings=[query_embedding], n_results=4)

print(f"\nHasil retrieval untuk: '{question}'\n")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
)):
    print(f"--- Hasil #{i+1} (jarak/distance: {dist:.4f}, sumber: {meta.get('source')}) ---")
    print(doc[:400], "..." if len(doc) > 400 else "")
    print()

print("💡 Semakin kecil 'distance', semakin relevan chunk tersebut dengan pertanyaan.")
print("   Kalau chunk yang muncul di atas ISINYA TIDAK NYAMBUNG dengan pertanyaan,")
print("   berarti masalahnya di kualitas embedding/chunking, bukan di collection kosong.")

# ============================================================
# CEK PALING PENTING: cari keyword secara LANGSUNG (bukan lewat embedding)
# supaya tahu pasti apakah kata kuncinya benar-benar ke-ingest atau tidak.
# ============================================================
print("\n\n🔎 Cari keyword langsung di SEMUA chunk (tanpa embedding)")
keyword = input("Masukkan kata kunci yang kamu tahu PASTI ada di dokumen (contoh: 'biaya pendaftaran'): ").strip().lower()

all_docs = collection.get(include=["documents", "metadatas"])
matches = []
for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
    if keyword in doc.lower():
        matches.append((doc, meta))

print(f"\nDitemukan {len(matches)} chunk yang MENGANDUNG kata '{keyword}':\n")
for doc, meta in matches:
    print(f"--- Sumber: {meta.get('source')} ---")
    print(doc)
    print()

if not matches:
    print(f"❌ TIDAK ADA chunk yang mengandung kata '{keyword}' sama sekali.")
    print("   Artinya: teks ini BELUM ke-ingest ke ChromaDB.")
    print("   Kemungkinan penyebab:")
    print("   1. Teksnya ada di dalam TABEL dengan format yang tidak terbaca rapi")
    print("   2. Teksnya ada di header/footer/text-box (python-docx tidak baca ini)")
    print("   3. Teksnya ada di file .docx yang BEDA dari yang kamu kira, atau file itu")
    print("      belum ke-copy ke folder documents/ saat rag_ingest.py dijalankan")
    print("   4. Kata kuncinya beda penulisan (misal 'Rp.' vs 'biaya', dsb)")
else:
    print("✅ Kata kunci DITEMUKAN di chunk di atas. Artinya datanya SUDAH ke-ingest,")
    print("   masalahnya murni di RANKING retrieval (embedding menganggap chunk lain")
    print("   lebih 'mirip' dengan pertanyaan kamu). Solusi: naikkan TOP_K di rag_query.py,")
    print("   atau perbaiki cara chunking supaya info biaya tidak tercampur/terpotong.")