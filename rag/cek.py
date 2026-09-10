"""
Skrip untuk mengecek dan mencetak seluruh isi chunk di ChromaDB Obsidian.
Jalankan: python check_chroma.py
"""

from __future__ import annotations
import os
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

# Tentukan direktori database ChromaDB yang digunakan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "obsidian_vault"

def inspect_chroma_db():
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"❌ Error: Folder database ChromaDB tidak ditemukan di '{CHROMA_DB_DIR}'")
        print("Pastikan Anda sudah menjalankan proses indexing sebelumnya.")
        return

    # Inisialisasi client ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error: Koleksi '{COLLECTION_NAME}' tidak ditemukan di database. ({e})")
        return

    total_chunks = collection.count()
    print("=" * 60)
    print(f"📊 LAPORAN INSPEKSI CHROMA DB: {COLLECTION_NAME}")
    print(f"📌 Total Keseluruhan Chunk: {total_chunks}")
    print("=" * 60)

    if total_chunks == 0:
        print("⚠️ Database kosong, belum ada dokumen yang di-index.")
        return

    # Ambil seluruh data (documents, metadatas, ids) dari collection
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    for idx, (chunk_id, doc, meta) in enumerate(zip(ids, documents, metadatas), 1):
        print(f"\n[{idx}] CHUNK ID : {chunk_id}")
        if meta:
            print(f"    📂 Path File : {meta.get('path', 'N/A')}")
            print(f"    🏷️ Judul Note: {meta.get('title', 'N/A')}")
            print(f"    🔖 Tags      : {meta.get('tags', 'N/A')}")
        print(f"    📄 Isi Teks  :")
        print("-" * 50)
        # Tampilkan isi teks chunk (batasi jika terlalu panjang agar rapi di terminal)
        formatted_doc = doc.strip().replace("\n", "\n    ")
        print(f"    {formatted_doc}")
        print("-" * 50)

    print("\n✅ Inspeksi selesai.")

if __name__ == "__main__":
    inspect_chroma_db()