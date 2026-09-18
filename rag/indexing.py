"""
Skrip indexing dokumen (.docx, .txt, .md) ke ChromaDB.
Jalankan ini: python indexing.py "/path/ke/folder/dokumen"
"""

from __future__ import annotations
import os
import re
import shutil
import sys
import time
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama

import config

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

DB_DIR = config.DB_DIR
COLLECTION_NAME = config.COLLECTION_NAME
EMBED_MODEL = config.EMBED_MODEL
SUPPORTED_EXTS = config.SUPPORTED_EXTS
CHUNK_SIZE_CHARS = config.CHUNK_SIZE_CHARS
CHUNK_OVERLAP_CHARS = config.CHUNK_OVERLAP_CHARS


def read_document(path: str) -> str:
    """Baca isi dokumen jadi teks polos. Tambah handler baru di sini kalau
    mau dukung format lain (pdf, pptx, dst)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        import docx2txt
        return docx2txt.process(path) or ""
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"format '{ext}' belum didukung")


def split_into_sentences(text: str) -> list[str]:
    """Pecah teks jadi kalimat. Sengaja sederhana (regex berbasis tanda baca
    akhir kalimat), cukup buat kebutuhan chunking tanpa perlu NLP lengkap."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Gabungkan kalimat jadi chunk sebesar `chunk_size`, dengan beberapa
    kalimat terakhir dari chunk sebelumnya diulang di awal chunk berikutnya
    (overlap) -- supaya konteks di perbatasan antar chunk tidak putus.
    Ini versi manual dari SentenceSplitter(chunk_size, chunk_overlap) di
    LlamaIndex, tanpa dependency tambahan."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current))

            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def get_collection():
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def reset_database() -> None:
    """Hapus database ChromaDB lama sebelum indexing penuh."""
    if not os.path.isdir(DB_DIR):
        return

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            shutil.rmtree(DB_DIR)
            print(f"🗑️ Database lama dihapus: {DB_DIR}")
            return
        except PermissionError as e:
            if attempt < max_attempts:
                print(
                    f"⏳ '{DB_DIR}' masih terkunci proses lain, coba lagi "
                    f"({attempt}/{max_attempts})..."
                )
                time.sleep(1.5)
                continue
            print(
                f"\n❌ Gagal menghapus '{DB_DIR}' karena masih dipakai proses lain:\n"
                f"   {e}\n\n"
                "   Tutup dulu semua proses yang membuka chroma_db ini "
                "(webhook, sesi chat, kernel Jupyter/VS Code), lalu jalankan "
                "ulang indexing.\n"
            )
            sys.exit(1)


def index_documents(source_dir: str, force: bool = False) -> None:
    collection = get_collection()
    existing = collection.get(include=["metadatas"])
    indexed_mtime = {
        m["path"]: m.get("mtime", 0)
        for m in existing.get("metadatas", [])
        if m
    }

    files = []
    for root, _, filenames in os.walk(source_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(root, fn))

    updated, skipped = 0, 0
    for path in files:
        rel_path = os.path.relpath(path, source_dir)
        mtime = os.path.getmtime(path)

        if not force and indexed_mtime.get(rel_path) == mtime:
            skipped += 1
            continue

        collection.delete(where={"path": rel_path})

        try:
            text = read_document(path)
        except ValueError as e:
            print(f"⚠️ Dilewati ({e}): {rel_path}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        title = os.path.splitext(os.path.basename(path))[0]

        for chunk_index, chunk in enumerate(chunks):
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
            collection.add(
                ids=[f"{rel_path}::{chunk_index}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "path": rel_path,
                    "title": title,
                    "mtime": mtime,
                }],
            )
        print(f"🔄 [INDEXED] {rel_path} ({len(chunks)} chunk)")
        updated += 1

    print(f"\n📁 [SELESAI] {updated} dokumen diperbarui, {skipped} dokumen dilewati.")


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if argument != "--force"]
    force = "--force" in sys.argv[1:]
    source_dir = arguments[0] if arguments else config.DEFAULT_SOURCE_DIR

    if not os.path.exists(source_dir):
        print(f"Error: Folder '{source_dir}' tidak ditemukan!")
        sys.exit(1)

    reset_database()
    print(f"🚀 Memulai indexing dokumen dari: {source_dir} ...")
    index_documents(source_dir, force=True)