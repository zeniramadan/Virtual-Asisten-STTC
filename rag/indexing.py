"""
Skrip khusus untuk Indexing/Sync Vault Obsidian ke ChromaDB.
Jalankan ini: python indexer.py "/path/ke/vault"
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
import yaml

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "obsidian_vault"
EMBED_MODEL = "bge-m3"

_HEADING_SPLIT_LEVEL = 6
_HEADING_LINE_RE = rf"(?m)^(#{{1,{_HEADING_SPLIT_LEVEL}}}\s+.+)$"
_HEADING_PREFIX_RE = rf"^(#{{1,{_HEADING_SPLIT_LEVEL}}}\s+.+)"

def parse_note(path: str) -> tuple[dict, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    frontmatter = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1].replace("\t", "  ")) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = parts[2]

    return frontmatter, body.strip()


def split_into_sections(body: str) -> list[str]:
    """Pisahkan body jadi section per heading level 1-2 (# / ##). Heading
    level 3-6 (### dst) dan isinya tetap MENYATU ke dalam section H1/H2
    induknya, bukan jadi section sendiri."""
    parts = re.split(_HEADING_LINE_RE, body)
    sections = []
    buffer = ""
    for part in parts:
        if re.match(rf"^#{{1,{_HEADING_SPLIT_LEVEL}}}\s+", part):
            if buffer.strip():
                sections.append(buffer.strip())
            buffer = part + "\n"
        else:
            buffer += part
    if buffer.strip():
        sections.append(buffer.strip())

    return sections or ([body] if body.strip() else [])


def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Kalau satu 'paragraf' (dipisah baris kosong) sendirian saja sudah
    lebih besar dari max_chars -- biasanya bullet list/daftar langkah yang
    tidak dipisah baris kosong antar item -- pecah lagi per baris supaya
    tetap bisa dikelompokkan jadi beberapa unit berukuran wajar, bukan jadi
    satu blok raksasa yang memaksa unit kecil sebelumnya (mis. baris tag)
    ke-flush sendirian jadi chunk nyaris kosong."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    lines = [line for line in paragraph.split("\n") if line.strip()]
    return lines if len(lines) > 1 else [paragraph]


def split_section_into_chunks(section: str, max_chars: int = 4000) -> list[str]:
    """Pecah SATU section jadi beberapa sub-chunk kalau kepanjangan. Heading
    section (kalau ada) diulang di SETIAP sub-chunk supaya konteksnya tidak
    hilang -- tanpa ini, heading gampang ke-flush sendirian jadi chunk
    terpisah tanpa isi begitu paragraf berikutnya sudah cukup besar untuk
    melewati max_chars."""
    if len(section) <= max_chars:
        return [section]

    heading_match = re.match(_HEADING_PREFIX_RE, section)
    if heading_match:
        heading_line = heading_match.group(1)
        rest = section[len(heading_line):].lstrip("\n")
    else:
        heading_line = ""
        rest = section

    sub_chunks: list[str] = []

    def _flush(text: str) -> None:
        text = text.strip()
        if not text:
            return
        sub_chunks.append(f"{heading_line}\n\n{text}" if heading_line else text)

    paragraphs = [p for p in rest.split("\n\n") if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_oversized_paragraph(p, max_chars))

    current = ""
    for p in units:
        if len(current) + len(p) > max_chars and current:
            _flush(current)
            current = ""
        current += p + "\n\n"
    _flush(current)

    return sub_chunks


def extract_inline_tags(text: str) -> list[str]:
    """Ambil tag Obsidian inline seperti #krs atau #biaya-kuliah dari chunk."""
    return re.findall(r"(?<![\w-])#[a-zA-Z0-9_-]+", text)


def normalize_tags(tags: list[str]) -> list[str]:
    normalized = []
    for tag in tags:
        tag = tag.strip().lstrip("#").lower()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized

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
                "   Tutup dulu SEMUA proses yang membuka chroma_db ini, lalu jalankan "
                "ulang indexer:\n"
                "   - webhook WhatsApp (uvicorn app:app) kalau sedang jalan\n"
                "   - sesi chat interaktif (query.py) yang masih terbuka di terminal lain\n"
                "   - kernel Jupyter/VS Code interactive yang pernah mengimpor "
                "query.py atau ingest.py\n"
            )
            sys.exit(1)


def index_vault(vault_dir: str, force: bool = False) -> None:
    collection = get_collection()
    existing = collection.get(include=["metadatas"])
    indexed_mtime = {
        m["path"]: m.get("mtime", 0)
        for m in existing.get("metadatas", [])
        if m
    }

    md_files = []
    for root, _, files in os.walk(vault_dir):
        if ".obsidian" in root or ".trash" in root:
            continue
        for fn in files:
            if fn.endswith(".md"):
                md_files.append(os.path.join(root, fn))

    updated, skipped = 0, 0
    for path in md_files:
        rel_path = os.path.relpath(path, vault_dir)
        mtime = os.path.getmtime(path)

        if not force and indexed_mtime.get(rel_path) == mtime:
            skipped += 1
            continue  

        collection.delete(where={"path": rel_path})

        frontmatter, body = parse_note(path)
        sections = split_into_sections(body)
        if not sections:
            continue

        note_tags = frontmatter.get("tags", [])
        if isinstance(note_tags, str):
            note_tags = [note_tags]
        note_tags = normalize_tags(note_tags)

        chunk_index = 0
        for section in sections:
            section_tags = normalize_tags(note_tags + extract_inline_tags(section))

            for chunk_text in split_section_into_chunks(section):
                embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk_text)["embedding"]
                collection.add(
                    ids=[f"{rel_path}::{chunk_index}"],
                    embeddings=[embedding],
                    documents=[chunk_text],
                    metadatas=[{
                        "path": rel_path,
                        "title": frontmatter.get("title", os.path.splitext(os.path.basename(path))[0]),
                        "tags": ", ".join(section_tags),
                        "mtime": mtime,
                    }],
                )
                chunk_index += 1
        print(f"🔄 [INDEXED] {rel_path}")
        updated += 1

    print(f"\n📁 [SELESAI] {updated} note diperbarui, {skipped} note dilewati.")

if __name__ == "__main__":
    DEFAULT_VAULT = r"C:\Users\ZENI RAMADAN\Documents\Skripsi\Virtual-Asisten-STTC\documents"
    arguments = [argument for argument in sys.argv[1:] if argument != "--force"]
    force = "--force" in sys.argv[1:]
    vault = arguments[0] if arguments else DEFAULT_VAULT

    if not os.path.exists(vault):
        print(f"Error: Folder vault '{vault}' tidak ditemukan!")
        sys.exit(1)

    reset_database()
    print(f"🚀 Memulai indexing vault dari: {vault} ...")
    index_vault(vault, force=True)