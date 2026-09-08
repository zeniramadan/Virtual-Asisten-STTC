"""
Skrip khusus untuk Indexing/Sync Vault Obsidian ke ChromaDB.
Jalankan ini: python indexer.py "/path/ke/vault"
"""

from __future__ import annotations
import os
import re
import shutil
import sys
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama
import yaml

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "obsidian_vault"
EMBED_MODEL = "bge-m3"

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

def chunk_by_heading(body: str, max_chars: int = 1500) -> list[str]:
    parts = re.split(r"(?m)^(#{1,6}\s+.+)$", body)
    sections = []
    buffer = ""
    for part in parts:
        if re.match(r"^#{1,6}\s+", part):
            if buffer.strip():
                sections.append(buffer.strip())
            buffer = part + "\n"
        else:
            buffer += part
    if buffer.strip():
        sections.append(buffer.strip())

    if not sections:
        sections = [body]

    chunks = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        paragraphs = [p for p in section.split("\n\n") if p.strip()]
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > max_chars and current:
                chunks.append(current.strip())
                current = ""
            current += p + "\n\n"
        if current.strip():
            chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


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
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"🗑️ Database lama dihapus: {DB_DIR}")


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
        # File tidak berubah, skip
            skipped += 1
            continue  

        # Hapus chunk lama sebelum re-index
        collection.delete(where={"path": rel_path})

        frontmatter, body = parse_note(path)
        chunks = chunk_by_heading(body)
        if not chunks:
            continue

        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags = normalize_tags(tags)

        inherited_inline_tags = []
        for i, chunk_text in enumerate(chunks):
            inline_tags = extract_inline_tags(chunk_text)
            if re.search(r"(?m)^#{1,6}\s+", chunk_text):
                inherited_inline_tags = inline_tags
            chunk_tags = normalize_tags(tags + inherited_inline_tags)
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk_text)["embedding"]
            collection.add(
                ids=[f"{rel_path}::{i}"],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{
                    "path": rel_path,
                    "title": frontmatter.get("title", os.path.splitext(os.path.basename(path))[0]),
                    "tags": ", ".join(chunk_tags),
                    "mtime": mtime,
                }],
            )
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