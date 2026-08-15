"""
ingest.py
=============
Script ini membaca semua file .docx di folder `documents/` (dokumen PMB & KRS),
memecahnya jadi potongan-potongan teks (chunking) sambil MEMPERTAHANKAN konteks
judul/sub-judul dan struktur tabel, lalu menyimpannya ke vector database lokal
(ChromaDB) dengan embedding dari Ollama (bge-m3 -- model multilingual, akurat untuk
Bahasa Indonesia).

Jalankan ini SETIAP KALI dokumen PMB/KRS berubah atau bertambah.

Cara pakai:
    1. Pastikan Ollama sudah jalan (`ollama serve`) dan sudah pull model embedding:
       ollama pull bge-m3
    2. Taruh semua file .docx (PMB & KRS) di folder documents/
    3. Jalankan: python rag_ingest.py

CATATAN PENTING soal metrik jarak (distance):
    ChromaDB secara default pakai metrik L2 (Euclidean distance) kalau tidak di-set
    eksplisit. Untuk pencarian makna teks (semantic search), metrik yang seharusnya
    dipakai adalah COSINE, karena L2 ikut terpengaruh "panjang" vektor embedding
    (yang bisa beda-beda tergantung panjang teks chunk), bukan murni kemiripan makna.
    Kalau tidak di-set cosine, hasil ranking retrieval bisa kacau/tidak konsisten --
    chunk yang tidak relevan bisa menang cuma karena kebetulan panjang vektornya mirip.
    Makanya di sini collection dibuat dengan metadata={"hnsw:space": "cosine"}.

CATATAN PENTING soal cara baca dokumen:
    Banyak dokumen Word (termasuk PMB.docx & KRS.docx) TIDAK pakai style "Heading"
    resmi dari Word untuk judul section -- judulnya cuma teks yang di-BOLD dan
    ditulis SEMUA HURUF BESAR secara manual. Script ini mendeteksi itu sebagai
    heuristik utama:
        - bold + SEMUA HURUF BESAR         -> dianggap JUDUL SECTION (H1)
        - bold + bukan semua huruf besar    -> dianggap SUB-JUDUL dalam section (H2)
        - style "Heading 1/2/3" resmi Word  -> tetap didukung juga sebagai H1
        - sisanya                           -> konten biasa
    Setiap potongan konten biasa otomatis diberi label "[Konteks: H1 >> H2]" di
    depannya, supaya walau nanti dipecah jadi chunk kecil-kecil, potongan itu
    TETAP tahu dia bagian dari section/sub-section mana.
"""

import os
import re
import glob
import docx
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.document import Document as _DocumentClass
import ollama
import chromadb
from chromadb.config import Settings

# ====== KONFIGURASI ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "..", "documents")
# Tambahkan "database" di tengah path-nya
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db") 
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"        # model embedding multilingual, jauh lebih akurat untuk
                               # Bahasa Indonesia dibanding nomic-embed-text (sebelumnya)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# bge-m3 TIDAK butuh prefix instruksi khusus (beda dengan nomic-embed-text yang
# wajib "search_document:"/"search_query:"). Dikosongkan saja di sini.
# Kalau nanti ganti model lagi dan modelnya butuh prefix, isi di sini.
EMBED_DOC_PREFIX = ""
# ==========================


# ============================================================
# BAGIAN 1: Deteksi struktur dokumen (heading, sub-heading, level bullet)
# ============================================================

# ============================================================
# BAGIAN 0: Jembatani angka romawi <-> angka arab
# (dokumen sering nulis "Gelombang I", tapi user nanya "gelombang 1" --
#  tanpa ini, embedding kadang gagal mencocokkan keduanya)
# ============================================================

ROMAN_TO_WORDS = {
    "I": ("1", "satu"),
    "II": ("2", "dua"),
    "III": ("3", "tiga"),
    "IV": ("4", "empat"),
    "V": ("5", "lima"),
    "VI": ("6", "enam"),
    "VII": ("7", "tujuh"),
    "VIII": ("8", "delapan"),
    "IX": ("9", "sembilan"),
    "X": ("10", "sepuluh"),
}

_ROMAN_PATTERN = re.compile(
    r"\b(Gelombang|Semester|Tahap|Angkatan)\s+(I|II|III|IV|V|VI|VII|VIII|IX|X)\b"
    r"(?!\s*/\s*\d)",  # skip kalau SUDAH ada format "/ <angka>" nyusul (hindari dobel anotasi)
    re.IGNORECASE,
)


def annotate_roman_numerals(text: str) -> str:
    """
    Ubah 'Gelombang I' jadi 'Gelombang I / 1 / satu' -- format ini sengaja disamakan
    dengan gaya yang sudah dipakai sendiri di dokumen (lihat 'Semester I / 1 / satu'
    di bagian rincian biaya), supaya query dengan angka arab ATAU kata (gelombang 1,
    gelombang satu) tetap match dengan dokumen yang nulisnya pakai angka romawi.
    """
    def repl(match):
        word, roman = match.group(1), match.group(2)
        pair = ROMAN_TO_WORDS.get(roman.upper())
        if pair is None:
            return match.group(0)
        arabic, kata = pair
        return f"{word} {roman} / {arabic} / {kata}"

    return _ROMAN_PATTERN.sub(repl, text)


def get_list_level(paragraph):
    """
    Cek level bullet/numbered list paragraf ini (0 = terluar, 1 = sub-bullet, dst).
    Return None kalau BUKAN bagian dari list.
    Coba beberapa cara berurutan karena tiap dokumen Word bisa beda gaya penulisannya.
    """
    p = paragraph._p
    pPr = p.find(qn("w:pPr"))

    if pPr is not None:
        numPr = pPr.find(qn("w:numPr"))
        if numPr is not None:
            ilvl = numPr.find(qn("w:ilvl"))
            if ilvl is not None:
                return int(ilvl.get(qn("w:val")))
            return 0

    indent = paragraph.paragraph_format.left_indent
    if indent is None and paragraph.style is not None:
        indent = paragraph.style.paragraph_format.left_indent
    if indent is not None and indent.inches > 0.15:
        return int(indent.inches // 0.5)

    style_name = (paragraph.style.name or "") if paragraph.style else ""
    style_name_lower = style_name.lower()
    if "list" in style_name_lower or "bullet" in style_name_lower:
        digits = "".join(ch for ch in style_name if ch.isdigit())
        if digits:
            return int(digits) - 1
        return 0

    return None


def is_paragraph_bold(paragraph) -> bool:
    """Cek apakah (sebagian besar) teks paragraf ini di-bold."""
    runs_with_text = [r for r in paragraph.runs if r.text.strip()]
    if not runs_with_text:
        return False
    bold_count = sum(1 for r in runs_with_text if r.bold)
    return bold_count >= len(runs_with_text) / 2  # mayoritas run di-bold


def is_word_heading_style(paragraph) -> bool:
    """Cek apakah paragraf ini pakai style resmi Word 'Heading 1/2/3' atau 'Title'."""
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    return style_name.startswith("heading") or style_name.startswith("title")


def classify_paragraph(paragraph, text: str) -> str:
    """
    Klasifikasikan paragraf jadi salah satu dari: 'h1', 'h2', 'content'.

    h1 = judul section utama (style Heading resmi Word, ATAU bold + SEMUA HURUF BESAR)
    h2 = sub-judul dalam section (bold, tapi TIDAK semua huruf besar)
    content = teks isi biasa
    """
    if is_word_heading_style(paragraph):
        return "h1"

    if is_paragraph_bold(paragraph):
        # Anggap heading kalau: SEMUA HURUF BESAR dan cukup panjang (hindari
        # singkatan pendek yang kebetulan huruf besar semua, misal "KRS", "PMB")
        if text.isupper() and len(text) > 8:
            return "h1"
        return "h2"

    return "content"


# ============================================================
# BAGIAN 2: Baca isi tabel Word (kalau ada), format jadi kalimat "kolom: nilai"
# ============================================================

def read_table_rows(table, context_prefix: str = ""):
    """
    Ubah tabel Word jadi baris-baris teks yang gampang dicari.
    Baris pertama tabel diasumsikan HEADER KOLOM (misal "Kode MK | Nama MK | SKS"),
    lalu tiap baris data diubah jadi format "Kode MK: TI101, Nama MK: Kalkulus I, SKS: 3"
    supaya tetap jelas maknanya walau baris ini nanti berdiri sendiri sebagai satu chunk.
    """
    rows_text = []
    if not table.rows:
        return rows_text

    header_cells = [c.text.strip() for c in table.rows[0].cells]
    has_header = any(header_cells)
    data_rows = table.rows[1:] if has_header else table.rows

    for row in data_rows:
        cells = [c.text.strip() for c in row.cells]
        if not any(cells):
            continue

        if has_header and len(cells) == len(header_cells):
            pairs = [f"{h}: {v}" for h, v in zip(header_cells, cells) if h and v]
            row_text = ", ".join(pairs) if pairs else " | ".join(c for c in cells if c)
        else:
            row_text = " | ".join(c for c in cells if c)

        if row_text:
            row_text = annotate_roman_numerals(row_text)
            if context_prefix:
                rows_text.append(f"[Konteks: {context_prefix}] {row_text}")
            else:
                rows_text.append(row_text)

    return rows_text


# ============================================================
# BAGIAN 3: Baca paragraf & tabel SESUAI URUTAN ASLI di dokumen
# (python-docx secara default memisahkan document.paragraphs dan document.tables,
#  padahal urutannya bisa selang-seling. iter_block_items menjaga urutan asli,
#  supaya konteks heading yang "aktif" saat itu tetap akurat.)
# ============================================================

def iter_block_items(document):
    parent_elm = document.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def read_docx_text(filepath: str) -> str:
    """Baca seluruh isi .docx (paragraf + tabel) sambil menyisipkan konteks heading/sub-heading."""
    document = docx.Document(filepath)
    full_text = []

    current_h1 = None
    current_h2 = None
    list_stack = {}  # ilvl -> teks bullet terakhir di level itu (fallback nested bullet non-bold)

    # Konteks terakhir yang SUDAH ditulis sebagai baris "[Konteks: ...]". Dipakai supaya
    # tidak nulis ulang konteks yang SAMA PERSIS berkali-kali di baris-baris berurutan --
    # ini penting karena embedding model (nomic-embed-text) jadi kurang bisa membedakan
    # antar section kalau tiap chunk isinya didominasi teks konteks yang berulang-ulang,
    # bukan konten sebenarnya. Header konteks cukup ditulis SEKALI tiap kali dia berubah.
    last_written_context = None

    for block in iter_block_items(document):

        # ---- Blok berupa TABEL ----
        if isinstance(block, Table):
            ctx_parts = [c for c in [current_h1, current_h2] if c]
            ctx_prefix = " >> ".join(ctx_parts)
            for row_text in read_table_rows(block, context_prefix=ctx_prefix):
                full_text.append(row_text)
            continue

        # ---- Blok berupa PARAGRAF ----
        para = block
        text = para.text.strip()
        if not text:
            continue

        text = annotate_roman_numerals(text)

        kind = classify_paragraph(para, text)

        if kind == "h1":
            current_h1 = text
            current_h2 = None
            list_stack = {}
            last_written_context = None
            full_text.append(f"\n## {text}")
            continue

        if kind == "h2":
            current_h2 = text
            list_stack = {}
            last_written_context = None
            if current_h1:
                full_text.append(f"[Bagian: {current_h1}] {text}")
            else:
                full_text.append(text)
            continue

        # ---- kind == "content" ----
        ctx_parts = [c for c in [current_h1, current_h2] if c]

        level = get_list_level(para)
        if level is not None and level > 0:
            parent_line = list_stack.get(level - 1)
            if parent_line and parent_line not in ctx_parts:
                ctx_parts.append(parent_line)
        if level is not None:
            list_stack[level] = text
            list_stack = {lvl: t for lvl, t in list_stack.items() if lvl <= level}

        if ctx_parts:
            ctx_str = " >> ".join(ctx_parts)
            if ctx_str != last_written_context:
                # Konteks berubah (baru masuk section/bullet lain) -> tulis header-nya SEKALI
                full_text.append(f"[Konteks: {ctx_str}]")
                last_written_context = ctx_str
            full_text.append(f"- {text}")
        else:
            full_text.append(text)

    return "\n".join(full_text)


# ============================================================
# BAGIAN 4: Chunking berbasis baris (tidak potong di tengah kalimat)
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    lines = [line for line in text.split("\n") if line.strip()]

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1

        if current_length + line_length > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))

            overlap_lines = []
            overlap_length = 0
            for prev_line in reversed(current_lines):
                if overlap_length + len(prev_line) > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_length += len(prev_line) + 1

            current_lines = overlap_lines
            current_length = overlap_length

        current_lines.append(line)
        current_length += line_length

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def get_embedding(text: str):
    # Prefix instruksi (kosong untuk bge-m3, tapi disiapkan di sini kalau nanti ganti
    # model lain yang butuh prefix seperti nomic-embed-text dulu).
    response = ollama.embeddings(model=EMBED_MODEL, prompt=f"{EMBED_DOC_PREFIX}{text}")
    return response["embedding"]


# ============================================================
# BAGIAN 5: Main — ingest semua dokumen ke ChromaDB
# ============================================================

def main():
    print(f"📁 BASE_DIR       : {BASE_DIR}")
    print(f"📁 DOCUMENTS_DIR  : {os.path.abspath(DOCUMENTS_DIR)}")
    print(f"📁 CHROMA_DB_DIR  : {CHROMA_DB_DIR}\n")

    docx_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.docx"))

    if not docx_files:
        print(f"⚠️  Tidak ada file .docx ditemukan di {os.path.abspath(DOCUMENTS_DIR)}")
        print("Taruh dokumen PMB dan KRS (.docx) di folder tersebut dulu ya.")
        print("(Pastikan juga bukan file yang lagi terbuka di Word/dimulai dengan '~$')")
        return

    print(f"📄 Ditemukan {len(docx_files)} file dokumen:")
    for f in docx_files:
        print(f"   - {os.path.basename(f)}")

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),  # matikan telemetry (sumber pesan error "Failed to send telemetry event")
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # PENTING: cosine, bukan default L2 -- lihat catatan di atas
    )

    total_chunks = 0
    doc_id_counter = 0

    for filepath in docx_files:
        filename = os.path.basename(filepath)
        print(f"\n🔄 Memproses: {filename}")

        raw_text = read_docx_text(filepath)

        if not raw_text.strip():
            print(f"   ⚠️  PERINGATAN: tidak ada teks terbaca dari {filename}!")
            continue

        chunks = chunk_text(raw_text)
        print(f"   -> {len(raw_text)} karakter teks terbaca, {len(chunks)} chunk dihasilkan")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            doc_id_counter += 1

            collection.add(
                ids=[f"{filename}_{i}_{doc_id_counter}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk_index": i}],
            )
            total_chunks += 1

    print(f"\n✅ Selesai! Total {total_chunks} chunk tersimpan di vector database ({CHROMA_DB_DIR})")


if __name__ == "__main__":
    main()