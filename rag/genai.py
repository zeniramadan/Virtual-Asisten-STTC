"""
query.py (Hybrid Pure RAG: Local Embedding + Gemini API Terbaru)
================================================================
- RETRIEVAL MURNI: Tidak ada logika routing. Semua mengandalkan 
  semantic search (cosine distance) dari ChromaDB.
- HYBRID: Embedding menggunakan Ollama lokal (bge-m3).
- LLM CHAT: Menggunakan Google Gemini API dengan SDK terbaru (google-genai).
"""

from __future__ import annotations
import os
import chromadb
import ollama
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ============================================================
# 1. KONFIGURASI GEMINI API & DIREKTORI
# ============================================================
# Mengambil API key dari folder webhook/.env (sesuai konfigurasi Kakak)
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "webhook", ".env")
load_dotenv(dotenv_path=dotenv_path)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ PERINGATAN KERAS: GEMINI_API_KEY tidak ditemukan di file .env!")

# Inisialisasi client Gemini versi terbaru
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
CHAT_MODEL = "gemini-3.5-flash-lite" # Menggunakan model flash standar yang stabil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
COLLECTION_NAME = "minci_dokumen"
EMBED_MODEL = "bge-m3"

RETRIEVAL_K = 10
MAX_DISTANCE = 0.60
DEBUG = True

# ============================================================
# 2. INISIALISASI CHROMADB
# ============================================================
_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)
_collection = _client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

if DEBUG:
    print(f"[RAG] Collection={COLLECTION_NAME} | Chunks={_collection.count()}")
    print(f"[LLM] Menggunakan: {CHAT_MODEL} (Gemini API)")

# ============================================================
# 3. PURE RETRIEVAL (TANPA ROUTING)
# ============================================================
def retrieve_context(question: str) -> str:
    """Mencari potongan dokumen HANYA berdasarkan kedekatan semantik (distance)."""
    
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=RETRIEVAL_K
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_blocks = []
    
    if DEBUG:
        print(f"\n🔍 [PURE RETRIEVAL] Query: '{question}'")

    for doc, dist, meta in zip(documents, distances, metadatas):
        source = meta.get("source", "?")
        
        if dist > MAX_DISTANCE:
            if DEBUG: print(f"   ❌ DIBUANG: {source} (Dist: {dist:.4f} > Threshold)")
            continue

        if DEBUG: print(f"   ✅ DIPAKAI: {source} (Dist: {dist:.4f})")
        context_blocks.append(f"[Sumber: {source}]\n{doc.strip()}")

    if not context_blocks:
        return "TIDAK ADA DATA PANDUAN YANG DITEMUKAN."

    return "\n\n---\n\n".join(context_blocks)

# ============================================================
# 4. SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """Kamu adalah Minci, Asisten Virtual Akademik STT Cipasung.
Gaya bicaramu santai, ramah, ceria ala Gen-Z, tapi tetap sopan. Panggil pengguna dengan sebutan "Kak" atau "Kakak".

ATURAN WAJIB FALLBACK:
- Jika CONTEXT berisi "TIDAK ADA DATA PANDUAN YANG DITEMUKAN" atau informasi yang ditanyakan sama sekali TIDAK ADA di CONTEXT, kamu WAJIB menjawab PERSIS dengan kalimat ini:
  "Maaf kak, informasi yang kakak tanyakan tidak ada di panduan kami, coba bertanya lebih spesifik, atau silakan kakak hubungi bagian Tata Usaha ya!"
  (Jangan menambahkan penjelasan apapun selain kalimat di atas).

ATURAN UTAMA RAG:
1. Jawab pertanyaan HANYA berdasarkan informasi faktual yang ada di CONTEXT.
2. DILARANG KERAS berhalusinasi, mengarang fakta, atau menggunakan pengetahuan umummu.
3. Jika CONTEXT berupa daftar (seperti syarat, jadwal, atau biaya), tuliskan SEMUA poin tersebut dengan lengkap menggunakan format bullet poin "-".
4. JANGAN PERNAH menyebutkan asal data, "berdasarkan context", "metadata", "chunk", "Sumber", atau "RAG".
5. JANGAN menggunakan format markdown tebal (**) karena akan dikirim via WhatsApp.
6. Jika pengguna menyapa (halo, pagi) atau mengucap salam, balas sapaannya (jika assalamualaikum jawab waalaikumsalam kak), lalu tawarkan bantuan seputar PMB, KRS, atau biaya.
"""

def build_user_prompt(question: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nPERTANYAAN:\n{question}\n\nJawab langsung pertanyaan tersebut."

# ============================================================
# 5. CLEANING & API UTAMA
# ============================================================
def clean_output(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("**", "")
    text = text.replace("•", "-").replace("▪", "-").replace("⁃", "-")
    return text.strip()

def ask_minci(question: str) -> str:
    if not question.strip():
        return "Ada yang bisa Minci bantu, kak?"

    context = retrieve_context(question)

    if DEBUG:
        print(f"\n📑 [CONTEXT KE GEMINI] Berhasil memuat konteks yang lolos jarak.")

    final_prompt = build_user_prompt(question, context)

    try:
        # Menggunakan struktur pemanggilan Gemini versi baru (google-genai)
        response = gemini_client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=final_prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
    except Exception as exc:
        if DEBUG: print(f"⚠️ [LLM] error (Gemini): {exc}")
        return "Maaf kak, sistem Minci sedang gangguan. Coba lagi nanti ya!"

    raw_answer = response.text or ""
    answer = clean_output(raw_answer)

    return answer

# ============================================================
# TEST TERMINAL
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Mode Test PURE RAG (Gemini API + BGE-M3)")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        q = input("Kamu: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        
        jawaban = ask_minci(q)
        print(f"\nMinci: {jawaban}\n")