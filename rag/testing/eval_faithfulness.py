"""
Tahap 2: Generation Quality Evaluation (Reference-free, metrik Faithfulness)
Mengukur sejauh mana jawaban model berakar pada CONTEXT yang ditarik retrieval,
bukan dibandingkan dengan kunci jawaban manusia.

Jawaban tetap dihasilkan oleh sistem Minci sendiri lewat minci.ask_minci()
(entah backend-nya Ollama atau Gemini, tergantung isi modul yang kamu import
lewat MODULE_NAME) -- itu yang sedang diuji. Yang bertindak sebagai JUDGE
(penilai faithfulness) memakai Gemini API secara terpisah, karena judge
idealnya model independen di luar sistem yang diuji, bukan model yang sama
dengan yang menghasilkan jawabannya.

Setup:
    export GEMINI_API_KEY="xxxxx"   # sudah kamu simpan di env

Cara pakai:
    python testing/eval_faithfulness.py
    python testing/eval_faithfulness.py --judge-model gemini-flash-latest
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", "webhook", ".env")
load_dotenv(dotenv_path=dotenv_path)

from google import genai
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODULE_NAME = os.environ.get("MODULE_NAME")
minci = __import__(MODULE_NAME)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"  # alias resmi Google ke Gemini Flash GA terbaru

# Free tier Gemini API cuma ~5 request/menit untuk model flash -- default di
# bawah ini dibuat konservatif (satu request per ~13 detik) supaya tidak
# terus-menerus kena 429 RESOURCE_EXHAUSTED. Kalau kamu pakai tier
# berbayar/kuota lebih tinggi, kecilkan lewat --request-delay.
DEFAULT_REQUEST_DELAY_SECONDS = 13
MAX_RETRIES = 4


JUDGE_PROMPT = """Kamu adalah evaluator faktual yang ketat. Diberikan CONTEXT dan JAWABAN.
Pecah JAWABAN menjadi klaim-klaim faktual terpisah (satu fakta = satu klaim),
lalu untuk tiap klaim tentukan apakah didukung PENUH oleh CONTEXT (SUPPORTED)
atau tidak didukung / ditambahkan sendiri oleh model (UNSUPPORTED).

Balas HANYA dengan JSON list, tanpa teks lain, contoh format:
[{{"claim": "...", "verdict": "SUPPORTED"}}, {{"claim": "...", "verdict": "UNSUPPORTED"}}]

CONTEXT:
{context}

JAWABAN:
{answer}
"""


def _parse_retry_delay_seconds(error_text: str) -> float:
    """Cari 'retryDelay': '8s' atau 'retry in 8.01...s' di pesan error Gemini."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", error_text)
    if not match:
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_text)
    return float(match.group(1)) if match else 15.0


def _call_gemini_judge(prompt: str, model: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY tidak ditemukan di environment. "
            "Set dulu: export GEMINI_API_KEY=xxxxx"
        )

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            error_text = str(exc)
            is_rate_limit = "RESOURCE_EXHAUSTED" in error_text or "429" in error_text

            if not is_rate_limit or attempt == MAX_RETRIES:
                raise

            wait = _parse_retry_delay_seconds(error_text) + 1  # +1s buffer
            print(f"   ⏳ Kena rate limit (percobaan {attempt}/{MAX_RETRIES}), "
                  f"tunggu {wait:.1f}s lalu retry...")
            time.sleep(wait)

    raise last_exc


def generate_with_context(question: str) -> tuple[str, str, str | None]:
    """
    Ambil jawaban lewat pipeline asli sistem (minci.ask_minci) -- apapun
    backend generation-nya (Ollama atau Gemini, tergantung isi modul yang
    di-import) -- lalu ambil context yang dipakai secara terpisah untuk
    keperluan penilaian faithfulness. Tidak lagi meniru manggil ollama.chat
    secara manual, supaya skrip ini tidak ikut rusak kalau backend
    generation di modul utama diganti (mis. dari Ollama ke Gemini).
    """
    normalized = minci.normalize_query(question)
    route_source, _ = minci.detect_route(normalized)

    try:
        chunks = minci.retrieve(normalized, route_source=route_source)
    except Exception as exc:
        print(f"⚠️ retrieval error saat ambil context: {exc}")
        chunks = []

    context = minci.build_context(chunks) if chunks else ""
    answer = minci.ask_minci(question)
    return answer, context, route_source


def faithfulness_score(context: str, answer: str, judge_model: str) -> tuple[float, list[dict]]:
    if not context.strip() or answer.strip() == minci.FALLBACK_TEXT:
        # Tidak ada context untuk dinilai (fallback) -> tidak relevan dihitung faithfulness
        return 1.0, []

    prompt = JUDGE_PROMPT.format(context=context, answer=answer)

    try:
        raw = _call_gemini_judge(prompt, judge_model)
    except Exception as exc:
        print(f"⚠️ Gemini API error, kasus ini dilewati: {exc}")
        return None, []

    raw = raw.strip().strip("`").removeprefix("json").strip()

    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        print(f"⚠️ Judge tidak mengembalikan JSON valid, dilewati. Raw: {raw[:200]}")
        return None, []

    if not claims:
        return 1.0, []

    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    return supported / len(claims), claims


def evaluate_faithfulness(
    dataset_path: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
) -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        cases = json.load(f)

    per_case = []
    scores = []

    for i, case in enumerate(cases):
        query = case["query"]
        answer, context, route_source = generate_with_context(query)

        if i > 0 and request_delay > 0:
            time.sleep(request_delay)  # jaga-jaga biar tidak kena limit free tier

        score, claims = faithfulness_score(context, answer, judge_model)

        if score is not None:
            scores.append(score)

        per_case.append({
            "query": query,
            "route_source": route_source,
            "answer": answer,
            "faithfulness_score": round(score, 3) if score is not None else None,
            "claims": claims,
        })

        label = "N/A (judge error)" if score is None else f"{score:.0%}"
        print(f"[{label:>18s}] '{query}'")

    n = len(scores)
    result = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": judge_model,
        "n_cases_scored": n,
        "faithfulness_avg": round(sum(scores) / n, 4) if n else None,
        "cases": per_case,
    }

    print(f"\n=== GENERATION QUALITY (Faithfulness, judge={judge_model}) ===")
    print(f"Rata-rata Faithfulness : {result['faithfulness_avg']:.2%}" if result["faithfulness_avg"] is not None else "Tidak ada skor valid")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "ground_truth.json"))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS,
                         help="Jeda (detik) antar-request ke Gemini, biar tidak kena rate limit free tier")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate_faithfulness(args.dataset, judge_model=args.judge_model, request_delay=args.request_delay)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nHasil detail disimpan ke: {args.out}")