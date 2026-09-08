"""
Tahap 1: Retrieval Quality Evaluation (Ground-Truth Evaluation)
Mengukur Hit Rate@k dan Precision@k dari komponen retrieval (routing + ChromaDB).

Cara pakai:
    python testing/eval_retrieval.py
    python testing/eval_retrieval.py --k 5 --dataset testing/ground_truth.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", "webhook", ".env")
load_dotenv(dotenv_path=dotenv_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ganti "backup" sesuai nama file utama sistem RAG kamu (mis. "query" kalau
# sudah kamu rename, atau biarkan "backup" kalau memang itu nama filenya).
MODULE_NAME = os.environ.get("MODULE_NAME")
minci = __import__(MODULE_NAME)


def evaluate_retrieval(dataset_path: str, k: int = 3) -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        cases = json.load(f)

    per_case = []
    hits = 0
    precisions = []

    for case in cases:
        query, expected = case["query"], case["expected_source"]

        route_source, route_reason = minci.detect_route(query)
        chunks = minci.retrieve(query, route_source=route_source)
        top_k = chunks[:k]
        sources = [c["metadata"].get("source") for c in top_k]

        hit = expected in sources
        precision = sources.count(expected) / len(top_k) if top_k else 0.0

        hits += hit
        precisions.append(precision)

        per_case.append({
            "query": query,
            "expected_source": expected,
            "route_source": route_source,
            "route_reason": route_reason,
            "top_k_sources": sources,
            "hit": hit,
            "precision_at_k": round(precision, 3),
        })

        status = "✅" if hit else "❌"
        print(f"{status} [{expected:15s}] route={route_source or 'GLOBAL':12s} "
              f"top{k}={sources} | '{query}'")

    n = len(cases)
    result = {
        "timestamp": datetime.now().isoformat(),
        "k": k,
        "n_cases": n,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "precision_at_k_avg": round(sum(precisions) / n, 4) if n else 0.0,
        "cases": per_case,
    }

    print(f"\n=== RETRIEVAL QUALITY (k={k}, n={n}) ===")
    print(f"Hit Rate@{k}      : {result['hit_rate']:.2%}")
    print(f"Precision@{k} avg : {result['precision_at_k_avg']:.2%}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "ground_truth.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=None, help="Simpan hasil detail ke file JSON")
    args = parser.parse_args()

    result = evaluate_retrieval(args.dataset, k=args.k)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nHasil detail disimpan ke: {args.out}")
