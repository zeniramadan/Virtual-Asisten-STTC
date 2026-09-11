"""
Tahap 1: Retrieval Quality Evaluation (Ground-Truth Evaluation)
Untuk obsidian_rag.py -- mengukur Hit Rate@k dan Precision@k berdasarkan
judul note (metadata['title']) yang seharusnya muncul di top-k hasil, serta
Recall@k dan NDCG@k.

PENTING: ground_truth.json di folder dataset isinya CONTOH/placeholder. Sesuaikan
"expected_title" dengan judul note yang SEBENARNYA ada di vault Obsidian kamu
(nilai metadata['title'] yang dihasilkan saat proses ingest ke ChromaDB).

Cara pakai:
    python testing_obsidian/eval_retrieval.py --k 3
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

MODULE_NAME = os.environ.get("MODULE_NAME")  # ganti kalau nama file utama kamu berbeda
rag = __import__(MODULE_NAME)


def evaluate_retrieval(dataset_path: str, k: int = 3) -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        cases = json.load(f)

    per_case = []
    hits = 0
    precisions = []
    recalls = []
    ndcgs = []

    for case in cases:
        query, expected = case["query"], case["expected_title"]

        chunks = rag.retrieve_with_debug(query)
        top_k = chunks[:k]
        titles = [c["metadata"].get("title") for c in top_k]

        hit = expected in titles
        precision = titles.count(expected) / len(top_k) if top_k else 0.0
        recall = 1.0 if hit else 0.0
        rank = titles.index(expected) + 1 if hit else None
        ndcg = 1.0 / math.log2(rank + 1) if rank else 0.0

        hits += hit
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

        per_case.append({
            "query": query,
            "expected_title": expected,
            "top_k_titles": titles,
            "hit": hit,
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(recall, 3),
            "ndcg_at_k": round(ndcg, 3),
            "rank": rank,
        })

        status = "✅" if hit else "❌"
        print(f"{status} [{expected:35s}] top{k}={titles} | '{query}'")

    n = len(cases)
    result = {
        "timestamp": datetime.now().isoformat(),
        "k": k,
        "n_cases": n,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "precision_at_k_avg": round(sum(precisions) / n, 4) if n else 0.0,
        "recall_at_k_avg": round(sum(recalls) / n, 4) if n else 0.0,
        "ndcg_at_k_avg": round(sum(ndcgs) / n, 4) if n else 0.0,
        "cases": per_case,
    }

    print(f"\n=== RETRIEVAL QUALITY (k={k}, n={n}) ===")
    print(f"Hit Rate@{k}      : {result['hit_rate']:.2%}")
    print(f"Precision@{k} avg : {result['precision_at_k_avg']:.2%}")
    print(f"Recall@{k} avg    : {result['recall_at_k_avg']:.2%}")
    print(f"NDCG@{k} avg      : {result['ndcg_at_k_avg']:.2%}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "ground_truth.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=None, help="Simpan hasil detail ke file JSON")
    args = parser.parse_args()

    result = evaluate_retrieval(args.dataset, k=args.k)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nHasil detail disimpan ke: {args.out}")
