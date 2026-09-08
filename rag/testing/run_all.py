"""
Menjalankan ketiga tahap pengujian sekaligus:
    1. Retrieval Quality (Hit Rate, Precision@k)
    2. Generation Quality (Faithfulness)
    3. Guardrail / Stress Test

Hasil detail tiap tahap disimpan ke testing/results/<timestamp>/*.json,
supaya bisa dibandingkan antar-run tiap kali ada perubahan di sistem
(mis. perubahan ROUTES, threshold, atau system prompt).

Cara pakai:
    python testing/run_all.py
"""

from __future__ import annotations
import json
import os
from datetime import datetime

from eval_retrieval import evaluate_retrieval
from eval_faithfulness import evaluate_faithfulness
from eval_guardrail import run_guardrail_test

HERE = os.path.dirname(__file__)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "results", ts)
    os.makedirs(out_dir, exist_ok=True)

    print("\n########## TAHAP 1: RETRIEVAL QUALITY ##########")
    retrieval_result = evaluate_retrieval(os.path.join(HERE, "ground_truth.json"), k=3)

    print("\n########## TAHAP 2: GENERATION QUALITY (FAITHFULNESS) ##########")
    faithfulness_result = evaluate_faithfulness(os.path.join(HERE, "ground_truth.json"))

    print("\n########## TAHAP 3: GUARDRAIL / STRESS TEST ##########")
    guardrail_result = run_guardrail_test(os.path.join(HERE, "stress_cases.json"))

    for name, result in [
        ("retrieval", retrieval_result),
        ("faithfulness", faithfulness_result),
        ("guardrail", guardrail_result),
    ]:
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n\n=== RINGKASAN ({ts}) ===")
    print(f"Hit Rate@3        : {retrieval_result['hit_rate']:.2%}")
    print(f"Precision@3 avg   : {retrieval_result['precision_at_k_avg']:.2%}")
    if faithfulness_result["faithfulness_avg"] is not None:
        print(f"Faithfulness avg  : {faithfulness_result['faithfulness_avg']:.2%}")
    print(f"Guardrail pass    : {guardrail_result['pass_rate']:.2%}")
    print(f"\nHasil lengkap disimpan di: {out_dir}")


if __name__ == "__main__":
    main()
