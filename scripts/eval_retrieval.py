"""
Run retrieval against the golden set and report hit rate, MRR, latency.
Re-run this after every retrieval change to measure improvement.

Output: data/eval/results_<stage>.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
load_dotenv()
from tenantmate.retrieve import search, search_bm25, search_hybrid, search_hybrid_rewritten

RETRIEVERS = {
    "dense": search,
    "bm25": search_bm25,
    "hybrid": search_hybrid,
    "hybrid_rewritten": search_hybrid_rewritten,
}


GOLDEN_PATH = Path("data/eval/golden_set.jsonl")
OUT_DIR = Path("data/eval/results")
K = 5


def evaluate(stage: str, retriever_name: str = "dense"):
    retriever_fn = RETRIEVERS[retriever_name]
    queries = [json.loads(line) for line in open(GOLDEN_PATH)]
    print(f"Loaded {len(queries)} queries from {GOLDEN_PATH}")
    print(f"Retriever: {retriever_name}")

    results = []
    latencies = []

    for q in queries:
        t0 = time.perf_counter()
        hits = retriever_fn(q["query"], k=K)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        top_section_nums = [h["section_number"] for h in hits]
        expected = q["expected_sections"]

        # A "hit" = ANY expected section appears in top-k
        matched = [s for s in expected if s in top_section_nums]
        is_hit = len(matched) > 0

        # Best rank of any expected section (1-indexed, None if missed)
        best_rank = None
        for i, sec in enumerate(top_section_nums):
            if sec in expected:
                best_rank = i + 1
                break

        results.append({
            "id": q["id"],
            "query": q["query"],
            "expected_sections": expected,
            "topic": q["topic"],
            "difficulty": q["difficulty"],
            "topk_sections": top_section_nums,
            "matched": matched,
            "hit": is_hit,
            "best_rank": best_rank,
            "top_hit": {
                "chunk_id": hits[0]["chunk_id"],
                "section_number": hits[0]["section_number"],
                "section_title": hits[0]["section_title"],
                "similarity": round(hits[0]["similarity"], 3),
            },
            "latency_seconds": round(elapsed, 3),
        })

    # Summary
    hit_rate = sum(1 for r in results if r["hit"]) / len(results)
    mrr = mean(1 / r["best_rank"] if r["best_rank"] else 0 for r in results)

    # Slice by difficulty and topic
    by_difficulty = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            by_difficulty[diff] = {
                "count": len(subset),
                "hit_rate": round(sum(1 for r in subset if r["hit"]) / len(subset), 3),
                "mrr": round(mean(1 / r["best_rank"] if r["best_rank"] else 0 for r in subset), 3),
            }

    by_topic = {}
    for topic in set(r["topic"] for r in results):
        subset = [r for r in results if r["topic"] == topic]
        by_topic[topic] = {
            "count": len(subset),
            "hit_rate": round(sum(1 for r in subset if r["hit"]) / len(subset), 3),
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "config": {
            "retriever": retriever_name,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "k": K,
            "vector_store": "pgvector (cosine)",
        },
        "overall": {
            "n_queries": len(results),
            "hit_rate_at_k": round(hit_rate, 3),
            "mean_reciprocal_rank": round(mrr, 3),
            "mean_latency_s": round(mean(latencies), 3),
            "p95_latency_s": round(sorted(latencies)[int(0.95 * len(latencies))], 3),
        },
        "by_difficulty": by_difficulty,
        "by_topic": by_topic,
        "results": results,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"results_{stage}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Print readable summary
    print(f"\n{'='*70}")
    print(f"  Stage: {stage}")
    print(f"  Generated: {summary['generated_at']}")
    print(f"{'='*70}")
    print(f"  Queries:                {len(results)}")
    print(f"  Hit rate @ {K}:           {summary['overall']['hit_rate_at_k']}")
    print(f"  MRR:                    {summary['overall']['mean_reciprocal_rank']}")
    print(f"  Mean latency:           {summary['overall']['mean_latency_s']}s")
    print(f"\n  By difficulty:")
    for diff, stats in by_difficulty.items():
        print(f"    {diff:8s} hit={stats['hit_rate']}  mrr={stats['mrr']}  (n={stats['count']})")
    print(f"\n  By topic:")
    for topic, stats in sorted(by_topic.items()):
        print(f"    {topic:18s} hit={stats['hit_rate']}  (n={stats['count']})")
    print(f"\n  Misses ({sum(1 for r in results if not r['hit'])}):")
    for r in results:
        if not r["hit"]:
            print(f"    {r['id']}  {r['query'][:55]:<55}  expected={r['expected_sections']}  got={r['topk_sections'][:3]}")
    print(f"\n  → Saved to {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="week1_baseline")
    parser.add_argument("--retriever", default="dense", choices=list(RETRIEVERS.keys()))
    args = parser.parse_args()
    evaluate(args.stage, args.retriever)