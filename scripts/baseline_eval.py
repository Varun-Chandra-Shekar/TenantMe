"""
Capturing retrieval baseline.

Runs a fixed set of representative questions through the dense-vector
retrieval (no rerank, no hybrid) and records:
- top result chunk_id, title, similarity
- latency per query

Output: data/processed/baseline_week1.json (commit this).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
load_dotenv()

from tenantmate.retrieve import search


# Fixed test queries — keep these stable so Week 2/3 results are comparable.
# Each entry: query + the section_number you'd EXPECT a good retriever to surface.
QUERIES = [
    {"q": "How much notice is required for a rent increase?",         "expect": "41"},
    {"q": "Can my landlord enter the property without notice?",       "expect": "55"},
    {"q": "What are the rules for getting my rental bond back?",      "expect": "187"},
    {"q": "Can I keep a pet in my rental?",                            "expect": "73B"},
    {"q": "How do I end my tenancy early?",                            "expect": "110"},
    {"q": "What if my landlord won't fix urgent repairs?",             "expect": "63"},
    {"q": "What grounds does a landlord need to evict me?",            "expect": "84"},
    {"q": "How is my rental bond paid and deposited?",                 "expect": "159"},
]

K = 5
OUT_PATH = Path("data/processed/baseline_week1.json")


def run():
    results = []
    latencies = []

    for item in QUERIES:
        q = item["q"]
        expect = item["expect"]

        t0 = time.perf_counter()
        hits = search(q, k=K)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        top = hits[0]
        top_section_nums = [h["section_number"] for h in hits]
        expected_in_topk = expect in top_section_nums
        expected_rank = (
            top_section_nums.index(expect) + 1 if expected_in_topk else None
        )

        results.append({
            "query": q,
            "expected_section": expect,
            "expected_in_topk": expected_in_topk,
            "expected_rank": expected_rank,
            "top_hit": {
                "chunk_id": top["chunk_id"],
                "section_number": top["section_number"],
                "section_title": top["section_title"],
                "similarity": round(top["similarity"], 3),
            },
            "topk_section_numbers": top_section_nums,
            "latency_seconds": round(elapsed, 3),
        })

    # Summary stats
    hit_rate = sum(1 for r in results if r["expected_in_topk"]) / len(results)
    mrr = mean(
        1 / r["expected_rank"] if r["expected_rank"] else 0
        for r in results
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "Week 1 baseline — dense vector only, no rerank, no hybrid",
        "config": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "k": K,
            "vector_store": "pgvector (cosine)",
        },
        "summary": {
            "n_queries": len(results),
            "hit_rate_at_k": round(hit_rate, 3),       # % of queries where expected section is in top-k
            "mean_reciprocal_rank": round(mrr, 3),     # 1.0 = always top-1, 0.5 = avg rank 2, etc.
            "mean_latency_seconds": round(mean(latencies), 3),
            "p95_latency_seconds": round(sorted(latencies)[int(0.95 * len(latencies))], 3),
        },
        "results": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    # Print a readable summary
    print(f"\n{'='*70}")
    print(f"Week 1 baseline — {summary['generated_at']}")
    print(f"{'='*70}")
    print(f"Queries:                  {summary['summary']['n_queries']}")
    print(f"Hit rate @ {K}:               {summary['summary']['hit_rate_at_k']}")
    print(f"Mean Reciprocal Rank:     {summary['summary']['mean_reciprocal_rank']}")
    print(f"Mean latency:             {summary['summary']['mean_latency_seconds']}s")
    print(f"P95 latency:              {summary['summary']['p95_latency_seconds']}s")
    print(f"\nDetail:")
    for r in results:
        flag = "Yes" if r["expected_in_topk"] else "No"
        rank = f"#{r['expected_rank']}" if r['expected_rank'] else "MISS"
        print(f"  {flag} {rank:>4}  s{r['top_hit']['section_number']:<5} "
              f"({r['top_hit']['similarity']})  {r['query']}")
    print(f"\n→ Saved to {OUT_PATH}")


if __name__ == "__main__":
    run()