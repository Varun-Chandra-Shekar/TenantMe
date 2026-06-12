"""
Compare two eval runs side-by-side.

Usage:
    python scripts/compare_results.py week1_baseline week2_hybrid
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("data/eval/results")


def load(stage):
    path = RESULTS_DIR / f"results_{stage}.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run eval_retrieval.py --stage {stage} first.")
    return json.loads(path.read_text())


def compare(stage_a, stage_b):
    a, b = load(stage_a), load(stage_b)

    print(f"\n{'='*70}")
    print(f"  {stage_a}  →  {stage_b}")
    print(f"{'='*70}")

    # Headline metrics
    print(f"\n{'Metric':<25} {'before':>12}  {'after':>12}  {'Δ':>10}")
    print(f"{'-'*60}")
    for key, label in [
        ("hit_rate_at_k", "Hit rate @ k"),
        ("mean_reciprocal_rank", "MRR"),
        ("mean_latency_s", "Mean latency (s)"),
    ]:
        v_a = a["overall"][key]
        v_b = b["overall"][key]
        delta = v_b - v_a
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
        print(f"{label:<25} {v_a:>12.3f}  {v_b:>12.3f}  {arrow} {delta:+.3f}")

    # By difficulty
    print(f"\nBy difficulty:")
    print(f"  {'level':<8} {'before':>12}  {'after':>12}  {'Δ':>10}")
    for level in ("easy", "medium", "hard"):
        ha = a["by_difficulty"].get(level, {}).get("hit_rate", 0)
        hb = b["by_difficulty"].get(level, {}).get("hit_rate", 0)
        delta = hb - ha
        print(f"  {level:<8} {ha:>12.3f}  {hb:>12.3f}  {delta:+.3f}")

    # By topic
    print(f"\nBy topic:")
    print(f"  {'topic':<18} {'before':>12}  {'after':>12}  {'Δ':>10}")
    topics = sorted(set(a["by_topic"]) | set(b["by_topic"]))
    for topic in topics:
        ha = a["by_topic"].get(topic, {}).get("hit_rate", 0)
        hb = b["by_topic"].get(topic, {}).get("hit_rate", 0)
        delta = hb - ha
        flag = "  ↑" if delta > 0.01 else ("  ↓" if delta < -0.01 else "")
        print(f"  {topic:<18} {ha:>12.3f}  {hb:>12.3f}  {delta:+.3f}{flag}")

    # Per-query changes
    res_a = {r["id"]: r for r in a["results"]}
    res_b = {r["id"]: r for r in b["results"]}

    became_hit, became_miss, rank_changes = [], [], []
    for qid in res_a:
        ra, rb = res_a[qid], res_b[qid]
        if not ra["hit"] and rb["hit"]:
            became_hit.append((qid, ra["query"], rb["best_rank"]))
        elif ra["hit"] and not rb["hit"]:
            became_miss.append((qid, ra["query"], ra["best_rank"]))
        elif ra["hit"] and rb["hit"] and ra["best_rank"] != rb["best_rank"]:
            rank_changes.append((qid, ra["query"], ra["best_rank"], rb["best_rank"]))

    print(f"\nQueries that flipped:")
    print(f"  Miss → Hit: {len(became_hit)}")
    for qid, q, r in became_hit:
        print(f"    {qid}  rank={r}  {q[:60]}")
    print(f"  Hit → Miss: {len(became_miss)}")
    for qid, q, r in became_miss:
        print(f"    {qid}  was rank={r}  {q[:60]}")
    print(f"  Rank changed (both hits): {len(rank_changes)}")
    for qid, q, ra, rb in rank_changes[:5]:
        print(f"    {qid}  {ra}→{rb}  {q[:55]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    compare(args.before, args.after)