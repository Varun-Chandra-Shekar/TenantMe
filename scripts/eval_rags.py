"""
Ragas evaluation — score the full RAG pipeline on faithfulness,
answer relevancy, context precision, context recall.

Usage:
    python scripts/eval_ragas.py --stage week2_full --retriever full
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from tenantmate.retrieve import (
    search, search_bm25, search_hybrid, search_hybrid_rewritten,
    search_hybrid_rerank, search_full,
)
from langchain_huggingface import HuggingFaceEmbeddings


RETRIEVERS = {
    "dense": search,
    "bm25": search_bm25,
    "hybrid": search_hybrid,
    "hybrid_rewritten": search_hybrid_rewritten,
    "hybrid_rerank": search_hybrid_rerank,
    "full": search_full,
}

GOLDEN_PATH = Path("data/eval/golden_set.jsonl")
OUT_DIR = Path("data/eval/results")

SYSTEM_PROMPT = """You are TenantMate, an assistant for NSW rental law questions.

Rules:
1. Answer ONLY from the provided context. If the context doesn't contain the answer, say so plainly.
2. Cite every claim with the section number (e.g. "Under s 41...").
3. Use plain English a tenant can understand.
4. End with: "This is general information, not legal advice."
"""


def build_dataset(queries, retriever_fn, k=5):
    """Run retrieval + answer generation for every golden query."""
    anthropic = Anthropic()
    rows = []

    for i, q in enumerate(queries, start=1):
        question = q["query"]
        print(f"[{i}/{len(queries)}] {question[:60]}...")

        # Retrieve
        chunks = retriever_fn(question, k=k)
        contexts = [
            f"[s{c['section_number']}] {c['text']}"
            for c in chunks
        ]

        # Generate answer with the same prompt /chat uses
        context_blob = "\n\n".join(contexts)
        user_prompt = f"CONTEXT:\n{context_blob}\n\nQUESTION: {question}\n\nAnswer using only the context above. Cite section numbers."

        response = anthropic.messages.create(
            model=os.getenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001"),
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer = response.content[0].text

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q["reference_answer"],
            # extras for our own analysis later
            "id": q["id"],
            "topic": q["topic"],
            "difficulty": q["difficulty"],
        })

    return Dataset.from_list(rows)


def run_ragas(dataset, judge_model_name):
    """Run Ragas with Claude as the judge."""
    judge = ChatAnthropic(
        model=judge_model_name,
        max_tokens=1024,
        temperature=0,
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    )
    return result


def main(stage, retriever_name):
    queries = [json.loads(line) for line in open(GOLDEN_PATH)]
    print(f"Loaded {len(queries)} queries\n")

    retriever_fn = RETRIEVERS[retriever_name]
    judge_model = os.getenv("LLM_MODEL_JUDGE", "claude-opus-4-7")

    print(f"Stage:    {stage}")
    print(f"Retriever:{retriever_name}")
    print(f"Judge:    {judge_model}\n")

    t0 = time.time()

    # Step 1: build the eval dataset (runs retrieval + LLM answer for each query)
    print("Building dataset (retrieval + answer generation)...")
    dataset = build_dataset(queries, retriever_fn)
    build_time = time.time() - t0

    # Step 2: run Ragas
    print(f"\nDataset built in {build_time:.1f}s. Running Ragas...\n")
    result = run_ragas(dataset, judge_model)

    print(f"\n{'='*60}")
    print(f"  Ragas — {stage}")
    print(f"{'='*60}")
    print(result)

    # Save full results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ragas_{stage}.json"
    df = result.to_pandas()
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "retriever": retriever_name,
        "judge_model": judge_model,
        "n_queries": len(queries),
        "scores": {
            "faithfulness":      float(df["faithfulness"].mean()),
            "answer_relevancy":  float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall":    float(df["context_recall"].mean()),
        },
        "per_query": df.to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n→ Saved to {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="ragas_baseline")
    parser.add_argument("--retriever", default="full", choices=list(RETRIEVERS.keys()))
    args = parser.parse_args()
    main(args.stage, args.retriever)