"""Retrieval layer — embed query, search pgvector, return top-k chunks."""

import os
from typing import Optional
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=Warning)
from anthropic import Anthropic
_anthropic_client = None

def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client

REWRITE_SYSTEM = """You rewrite tenant questions into search queries for the NSW Residential Tenancies Act 2010.

The Act uses formal legal language. Rewrite the user's plain-English question into the keywords and phrasing the Act itself would use.

Rules:
- Output ONLY the rewritten query. No explanation, no quotes, no preamble.
- Keep it short — 5 to 12 words.
- Use legal terms the Act uses (e.g. "termination" not "eviction", "rental bond" not "deposit").
- Drop conversational words ("how do I", "can my", "what if").

Examples:
User: How much notice for a rent increase?
Rewrite: rent increase notice period

User: Can the landlord enter without telling me?
Rewrite: landlord access premises without consent

User: How do I get my bond back?
Rewrite: rental bond claim payment tenant

User: What grounds does a landlord need to evict me?
Rewrite: grounds for termination notice landlord"""


def rewrite_query(query: str) -> str:
    """Rewrite a user's plain-English query into corpus-aligned legal terms."""
    client = _get_anthropic()
    response = client.messages.create(
        model=os.getenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001"),
        max_tokens=60,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text.strip()


# Load model once at import time (not per-query)
_MODEL: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL


def search(query: str, k: int = 5) -> list[dict]:
    """
    Return top-k chunks most similar to the query.
    Each result has chunk_id, section_title, part, similarity, text.
    """
    model = _get_model()
    q_vec = model.encode(query)

    sql = """
        SELECT chunk_id, section_title, part, schedule, section_number,
               1 - (embedding <=> %s) AS similarity,
               text
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s;
    """

    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, (q_vec, q_vec, k))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "section_title": r[1],
            "part": r[2],
            "schedule": r[3],
            "section_number": r[4],
            "similarity": float(r[5]),
            "text": r[6],
        }
        for r in rows
    ]


"""
Seach function - the concept of BM25
plainto_tsquery('english', %s) — turns natural-language input ("how much notice for a rent increase") into a tsquery, handling stemming and stopwords.
The english config matches the one we used on the column.
@@ — Postgres's "matches" operator, returns true if the doc contains the query terms.
ts_rank_cd — cover density ranking; similar to BM25 family. Higher = more relevant.
"""

def search_bm25(query: str, k: int = 5) -> list[dict]:
    """
    Keyword search using Postgres full-text search(ts_rank_cd).
    Complements dense vector seach - strong on exact terms, weak on synonyms
    """

    sql = """
        SELECT chunk_id, section_title, part, schedule, section_number,
            ts_rank_cd(text_tsv, plainto_tsquery('english', %s)) AS score,
            text
        FROM chunks
        WHERE text_tsv @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s;
    """

    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, k))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "section_title": r[1],
            "part": r[2],
            "schedule": r[3],
            "section_number": r[4],
            "score": float(r[5]),
            "text": r[6],
        }
        for r in rows
    ]

def search_hybrid(query: str, k: int = 5, candidates: int = 20, rrf_k: int = 60) -> list[dict]:
    """
    Hybrid retrieval: dense vector + BM25, merged with Reciprocal Rank Fusion.

    candidates: how many to pull from EACH retriever before merging.
    rrf_k:      RRF constant (60 is the canonical default).
    """
    dense_hits = search(query, k=candidates)
    bm25_hits = search_bm25(query, k=candidates)

    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)
        chunks_by_id[cid] = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)
        chunks_by_id.setdefault(cid, hit)

    ranked_ids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:k]
    return [
        {**chunks_by_id[cid], "rrf_score": scores[cid]}
        for cid in ranked_ids
    ]


def search_hybrid_rewritten(query: str, k: int = 5, candidates: int = 20, rrf_k: int = 60) -> list[dict]:
    """Hybrid retrieval with LLM query rewriting upfront."""
    rewritten = rewrite_query(query)
    results = search_hybrid(rewritten, k=k, candidates=candidates, rrf_k=rrf_k)
    # Attach the rewritten query so callers (and eval) can inspect it
    for r in results:
        r["original_query"] = query
        r["rewritten_query"] = rewritten
    return results