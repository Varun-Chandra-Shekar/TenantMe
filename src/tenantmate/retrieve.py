"""Retrieval layer — embed query, search pgvector, return top-k chunks."""

import os
from typing import Optional
import psycopgs
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=Warning)

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


#Seach function - the concept of BM25

"""
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