"""Retrieval layer — embed query, search pgvector, return top-k chunks."""

import os
from typing import Optional
import psycopg
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