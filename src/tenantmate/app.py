"""FastAPI app — POST /chat returns a grounded, cited answer."""

import os
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from tenantmate.retrieve import search

load_dotenv()

app = FastAPI(title="TenantMate", version="0.1.0")
client = Anthropic()

SYSTEM_PROMPT = """You are TenantMate, an assistant for NSW rental law questions.

Rules:
1. Answer ONLY from the provided context. If the context does not contain the answer, say so plainly.
2. Cite every claim with the section number from the context (e.g. "Under s 41…").
3. Use plain English a tenant can understand.
4. Always end with this disclaimer: "This is general information, not legal advice."
"""


class ChatRequest(BaseModel):
    question: str
    k: int = 5


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = search(req.question, k=req.k)

    context = "\n\n".join(
        f"[{c['chunk_id']}] Section {c['section_number']} — {c['section_title']}\n{c['text']}"
        for c in chunks
    )

    user_prompt = f"""CONTEXT:
{context}

QUESTION: {req.question}

Answer using only the context above. Cite section numbers."""

    response = client.messages.create(
        model=os.getenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001"),
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return ChatResponse(
        answer=response.content[0].text,
        citations=[c["chunk_id"] for c in chunks],
    )


@app.get("/health")
def health():
    return {"status": "ok"}