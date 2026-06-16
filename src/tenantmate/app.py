"""FastAPI app — POST /chat returns a grounded, cited answer via the agent."""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from tenantmate.agent.graph import run_agent

load_dotenv()

app = FastAPI(title="TenantMate", version="0.2.0")
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")



class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]            # chunk_ids from retrieval (the Act sections)
    rule_citations: list[str] = []  # section refs from tools (calculator)
    tools_used: list[str]
    tool_results: list[dict] = []   # structured tool outputs for UI consumers
    hops: int


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = run_agent(req.question)

    citations = [c["chunk_id"] for c in state["retrieved_chunks"]]
    tools_used = [t["tool"] for t in state["tool_results"]]

    # Collect rule citations from every tool that produced them
    rule_citations: list[str] = []
    for t in state["tool_results"]:
        rule_citations.extend(t.get("rule_citations", []))
    # Deduplicate while preserving order
    seen = set()
    rule_citations = [c for c in rule_citations if not (c in seen or seen.add(c))]

    return ChatResponse(
        answer=state["final_answer"],
        citations=citations,
        rule_citations=rule_citations,
        tools_used=tools_used,
        tool_results=state["tool_results"],
        hops=state["hop_count"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}