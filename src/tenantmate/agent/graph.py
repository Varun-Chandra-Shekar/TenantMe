"""
LangGraph state machine for TenantMate.

Replaces the linear /chat flow with a planner-driven agent that decides
per query whether to retrieve, call the calculator, or answer directly.

Bounded at MAX_HOPS to keep latency/cost predictable.
"""

import json
import os
import re
from datetime import date
from typing import TypedDict, Optional

from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from tenantmate.retrieve import search_full
from tenantmate.agent.tools import check_rent_increase


MAX_HOPS = 4
ACTIONS = ("retrieve", "use_calculator", "answer")


# ─── State ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    retrieved_chunks: list[dict]
    tool_results: list[dict]
    next_action: str
    final_answer: Optional[str]
    hop_count: int


# ─── Helpers ────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers an LLM may add."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─── Planner ────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are the planner of a NSW rental-law agent.

Decide what the agent should do next, given the user query and what
has already been gathered.

Available actions:
- retrieve: search the Residential Tenancies Act for relevant sections
- use_calculator: invoke the rent-increase calculator (needs dates from the query)
- answer: produce the final answer

Use the calculator ONLY when the query involves a specific proposed rent
change with dates (e.g. "my rent is going from $600 to $680 next month").

Use retrieve when you don't yet have relevant law in the context.

Use answer when you have enough information to respond.

RULES:
- If a tool has ALREADY FAILED, do NOT pick it again. Use the retrieved law instead.
- If the calculator has already run successfully, do not run it again.
- If you have both retrieved law AND tool results (success or failure), pick "answer".

Respond in this exact format:
<action>retrieve|use_calculator|answer</action>
<reason>one short sentence</reason>"""


def _parse_planner(text: str) -> str:
    """Naive but reliable XML-ish parsing — no library dependency."""
    if "<action>" not in text:
        return "answer"
    inner = text.split("<action>")[1].split("</action>")[0].strip()
    return inner if inner in ACTIONS else "answer"


def _summarise_state(state: AgentState) -> str:
    """Make tool failures visible to the planner so it won't retry."""
    parts = []

    if state["retrieved_chunks"]:
        names = ", ".join(c["chunk_id"] for c in state["retrieved_chunks"][:5])
        parts.append(f"- Retrieved chunks: {names}")
    else:
        parts.append("- No chunks retrieved yet.")

    if state["tool_results"]:
        failed = sorted({r["tool"] for r in state["tool_results"] if "error" in r})
        succeeded = sorted({r["tool"] for r in state["tool_results"] if "error" not in r})
        if succeeded:
            parts.append(f"- Tools succeeded: {succeeded}")
        if failed:
            parts.append(f"- Tools ALREADY FAILED (do NOT retry): {failed}")
    else:
        parts.append("- No tools used yet.")

    return "\n".join(parts)


def planner_node(state: AgentState) -> dict:
    """Decide the next action by asking the LLM. Bounded by MAX_HOPS."""
    new_hop = state["hop_count"] + 1

    # Hard stop BEFORE wasting another LLM call.
    if new_hop > MAX_HOPS:
        return {"next_action": "answer", "hop_count": new_hop}

    client = Anthropic()
    summary = _summarise_state(state)

    response = client.messages.create(
        model=os.getenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001"),
        max_tokens=120,
        system=PLANNER_PROMPT,
        messages=[{"role": "user", "content":
            f"Query: {state['query']}\n\nProgress so far:\n{summary}\n\n"
            "Pick the next action."
        }],
    )
    decision = _parse_planner(response.content[0].text)
    return {"next_action": decision, "hop_count": new_hop}


# ─── Retrieve node ──────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    """Run the existing search_full pipeline."""
    chunks = search_full(state["query"], k=5)
    return {"retrieved_chunks": chunks}


# ─── Calculator node ────────────────────────────────────────────────

CALC_EXTRACT_PROMPT = """Extract the dates from this NSW rent-increase
question and return JSON with these fields. Use ISO format YYYY-MM-DD.
If a field isn't mentioned, use null.

{
  "notice_date": "YYYY-MM-DD or null",
  "increase_takes_effect_on": "YYYY-MM-DD or null",
  "tenancy_start_date": "YYYY-MM-DD or null",
  "last_increase_date": "YYYY-MM-DD or null"
}

Return ONLY the JSON. No prose, no markdown fences."""


def calculator_node(state: AgentState) -> dict:
    """Extract dates from the query, call the calculator, attach the result."""
    client = Anthropic()
    response = client.messages.create(
        model=os.getenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001"),
        max_tokens=200,
        system=CALC_EXTRACT_PROMPT,
        messages=[{"role": "user", "content": state["query"]}],
    )
    raw = response.content[0].text
    cleaned = _strip_code_fences(raw)

    try:
        params = json.loads(cleaned)
    except json.JSONDecodeError:
        # Include the raw response so the next debugger can see what came back.
        return {"tool_results": state["tool_results"] + [
            {"tool": "rent_calculator",
             "error": f"Could not parse dates. LLM returned: {raw[:200]}"}
        ]}

    # Best-effort: only call the calculator if we have the minimum fields.
    if not (params.get("notice_date")
            and params.get("increase_takes_effect_on")
            and params.get("tenancy_start_date")):
        return {"tool_results": state["tool_results"] + [
            {"tool": "rent_calculator",
             "error": f"Insufficient dates in the query. Extracted: {params}"}
        ]}

    result = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date.fromisoformat(params["notice_date"]),
        increase_takes_effect_on=date.fromisoformat(params["increase_takes_effect_on"]),
        tenancy_start_date=date.fromisoformat(params["tenancy_start_date"]),
        last_increase_date=(date.fromisoformat(params["last_increase_date"])
                            if params.get("last_increase_date") else None),
    )
    return {"tool_results": state["tool_results"] + [{
        "tool": "rent_calculator",
        "is_allowed": result.is_allowed,
        "reasons": result.reasons,
        "earliest_lawful_date": result.earliest_lawful_increase_date,
        "rule_citations": result.rule_citations,
    }]}


# ─── Answer node ────────────────────────────────────────────────────

ANSWER_PROMPT = """You are TenantMate, an assistant for NSW rental law.

Use the retrieved context and any tool results to answer the user's
question. Cite the section numbers from the retrieved context, and any
rule citations from the calculator output.

End with: "This is general information, not legal advice."
"""


def answer_node(state: AgentState) -> dict:
    """Compose the final grounded answer."""
    client = Anthropic()

    context_blob = "\n\n".join(
        f"[{c['chunk_id']}] {c.get('section_title','')} — {c['text']}"
        for c in state["retrieved_chunks"]
    ) or "(no retrieved law)"

    tool_blob = json.dumps(state["tool_results"], indent=2, default=str) \
        if state["tool_results"] else "(no tool results)"

    response = client.messages.create(
        model=os.getenv("LLM_MODEL_DEV", "claude-haiku-4-5-20251001"),
        max_tokens=600,
        system=ANSWER_PROMPT,
        messages=[{"role": "user", "content":
            f"QUESTION: {state['query']}\n\n"
            f"RETRIEVED LAW:\n{context_blob}\n\n"
            f"TOOL RESULTS:\n{tool_blob}\n\n"
            "Answer the question grounded in the law and any tool output."
        }],
    )
    return {"final_answer": response.content[0].text}


# ─── Edges (routing) ────────────────────────────────────────────────

def route_from_planner(state: AgentState) -> str:
    """Map the planner's decision to the next node name."""
    return {
        "retrieve": "retrieve",
        "use_calculator": "calculator",
        "answer": "answer",
    }.get(state["next_action"], "answer")


# ─── Build the graph ────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("calculator", calculator_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", route_from_planner, {
        "retrieve": "retrieve",
        "calculator": "calculator",
        "answer": "answer",
    })

    # After any tool, loop back to the planner.
    g.add_edge("retrieve", "planner")
    g.add_edge("calculator", "planner")
    g.add_edge("answer", END)

    return g.compile()


_GRAPH = None

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ─── Public entry point ─────────────────────────────────────────────

def run_agent(query: str) -> dict:
    """Run the agent end to end. Returns the final state."""
    graph = get_graph()
    initial = AgentState(
        query=query,
        retrieved_chunks=[],
        tool_results=[],
        next_action="",
        final_answer=None,
        hop_count=0,
    )
    return graph.invoke(initial)