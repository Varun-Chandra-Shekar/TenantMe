"""
TenantMate — Streamlit UI for the agentic RAG system.
Talks to the FastAPI /chat endpoint on Railway.
"""

import os
import time
import requests
import streamlit as st


# ─── Config ──────────────────────────────────────────────────────────

# Default points at the live Railway API. Override via Streamlit Cloud
# secrets or local env var.
DEFAULT_API = "https://tenantme-production.up.railway.app"

try:
    API_BASE = st.secrets.get("API_BASE")
except Exception:
    API_BASE = None
API_BASE = API_BASE or os.getenv("API_BASE", DEFAULT_API)

st.set_page_config(
    page_title="TenantMate",
    page_icon="🏠",
    layout="centered",
)


# ─── Helpers ─────────────────────────────────────────────────────────

def render_metadata(entry):
    """Three expandable columns: citations, rule cites, tool results."""
    cols = st.columns([1, 1, 1])

    with cols[0]:
        if entry["citations"]:
            with st.expander(f"📋 {len(entry['citations'])} citations"):
                for c in entry["citations"]:
                    st.code(c, language=None)

    with cols[1]:
        if entry.get("rule_citations"):
            with st.expander(f"⚖️ {len(entry['rule_citations'])} rules"):
                for r in entry["rule_citations"]:
                    st.write(f"• {r}")

    with cols[2]:
        if entry["tools_used"]:
            with st.expander(f"🔧 {len(entry['tools_used'])} tools"):
                for t in entry["tool_results"]:
                    if "error" in t:
                        st.error(f"{t['tool']}: {t['error']}")
                    else:
                        st.success(f"**{t['tool']}**")
                        st.json({k: v for k, v in t.items() if k != "tool"})

    st.caption(f"⏱ {entry['hops']} hops · {entry.get('latency_ms', 0)} ms")


# ─── Header ──────────────────────────────────────────────────────────

st.title("🏠 TenantMate")
st.caption("Ask a question about NSW residential tenancy law. Answers cite sections of the Act.")


# ─── State ───────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


# ─── Render history ──────────────────────────────────────────────────

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        render_metadata(entry)


# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Try asking…")
    examples = [
        "How much notice for a rent increase in NSW?",
        "Can the landlord enter without telling me?",
        "How do I get my rental bond back?",
        "Can I keep a pet in my rental?",
        "My tenancy started 1 March 2024. The landlord gave notice on 1 May 2026 that rent goes from $600 to $680 from 1 June 2026. The last increase was 1 December 2025. Is this allowed?",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(
        "Portfolio project: agentic RAG over the NSW Residential Tenancies Act 2010. "
        "General information only — not legal advice."
    )
    st.caption(f"API: `{API_BASE}`")


# ─── Input ───────────────────────────────────────────────────────────

user_question = st.chat_input("Ask a question…")
pending = st.session_state.pop("pending_question", None)
question = pending or user_question

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                latency_ms = int((time.time() - t0) * 1000)

                entry = {
                    "question": question,
                    "answer": data["answer"],
                    "citations": data.get("citations", []),
                    "rule_citations": data.get("rule_citations", []),
                    "tools_used": data.get("tools_used", []),
                    "tool_results": data.get("tool_results", []),
                    "hops": data.get("hops", 0),
                    "latency_ms": latency_ms,
                }
                st.session_state.history.append(entry)

                st.markdown(data["answer"])
                render_metadata(entry)

            except requests.exceptions.Timeout:
                st.error("Request timed out — the server may be cold-starting. Try again in 30s.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")