"""Minimal Streamlit demo for the PaperTrail agent.

Usage: streamlit run app.py
Requires: `make up` (Postgres running) and OPENAI_API_KEY in .env.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import run as run_agent  # noqa: E402

st.set_page_config(page_title="PaperTrail", page_icon="🧬", layout="wide")

st.title("🧬 PaperTrail")
st.caption(
    "Citation-grounded biomedical RAG agent over 8,660 PubMed abstracts + 50 PMC "
    "full-text papers on GLP-1 receptor agonists (semaglutide, tirzepatide, "
    "liraglutide) — refuses when the corpus doesn't support an answer."
)

EXAMPLE_QUESTIONS = [
    "What weight reduction did SURMOUNT-1 report for 15mg tirzepatide?",
    "How do semaglutide and tirzepatide compare on weight loss efficacy?",
    "What mechanisms are proposed for GLP-1 effects on cardiovascular outcomes?",
    "What did the 2031 phase III trial of retatrutide-XR show for cardiovascular mortality?",
]

with st.sidebar:
    st.subheader("Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q
    st.divider()
    st.caption(
        "PLAN → RETRIEVE (hybrid RRF + cross-encoder rerank) → SYNTHESIZE "
        "(cited or INSUFFICIENT_EVIDENCE) → VERIFY (one retry on failure)."
    )
    st.caption("See the README for the eval harness, ablations, and failure gallery.")

question = st.text_input(
    "Ask a question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How do semaglutide and liraglutide compare on HbA1c reduction?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Planning → retrieving → synthesizing → verifying..."):
        state = run_agent(question)

    if not state.get("sufficient", False):
        st.warning("**INSUFFICIENT_EVIDENCE**")
        st.write(state.get("answer", ""))
    else:
        st.success("Answer")
        st.write(state.get("answer", ""))
        verdicts = [c.get("verdict") for c in state.get("verify_claims", [])]
        if state.get("retry_count", 0) > 0:
            st.caption(f"retried retrieval after initial verify failure ({state['retry_count']}x)")
        st.caption(f"self-verify: all_supported={state.get('all_supported')} · claim verdicts={verdicts}")

    retrieved = state.get("retrieved", [])
    with st.expander(f"Retrieved evidence ({len(retrieved)} chunks)"):
        for r in retrieved:
            pmid = r["pmid"]
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            st.markdown(f"**[PMID:{pmid}]({url})** ({r.get('year')}) {r.get('title')}")
            st.caption(f"rerank score: {r.get('rerank_score', 0):.3f}")
            snippet = r["text"][:400] + ("..." if len(r["text"]) > 400 else "")
            st.text(snippet)
            st.divider()

    with st.expander("Sub-queries used (PLAN node output)"):
        st.write(state.get("sub_queries", []))

    st.caption(
        f"run_id: {state['run_id']} · {state.get('n_llm_calls', '?')} LLM calls · "
        f"{state.get('wall_clock_ms', 0) / 1000:.1f}s · trace: runs/{state['run_id']}.jsonl"
    )
