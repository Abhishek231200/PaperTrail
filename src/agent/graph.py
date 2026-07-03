"""The PaperTrail agent: PLAN -> RETRIEVE -> SYNTHESIZE -> VERIFY, with one
retry loop back to RETRIEVE if VERIFY finds an unsupported claim.

Refusal (INSUFFICIENT_EVIDENCE) is a first-class terminal state, not an error.
"""
import time
from typing import TypedDict

import yaml
from langgraph.graph import END, StateGraph

from src.agent.llm import call_json, get_usage_log, reset_usage_log
from src.agent.prompts import PLAN_SYSTEM, SYNTHESIZE_SYSTEM, VERIFY_SYSTEM
from src.agent.trace import RunTracer
from src.corpus.db import get_papers_by_pmid
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank


def load_config() -> dict:
    return yaml.safe_load(open("config.yaml"))


class AgentState(TypedDict, total=False):
    question: str
    sub_queries: list[str]
    retrieved: list[dict]
    answer: str
    sufficient: bool
    verify_claims: list[dict]
    all_supported: bool
    should_retry: bool
    retry_count: int
    tracer: RunTracer


def format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for r in retrieved:
        lines.append(f"[PMID:{r['pmid']}] ({r.get('year')}) {r.get('title')}\n{r['text']}")
    return "\n\n".join(lines)


def plan_node(state: AgentState) -> dict:
    config = load_config()
    model = config["generation"]["model"]
    result = call_json(model, PLAN_SYSTEM, state["question"])
    sub_queries = (result.get("sub_queries") or [state["question"]])[:3]
    state["tracer"].log("plan", {"question": state["question"]}, {"sub_queries": sub_queries})
    return {"sub_queries": sub_queries, "retry_count": 0}


def retrieve_node(state: AgentState) -> dict:
    config = load_config()
    r = config["retrieval"]
    strategy = config["chunking"]["strategy"]
    embed_model = config["embedding"]["model"]

    candidates = []
    for sub_query in state["sub_queries"]:
        fused = hybrid_search(sub_query, strategy, embed_model, r["dense_top_k"], r["lexical_top_k"], r["rrf_k"])
        candidates.extend(rerank(sub_query, fused, r["reranker_model"], r["rerank_top_k"]))

    best_by_pmid: dict[int, dict] = {}
    for item in candidates:
        pmid = item["pmid"]
        if pmid not in best_by_pmid or item["rerank_score"] > best_by_pmid[pmid]["rerank_score"]:
            best_by_pmid[pmid] = item
    retrieved = sorted(best_by_pmid.values(), key=lambda x: x["rerank_score"], reverse=True)

    meta = get_papers_by_pmid([item["pmid"] for item in retrieved])
    for item in retrieved:
        m = meta.get(item["pmid"], {})
        item["title"] = m.get("title")
        item["year"] = m.get("year")

    state["tracer"].log(
        "retrieve",
        {"sub_queries": state["sub_queries"]},
        {"pmids": [item["pmid"] for item in retrieved]},
    )
    return {"retrieved": retrieved}


def synthesize_node(state: AgentState) -> dict:
    config = load_config()
    model = config["generation"]["model"]
    evidence = format_evidence(state["retrieved"])
    user = f"Question: {state['question']}\n\nEvidence:\n{evidence}"
    result = call_json(model, SYNTHESIZE_SYSTEM, user)
    answer = result.get("answer", "")
    sufficient = bool(result.get("sufficient", False))
    state["tracer"].log("synthesize", {"n_evidence": len(state["retrieved"])},
                         {"answer": answer, "sufficient": sufficient})
    return {"answer": answer, "sufficient": sufficient}


def verify_node(state: AgentState) -> dict:
    if not state.get("sufficient", False):
        state["tracer"].log("verify", {"sufficient": False}, {"skipped": True})
        return {"verify_claims": [], "all_supported": True, "should_retry": False}

    config = load_config()
    model = config["generation"]["model"]
    evidence = format_evidence(state["retrieved"])
    user = f"Question: {state['question']}\n\nAnswer: {state['answer']}\n\nEvidence:\n{evidence}"
    result = call_json(model, VERIFY_SYSTEM, user)
    claims = result.get("claims", [])
    all_supported = bool(result.get("all_supported", False))

    retry_count = state.get("retry_count", 0)
    should_retry = (not all_supported) and retry_count < 1
    update = {"verify_claims": claims, "all_supported": all_supported, "should_retry": should_retry}
    if should_retry:
        unsupported = [c["claim"] for c in claims if c.get("verdict") == "unsupported"]
        update["sub_queries"] = state["sub_queries"] + unsupported[:3]
        update["retry_count"] = retry_count + 1

    state["tracer"].log("verify", {"answer": state["answer"]},
                         {"all_supported": all_supported, "should_retry": should_retry, "claims": claims})
    return update


def route_after_verify(state: AgentState) -> str:
    return "retry" if state.get("should_retry") else "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"retry": "retrieve", "end": END})

    return graph.compile()


def run(question: str) -> AgentState:
    tracer = RunTracer(question)
    app = build_graph()

    reset_usage_log()
    wall_clock_start = time.monotonic()
    final_state = app.invoke({"question": question, "tracer": tracer})
    wall_clock_ms = (time.monotonic() - wall_clock_start) * 1000
    usage = get_usage_log()

    tracer.finish({k: v for k, v in final_state.items() if k != "tracer"})
    final_state["run_id"] = tracer.run_id
    final_state["wall_clock_ms"] = wall_clock_ms
    final_state["prompt_tokens"] = sum(u["prompt_tokens"] for u in usage)
    final_state["completion_tokens"] = sum(u["completion_tokens"] for u in usage)
    final_state["n_llm_calls"] = len(usage)
    return final_state
