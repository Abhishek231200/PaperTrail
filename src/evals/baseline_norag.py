"""No-RAG baseline: ask the SAME generator model the SAME golden-set
questions with NO retrieved evidence, using an equivalent refusal-affordance
contract (still free to say "insufficient information"), so the only
variable that differs from the shipped agent is whether grounding evidence
was provided. This isolates "does retrieval actually reduce hallucination"
as a measured question instead of an assertion.

Faithfulness is judged against the SAME real corpus evidence the shipped
agent would have retrieved (fetched fresh here, never shown to the baseline
model) -- using a citation-blind judge prompt, since a baseline with no
evidence can't produce [PMID:...] citations and shouldn't be penalized for
that by the main judge's "no citation = unsupported" rule, which exists to
catch a *grounded* system that forgot to cite, not to fairly grade an
ungrounded one.

Usage: python -m src.evals.baseline_norag
"""
import json

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.agent.llm import call_json
from src.evals.run import retrieval_only
from src.agent.graph import format_evidence
from src.corpus.db import get_papers_by_pmid
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank

load_dotenv()

NOEVIDENCE_SYSTEM = """You are a helpful biomedical research assistant. Answer the question
directly using your own knowledge -- no source documents are provided to you.

If you are confident you know the answer, give a specific, direct answer (include numbers,
trial names, etc. if you know them). If you do not have reliable knowledge to answer the
question, set "sufficient": false and briefly say so, rather than guessing.

Respond with a JSON object:
{"answer": "<answer text, no citations>", "sufficient": true|false}"""

BASELINE_JUDGE_SYSTEM = """You are an independent fact-checking judge. You are given a question,
an answer written WITHOUT access to any source documents (so it contains no citations), and a
set of real evidence chunks from the actual literature on this question -- evidence the answer's
author never saw.

Break the answer into its individual factual claims. For each claim, decide whether the
provided evidence supports it:
- "supported": the evidence confirms this specific claim.
- "partially_supported": the evidence is related but doesn't fully confirm the specific
  figure/detail, or the claim is a reasonable but unconfirmed generalization.
- "unsupported": the evidence contradicts the claim, or doesn't address it at all so it
  cannot be verified either way.

Respond with JSON: {"claims": [{"claim": "...", "verdict": "supported"|"partially_supported"|"unsupported"}, ...]}
If the answer makes no factual claims (e.g. it's a refusal), respond with {"claims": []}."""


def run_baseline_one(question_row: dict, config: dict) -> dict:
    question = question_row["question"]
    model = config["generation"]["model"]
    judge_model = config["judge"]["model"]

    result = call_json(model, NOEVIDENCE_SYSTEM, question)
    answer = result.get("answer", "")
    sufficient = bool(result.get("sufficient", False))

    claims = []
    if sufficient and answer:
        retrieved_pmids = retrieval_only(question, config)
        meta = get_papers_by_pmid(retrieved_pmids)
        strategy = config["chunking"]["strategy"]
        embed_model = config["embedding"]["model"]
        r = config["retrieval"]
        fused = hybrid_search(question, strategy, embed_model, r["dense_top_k"], r["lexical_top_k"], r["rrf_k"])
        top = rerank(question, fused, r["reranker_model"], 10)
        seen, evidence_chunks = set(), []
        for item in top:
            if item["pmid"] in seen:
                continue
            seen.add(item["pmid"])
            m = meta.get(item["pmid"], {})
            evidence_chunks.append({**item, "title": m.get("title"), "year": m.get("year")})
        evidence = format_evidence(evidence_chunks)
        judge_result = call_json(judge_model, BASELINE_JUDGE_SYSTEM,
                                  f"Question: {question}\n\nAnswer: {answer}\n\nEvidence:\n{evidence}",
                                  temperature=0.0)
        claims = judge_result.get("claims", [])

    return {
        "qid": question_row["qid"], "type": question_row["type"],
        "answerable_gold": question_row["answerable"],
        "answer": answer, "sufficient": sufficient, "judge_claims": claims,
    }


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    questions = [json.loads(line) for line in open(config["eval"]["golden_set_path"])]

    rows = []
    for q in tqdm(questions, desc="no-RAG baseline"):
        rows.append(run_baseline_one(q, config))

    out_path = "evals/results/baseline_norag.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
