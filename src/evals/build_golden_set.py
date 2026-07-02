"""Semi-automated golden-set drafting (spec section 7.1):
1. Sample real abstracts from the corpus.
2. Have an LLM draft a Q/A grounded in that text, forced to also emit a
   verbatim "supporting_quote" per cited PMID.
3. Programmatically verify each quote is actually a substring of the source
   abstract -- any draft that fails this check is auto-rejected rather than
   trusted. Survivors still get a manual read (see review_golden_set.py).

This produces evals/results/golden_set_candidates.jsonl for hand review;
it does NOT write golden_set.jsonl directly.
"""
import json
import re

import yaml
from dotenv import load_dotenv

from src.agent.llm import call_json
from src.corpus.db import get_papers_by_pmid
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank

load_dotenv()

FACTOID_SYSTEM = """You draft a single factoid question-answer pair for an evaluation set,
grounded ONLY in the one abstract given to you.

Requirements:
- The question must be answerable with a specific fact stated in the abstract
  (a number, percentage, trial outcome, dose, or defined comparison).
- The answer must be fully supported by the abstract text -- do not add outside knowledge.
- Include "supporting_quote": a verbatim excerpt (<=30 words) copied EXACTLY
  (same words, same order) from the abstract that contains the fact.
- Do not ask about the paper's title, authors, or journal.

Respond with JSON: {"question": "...", "answer": "...", "supporting_quote": "..."}
If the abstract has no clean extractable factoid, respond with {"skip": true}."""

SYNTHESIS_SYSTEM = """You draft a single synthesis question-answer pair for an evaluation set.
You are given several abstracts on a shared topic. Draft a question that requires
aggregating findings from AT LEAST TWO of the abstracts to answer well (e.g. "what
mechanisms are proposed for X" where different abstracts propose different mechanisms).

Requirements:
- The answer must be fully supported by the given abstracts -- no outside knowledge,
  no generalizations beyond what the abstracts state.
- "expected_pmids": the PMIDs (integers) you actually drew the answer from (>=2).

Respond with JSON:
{"question": "...", "answer": "...", "expected_pmids": [...]}
If the abstracts don't support a good synthesis question, respond with {"skip": true}."""

COMPARATIVE_SYSTEM = """You draft a single comparative question-answer pair for an evaluation set,
comparing two drugs/entities on a shared endpoint, using ONLY the given abstracts.

Requirements:
- The question must ask for a comparison (e.g. "how do X and Y compare on Z").
- The answer must be fully supported by the given abstracts -- no outside knowledge,
  no generalizations beyond what the abstracts state.
- "expected_pmids": the PMIDs (integers) you drew the answer from (should cover both entities).

Respond with JSON:
{"question": "...", "answer": "...", "expected_pmids": [...]}
If the abstracts don't support a good comparative question, respond with {"skip": true}."""

CLAIM_VERIFY_SYSTEM = """You are a skeptical fact-checker. You are given a question, an
answer someone drafted for it, and ONE abstract that was supposedly used as a source
for that answer. Decide whether this specific abstract actually supports (at least part
of) the answer -- not whether it's a good abstract in general.

Respond with JSON: {"supported": true|false, "reason": "<one sentence>"}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_supported(quote: str, source_text: str) -> bool:
    return _norm(quote) in _norm(source_text)


def draft_factoid(model: str, pmid: int, title: str, abstract: str) -> dict | None:
    user = f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract}"
    result = call_json(model, FACTOID_SYSTEM, user)
    if result.get("skip"):
        return None
    if not quote_supported(result.get("supporting_quote", ""), abstract):
        return None
    return {
        "type": "factoid",
        "question": result["question"],
        "gold_answer": result["answer"],
        "expected_pmids": [pmid],
        "answerable": True,
        "_supporting_quotes": {str(pmid): result["supporting_quote"]},
    }


def verify_claim_against_pmid(model: str, question: str, answer: str, pmid: int, abstract: str) -> bool:
    user = f"Question: {question}\nAnswer: {answer}\n\nAbstract (PMID {pmid}): {abstract}"
    result = call_json(model, CLAIM_VERIFY_SYSTEM, user, temperature=0.0)
    return bool(result.get("supported", False))


def draft_multi(model: str, system: str, qtype: str, topic: str, papers: list[dict]) -> dict | None:
    blocks = "\n\n".join(f"PMID: {p['pmid']}\nTitle: {p['title']}\nAbstract: {p['abstract']}" for p in papers)
    user = f"Topic: {topic}\n\n{blocks}"
    result = call_json(model, system, user)
    if result.get("skip"):
        return None
    by_pmid = {p["pmid"]: p["abstract"] for p in papers}
    expected = [int(p) for p in result.get("expected_pmids", []) if int(p) in by_pmid]
    if len(expected) < 2:
        return None

    verified = [
        pmid for pmid in expected
        if verify_claim_against_pmid(model, result["question"], result["answer"], pmid, by_pmid[pmid])
    ]
    if len(verified) < 2:
        return None

    return {
        "type": qtype,
        "question": result["question"],
        "gold_answer": result["answer"],
        "expected_pmids": verified,
        "answerable": True,
    }


def sample_trial_abstracts(dsn_conn, n_per_family: int = 4) -> list[dict]:
    import psycopg
    families = ["SURMOUNT", "STEP ", "SUSTAIN", "PIONEER", "SELECT", "SURPASS"]
    out = []
    with psycopg.connect(dsn_conn) as conn, conn.cursor() as cur:
        for fam in families:
            cur.execute(
                """
                SELECT pmid, title, abstract FROM papers
                WHERE title ILIKE %s
                  AND title NOT ILIKE '%%post hoc%%' AND title NOT ILIKE '%%subgroup%%'
                  AND title NOT ILIKE '%%review%%' AND title NOT ILIKE '%%protocol%%'
                  AND title NOT ILIKE '%%rationale%%' AND title NOT ILIKE '%%meta-analysis%%'
                ORDER BY random() LIMIT %s
                """,
                (f"%{fam}%", n_per_family),
            )
            for pmid, title, abstract in cur.fetchall():
                out.append({"pmid": pmid, "title": title, "abstract": abstract})
    return out


SYNTHESIS_TOPICS = [
    "GLP-1 receptor agonist cardiovascular protection mechanism",
    "GLP-1 receptor agonist gastrointestinal side effect mechanism",
    "GLP-1 receptor agonist effect on lean muscle mass and body composition",
    "GLP-1 receptor agonist central nervous system appetite regulation mechanism",
    "GLP-1 receptor agonist beta cell function and insulin secretion mechanism",
    "GLP-1 receptor agonist effect on liver fat and MASLD/NAFLD",
    "GLP-1 receptor agonist effect on kidney outcomes and nephropathy",
    "GLP-1 receptor agonist bone density and fracture risk",
    "GLP-1 receptor agonist use in adolescents and pediatric obesity",
    "GLP-1 receptor agonist discontinuation and weight regain",
]

COMPARATIVE_PAIRS = [
    ("semaglutide", "tirzepatide", "weight loss efficacy"),
    ("semaglutide", "liraglutide", "HbA1c reduction"),
    ("tirzepatide", "dulaglutide", "glycemic control"),
    ("oral semaglutide", "injectable semaglutide", "efficacy and adherence"),
    ("semaglutide", "tirzepatide", "gastrointestinal adverse events"),
    ("liraglutide", "dulaglutide", "cardiovascular outcomes"),
    ("tirzepatide", "semaglutide", "cardiometabolic risk factor improvement"),
    ("semaglutide", "bariatric surgery", "weight loss magnitude"),
    ("tirzepatide", "phentermine-topiramate", "weight loss efficacy"),
    ("semaglutide", "dulaglutide", "cardiovascular risk reduction"),
]


def build_candidates(dsn: str, config: dict) -> list[dict]:
    model = config["generation"]["model"]
    strategy = config["chunking"]["strategy"]
    embed_model = config["embedding"]["model"]
    r = config["retrieval"]

    candidates = []

    for paper in sample_trial_abstracts(dsn, n_per_family=4):
        c = draft_factoid(model, paper["pmid"], paper["title"], paper["abstract"])
        if c:
            candidates.append(c)

    for topic in SYNTHESIS_TOPICS:
        fused = hybrid_search(topic, strategy, embed_model, r["dense_top_k"], r["lexical_top_k"], r["rrf_k"])
        top = rerank(topic, fused, r["reranker_model"], 6)
        pmids = list({item["pmid"] for item in top})
        meta = get_papers_by_pmid(pmids)
        papers = [{"pmid": p, "title": meta[p]["title"], "abstract": next(t["text"] for t in top if t["pmid"] == p)}
                  for p in pmids if p in meta]
        c = draft_multi(model, SYNTHESIS_SYSTEM, "synthesis", topic, papers)
        if c:
            candidates.append(c)

    for drug_a, drug_b, endpoint in COMPARATIVE_PAIRS:
        papers = []
        for drug in (drug_a, drug_b):
            fused = hybrid_search(f"{drug} {endpoint}", strategy, embed_model, r["dense_top_k"], r["lexical_top_k"], r["rrf_k"])
            top = rerank(f"{drug} {endpoint}", fused, r["reranker_model"], 3)
            pmids = list({item["pmid"] for item in top})
            meta = get_papers_by_pmid(pmids)
            for p in pmids:
                if p in meta:
                    papers.append({"pmid": p, "title": meta[p]["title"],
                                    "abstract": next(t["text"] for t in top if t["pmid"] == p)})
        c = draft_multi(model, COMPARATIVE_SYSTEM, "comparative", f"{drug_a} vs {drug_b}: {endpoint}", papers)
        if c:
            candidates.append(c)

    return candidates


def main() -> None:
    import os
    config = yaml.safe_load(open("config.yaml"))
    dsn = os.environ["POSTGRES_DSN"]
    candidates = build_candidates(dsn, config)
    out_path = "evals/results/golden_set_candidates.jsonl"
    with open(out_path, "w") as f:
        for i, c in enumerate(candidates):
            c["qid"] = f"cand-{i:03d}"
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(candidates)} candidates to {out_path}")


if __name__ == "__main__":
    main()
