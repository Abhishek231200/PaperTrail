"""Build a small (~6-8) golden mini-set of "detail" questions whose answers
live only in a paper's full text, not its abstract -- demonstrating why
full-text ingestion matters for anything beyond headline trial results.

For each candidate: draft a question from a full-text paragraph (LLM,
grounded, verbatim-quote-checked like the main golden set), then verify the
SAME fact is NOT already stated in that paper's abstract (a second LLM check).
Only survivors that pass both checks go in the final set.

Usage: python -m src.evals.build_fulltext_questions
"""
import json
import random

import psycopg
import yaml
from dotenv import load_dotenv

from src.agent.llm import call_json
from src.evals.build_golden_set import quote_supported

load_dotenv()

DETAIL_DRAFT_SYSTEM = """You draft a single factoid question-answer pair for an evaluation set,
grounded ONLY in the one paragraph given to you (a paragraph from a paper's full text, NOT its
abstract). Prefer specific, narrow methodological or results details that a paper's abstract
would typically summarize away or omit entirely (e.g. an exact sub-analysis number, a specific
inclusion/exclusion criterion, a specific statistical test used, a specific secondary finding) --
NOT a paper's headline result, which usually IS in the abstract.

Requirements:
- The question must be answerable with a specific fact stated in this paragraph.
- Include "supporting_quote": a verbatim excerpt (<=30 words) copied EXACTLY from the paragraph.

Respond with JSON: {"question": "...", "answer": "...", "supporting_quote": "..."}
If the paragraph has no clean extractable detail fact, respond with {"skip": true}."""

NOT_IN_ABSTRACT_SYSTEM = """You are given a question, an answer, and a paper's ABSTRACT (not its
full text). Decide whether the abstract ALONE already contains enough information to answer the
question. Respond with JSON: {"answerable_from_abstract": true|false}"""


def draft_detail_question(model: str, pmid: int, title: str, paragraph: str) -> dict | None:
    user = f"PMID: {pmid}\nTitle: {title}\nParagraph: {paragraph}"
    result = call_json(model, DETAIL_DRAFT_SYSTEM, user)
    if result.get("skip"):
        return None
    if not quote_supported(result.get("supporting_quote", ""), paragraph):
        return None
    return {"question": result["question"], "answer": result["answer"], "pmid": pmid}


def check_not_in_abstract(model: str, question: str, answer: str, abstract: str) -> bool:
    user = f"Question: {question}\nAnswer: {answer}\n\nAbstract: {abstract}"
    result = call_json(model, NOT_IN_ABSTRACT_SYSTEM, user, temperature=0.0)
    return not result.get("answerable_from_abstract", True)


def main() -> None:
    random.seed(7)
    config = yaml.safe_load(open("config.yaml"))
    model = config["generation"]["model"]
    dsn = __import__("os").environ["POSTGRES_DSN"]

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT c.pmid, p.title, p.abstract, c.chunk_ix, c.text
            FROM chunks c JOIN papers p ON p.pmid = c.pmid
            WHERE c.strategy = 'fulltext_paragraph'
            ORDER BY random()
        """)
        rows = cur.fetchall()

    candidates = []
    n_pmids_seen = set()
    for pmid, title, abstract, chunk_ix, text in rows:
        if pmid in n_pmids_seen or len(candidates) >= 20:
            continue
        n_pmids_seen.add(pmid)
        candidates.append((pmid, title, abstract, text))

    final = []
    for pmid, title, abstract, paragraph in candidates:
        if len(final) >= 8:
            break
        draft = draft_detail_question(model, pmid, title, paragraph)
        if not draft:
            continue
        if not check_not_in_abstract(model, draft["question"], draft["answer"], abstract):
            continue
        final.append({
            "qid": f"ft-{len(final)+1:03d}",
            "type": "fulltext_detail",
            "question": draft["question"],
            "gold_answer": draft["answer"],
            "expected_pmids": [pmid],
            "answerable": True,
        })

    out_path = "src/evals/golden_set_fulltext.jsonl"
    with open(out_path, "w") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(final)} full-text-only detail questions to {out_path}")


if __name__ == "__main__":
    main()
