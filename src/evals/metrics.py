"""Compute the five headline metrics from a raw eval run."""
import json
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compute_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable_gold"]]
    unanswerable = [r for r in rows if not r["answerable_gold"]]

    recall_5 = _mean([r["recall_at_5"] for r in answerable])
    recall_10 = _mean([r["recall_at_10"] for r in answerable])

    # Citation precision: of PMIDs cited, fraction that are both grounded
    # (actually in the retrieved set, i.e. not a fabricated ID) AND judged relevant.
    total_cites, correct_cites = 0, 0
    all_claims = []
    for r in rows:
        cited = set(r["cited_pmids"])
        grounded = set(p for p in r["agent_retrieved_pmids"] if p in cited)
        judged_relevant = {
            c["cited_pmid"] for c in r["judge_claims"]
            if c.get("verdict") in ("supported", "partially_supported") and c.get("cited_pmid") is not None
        }
        for pmid in cited:
            total_cites += 1
            if pmid in grounded and pmid in judged_relevant:
                correct_cites += 1
        all_claims.extend(r["judge_claims"])
    citation_precision = correct_cites / total_cites if total_cites else None

    # Faithfulness: % of judged claims that are fully "supported".
    n_claims = len(all_claims)
    n_supported = sum(1 for c in all_claims if c.get("verdict") == "supported")
    faithfulness = n_supported / n_claims if n_claims else None

    # Hallucination rate: % of non-refused answers with >=1 unsupported claim.
    answered_rows = [r for r in rows if r["sufficient"]]
    n_hallucinated = sum(
        1 for r in answered_rows
        if any(c.get("verdict") == "unsupported" for c in r["judge_claims"])
    )
    hallucination_rate = n_hallucinated / len(answered_rows) if answered_rows else None

    # Refusal correctness: true-refusal rate (on unanswerable) and over-refusal rate (on answerable).
    true_refusals = sum(1 for r in unanswerable if not r["sufficient"])
    true_refusal_rate = true_refusals / len(unanswerable) if unanswerable else None

    over_refusals = sum(1 for r in answerable if not r["sufficient"])
    over_refusal_rate = over_refusals / len(answerable) if answerable else None

    return {
        "n_questions": len(rows),
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "citation_precision": citation_precision,
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "true_refusal_rate": true_refusal_rate,
        "over_refusal_rate": over_refusal_rate,
        "n_claims_judged": n_claims,
        "n_citations_judged": total_cites,
    }


def load_raw(run_id: str) -> list[dict]:
    path = Path("evals/results") / run_id / "raw.jsonl"
    return [json.loads(line) for line in open(path)]
