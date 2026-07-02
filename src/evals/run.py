"""Run the golden set end-to-end: isolated retrieval recall + full agent run
+ independent judge faithfulness scoring. Writes one raw JSONL row per
question plus a config snapshot, keyed by a config-hash run_id so every
number in the README is reproducible.

Usage: python -m src.evals.run [--limit N]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.agent.graph import format_evidence, run as run_agent
from src.evals.citations import extract_cited_pmids
from src.evals.judges import judge_faithfulness
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import rerank

load_dotenv()

RESULTS_DIR = Path("evals/results")


def config_run_id(config: dict) -> str:
    relevant = {
        "chunking": config["chunking"],
        "retrieval": config["retrieval"],
        "embedding": config["embedding"],
        "generation": config["generation"],
        "judge": config["judge"],
    }
    blob = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def retrieval_only(question: str, config: dict) -> list[int]:
    strategy = config["chunking"]["strategy"]
    embed_model = config["embedding"]["model"]
    r = config["retrieval"]
    fused = hybrid_search(question, strategy, embed_model, r["dense_top_k"], r["lexical_top_k"], r["rrf_k"])
    top = rerank(question, fused, r["reranker_model"], 10)
    return [item["pmid"] for item in top]


def recall_at_k(retrieved_pmids: list[int], expected_pmids: list[int], k: int) -> int:
    if not expected_pmids:
        return None
    return int(any(pmid in retrieved_pmids[:k] for pmid in expected_pmids))


def eval_one(question_row: dict, config: dict) -> dict:
    question = question_row["question"]
    expected_pmids = question_row.get("expected_pmids", [])

    retrieved_top10 = retrieval_only(question, config)
    recall5 = recall_at_k(retrieved_top10, expected_pmids, 5)
    recall10 = recall_at_k(retrieved_top10, expected_pmids, 10)

    agent_state = run_agent(question)
    answer = agent_state.get("answer", "")
    sufficient = bool(agent_state.get("sufficient", False))
    retrieved = agent_state.get("retrieved", [])
    agent_retrieved_pmids = [r["pmid"] for r in retrieved]

    claims = []
    if sufficient and answer:
        evidence = format_evidence(retrieved)
        judge_model = config["judge"]["model"]
        claims = judge_faithfulness(judge_model, question, answer, evidence)

    cited_pmids = extract_cited_pmids(answer) if sufficient else []
    grounded_citations = [p for p in cited_pmids if p in agent_retrieved_pmids]

    return {
        "qid": question_row["qid"],
        "type": question_row["type"],
        "question": question,
        "answerable_gold": question_row["answerable"],
        "expected_pmids": expected_pmids,
        "retrieved_top10_pmids": retrieved_top10,
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "agent_run_id": agent_state.get("run_id"),
        "answer": answer,
        "sufficient": sufficient,
        "agent_retrieved_pmids": agent_retrieved_pmids,
        "retry_count": agent_state.get("retry_count", 0),
        "cited_pmids": cited_pmids,
        "grounded_citation_count": len(grounded_citations),
        "total_citation_count": len(cited_pmids),
        "judge_claims": claims,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--golden-set", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(open("config.yaml"))
    golden_path = args.golden_set or config["eval"]["golden_set_path"]
    questions = [json.loads(line) for line in open(golden_path)]
    if args.limit:
        questions = questions[: args.limit]

    run_id = config_run_id(config)
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config))

    raw_path = run_dir / "raw.jsonl"
    with raw_path.open("w") as f:
        for q in tqdm(questions, desc=f"eval run {run_id}"):
            row = eval_one(q, config)
            row["ts"] = time.time()
            f.write(json.dumps(row) + "\n")

    print(f"run_id={run_id}  raw results -> {raw_path}")


if __name__ == "__main__":
    main()
