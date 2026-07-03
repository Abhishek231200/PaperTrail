"""Retrieval-quality ablation grid (spec section 8, runs A-E).

Isolates the retrieval step from generation: for each config, retrieve top-10
paper PMIDs per answerable golden-set question and compute recall@5/@10.
Fast and cheap (no agent/judge calls) since it reuses the Phase 1 retrieval
primitives directly rather than running the full agent per config -- this
mirrors real practice: use cheap retrieval metrics to pick a retrieval config,
then run the expensive full agent+judge eval (Phase 3's run.py) only on the
finalist. Faithfulness/hallucination/refusal numbers for the shipped config
live in the main eval run's results, not here.

Usage: python -m src.evals.ablate
"""
import json

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.retrieval.dense import dense_search
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.lexical import lexical_search
from src.retrieval.rerank import rerank

load_dotenv()

CONFIGS = [
    # label, chunking_strategy, use_dense, use_lexical, use_rerank
    ("A: dense only", "whole_abstract", True, False, False),
    ("B: lexical only", "whole_abstract", False, True, False),
    ("C: hybrid RRF (no rerank)", "whole_abstract", True, True, False),
    ("D: hybrid + rerank", "whole_abstract", True, True, True),
    ("E: sentence-window + hybrid + rerank", "sentence_window", True, True, True),
]


def retrieve_top10(question: str, config: dict, strategy: str, use_dense: bool, use_lexical: bool, use_rerank: bool) -> list[int]:
    r = config["retrieval"]
    embed_model = config["embedding"]["model"]

    dense_results = dense_search(question, strategy, embed_model, r["dense_top_k"]) if use_dense else []
    lexical_results = lexical_search(question, strategy, r["lexical_top_k"]) if use_lexical else []

    if use_dense and use_lexical:
        fused = rrf_fuse([dense_results, lexical_results], r["rrf_k"])
    else:
        fused = dense_results or lexical_results

    if use_rerank:
        candidates = rerank(question, fused[:20], r["reranker_model"], 15)
    else:
        candidates = fused

    pmids: list[int] = []
    for item in candidates:
        if item["pmid"] not in pmids:
            pmids.append(item["pmid"])
        if len(pmids) >= 10:
            break
    return pmids


def recall_at_k(retrieved: list[int], expected: list[int], k: int) -> int:
    return int(any(pmid in retrieved[:k] for pmid in expected))


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    questions = [json.loads(line) for line in open(config["eval"]["golden_set_path"])]
    answerable = [q for q in questions if q["answerable"]]

    results = []
    for label, strategy, use_dense, use_lexical, use_rerank in CONFIGS:
        r5, r10 = [], []
        for q in tqdm(answerable, desc=label):
            retrieved = retrieve_top10(q["question"], config, strategy, use_dense, use_lexical, use_rerank)
            r5.append(recall_at_k(retrieved, q["expected_pmids"], 5))
            r10.append(recall_at_k(retrieved, q["expected_pmids"], 10))
        results.append({
            "run": label, "chunking": strategy,
            "dense": use_dense, "lexical": use_lexical, "rerank": use_rerank,
            "recall_at_5": sum(r5) / len(r5), "recall_at_10": sum(r10) / len(r10),
            "n": len(answerable),
        })

    out_path = "evals/results/ablation_recall.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Run':<40} {'Recall@5':>10} {'Recall@10':>10}")
    for r in results:
        print(f"{r['run']:<40} {r['recall_at_5']*100:>9.1f}% {r['recall_at_10']*100:>9.1f}%")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
