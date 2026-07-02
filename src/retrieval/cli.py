"""Compare dense / lexical / hybrid+rerank retrieval for a question.

Usage: python -m src.retrieval.cli "your question"
"""
import sys

import yaml
from dotenv import load_dotenv

from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.lexical import lexical_search
from src.retrieval.rerank import rerank

load_dotenv()


def _print_results(label: str, results: list[dict], score_key: str) -> None:
    print(f"\n=== {label} (top {len(results)}) ===")
    for r in results:
        snippet = r["text"].replace("\n", " ")[:110]
        print(f"  [{r[score_key]:.4f}] PMID:{r['pmid']} chunk:{r['chunk_ix']} {snippet}...")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    config = yaml.safe_load(open("config.yaml"))
    strategy = config["chunking"]["strategy"]
    model = config["embedding"]["model"]
    r = config["retrieval"]

    dense_results = dense_search(query, strategy, model, r["dense_top_k"])
    _print_results("DENSE", dense_results, "score")

    lexical_results = lexical_search(query, strategy, r["lexical_top_k"])
    _print_results("LEXICAL", lexical_results, "score")

    fused = hybrid_search(query, strategy, model, r["dense_top_k"], r["lexical_top_k"], r["rrf_k"])
    _print_results("HYBRID (RRF)", fused[:10], "rrf_score")

    reranked = rerank(query, fused, r["reranker_model"], r["rerank_top_k"])
    _print_results("HYBRID + RERANK (final)", reranked, "rerank_score")


if __name__ == "__main__":
    main()
