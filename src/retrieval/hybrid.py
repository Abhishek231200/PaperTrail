from src.retrieval.dense import dense_search
from src.retrieval.lexical import lexical_search


def rrf_fuse(result_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over ranked result lists, keyed by (pmid, chunk_ix)."""
    fused: dict[tuple[int, int], dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = (item["pmid"], item["chunk_ix"])
            if key not in fused:
                fused[key] = {"pmid": item["pmid"], "chunk_ix": item["chunk_ix"],
                               "text": item["text"], "rrf_score": 0.0}
            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def hybrid_search(query: str, strategy: str, model: str, dense_k: int, lexical_k: int,
                   rrf_k: int, dsn: str | None = None) -> list[dict]:
    dense_results = dense_search(query, strategy, model, dense_k, dsn)
    lexical_results = lexical_search(query, strategy, lexical_k, dsn)
    return rrf_fuse([dense_results, lexical_results], rrf_k)
