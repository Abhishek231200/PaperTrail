from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=2)
def _model(name: str) -> CrossEncoder:
    return CrossEncoder(name)


def rerank(query: str, candidates: list[dict], model_name: str, top_k: int) -> list[dict]:
    if not candidates:
        return []
    model = _model(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    out = []
    for c, score in ranked[:top_k]:
        out.append({**c, "rerank_score": float(score)})
    return out
