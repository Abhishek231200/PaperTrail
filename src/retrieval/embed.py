"""OpenAI embedding client with batching + retry."""
import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def embed_batch(texts: list[str], model: str) -> list[list[float]]:
    resp = client().embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def embed_many(texts: list[str], model: str, batch_size: int = 100) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(embed_batch(texts[i:i + batch_size], model))
    return out
