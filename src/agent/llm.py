import json
import os
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

_client: OpenAI | None = None
_usage_log: list[dict] = []


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def reset_usage_log() -> None:
    """Call at the start of a unit of work (one agent run, one judge call)
    whose token/latency cost you want to measure in isolation."""
    global _usage_log
    _usage_log = []


def get_usage_log() -> list[dict]:
    return list(_usage_log)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_json(model: str, system: str, user: str, temperature: float = 0.0) -> dict:
    """Chat completion constrained to a single JSON object response."""
    start = time.monotonic()
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    _usage_log.append({
        "model": model,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "latency_ms": (time.monotonic() - start) * 1000,
    })
    return json.loads(resp.choices[0].message.content)
