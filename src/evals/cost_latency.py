"""Measure per-query cost and latency of the deployed agent (PLAN/RETRIEVE/
SYNTHESIZE/VERIFY chat-completion calls only -- not judge calls, since the
judge is an eval-time cost never incurred in production; not embedding calls,
which are negligible at this token volume with text-embedding-3-small).

Usage: python -m src.evals.cost_latency
"""
import json

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.agent.graph import run as run_agent

load_dotenv()

# Illustrative OpenAI pricing as of this writing (per 1M tokens); update if
# pricing changes -- this is a documented approximation, not a live lookup.
PRICING_PER_1M = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
}


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    generator_model = config["generation"]["model"]
    questions = [json.loads(line) for line in open(config["eval"]["golden_set_path"])]

    rows = []
    for q in tqdm(questions, desc="cost/latency pass"):
        state = run_agent(q["question"])
        rows.append({
            "qid": q["qid"], "type": q["type"],
            "prompt_tokens": state["prompt_tokens"],
            "completion_tokens": state["completion_tokens"],
            "n_llm_calls": state["n_llm_calls"],
            "wall_clock_ms": state["wall_clock_ms"],
            "retry_count": state.get("retry_count", 0),
        })

    out_path = "evals/results/cost_latency.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    mean_prompt = sum(r["prompt_tokens"] for r in rows) / n
    mean_completion = sum(r["completion_tokens"] for r in rows) / n
    mean_calls = sum(r["n_llm_calls"] for r in rows) / n
    mean_latency = sum(r["wall_clock_ms"] for r in rows) / n
    p50_latency = sorted(r["wall_clock_ms"] for r in rows)[n // 2]
    p95_latency = sorted(r["wall_clock_ms"] for r in rows)[int(n * 0.95)]
    retried = sum(1 for r in rows if r["retry_count"] > 0)

    pricing = PRICING_PER_1M.get(generator_model)
    cost_per_query = None
    if pricing:
        cost_per_query = (mean_prompt / 1e6) * pricing["prompt"] + (mean_completion / 1e6) * pricing["completion"]

    print(f"\nmean prompt tokens/query:     {mean_prompt:.0f}")
    print(f"mean completion tokens/query: {mean_completion:.0f}")
    print(f"mean LLM calls/query:         {mean_calls:.1f}  (plan + synthesize + verify, +more on retry)")
    print(f"retried at least once:        {retried}/{n} ({100*retried/n:.1f}%)")
    print(f"mean wall-clock latency:      {mean_latency:.0f} ms")
    print(f"p50 / p95 latency:            {p50_latency:.0f} ms / {p95_latency:.0f} ms")
    if cost_per_query is not None:
        print(f"estimated cost/query ({generator_model}): ${cost_per_query:.5f}")
        print(f"estimated cost per 1,000 queries:          ${cost_per_query*1000:.2f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
