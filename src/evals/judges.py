"""Independent LLM-judge for post-hoc evaluation. Deliberately uses a
DIFFERENT (stronger) model than the generator, and runs AFTER the agent has
already produced its final answer -- this is not the same as the agent's own
VERIFY node, which is a same-model self-check used for in-flight hallucination
control. Reusing the generator as its own judge would bias faithfulness
numbers upward; a stronger, independent judge is the whole point of Phase 3.
"""
from src.agent.llm import call_json

JUDGE_VERSION = "judge-v1"

JUDGE_FAITHFULNESS_SYSTEM = """You are an independent, skeptical fact-checking judge for a
biomedical RAG system. You did NOT write the answer being evaluated -- you are auditing it.

You are given a question, an answer (which cites evidence using [PMID:xxxxx] tags), and the
full text of every evidence chunk that was available to the system when it wrote the answer.

Break the answer into its individual factual claims. For each claim:
- Identify the PMID it cites (if any).
- Decide whether the cited evidence chunk actually supports that specific claim:
  "supported" (chunk clearly states it), "partially_supported" (chunk is related but
  doesn't fully establish the claim, or the claim overgeneralizes/overstates the evidence),
  or "unsupported" (chunk does not support it, or the claim cites no PMID, or cites a
  PMID that is not among the provided evidence).
- Be strict: a claim that combines two facts where only one is supported is "partially_supported".

Respond with JSON:
{"claims": [{"claim": "<short paraphrase>", "cited_pmid": <int or null>,
             "verdict": "supported"|"partially_supported"|"unsupported"}, ...]}
If the answer makes no factual claims (e.g. it is a refusal), respond with {"claims": []}."""


def judge_faithfulness(model: str, question: str, answer: str, evidence: str) -> list[dict]:
    user = f"Question: {question}\n\nAnswer: {answer}\n\nEvidence:\n{evidence}"
    result = call_json(model, JUDGE_FAITHFULNESS_SYSTEM, user, temperature=0.0)
    return result.get("claims", [])
