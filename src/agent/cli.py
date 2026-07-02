"""Usage: python -m src.agent.cli "your question" """
import sys

from dotenv import load_dotenv

from src.agent.graph import run

load_dotenv()


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    state = run(question)

    print(f"\nQuestion: {question}")
    print(f"Sub-queries: {state.get('sub_queries')}")
    print(f"Retrieved PMIDs: {[r['pmid'] for r in state.get('retrieved', [])]}")

    if not state.get("sufficient", False):
        print("\n[INSUFFICIENT_EVIDENCE]")
        print(state.get("answer", ""))
    else:
        print(f"\nAnswer:\n{state.get('answer', '')}")
        verdicts = [c.get("verdict") for c in state.get("verify_claims", [])]
        print(f"\nVerify: all_supported={state.get('all_supported')} claim_verdicts={verdicts}")
        if state.get("retry_count", 0) > 0:
            print(f"(retried retrieval {state['retry_count']}x after first verify failure)")

    print(f"\nrun_id: {state['run_id']}  (trace: runs/{state['run_id']}.jsonl)")


if __name__ == "__main__":
    main()
