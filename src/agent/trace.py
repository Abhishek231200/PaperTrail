"""Per-run JSONL tracing: one line per node execution, keyed by run_id.
Traces are the debugging substrate for eval failure analysis (Phase 3/4)."""
import json
import time
import uuid
from pathlib import Path

RUNS_DIR = Path("runs")


class RunTracer:
    def __init__(self, question: str, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.question = question
        RUNS_DIR.mkdir(exist_ok=True)
        self.path = RUNS_DIR / f"{self.run_id}.jsonl"
        self._append({"event": "run_start", "question": question})

    def log(self, node: str, input_summary: dict, output_summary: dict) -> None:
        self._append({"event": "node", "node": node, "input": input_summary, "output": output_summary})

    def finish(self, final_state: dict) -> None:
        self._append({"event": "run_end", "final": final_state})

    def _append(self, record: dict) -> None:
        record = {"run_id": self.run_id, "ts": time.time(), **record}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
