"""Render a results table (CSV + Markdown) for one or more eval runs.

Usage: python -m src.evals.report <run_id> [<run_id> ...] [--label A --label B ...]
If no run_ids given, uses every run under evals/results/.
"""
import argparse
import csv
from pathlib import Path

from src.evals.metrics import compute_metrics, load_raw

RESULTS_DIR = Path("evals/results")

METRIC_FIELDS = [
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("citation_precision", "Citation Precision"),
    ("faithfulness", "Faithfulness"),
    ("hallucination_rate", "Hallucination Rate"),
    ("true_refusal_rate", "True-Refusal Rate"),
    ("over_refusal_rate", "Over-Refusal Rate"),
]


def fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="*")
    parser.add_argument("--label", action="append", default=None)
    args = parser.parse_args()

    run_ids = args.run_ids or sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "raw.jsonl").exists())
    labels = args.label or run_ids

    rows = []
    for run_id, label in zip(run_ids, labels):
        metrics = compute_metrics(load_raw(run_id))
        rows.append((label, run_id, metrics))

    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "run_id", "n_questions"] + [name for _, name in METRIC_FIELDS])
        for label, run_id, m in rows:
            writer.writerow([label, run_id, m["n_questions"]] + [m[k] for k, _ in METRIC_FIELDS])

    md_lines = ["| Run | " + " | ".join(name for _, name in METRIC_FIELDS) + " |",
                "|---|" + "---|" * len(METRIC_FIELDS)]
    for label, run_id, m in rows:
        md_lines.append("| " + label + " | " + " | ".join(fmt(m[k]) for k, _ in METRIC_FIELDS) + " |")
    md_path = RESULTS_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print("\n".join(md_lines))
    print(f"\nwrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
