# Judge Spot-Check — run 88befd366b

Sampled 34/169 judged claims (20%, seed=42) and independently re-read each
against its cited abstract (fetched fresh from Postgres, not from the judge's
context) to form an independent verdict before comparing to the judge's call.

**Agreement: 34/34 (100%)** — includes the judge's more nuanced calls:
- 2 `partially_supported` verdicts (generalizations that overstate a narrower
  finding) — both matched independent read.
- 3 `unsupported` verdicts, including catching a sentence in u-002's answer
  that generalizes a narrow in-vitro finding (LDL receptor levels in
  pancreatic beta cells from one exenatide/statin interaction paper) into a
  general "how statins lower LDL cholesterol" claim — a real hallucination,
  correctly flagged.

Two claims (p<0.0001 treatment difference; apoptosis suppression + beta-cell
proliferation) needed the full abstract (evidence chunk in the agent's context
is truncated in this spot-check log at 500 chars for readability) to confirm;
both checked out exact on the untruncated text.

No judge prompt revision needed (100% >> the 85% threshold).

Sample: evals/results/spotcheck_sample.jsonl
