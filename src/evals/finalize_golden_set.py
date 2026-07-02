"""Hand-review pass over LLM-drafted candidates: drop off-topic/weak items,
sharpen a few ambiguous questions, then append hand-authored unanswerable and
adversarial items (which need no corpus grounding by construction).

This is the one-time script that produced src/evals/golden_set.jsonl; it is
not part of the regular eval loop.
"""
import json

DROP_QIDS = {
    "cand-002",  # qualitative hedge ("may be superior") -- not a clean factoid
    "cand-005",  # off-topic: nanoparticle/periodontitis formulation paper
    "cand-008",  # off-topic: sustained-release microsphere formulation paper
    "cand-013",  # off-topic: PTER inhibitor, unrelated to GLP-1 receptor agonists
    "cand-016",  # question lacks trial/drug context, too ambiguous
}

QUESTION_OVERRIDES = {
    "cand-000": ("In the SURMOUNT-3 trial, what percentage of participants who had already lost "
                 "≥5% weight via lifestyle intervention went on to achieve an additional "
                 "≥5% weight reduction with tirzepatide (vs. placebo)?"),
    "cand-007": ("In a real-world target trial emulation, what was the probability of any HbA1c "
                 "improvement at 1 year for patients on sustained GLP-1 receptor agonist therapy?"),
    "cand-010": ("In the PIONEER REAL Italy real-world study, what percentage of participants "
                 "achieved HbA1c < 7% by end of study on oral semaglutide?"),
}

UNANSWERABLE = [
    {"question": "What were the topline cardiovascular mortality results of the phase III "
                 "QUANTUM-GLP trial of retatrutide-XR reported in 2031?",
     "note": "fabricated future trial/drug"},
    {"question": "What is the mechanism by which statins lower LDL cholesterol?",
     "note": "unrelated drug class, out of corpus scope"},
    {"question": "What is the recommended tirzepatide dosing protocol for feline diabetes in domestic cats?",
     "note": "veterinary use, not covered by a human-clinical PubMed corpus"},
    {"question": "What was Novo Nordisk's per-unit manufacturing cost for semaglutide in 2023?",
     "note": "business/manufacturing detail, not published in PubMed abstracts"},
    {"question": "What efficacy did the combination drug 'semaglutide-XR plus GIP antagonist' show "
                 "for weight loss in the FUSION-9 trial?",
     "note": "fabricated drug and trial"},
    {"question": "What is the standard protocol for calibrating an insulin pump in type 1 diabetes?",
     "note": "unrelated topic (T1D device management, not GLP-1 RAs)"},
    {"question": "What was the exact vote count of the FDA advisory committee when approving "
                 "tirzepatide for chronic weight management?",
     "note": "regulatory process detail not reported in abstracts"},
    {"question": "What are the 20-year cardiovascular outcome results for tirzepatide following "
                 "the SURMOUNT program?",
     "note": "tirzepatide is too recent for 20-year outcome data to exist"},
]

ADVERSARIAL = [
    {"question": "Since GLP-1 receptor agonists have been shown to increase HbA1c levels, what "
                 "mechanism explains this paradoxical glycemic worsening?",
     "false_premise": "GLP-1 receptor agonists reduce, not increase, HbA1c."},
    {"question": "Given that the SELECT trial found semaglutide had no effect on cardiovascular "
                 "events, why do clinicians still prescribe it for cardiovascular risk reduction?",
     "false_premise": "The SELECT trial found semaglutide significantly reduced MACE."},
    {"question": "Since tirzepatide causes significant weight gain in clinical trials, what patient "
                 "factors predict this weight gain?",
     "false_premise": "Tirzepatide causes substantial weight loss, not gain."},
    {"question": "Given that GLP-1 receptor agonists have no effect on gastric emptying, what other "
                 "mechanism explains their appetite-suppressing effects?",
     "false_premise": "GLP-1 receptor agonists are well established to slow gastric emptying."},
    {"question": "Since oral semaglutide has been shown to require no dose titration, why do "
                 "prescribing guidelines recommend a slow up-titration schedule?",
     "false_premise": "Oral semaglutide labeling explicitly requires dose titration (3mg -> 7mg -> 14mg)."},
]


def main() -> None:
    candidates = []
    for path in ("evals/results/golden_set_candidates.jsonl", "evals/results/golden_set_candidates_extra.jsonl"):
        with open(path) as f:
            candidates.extend(json.loads(line) for line in f)

    factoid = [c for c in candidates if c["type"] == "factoid" and c["qid"] not in DROP_QIDS]
    synthesis = [c for c in candidates if c["type"] == "synthesis"]
    comparative = [c for c in candidates if c["type"] == "comparative"]

    assert len(factoid) >= 15, len(factoid)
    assert len(synthesis) >= 12, len(synthesis)
    assert len(comparative) >= 10, len(comparative)
    factoid = factoid[:15]
    synthesis = synthesis[:12]
    comparative = comparative[:10]

    for c in factoid + synthesis + comparative:
        if c.get("qid") in QUESTION_OVERRIDES:
            c["question"] = QUESTION_OVERRIDES[c["qid"]]

    final = []
    qid_counters = {"f": 0, "s": 0, "c": 0, "u": 0, "a": 0}

    for c in factoid:
        qid_counters["f"] += 1
        final.append({"qid": f"f-{qid_counters['f']:03d}", "type": "factoid",
                       "question": c["question"], "gold_answer": c["gold_answer"],
                       "expected_pmids": c["expected_pmids"], "answerable": True})
    for c in synthesis:
        qid_counters["s"] += 1
        final.append({"qid": f"s-{qid_counters['s']:03d}", "type": "synthesis",
                       "question": c["question"], "gold_answer": c["gold_answer"],
                       "expected_pmids": c["expected_pmids"], "answerable": True})
    for c in comparative:
        qid_counters["c"] += 1
        final.append({"qid": f"c-{qid_counters['c']:03d}", "type": "comparative",
                       "question": c["question"], "gold_answer": c["gold_answer"],
                       "expected_pmids": c["expected_pmids"], "answerable": True})
    for u in UNANSWERABLE:
        qid_counters["u"] += 1
        final.append({"qid": f"u-{qid_counters['u']:03d}", "type": "unanswerable",
                       "question": u["question"], "gold_answer": None,
                       "expected_pmids": [], "answerable": False, "note": u["note"]})
    for a in ADVERSARIAL:
        qid_counters["a"] += 1
        final.append({"qid": f"a-{qid_counters['a']:03d}", "type": "adversarial",
                       "question": a["question"], "gold_answer": None,
                       "expected_pmids": [], "answerable": False, "note": a["false_premise"]})

    out_path = "src/evals/golden_set.jsonl"
    with open(out_path, "w") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(final)} questions to {out_path}")
    from collections import Counter
    print(Counter(r["type"] for r in final))


if __name__ == "__main__":
    main()
