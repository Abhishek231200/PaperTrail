from src.evals.metrics import compute_baseline_metrics, compute_metrics


def _row(answerable_gold, sufficient, recall5=1, recall10=1, cited_pmids=None,
         agent_retrieved_pmids=None, judge_claims=None):
    return {
        "answerable_gold": answerable_gold,
        "sufficient": sufficient,
        "recall_at_5": recall5,
        "recall_at_10": recall10,
        "cited_pmids": cited_pmids or [],
        "agent_retrieved_pmids": agent_retrieved_pmids or [],
        "judge_claims": judge_claims or [],
    }


def test_recall_only_averaged_over_answerable_rows():
    rows = [
        _row(True, True, recall5=1, recall10=1),
        _row(True, True, recall5=0, recall10=1),
        _row(False, False, recall5=None, recall10=None),  # unanswerable: no expected pmids
    ]
    m = compute_metrics(rows)
    assert m["recall_at_5"] == 0.5
    assert m["recall_at_10"] == 1.0


def test_true_refusal_and_over_refusal_rates():
    rows = [
        _row(False, sufficient=False),  # correctly refused
        _row(False, sufficient=True),   # should have refused, didn't
        _row(True, sufficient=True),    # correctly answered
        _row(True, sufficient=False),   # over-refused
    ]
    m = compute_metrics(rows)
    assert m["true_refusal_rate"] == 0.5
    assert m["over_refusal_rate"] == 0.5


def test_hallucination_rate_counts_answers_with_unsupported_claims():
    rows = [
        _row(True, True, judge_claims=[{"verdict": "supported", "cited_pmid": 1}]),
        _row(True, True, judge_claims=[{"verdict": "unsupported", "cited_pmid": None}]),
        _row(False, False, judge_claims=[]),  # refused, not counted (not "answered")
    ]
    m = compute_metrics(rows)
    assert m["hallucination_rate"] == 0.5


def test_faithfulness_is_fraction_of_supported_claims():
    rows = [_row(True, True, judge_claims=[
        {"verdict": "supported", "cited_pmid": 1},
        {"verdict": "unsupported", "cited_pmid": None},
        {"verdict": "partially_supported", "cited_pmid": 2},
    ])]
    m = compute_metrics(rows)
    assert m["faithfulness"] == 1 / 3


def test_citation_precision_requires_grounded_and_judged_relevant():
    rows = [_row(
        True, True,
        cited_pmids=[1, 2],
        agent_retrieved_pmids=[1],  # pmid 2 was cited but never actually retrieved -> hallucinated ID
        judge_claims=[
            {"verdict": "supported", "cited_pmid": 1},
            {"verdict": "supported", "cited_pmid": 2},
        ],
    )]
    m = compute_metrics(rows)
    # pmid 1: grounded + judged relevant -> correct. pmid 2: judged relevant but
    # never retrieved (hallucinated citation) -> not grounded, so not correct.
    assert m["citation_precision"] == 0.5


def test_metrics_are_none_when_denominator_is_empty():
    m = compute_metrics([_row(False, sufficient=False)])
    assert m["recall_at_5"] is None
    assert m["over_refusal_rate"] is None


def test_baseline_metrics_has_no_recall_or_citation_fields():
    rows = [_row(True, True, judge_claims=[{"verdict": "supported"}])]
    m = compute_baseline_metrics(rows)
    assert "recall_at_5" not in m
    assert "citation_precision" not in m
    assert m["faithfulness"] == 1.0


def test_baseline_metrics_hallucination_and_refusal_match_compute_metrics_semantics():
    rows = [
        _row(False, sufficient=False),  # correctly refused
        _row(False, sufficient=True, judge_claims=[{"verdict": "unsupported"}]),  # hallucinated instead
    ]
    m = compute_baseline_metrics(rows)
    assert m["true_refusal_rate"] == 0.5
    assert m["hallucination_rate"] == 1.0
