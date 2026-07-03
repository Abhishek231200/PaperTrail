from src.evals.run import recall_at_k


def test_recall_hit_within_k():
    assert recall_at_k([1, 2, 3, 4, 5], expected_pmids=[3], k=5) == 1


def test_recall_miss_outside_k():
    assert recall_at_k([1, 2, 3, 4, 5, 6], expected_pmids=[6], k=5) == 0


def test_recall_any_of_multiple_expected():
    assert recall_at_k([1, 2, 3], expected_pmids=[99, 2], k=5) == 1


def test_recall_none_when_no_expected_pmids():
    # unanswerable golden-set questions have no expected_pmids -- recall is
    # undefined for them, not zero, so they're excluded from the average.
    assert recall_at_k([1, 2, 3], expected_pmids=[], k=5) is None
