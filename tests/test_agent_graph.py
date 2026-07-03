from src.agent.graph import format_evidence, route_after_verify


def test_format_evidence_includes_pmid_year_title_and_text():
    retrieved = [{"pmid": 123, "year": 2023, "title": "A Trial", "text": "Some finding."}]
    out = format_evidence(retrieved)
    assert "[PMID:123]" in out
    assert "2023" in out
    assert "A Trial" in out
    assert "Some finding." in out


def test_format_evidence_joins_multiple_chunks():
    retrieved = [
        {"pmid": 1, "year": 2020, "title": "T1", "text": "First."},
        {"pmid": 2, "year": 2021, "title": "T2", "text": "Second."},
    ]
    out = format_evidence(retrieved)
    assert out.index("[PMID:1]") < out.index("[PMID:2]")


def test_route_after_verify_retries_only_when_flagged():
    assert route_after_verify({"should_retry": True}) == "retry"
    assert route_after_verify({"should_retry": False}) == "end"
    assert route_after_verify({}) == "end"
