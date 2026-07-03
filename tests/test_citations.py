from src.evals.citations import extract_cited_pmids


def test_extract_single_citation():
    assert extract_cited_pmids("Weight loss was 20% [PMID:12345678].") == [12345678]


def test_extract_multiple_citations():
    text = "Drug A helps [PMID:111] and drug B also helps [PMID:222][PMID:333]."
    assert extract_cited_pmids(text) == [111, 222, 333]


def test_extract_no_citations():
    assert extract_cited_pmids("INSUFFICIENT_EVIDENCE: nothing in the corpus.") == []


def test_extract_tolerates_internal_whitespace():
    assert extract_cited_pmids("See [PMID: 987654].") == [987654]


def test_extract_ignores_malformed_tag():
    assert extract_cited_pmids("Not a real citation [PMID:] or [PMID]") == []
