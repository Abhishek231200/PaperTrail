import re

CITATION_RE = re.compile(r"\[PMID:\s*(\d+)\s*\]")


def extract_cited_pmids(answer: str) -> list[int]:
    return [int(m) for m in CITATION_RE.findall(answer)]
