"""Ingest ~50 open-access PMC full-text articles to demonstrate chunking on
long documents, not just abstracts.

Prefers papers whose PMID already exists in our abstract corpus (papers
table), so we get an apples-to-apples comparison: the same paper available
both as a single abstract chunk and as many paragraph-level full-text chunks.
Papers not already in the corpus are inserted fresh, using PMC's own
title/abstract front-matter.

Usage: python -m src.corpus.fetch_fulltext
"""
import os
import time
from pathlib import Path

import psycopg
import requests
import yaml
from dotenv import load_dotenv
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.corpus.fetch import NCBI_API_KEY, NCBI_EMAIL, REQUEST_DELAY, _params  # reuse rate-limit config

load_dotenv()

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RAW_DIR = Path("data/raw_fulltext")
TARGET_SIZE = 50
FULLTEXT_STRATEGY = "fulltext_paragraph"
MIN_PARAGRAPH_WORDS = 25
MAX_PARAGRAPH_WORDS = 220


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(url: str, params: dict) -> requests.Response:
    resp = requests.get(url, params={**params, "db": "pmc"}, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp


def esearch_pmc(query: str, target_size: int) -> list[str]:
    resp = _get(f"{EUTILS}/esearch.fcgi", _params({"term": query, "retmax": target_size}))
    root = etree.fromstring(resp.content)
    return [el.text for el in root.findall(".//Id")]


def efetch_pmc_batches(pmcids: list[str], batch_size: int = 20) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    batches = [pmcids[i:i + batch_size] for i in range(0, len(pmcids), batch_size)]
    for i, batch in enumerate(tqdm(batches, desc="efetch fulltext batches")):
        out_path = RAW_DIR / f"batch_{i:04d}.xml"
        if out_path.exists():
            continue
        resp = _get(f"{EUTILS}/efetch.fcgi", _params({
            "id": ",".join(batch), "rettype": "full", "retmode": "xml",
        }))
        try:
            etree.fromstring(resp.content)
        except etree.XMLSyntaxError:
            continue  # skip malformed batch rather than caching garbage
        out_path.write_bytes(resp.content)


def _paragraph_chunks(body: etree._Element) -> list[str]:
    chunks = []
    for p in body.findall(".//p"):
        text = " ".join("".join(p.itertext()).split())
        n_words = len(text.split())
        if n_words < MIN_PARAGRAPH_WORDS:
            continue
        if n_words > MAX_PARAGRAPH_WORDS:
            words = text.split()
            for start in range(0, len(words), MAX_PARAGRAPH_WORDS):
                chunks.append(" ".join(words[start:start + MAX_PARAGRAPH_WORDS]))
        else:
            chunks.append(text)
    return chunks


def _abstract_text(article_meta: etree._Element) -> str:
    abstract_el = article_meta.find("abstract")
    if abstract_el is None:
        return ""
    return " ".join("".join(abstract_el.itertext()).split())


def parse_fulltext_batch(xml_path: Path) -> list[dict]:
    root = etree.parse(str(xml_path)).getroot()
    out = []
    for article in root.findall(".//article"):
        article_meta = article.find(".//article-meta")
        body = article.find(".//body")
        if article_meta is None or body is None:
            continue
        pmid_el = article_meta.find('.//article-id[@pub-id-type="pmid"]')
        pmcid_el = article_meta.find('.//article-id[@pub-id-type="pmc"]')
        title_el = article_meta.find(".//title-group/article-title")
        if pmid_el is None or title_el is None:
            continue
        title = " ".join("".join(title_el.itertext()).split())
        journal_el = article.find('.//journal-title')
        year_el = article_meta.find('.//pub-date/year')
        paragraphs = _paragraph_chunks(body)
        if not paragraphs:
            continue
        out.append({
            "pmid": int(pmid_el.text),
            "pmcid": pmcid_el.text if pmcid_el is not None else None,
            "title": title,
            "abstract": _abstract_text(article_meta) or title,
            "journal": journal_el.text if journal_el is not None else None,
            "year": int(year_el.text) if year_el is not None and year_el.text.isdigit() else None,
            "paragraphs": paragraphs,
        })
    return out


def upsert_paper_if_missing(cur, paper: dict) -> None:
    cur.execute("SELECT 1 FROM papers WHERE pmid = %s", (paper["pmid"],))
    if cur.fetchone():
        return
    cur.execute(
        """
        INSERT INTO papers (pmid, title, abstract, journal, year, mesh_terms)
        VALUES (%s, %s, %s, %s, %s, '{}')
        ON CONFLICT (pmid) DO NOTHING
        """,
        (paper["pmid"], paper["title"], paper["abstract"], paper["journal"], paper["year"]),
    )


def insert_fulltext_chunks(cur, paper: dict) -> None:
    rows = [(paper["pmid"], ix, FULLTEXT_STRATEGY, text) for ix, text in enumerate(paper["paragraphs"])]
    cur.executemany(
        """
        INSERT INTO chunks (pmid, chunk_ix, strategy, text)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (pmid, chunk_ix, strategy) DO NOTHING
        """,
        rows,
    )


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    dsn = os.environ["POSTGRES_DSN"]

    # Same drug terms as the main corpus query, open-access only, PMC's own filter syntax.
    query = ('(semaglutide[Title/Abstract] OR tirzepatide[Title/Abstract] OR '
             'liraglutide[Title/Abstract]) AND open access[filter] AND 2019:2026[PDAT]')
    pmcids = esearch_pmc(query, TARGET_SIZE * 3)  # oversample; many won't have a usable <body>
    efetch_pmc_batches(pmcids)

    n_papers, n_chunks, n_matched_existing = 0, 0, 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT pmid FROM papers")
        existing_pmids = {r[0] for r in cur.fetchall()}

        for xml_path in tqdm(sorted(RAW_DIR.glob("batch_*.xml")), desc="parse+load fulltext"):
            for paper in parse_fulltext_batch(xml_path):
                if n_papers >= TARGET_SIZE:
                    break
                if paper["pmid"] in existing_pmids:
                    n_matched_existing += 1
                upsert_paper_if_missing(cur, paper)
                insert_fulltext_chunks(cur, paper)
                n_papers += 1
                n_chunks += len(paper["paragraphs"])
            conn.commit()
            if n_papers >= TARGET_SIZE:
                break

    print(f"loaded {n_papers} full-text papers ({n_matched_existing} already existed in the abstract "
          f"corpus, {n_papers - n_matched_existing} newly added), {n_chunks} paragraph chunks")


if __name__ == "__main__":
    main()
