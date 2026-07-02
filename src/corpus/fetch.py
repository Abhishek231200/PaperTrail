"""PubMed E-utilities client: esearch + efetch -> raw XML on disk -> Postgres.

Usage: python -m src.corpus.fetch
Re-running is safe/cheap: raw XML batches already on disk are not re-fetched,
and Postgres upserts are idempotent on pmid.
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

load_dotenv()

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RAW_DIR = Path("data/raw")
EFETCH_BATCH_SIZE = 200

NCBI_API_KEY = os.environ.get("NCBI_API_KEY") or None
NCBI_EMAIL = os.environ.get("NCBI_EMAIL") or None
REQUEST_DELAY = 1.0 / 10 if NCBI_API_KEY else 1.0 / 3


def _params(extra: dict) -> dict:
    p = {"db": "pubmed", "tool": "papertrail", **extra}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(url: str, params: dict) -> requests.Response:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp


def esearch_all_pmids(query: str, target_size: int) -> list[str]:
    resp = _get(f"{EUTILS}/esearch.fcgi", _params({
        "term": query, "retmax": 0, "usehistory": "y",
    }))
    root = etree.fromstring(resp.content)
    count = int(root.findtext("Count"))
    webenv = root.findtext("WebEnv")
    query_key = root.findtext("QueryKey")
    total = min(count, target_size)
    print(f"esearch: {count} PMIDs match query; fetching {total}")

    pmids: list[str] = []
    retmax = 10000
    for start in range(0, total, retmax):
        resp = _get(f"{EUTILS}/esearch.fcgi", _params({
            "term": query, "retstart": start, "retmax": min(retmax, total - start),
            "WebEnv": webenv, "query_key": query_key,
        }))
        root = etree.fromstring(resp.content)
        pmids.extend(id_el.text for id_el in root.findall(".//Id"))
    return pmids[:target_size]


def efetch_batches(pmids: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    batches = [pmids[i:i + EFETCH_BATCH_SIZE] for i in range(0, len(pmids), EFETCH_BATCH_SIZE)]
    for i, batch in enumerate(tqdm(batches, desc="efetch batches")):
        out_path = RAW_DIR / f"batch_{i:04d}.xml"
        if out_path.exists():
            continue
        content = _efetch_batch_validated(batch)
        out_path.write_bytes(content)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _efetch_batch_validated(batch: list[str]) -> bytes:
    """NCBI occasionally injects a proxy error message into an otherwise-200
    response, corrupting the XML. Validate before trusting the response."""
    resp = _get(f"{EUTILS}/efetch.fcgi", _params({
        "id": ",".join(batch), "rettype": "abstract", "retmode": "xml",
    }))
    try:
        etree.fromstring(resp.content)
    except etree.XMLSyntaxError as e:
        raise RuntimeError(f"NCBI returned malformed XML for batch: {e}") from e
    return resp.content


def _abstract_text(article: etree._Element) -> str:
    parts = []
    for ab in article.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        text = "".join(ab.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def _year(article: etree._Element) -> int | None:
    year = article.findtext(".//JournalIssue/PubDate/Year")
    if year and year.isdigit():
        return int(year)
    medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate")
    if medline_date and medline_date[:4].isdigit():
        return int(medline_date[:4])
    return None


def parse_batch(xml_path: Path) -> list[dict]:
    root = etree.parse(str(xml_path)).getroot()
    papers = []
    for pubmed_article in root.findall(".//PubmedArticle"):
        article = pubmed_article.find(".//Article")
        if article is None:
            continue
        pmid_el = pubmed_article.find(".//PMID")
        title = article.findtext("ArticleTitle") or ""
        abstract = _abstract_text(article)
        if pmid_el is None or not abstract:
            continue
        journal = article.findtext(".//Journal/Title")
        mesh_terms = [
            d.text for d in pubmed_article.findall(".//MeshHeading/DescriptorName")
            if d.text
        ]
        papers.append({
            "pmid": int(pmid_el.text),
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": _year(article),
            "mesh_terms": mesh_terms,
        })
    return papers


def upsert_papers(papers: list[dict], dsn: str) -> None:
    if not papers:
        return
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO papers (pmid, title, abstract, journal, year, mesh_terms)
            VALUES (%(pmid)s, %(title)s, %(abstract)s, %(journal)s, %(year)s, %(mesh_terms)s)
            ON CONFLICT (pmid) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                journal = EXCLUDED.journal,
                year = EXCLUDED.year,
                mesh_terms = EXCLUDED.mesh_terms
            """,
            papers,
        )
        conn.commit()


def main() -> None:
    config = yaml.safe_load(Path("config.yaml").read_text())
    dsn = os.environ["POSTGRES_DSN"]

    pmids = esearch_all_pmids(config["corpus"]["query"], config["corpus"]["target_size"])
    efetch_batches(pmids)

    total_papers = 0
    for xml_path in tqdm(sorted(RAW_DIR.glob("batch_*.xml")), desc="parse+upsert"):
        papers = parse_batch(xml_path)
        upsert_papers(papers, dsn)
        total_papers += len(papers)
    print(f"Ingested/updated {total_papers} papers with non-empty abstracts.")


if __name__ == "__main__":
    main()
