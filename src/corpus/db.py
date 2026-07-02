import os

import psycopg


def get_papers_by_pmid(pmids: list[int], dsn: str | None = None) -> dict[int, dict]:
    if not pmids:
        return {}
    dsn = dsn or os.environ["POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pmid, title, year, journal FROM papers WHERE pmid = ANY(%s)",
            (list(set(pmids)),),
        )
        rows = cur.fetchall()
    return {r[0]: {"title": r[1], "year": r[2], "journal": r[3]} for r in rows}
