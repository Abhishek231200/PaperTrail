import os

import psycopg


def lexical_search(query: str, strategy: str, top_k: int, dsn: str | None = None) -> list[dict]:
    dsn = dsn or os.environ["POSTGRES_DSN"]

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT pmid, chunk_ix, text, ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE strategy = %s AND tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, strategy, query, top_k),
        )
        rows = cur.fetchall()

    return [{"pmid": r[0], "chunk_ix": r[1], "text": r[2], "score": float(r[3])} for r in rows]
