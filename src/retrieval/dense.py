import os

import psycopg

from src.retrieval.embed import embed_batch


def dense_search(query: str, strategy: str, model: str, top_k: int, dsn: str | None = None) -> list[dict]:
    dsn = dsn or os.environ["POSTGRES_DSN"]
    query_vec = embed_batch([query], model)[0]
    vec_literal = "[" + ",".join(f"{v:.8f}" for v in query_vec) + "]"

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.pmid, c.chunk_ix, c.text, 1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            WHERE c.strategy = %s AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_literal, strategy, vec_literal, top_k),
        )
        rows = cur.fetchall()

    return [{"pmid": r[0], "chunk_ix": r[1], "text": r[2], "score": float(r[3])} for r in rows]
