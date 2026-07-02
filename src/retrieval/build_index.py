"""Chunk every paper under the configured strategy and embed any chunks
that don't have an embedding yet. Idempotent - safe to re-run.

Usage: python -m src.retrieval.build_index
"""
import os

import psycopg
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.retrieval.chunk import chunk_paper
from src.retrieval.embed import embed_many

load_dotenv()


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def ensure_chunks(dsn: str, strategy: str, window: int, overlap: int) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT pmid, title, abstract FROM papers")
        papers = cur.fetchall()

        rows = []
        for pmid, title, abstract in tqdm(papers, desc=f"chunking ({strategy})"):
            for ix, text in enumerate(chunk_paper(strategy, title, abstract, window, overlap)):
                rows.append((pmid, ix, strategy, text))

        cur.executemany(
            """
            INSERT INTO chunks (pmid, chunk_ix, strategy, text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (pmid, chunk_ix, strategy) DO NOTHING
            """,
            rows,
        )
        conn.commit()


def embed_missing(dsn: str, strategy: str, model: str) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, text FROM chunks WHERE strategy = %s AND embedding IS NULL",
            (strategy,),
        )
        rows = cur.fetchall()

    if not rows:
        print("no chunks pending embedding")
        return

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"embedding {len(texts)} chunks with {model}...")
    vectors = embed_many(texts, model)

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
            [(_vector_literal(v), i) for v, i in zip(vectors, ids)],
        )
        conn.commit()


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    dsn = os.environ["POSTGRES_DSN"]
    strategy = config["chunking"]["strategy"]
    window = config["chunking"]["window_sentences"]
    overlap = config["chunking"]["window_overlap"]
    model = config["embedding"]["model"]

    ensure_chunks(dsn, strategy, window, overlap)
    embed_missing(dsn, strategy, model)


if __name__ == "__main__":
    main()
