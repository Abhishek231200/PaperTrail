"""Print corpus counts. Usage: python -m src.corpus.stats"""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    dsn = os.environ["POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers")
        n_papers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        cur.execute("SELECT min(year), max(year) FROM papers")
        year_min, year_max = cur.fetchone()
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        n_embedded = cur.fetchone()[0]

    print(f"papers:            {n_papers}")
    print(f"chunks:            {n_chunks}")
    print(f"chunks embedded:   {n_embedded}")
    print(f"year range:        {year_min}-{year_max}")


if __name__ == "__main__":
    main()
