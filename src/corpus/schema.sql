CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
  pmid        BIGINT PRIMARY KEY,
  title       TEXT NOT NULL,
  abstract    TEXT NOT NULL,
  journal     TEXT,
  year        INT,
  mesh_terms  TEXT[] DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
  id          BIGSERIAL PRIMARY KEY,
  pmid        BIGINT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
  chunk_ix    INT NOT NULL,
  strategy    TEXT NOT NULL DEFAULT 'whole_abstract',
  text        TEXT NOT NULL,
  tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  embedding   VECTOR(1536),
  UNIQUE (pmid, chunk_ix, strategy)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_pmid_idx ON chunks (pmid);
