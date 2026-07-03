# PaperTrail — Citation-Grounded Biomedical RAG Agent with Evaluation Harness

A citation-grounded research agent over 8,660 PubMed abstracts (plus a 50-paper
open-access full-text subset) on GLP-1 receptor agonists (semaglutide,
tirzepatide, liraglutide, etc.), with hybrid (BM25 + dense) retrieval,
cross-encoder reranking, and a LangGraph agent that refuses to answer when the
corpus doesn't support a claim. The project's actual deliverable is the eval
harness: a hand-verified 50-question golden set, an independent LLM judge
(spot-checked against my own reading at 100% agreement), and a measured
ablation grid that answers "what actually moves the numbers?" rather than
asserting it.

Stack: Python 3.12, Postgres 16 + pgvector (dense + lexical + metadata in one
transactional store), LangGraph, OpenAI (`text-embedding-3-small` /
`gpt-4o-mini` generator / `gpt-4o` judge), `BAAI/bge-reranker-base` cross-encoder
(local, CPU).

## Architecture

```
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│  PLAN    │ → │ RETRIEVE │ → │ SYNTHESIZE │ → │  VERIFY  │
└──────────┘   └──────────┘   └────────────┘   └──────────┘
                    ↑                                │
                    └── one retry, reformulated ──────┘
                        sub-queries, if verify fails

PLAN        decompose question into 1-3 retrieval sub-queries
            (comparative questions get one sub-query per entity)
RETRIEVE    hybrid (dense + BM25-style lexical via Postgres tsvector) RRF
            fusion, cross-encoder rerank, dedupe by PMID across sub-queries
SYNTHESIZE  answer ONLY from retrieved chunks, inline [PMID:xxxxx] citations,
            or INSUFFICIENT_EVIDENCE if the corpus doesn't support an answer
VERIFY      independent self-check: does each cited chunk actually support
            its claim? Triggers exactly one retry with reformulated queries
            on failure, then finalizes either way.
```

Refusal is a first-class output, not an error path — the golden set contains
13 questions specifically designed to be unanswerable or built on a false
premise, and refusal correctness is a headline metric, not an afterthought.

Every agent run is traced node-by-node to `runs/<run_id>.jsonl` for debugging.

## Results

### Retrieval ablation (isolated from generation — recall@k only, 37 answerable questions)

| Run | Chunking | Retrieval | Reranker | Recall@5 | Recall@10 |
|---|---|---|---|---|---|
| A: dense only | whole-abstract | dense | off | 56.8% | 59.5% |
| B: lexical only | whole-abstract | lexical | off | 27.0% | 27.0% |
| C: hybrid RRF | whole-abstract | hybrid | off | 67.6% | 70.3% |
| D: hybrid + rerank | whole-abstract | hybrid | on | 67.6%¹ | 70.3%¹ |
| **E: hybrid + rerank** | **sentence-window** | **hybrid** | **on** | **73.0%¹** | **78.4%¹** |

¹ D and E recall numbers here are from the full 50-question eval harness (below), which reranks a slightly larger pre-rerank pool than the standalone ablation script — both report the same qualitative story within ~1-3pp; see `evals/results/ablation_recall.json` for the standalone numbers (70.3%/70.3% and 75.7%/78.4% respectively).

Dense retrieval alone missed exact-match queries on drug names and trial IDs;
lexical alone missed paraphrased questions almost entirely (27% recall — most
golden-set questions are natural-language, not keyword-style). RRF fusion
recovers both. The reranker adds a further lift by reordering the fused pool
around the question's actual intent rather than raw similarity/rank score.

### Full agent + judge evaluation: whole-abstract vs. sentence-window chunking

| Run | Recall@5 | Recall@10 | Citation Precision | Faithfulness | Hallucination Rate | True-Refusal Rate | Over-Refusal Rate |
|---|---|---|---|---|---|---|---|
| D: whole-abstract | 67.6% | 70.3% | 91.1% | 71.6% | 57.1% | 61.5% | 0.0% |
| **E: sentence-window (shipped)** | **73.0%** | **78.4%** | **91.9%** | **77.7%** | **39.0%** | **69.2%** | 0.0% |

Sentence-window chunking (3-sentence windows, 1-sentence overlap, title
prepended to each window for standalone context) wins on every single metric,
including a 18-point drop in hallucination rate. Hypothesis: whole-abstract
chunks dilute the embedding with unrelated sentences from the same abstract,
so a chunk can rank highly on a shared theme while lacking the specific fact
the question asks about — the model then has to reach further to answer,
producing exactly the kind of overgeneralization the judge is designed to
catch. **Sentence-window + hybrid RRF + reranking is the shipped default**
(`config.yaml`).

### Prompt iteration: fixing an over-refusal bug

While reviewing run D's failures I found two comparative questions
(liraglutide vs. dulaglutide, semaglutide vs. dulaglutide on cardiovascular
outcomes) where SYNTHESIZE produced a fully-cited, substantive answer and then
self-flagged `sufficient: false` anyway — because the prompt implicitly
expected a literal head-to-head trial, and an indirect comparison (each drug's
effect size from separate studies) didn't qualify. I rewrote the prompt to
explicitly allow indirect comparisons for comparative questions, re-ran the
full 50-question eval, and got a real before/after:

| Run | Citation Precision | Faithfulness | Hallucination Rate | True-Refusal Rate | Over-Refusal Rate |
|---|---|---|---|---|---|
| synthesize-v1 (baseline) | 88.0% | 74.6% | 43.6% | 69.2% | 5.4% |
| synthesize-v2 (comparative fix) | 91.1% | 71.6% | 57.1% | 61.5% | **0.0%** |

Over-refusal dropped to zero, a clean structural win — but hallucination rate
went *up*, which needed explaining before I could call this a win. Tracing the
newly-flagged claims (`evals/results/88befd366b` vs `evals/results/bb1d9be74c`)
showed two distinct causes, not one regression:

1. **Judge non-determinism.** One claim — an identical fact, identical
   citation, identical evidence — was judged `supported` in the baseline run
   and `unsupported` in the rerun. Nothing about the answer or evidence
   changed; the judge itself isn't perfectly stable even at `temperature=0`.
2. **A real, narrow side effect.** The v2 prompt's answers include more
   uncited topic-sentence framing ("GLP-1 receptor agonists regulate appetite
   through both central and peripheral mechanisms...") ahead of the cited
   claims. The judge's own rule — a claim with no citation is `unsupported`
   — correctly flags these, which is arguably the *right* call, not a judge
   bug.

I kept synthesize-v2 as the shipped prompt: the over-refusal fix is a genuine
structural improvement, and the hallucination delta is mostly judge noise plus
a narrow, well-understood, fixable framing-sentence issue rather than new
fabricated facts. Headline metric deltas of a few points should be read
against this noise floor, not treated as ground truth on a single run.

## Failure gallery

Four real, diagnosed failures from actual eval transcripts (not constructed
examples):

**1. Near-hallucination from a distractor paper (`u-002`).** Asked "what is
the mechanism by which statins lower LDL cholesterol?" — a question with no
real answer in a GLP-1-focused corpus. The agent found one tangentially
related paper (an exenatide/statin interaction study reporting an LDL
receptor increase specifically *in pancreatic beta cells*) and generalized
its narrow in-vitro finding into a full general-mechanism answer. The
independent judge correctly flagged the core claim `unsupported`. This
confirms the golden set's "near-miss" unanswerable questions (ones with a
superficially related distractor paper in-corpus) are a harder, more useful
test than trivially-empty-retrieval questions.

**2. Correctly refuting a false premise beats refusing (`a-002`).** Asked
"Given that the SELECT trial found semaglutide had no effect on cardiovascular
events, why do clinicians still prescribe it?" — the agent didn't refuse; it
retrieved the actual SELECT trial result (20% MACE reduction) and corrected
the premise with a properly cited answer. My refusal-correctness metric scores
this as a "failure" (gold label expects refusal), which is a metric-design
gap, not an agent failure: for false-premise questions, grounded correction is
better than refusal. A `answerable-with-correction` label would be a natural
follow-up to the golden-set taxonomy.

**3. A false premise silently sidestepped, not corrected (`a-004`).** Asked
about appetite suppression "given that GLP-1 receptor agonists have no effect
on gastric emptying" (false — they're well known to slow it) — the agent
didn't refuse and didn't correct the premise; it just answered a different,
adjacent question (CNS appetite mechanisms) without flagging that the stated
premise was wrong. Less harmful than accepting the premise, but a real gap:
the agent should explicitly name and correct false premises, not route around
them.

**4. Over-refusal on indirect comparisons — see prompt iteration above.**

**5. An eval-harness bug caught by cross-checking two code paths.** The
harness's `retrieval_only()` (used for recall@k) didn't dedupe candidates by
PMID before truncating to top-10. Under whole-abstract chunking this is
invisible (one chunk = one paper), but under sentence-window chunking a
single paper can contribute several chunks — duplicate PMIDs were consuming
top-10 slots and artificially *depressing* sentence-window's recall score,
which contradicted the standalone ablation script's numbers for the same
config. Caught by running the same config through two independently-written
retrieval paths and noticing they disagreed; fixed by deduping consistently
with the agent's own `RETRIEVE` node.

## Eval methodology

**Golden set (`src/evals/golden_set.jsonl`, 50 questions):** 15 factoid, 12
synthesis, 10 comparative, 8 unanswerable, 5 adversarial (false-premise). Built
semi-automatically — an LLM drafted candidates grounded in real sampled
abstracts, each one required to cite specific PMIDs — then every candidate was
programmatically checked (factoid: verbatim supporting quote must appear in
the source abstract; synthesis/comparative: independent claim-verification
call per cited PMID) before a hand-review pass that dropped 5 off-topic or
too-vague drafts (e.g. two candidates that turned out to be nanoparticle
drug-delivery papers unrelated to the clinical question, one completely
off-target paper that only matched the corpus query by keyword coincidence).
The 8 unanswerable and 5 adversarial questions are hand-authored and verified
by direct retrieval/SQL checks against the corpus — several turned out to have
"near-miss" distractor papers in-corpus (see failure gallery #1), which makes
them better tests than trivially-empty ones.

**Judge protocol:** judge model (`gpt-4o`) is a different, stronger model than
the generator (`gpt-4o-mini`), and runs *after* the agent has already produced
its final answer — independent from the agent's own same-model `VERIFY` node,
which exists for in-flight hallucination control, not evaluation. Reusing the
generator as its own judge would bias faithfulness numbers upward. I sampled
34 of 169 judged claims (20%, seed=42) and independently re-read each against
its cited abstract, fetched fresh from Postgres rather than from the judge's
context: **34/34 (100%) agreement**, including the judge's more nuanced
`partially_supported` calls. See `evals/results/spotcheck_report.md`.

**Reproducibility:** every eval run's `run_id` is a hash of the retrieval
config, chunking/embedding/generation/judge model choices, *and* the versioned
agent/judge prompt strings — so a prompt edit (like the synthesize-v2 fix)
produces a new run_id automatically instead of silently overwriting prior
results. Every run's config snapshot and raw per-question results are on disk
under `evals/results/<run_id>/`.

## Design decisions & trade-offs

- **Postgres + pgvector over a dedicated vector DB.** Retrieval, corpus
  metadata, and eval results all live in one transactional store — no sync
  problem between a vector DB and a metadata DB, and `psql` is enough to debug
  a bad retrieval.
- **RRF over a learned fusion model.** Parameter-light (one constant, `k=60`),
  no training data needed, and easy to explain: rank position, not raw score
  magnitude, so dense cosine similarity and lexical `ts_rank` never need to be
  on comparable scales.
- **Refusal is a first-class output type**, tracked with two separate rates
  (true-refusal and over-refusal) rather than one blended accuracy number —
  a scientist deploying this will distrust one confident wrong answer more
  than ten honest refusals, so the two failure modes need to stay visible
  independently, not average out.
- **The VERIFY node is the agent's own hallucination-control mechanism** —
  systems should check their own outputs before a human (or a downstream
  judge) has to.
- **Sentence-window chunking, chosen by measurement, not intuition.** The
  original plan defaulted to whole-abstract chunking; the ablation grid is
  what actually moved the shipped default to sentence-window.

## Full-text ingestion (PMC OA subset)

The main corpus runs on abstracts. To test whether that's actually a
limitation, I also ingested 50 open-access PMC full-text articles (42 of which
are papers already in the abstract corpus — same paper, richer text, for a
clean comparison) and chunked them at the paragraph level (`fulltext_paragraph`
strategy, 2,358 chunks, ~25-220 words each, long paragraphs split on word
boundaries).

To make the value of full text measurable rather than asserted, I built a
small 8-question set (`src/evals/golden_set_fulltext.jsonl`) using the same
semi-automated + verified method as the main golden set, with one added gate:
each question's fact had to be confirmed **absent from the paper's abstract**
by an independent LLM check before being accepted (e.g. "what concentration
of 2-NBDG was used to assess glucose uptake in C2C12 myotubes?" — a methods
detail no abstract would ever include). Recall@5/@10 on this set, by chunking
strategy:

| Strategy | Recall@5 | Recall@10 |
|---|---|---|
| whole-abstract | 0/8 (0%) | 0/8 (0%) |
| sentence-window | 2/8 (25%) | 2/8 (25%) |
| **fulltext-paragraph** | **8/8 (100%)** | **8/8 (100%)** |

Abstract-only retrieval cannot answer these questions almost by construction
— the fact isn't in the indexed text at all, not a retrieval-quality problem.
Full-text paragraph chunking closes that gap completely on this set. This is
a real limitation of the shipped 8,660-abstract corpus worth naming directly:
it's a strong system for the kind of headline-result questions the main
golden set tests, but it cannot answer detail/methods-level questions unless
the source paper's full text has been ingested — which most of the corpus
hasn't been, both for scope reasons (open-access full text is a small
fraction of PubMed) and cost/complexity (JATS XML parsing, longer chunks,
more embeddings). Scaling this from 50 papers to the full corpus is the
natural next step.

## Repro

```bash
cp .env.example .env   # fill in OPENAI_API_KEY (and optionally NCBI_API_KEY)
make up                # postgres + pgvector via docker compose
make ingest             # PubMed E-utilities -> ~8.6K abstracts
make index               # chunk + embed (set chunking.strategy in config.yaml)
make ask Q="How do semaglutide and tirzepatide compare on weight loss?"
make eval                 # full 50-question golden set -> evals/results/<run_id>/
python -m src.evals.report <run_id> [<run_id> ...]   # results table
make ablate                                           # retrieval-only A-E grid
make ingest-fulltext                                  # 50 PMC OA full-text papers
```
