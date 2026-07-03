.PHONY: up down ingest ingest-fulltext stats index ask eval report ablate clean

up:
	docker compose up -d
	@echo "waiting for postgres..."
	@until docker compose exec -T db pg_isready -U papertrail -d papertrail > /dev/null 2>&1; do sleep 1; done
	@echo "postgres ready"

down:
	docker compose down

ingest: up
	uv run python -m src.corpus.fetch

ingest-fulltext: up
	uv run python -m src.corpus.fetch_fulltext
	uv run python -m src.evals.build_fulltext_questions

stats:
	uv run python -m src.corpus.stats

index:
	uv run python -m src.retrieval.build_index

retrieve:
	uv run python -m src.retrieval.cli "$(Q)"

ask:
	uv run python -m src.agent.cli "$(Q)"

eval:
	uv run python -m src.evals.run

report:
	uv run python -m src.evals.report

ablate:
	uv run python -m src.evals.ablate

cost-latency:
	uv run python -m src.evals.cost_latency

baseline-norag:
	uv run python -m src.evals.baseline_norag

clean:
	docker compose down -v
