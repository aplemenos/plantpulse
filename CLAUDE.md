# PlantPulse: notes for Claude

Context handed over from an earlier Cowork conversation, so a new session can continue the work.

## What this is

An agentic analytics service for solar and wind sites, built as a portfolio project for ML-engineering and AI-agent roles (for example the Kimchi/CAST AI "Harness" team). The owner asks plain-English questions ("is anything wrong at wind-02, and have we seen it before?") and an agent answers using ML models, retrieval and tools.

The data is synthetic and seeded: the whole pipeline rebuilds identically, which the CI regression gate relies on.

## Architecture (where things live)

- `ml/`: data generation, features, training (MLflow registry with `@champion` aliases and signatures), `promote.py` (champion/challenger, rollback), `vector_store.py` (hybrid search: FAISS vectors + BM25 with RRF; bge-small via fastembed, falls back to TF-IDF), `retrieval_eval.py`, `regression_check.py` (gate), `benchmark_parallel.py`.
- `app/`: FastAPI service. `registry.py` loads `@champion` models and checks signatures against computed features; `state.py`, `inference.py`, `cache.py` (Redis or in-memory), `main.py` (routes, SSE stream), `static/index.html` (UI).
- `agent/`:
  - `tools.py`: tool specs + `dispatch` (each call is an MLflow span).
  - `orchestrator.py`: `/agent/query`, Claude tool-use loop plus a rule-based fallback, citation checking.
  - `investigate.py`: `/agent/investigate`, planner → executor → independent verifier, one revision round; three model roles (Sonnet plans and judges, Haiku drafts; configurable via env).
  - `verifier.py`: deterministic checks (numbers, citations, health status; quoted log text is ignored).
  - `sessions.py`: multi-turn conversations; append-only SQLite event log (`var/sessions.db`), context built from the log, compaction after 3 turns, resume of interrupted turns.
  - `mcp_server.py`: MCP server (FastMCP, `mcp==1.12.4`): the three model tools via `dispatch`, `investigate` via sessions (returns `session_id`), `plantpulse://sites`; stdio or `--http`. No `from __future__ import annotations` there: FastMCP needs real types.
  - `tracing.py`: MLflow tracing setup.
- `tests/`: 83 tests. `typings/`: local type stubs.

## Commands

```bash
make install     # pip install -r requirements-dev.txt
make pipeline    # data -> train (MLflow) -> vector index
make test        # pytest
make lint        # pyright (strict) + ruff
make regress     # regression gate vs ml/regression_baseline.json
make baseline    # accept current numbers (merges per-embedder keys)
make retrieval   # vector vs BM25 vs hybrid on golden queries
make models      # registry versions; python -m ml.promote <model> <version> to roll back
make serve       # http://localhost:8000
make mcp         # MCP server, stdio (mcp-http: :8765/mcp, mcp-inspect: Inspector)
make mlflow-ui   # http://127.0.0.1:5050 (not 5000: AirPlay owns that port on macOS)
```

## Conventions

- Pyright strict and ruff must stay at zero findings; add stubs to `typings/` rather than ignoring errors.
- Every change keeps `make test` and `make regress` green. The gate covers model quality, p95 latency, agent routing, retrieval (per embedder), and the share of golden investigations that pass the verifier (must be 1.0).
- Tests never call the real Anthropic API: `conftest.py` removes `ANTHROPIC_API_KEY`; LLM paths are tested with a scripted fake client. The regression gate runs investigations with `allow_llm=False`.
- `mcp` is pinned to 1.x: 2.x requires starlette>=1, which needs a FastAPI upgrade first.
- Never commit secrets (`.env`, API keys). `var/`, `.cache/`, `mlruns/` are ignored.
- Commit messages: imperative summary line, then a short body explaining why.

## Decisions worth knowing

- Two-stage anomaly detection (expected-power regressor, then a classifier on residuals) so curtailment is not mistaken for a fault.
- Models are saved with skops (not pickle), with `sklearn.tree._tree.Tree` as the only trusted type.
- The verifier never sees the drafter's reasoning: no self-verification. Unverified answers are returned marked as such, never hidden.
- Session context is a view built from the log; compaction adds a summary event and never deletes events.
- With the bge embedder, a curtailment log ending "equipment healthy" once made the verifier reject a correct answer; quoted log text is now stripped before status checks (regression test included).

## Status and next steps

Done: MLflow signatures/registry/tracing, hybrid RAG with citations and retrieval eval, plan → execute → verify, sessions, MCP server.

Next, in order:
1. **Action with human approval**: `create_maintenance_ticket` waits for approval; audit log.
2. **Self-hosted model**: Ollama or vLLM on k3s behind an OpenAI-compatible endpoint, selectable as the agent's model.
3. **Prometheus `/metrics` + Grafana**: latency, cache hits, tokens and cost.
4. Optional: let plain Ask mode use sessions too (today only Investigate mode keeps a conversation).

Not yet verified against the real Claude API: the investigate pipeline's LLM mode. Run one real investigation with a key and check `"mode": "llm"` and `verification.passed`.

## Owner

Anargyros: senior Go backend engineer (Power Factors, SCADA, MCP servers), building ML and agent skills. Prefers short, clean answers.
