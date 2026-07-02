# SHL Assessment Recommender

## Architecture

```
POST /chat
    │
    ▼
supervisor (1 LLM call: in_scope, mode, stance, constraints, end_of_conversation)
    │
    ├─ not in_scope ──────────► refuse (deterministic template, no LLM) ─► END
    ├─ mode == clarify ───────► generate (asks 1 question, full history) ─► END
    └─ mode in {recommend,compare,refine}
              │
              ▼
          retrieve (hybrid BM25+vector for recommend/refine,
                     exact alias lookup for compare)
              │
              ▼
          generate (selects ONLY from retrieved candidates) ─► END
```

Tier-1 deterministic keyword check runs before the supervisor LLM call and
only catches literal prompt-injection phrasing — it does NOT and cannot
reliably catch off-topic/legal/hiring-advice requests, which are semantic
judgments handled by the supervisor's own `in_scope` field.

Constraints are re-derived from the FULL conversation on every turn, never
patched incrementally — required for cases where a later choice makes an
earlier item redundant without the user explicitly saying "remove X".

## Setup

```bash
# 1. Place the real catalog JSON
mkdir -p catalog/raw
# download shl_product_catalog.json there yourself

# 2. Verify the schema
python scripts/inspect_catalog.py

# 3. Fix field names in app/catalog.py and app/nodes.py if needed

# 4. Build filtered catalog + search index (offline, not at boot)
python -m app.catalog
python -m app.retrieval build

# 5. Set your API key
export ANTHROPIC_API_KEY=sk-...

# 6. Run
uvicorn app.main:app 
```

## Testing

```bash
# Structural smoke test — mocked LLM/retrieval, verifies graph wiring only
python tests/smoke_test_graph.py

# Real behavior tests — write these against the 10 provided traces AND
# your own adversarial cases (injection, off-topic, legal questions).
# The 10 traces do not cover injection/off-topic — you must write those
# probes yourself.
```

## Known open design decisions (documented, not silently resolved)

- Refusal replies are fixed templates per category, not model-generated —
  chosen to avoid an adversarial conversation talking the model into
  softening a refusal over multiple turns.
- Soft internal timeout is 22s (`SOFT_TIMEOUT_SECONDS`), under the
  evaluator's 30s hard limit, so a schema-valid fallback can return before
  the external connection is killed.
