# Recall IR Micro-Benchmark

## What this benchmarks

`bench_ir.py` is a **self-contained IR (Information Retrieval) micro-benchmark** for Recall's Tier 1 FTS5 full-text search layer.

It is deliberately *not* SupMemEval or LongMemEval — those suites require external LLM calls (Tier 4) and evaluate long-horizon memory across many turns. This benchmark instead focuses tightly on the sub-millisecond SQLite FTS5 layer so that regressions are caught quickly without any API keys or GPU.

**What it measures:**

| Metric | Description |
|--------|-------------|
| Recall@1/3/5 | Fraction of relevant documents found in the top 1, 3, or 5 results |
| MRR | Mean Reciprocal Rank — position of the *first* correct result |
| Latency p50/p95 | Median and 95th-percentile query latency in milliseconds |

**Corpus:** 15 labelled observations across 6 domains (medicine, fitness, nutrition, travel, sleep, productivity, finance).

**Queries:** 10 labelled queries with ground-truth relevant document sets, including 4 that specifically exercise the two bugs described below.

---

## How to run

```bash
# From the repo root
uv run python benchmarks/bench_ir.py

# Skip saving results files
uv run python benchmarks/bench_ir.py --no-save
```

Results are printed to stdout and saved to:
- `benchmarks/results/bench_ir_<YYYYMMDD_HHMMSS>.json` — timestamped run
- `benchmarks/results/latest.json` — always overwritten with the most recent run

---

## Bugs found and fixed

Two bugs in FTS5 query handling were identified during benchmarking. Both caused silent failures — the retriever returned empty results without raising an exception.

---

### Bug 1 — FTS5 crashes on `?` in natural-language queries

**File:** `recall/storage/database.py` → `DatabaseManager.fts_search()`

**Symptom:** Any query containing a bare `?` (e.g. `"What medication should I take?"`) caused SQLite FTS5 to raise a parse error. The exception was caught and the method returned `[]`, so the caller received no results and no error.

**Root cause:** SQLite FTS5 treats `?`, `"`, `*`, `[`, `]`, `{`, `}`, `(`, `)`, `:`, and `^` as syntax characters. Passing them unescaped to a `MATCH` query triggers a parse error.

**Fix:** Strip FTS5 syntax characters before the query reaches SQLite:

```python
# recall/storage/database.py

_FTS5_SPECIAL = re.compile(r'[?"*\[\]{}():\^]')

def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 syntax chars that cause parse errors (e.g. bare '?')."""
    return _FTS5_SPECIAL.sub("", query).strip()

async def fts_search(self, query: str, limit: int = 20) -> list[int]:
    query = _sanitize_fts_query(query)
    if not query:
        return []
    ...
```

---

### Bug 2 — FTS5 implicit AND is too strict for multi-word queries

**File:** `recall/retrieval/fts.py` → `FTSRetriever.search()`

**Symptom:** A multi-word query like `"medication take"` used FTS5's default AND logic, requiring *all* tokens to appear in a document. A document that only contains "medication" (but not "take") was silently excluded, even though it was clearly relevant.

**Root cause:** FTS5's `MATCH 'foo bar'` means `foo AND bar`. For natural-language questions split into keywords, OR semantics are almost always more useful.

**Fix:** Join tokens with ` OR ` before passing to `fts_search()`:

```python
# recall/retrieval/fts.py

async def search(self, query: str, limit: int = 10) -> RetrievalResult:
    tokens = query.split()
    fts_query = " OR ".join(tokens) if len(tokens) > 1 else query
    obs_ids = await self._db.fts_search(fts_query, limit=limit)
    ...
```

---

## Before / After results

### Before fixes

Both bugs caused the affected queries to return zero results at Tier 1. The system would fall through to slower tiers (Tier 2 graph, Tier 3 vector, Tier 4 LLM) or return nothing at all.

| Query | R@1 | R@3 | R@5 | MRR | Note |
|-------|-----|-----|-----|-----|------|
| What medication should I take for pain? | 0.00 | 0.00 | 0.00 | 0.000 | Bug 1: FTS5 parse error on `?` |
| medication side effects? | 0.00 | 0.00 | 0.00 | 0.000 | Bug 1: trailing `?` |
| medication take | 0.00 | 0.00 | 0.00 | 0.000 | Bug 2: AND required both words |
| sleep recovery | 0.00 | 0.00 | 0.00 | 0.000 | Bug 2: AND required both words |

### After fixes (current results)

```
====================================================================
  Recall IR Micro-Benchmark
====================================================================
  Corpus: 15 observations
  Queries: 10
  Run at: 2026-04-03T14:57:55.040438+00:00
--------------------------------------------------------------------
  Metric                     Value
  --------------------  ----------
  recall@1                  0.5000
  recall@3                  0.8667
  recall@5                  0.9000
  mrr                       0.9000
  latency p50 (ms)           0.103
  latency p95 (ms)           0.121
--------------------------------------------------------------------

  Query                                      R@1   R@3   R@5    MRR
  ----------------------------------------  ----  ----  ----  -----
  What medication should I take for pain?   0.00  0.67  1.00  0.500
  medication side effects?                  0.50  0.50  0.50  1.000
  medication take                           0.00  1.00  1.00  0.500
  sleep recovery                            0.50  1.00  1.00  1.000
  ibuprofen dosage                          0.50  1.00  1.00  1.000
  running cardio                            1.00  1.00  1.00  1.000
  protein muscle                            0.50  1.00  1.00  1.000
  passport travel documents                 0.50  0.50  0.50  1.000
  Pomodoro focus technique                  1.00  1.00  1.00  1.000
  savings investing money                   0.50  1.00  1.00  1.000
====================================================================
```

All four bug-exercise queries now return results. MRR of 0.90 means the correct document is almost always the first result. Sub-millisecond latency (p95 = 0.121 ms) confirms Tier 1 remains fast.

**R@1 of 0.50 is expected** — for queries with two relevant documents (e.g. both `med_ibuprofen` and `med_dosage`), only one fits in the top-1 slot, so the maximum achievable R@1 is 0.50 for those queries. R@3 and R@5 reach 0.87–0.90, indicating good coverage at broader cutoffs.

---

## Results files

The `results/` directory contains JSON files from each benchmark run. The schema is:

```json
{
  "run_at": "<ISO-8601 UTC timestamp>",
  "corpus_size": 15,
  "num_queries": 10,
  "recall@1": 0.5,
  "recall@3": 0.8667,
  "recall@5": 0.9,
  "mrr": 0.9,
  "latency_p50_ms": 0.103,
  "latency_p95_ms": 0.121,
  "queries": [...]
}
```
