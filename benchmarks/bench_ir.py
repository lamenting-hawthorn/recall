"""
bench_ir.py — IR micro-benchmark for Recall's FTS5 Tier-1 retrieval.

Measures Recall@1/3/5, MRR, and p50/p95 latency against a small labelled
corpus. Designed to be fast (<5 s total) and dependency-free beyond the
recall package itself.

Usage:
    uv run python benchmarks/bench_ir.py
    uv run python benchmarks/bench_ir.py --no-save   # skip writing JSON

Results are saved to:
    benchmarks/results/bench_ir_<YYYYMMDD_HHMMSS>.json
    benchmarks/results/latest.json  (always overwritten)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve project root so the benchmark works whether run from repo root
# or from the benchmarks/ subdirectory.
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from recall.storage.database import DatabaseManager  # noqa: E402
from recall.retrieval.fts import FTSRetriever  # noqa: E402

# ── Corpus ────────────────────────────────────────────────────────────────────
# Each entry is the text that will be stored as an observation.
# Keys are short labels used only for ground-truth wiring below.

CORPUS: dict[str, str] = {
    "med_ibuprofen": (
        "Ibuprofen is a nonsteroidal anti-inflammatory drug used to relieve pain, "
        "reduce fever, and decrease inflammation."
    ),
    "med_dosage": (
        "The recommended adult dosage for ibuprofen is 200–400 mg every 4–6 hours. "
        "Do not exceed 1200 mg per day without medical supervision."
    ),
    "med_side_effects": (
        "Common side effects of medication include nausea, dizziness, and stomach "
        "upset. Serious effects are rare but include GI bleeding."
    ),
    "fitness_running": (
        "Running three times a week improves cardiovascular endurance and helps "
        "maintain a healthy body weight."
    ),
    "fitness_strength": (
        "Strength training with progressive overload increases muscle mass and "
        "improves metabolic rate over time."
    ),
    "nutrition_protein": (
        "Protein intake of 1.6–2.2 g per kg of body weight supports muscle protein "
        "synthesis after resistance exercise."
    ),
    "nutrition_hydration": (
        "Staying hydrated is essential for cognitive performance. Drink at least "
        "eight glasses of water per day."
    ),
    "travel_packing": (
        "When packing for a long trip, roll clothes instead of folding to save "
        "space and reduce wrinkles."
    ),
    "travel_documents": (
        "Always keep travel documents — passport, boarding pass, and travel "
        "insurance — in your carry-on bag."
    ),
    "sleep_hygiene": (
        "Good sleep hygiene includes maintaining a consistent bedtime, avoiding "
        "screens one hour before sleep, and keeping the bedroom cool and dark."
    ),
    "sleep_stages": (
        "Deep sleep (slow-wave sleep) is critical for physical recovery. Most "
        "adults need 7–9 hours of total sleep per night."
    ),
    "productivity_pomodoro": (
        "The Pomodoro technique involves working for 25 minutes then taking a "
        "5-minute break to maintain focus and avoid burnout."
    ),
    "productivity_priorities": (
        "Prioritise tasks using the Eisenhower matrix: urgent+important first, "
        "then important but not urgent, then delegate or drop the rest."
    ),
    "finance_savings": (
        "The 50/30/20 rule allocates 50 % of income to needs, 30 % to wants, "
        "and 20 % to savings and debt repayment."
    ),
    "finance_investing": (
        "Index funds offer broad market diversification at low cost, making them "
        "suitable as a core long-term investing strategy."
    ),
}

# ── Query set ─────────────────────────────────────────────────────────────────
# Each query has:
#   query       — the string sent to FTSRetriever.search()
#   relevant    — set of corpus keys that count as a correct retrieval
#   note        — human-readable description (appears in JSON output)

QUERIES: list[dict] = [
    # ── Bug-exercise queries ───────────────────────────────────────────────
    {
        "query": "What medication should I take for pain?",
        "relevant": {"med_ibuprofen", "med_dosage", "med_side_effects"},
        "note": "Bug 1: bare '?' crashed FTS5 before fix",
    },
    {
        "query": "medication side effects?",
        "relevant": {"med_side_effects", "med_ibuprofen"},
        "note": "Bug 1: trailing '?' on multi-word query",
    },
    {
        "query": "medication take",
        "relevant": {"med_dosage", "med_side_effects"},
        "note": "Bug 2: AND logic missed docs with only one keyword",
    },
    {
        "query": "sleep recovery",
        "relevant": {"sleep_stages", "sleep_hygiene"},
        "note": "Bug 2: multi-word OR should match both sleep docs",
    },
    # ── Standard retrieval queries ─────────────────────────────────────────
    {
        "query": "ibuprofen dosage",
        "relevant": {"med_dosage", "med_ibuprofen"},
        "note": "Exact keyword match",
    },
    {
        "query": "running cardio",
        "relevant": {"fitness_running"},
        "note": "Single-domain fitness query",
    },
    {
        "query": "protein muscle",
        "relevant": {"nutrition_protein", "fitness_strength"},
        "note": "Cross-domain keyword overlap",
    },
    {
        "query": "passport travel documents",
        "relevant": {"travel_documents", "travel_packing"},
        "note": "Travel domain",
    },
    {
        "query": "Pomodoro focus technique",
        "relevant": {"productivity_pomodoro"},
        "note": "Proper noun lookup",
    },
    {
        "query": "savings investing money",
        "relevant": {"finance_savings", "finance_investing"},
        "note": "Finance domain, three keywords",
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant docs found in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved: list[int], relevant_ids: set[int]) -> float:
    """Position of the first relevant result (1-indexed), or 0 if none found."""
    for rank, obs_id in enumerate(retrieved, start=1):
        if obs_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = max(0, int(len(data_sorted) * pct / 100) - 1)
    return data_sorted[idx]


# ── Main benchmark ────────────────────────────────────────────────────────────

async def run_benchmark() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.db"
        async with DatabaseManager(db_path=db_path) as db:
            # ── Load corpus ────────────────────────────────────────────────
            key_to_id: dict[str, int] = {}
            for key, text in CORPUS.items():
                obs_id = await db.write_observation(text)
                key_to_id[key] = obs_id

            retriever = FTSRetriever(db)

            # ── Run queries ────────────────────────────────────────────────
            r1_scores, r3_scores, r5_scores, mrr_scores = [], [], [], []
            latencies: list[float] = []
            query_results = []

            for q in QUERIES:
                relevant_ids = {key_to_id[k] for k in q["relevant"]}

                t0 = time.perf_counter()
                result = await retriever.search(q["query"], limit=10)
                latency_ms = (time.perf_counter() - t0) * 1000

                retrieved = result.obs_ids
                r1 = recall_at_k(retrieved, relevant_ids, 1)
                r3 = recall_at_k(retrieved, relevant_ids, 3)
                r5 = recall_at_k(retrieved, relevant_ids, 5)
                rr = reciprocal_rank(retrieved, relevant_ids)

                r1_scores.append(r1)
                r3_scores.append(r3)
                r5_scores.append(r5)
                mrr_scores.append(rr)
                latencies.append(latency_ms)

                query_results.append(
                    {
                        "query": q["query"],
                        "note": q["note"],
                        "recall@1": round(r1, 3),
                        "recall@3": round(r3, 3),
                        "recall@5": round(r5, 3),
                        "mrr": round(rr, 3),
                        "latency_ms": round(latency_ms, 3),
                        "retrieved_ids": retrieved[:5],
                        "relevant_ids": sorted(relevant_ids),
                    }
                )

    n = len(QUERIES)
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(CORPUS),
        "num_queries": n,
        "recall@1": round(sum(r1_scores) / n, 4),
        "recall@3": round(sum(r3_scores) / n, 4),
        "recall@5": round(sum(r5_scores) / n, 4),
        "mrr": round(sum(mrr_scores) / n, 4),
        "latency_p50_ms": round(percentile(latencies, 50), 3),
        "latency_p95_ms": round(percentile(latencies, 95), 3),
        "queries": query_results,
    }
    return summary


def print_report(results: dict) -> None:
    print("\n" + "=" * 68)
    print("  Recall IR Micro-Benchmark")
    print("=" * 68)
    print(f"  Corpus: {results['corpus_size']} observations")
    print(f"  Queries: {results['num_queries']}")
    print(f"  Run at: {results['run_at']}")
    print("-" * 68)
    print(f"  {'Metric':<20}  {'Value':>10}")
    print(f"  {'-'*20}  {'-'*10}")
    for metric in ("recall@1", "recall@3", "recall@5", "mrr"):
        print(f"  {metric:<20}  {results[metric]:>10.4f}")
    print(f"  {'latency p50 (ms)':<20}  {results['latency_p50_ms']:>10.3f}")
    print(f"  {'latency p95 (ms)':<20}  {results['latency_p95_ms']:>10.3f}")
    print("-" * 68)
    print(f"\n  {'Query':<40}  {'R@1':>4}  {'R@3':>4}  {'R@5':>4}  {'MRR':>5}")
    print(f"  {'-'*40}  {'----':>4}  {'----':>4}  {'----':>4}  {'-----':>5}")
    for q in results["queries"]:
        label = q["query"][:38] + ".." if len(q["query"]) > 40 else q["query"]
        print(
            f"  {label:<40}  {q['recall@1']:>4.2f}  {q['recall@3']:>4.2f}"
            f"  {q['recall@5']:>4.2f}  {q['mrr']:>5.3f}"
        )
    print("=" * 68 + "\n")


def save_results(results: dict, results_dir: Path, no_save: bool) -> None:
    if no_save:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = results_dir / f"bench_ir_{ts}.json"
    latest = results_dir / "latest.json"
    payload = json.dumps(results, indent=2)
    stamped.write_text(payload)
    latest.write_text(payload)
    print(f"  Results saved to:\n    {stamped}\n    {latest}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Recall IR micro-benchmark.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write JSON results files.",
    )
    args = parser.parse_args()

    results = asyncio.run(run_benchmark())
    print_report(results)

    results_dir = Path(__file__).parent / "results"
    save_results(results, results_dir, args.no_save)
