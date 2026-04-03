"""FTSRetriever — Tier 1: SQLite FTS5 keyword search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from recall.core.retriever import BaseRetriever, RetrievalResult
from recall.logging import get_logger

if TYPE_CHECKING:
    from recall.storage.database import DatabaseManager

log = get_logger(__name__)


class FTSRetriever(BaseRetriever):
    """
    Tier 1 — SQLite FTS5 full-text search.

    Always on. Sub-millisecond. No external dependencies.
    Returns candidate observation IDs ranked by BM25.
    """

    def __init__(self, db: "DatabaseManager") -> None:
        self._db = db

    @property
    def tier(self) -> int:
        return 1

    @property
    def available(self) -> bool:
        return True  # SQLite FTS5 always available

    async def search(self, query: str, limit: int = 10) -> RetrievalResult:
        t0 = self._now_ms()
        try:
            # OR-join tokens so any matching keyword returns a result.
            # FTS5's default AND logic misses documents that contain only a
            # subset of the query words (e.g. "medication take" misses a doc
            # that only mentions "medication").
            tokens = query.split()
            fts_query = " OR ".join(tokens) if len(tokens) > 1 else query
            obs_ids = await self._db.fts_search(fts_query, limit=limit)
            latency = self._now_ms() - t0
            log.debug(
                "fts_search",
                query=query,
                found=len(obs_ids),
                latency_ms=round(latency, 2),
            )
            return RetrievalResult(
                obs_ids=obs_ids,
                source_tier=self.tier,
                latency_ms=latency,
            )
        except Exception as exc:
            log.warning("fts_retriever_failed", error=str(exc))
            return RetrievalResult(
                source_tier=self.tier, latency_ms=self._now_ms() - t0
            )
