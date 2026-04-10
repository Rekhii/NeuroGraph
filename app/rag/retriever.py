"""
Retrieval module — app/rag/retriever.py (v3)

Single interface for all retrieval in NeuroGraph. Every agent
(planner, critic, writer) talks to the Retriever, never to
ChromaDB or BM25 directly.

v3 changes (architecture fixes):
    1. No more tight coupling: uses store.get_all_documents()
       instead of store._collection.get()
    2. Planner integration: search_with_plan() accepts a Plan
       directly, AdaptiveRouter is fallback only
    3. Feedback loop: search_with_feedback() supports critic
       re-retrieval with refined queries
    4. Consistent scoring: ScoreNormalizer calibrates across
       retrieval methods and query runs
    5. Context optimization: integrates with ContextOptimizer
       for LLM-ready output

Pipeline:
    planner.Plan (or raw query)
      -> strategy selection (plan > router > default)
      -> retrieval (vector / keyword / hybrid)
      -> score normalization
      -> reranking (cross-encoder)
      -> context optimization (relevance + dedup + compression + budget)
      -> RetrievalResponse

Feedback loop (critic integration):
    critic says "evidence insufficient"
      -> search_with_feedback(query, previous_results, feedback)
      -> excludes already-seen chunks
      -> tries alternative strategies
      -> merges new results with previous best

Usage:
    from app.rag.retriever import Retriever

    retriever = Retriever()

    # Simple search
    response = retriever.search("What is active inference?")

    # Search with planner
    plan = planner.plan("Compare GWT and IIT")
    response = retriever.search_with_plan(plan)

    # Critic feedback loop
    response2 = retriever.search_with_feedback(
        query="What is active inference?",
        previous_results=response.results,
        feedback="Need more detail on free energy minimization",
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.embedder import VectorStore, get_store

if TYPE_CHECKING:
    from app.agents.planner import Plan

logger = logging.getLogger(__name__)


# Score normalization

class ScoreNormalizer:
    """
    Calibrates scores across retrieval methods and queries.

    Problem: vector score 0.8 != BM25 score 0.8 != rerank score 0.8.
    Each method has different score distributions. Without
    normalization, merging results from different methods or
    comparing scores across queries is unreliable.

    Solution: track score distributions per method and normalize
    using percentile-based calibration. A score of 0.8 means
    "better than 80% of results from this method."

    The normalizer maintains a rolling window of recent scores
    per method. After enough samples, it can map raw scores to
    calibrated percentiles. Before calibration, it uses min-max
    normalization as a baseline.
    """

    WINDOW_SIZE = 200

    def __init__(self):
        self._history: dict[str, list[float]] = {
            "vector": [],
            "keyword": [],
            "rerank": [],
        }

    def record(self, method: str, scores: list[float]) -> None:
        """Add scores to the rolling history for a method."""
        if method not in self._history:
            self._history[method] = []
        window = self._history[method]
        window.extend(scores)
        if len(window) > self.WINDOW_SIZE:
            self._history[method] = window[-self.WINDOW_SIZE:]

    def normalize(self, method: str, score: float) -> float:
        """
        Normalize a raw score to a calibrated 0-1 range.

        If enough history exists (50+ scores), uses percentile
        calibration. Otherwise, returns the raw score (already
        0-1 from per-query normalization).
        """
        window = self._history.get(method, [])

        if len(window) < 50:
            # Not enough data for calibration, return as-is
            return score

        # Percentile: what fraction of historical scores is this above?
        below = sum(1 for s in window if s < score)
        return round(below / len(window), 4)

    def is_calibrated(self, method: str) -> bool:
        """Whether enough data exists for reliable calibration."""
        return len(self._history.get(method, [])) >= 50


# Result data structures

@dataclass
class RetrievalResult:
    """
    One retrieved chunk with all metadata.

    Score fields:
        score: initial retrieval score (0-1, per-query normalized)
        rerank_score: cross-encoder score (0-1, per-batch normalized)
        calibrated_score: percentile across queries (0-1, if calibrated)
        effective_score: best available score for RANKING within a single query

    IMPORTANT — score interpretation:
        These scores are reliable for ORDERING results within one search.
        They are NOT globally calibrated confidence values. A score of 0.8
        from query A is not comparable to 0.8 from query B unless the
        ScoreNormalizer has accumulated enough history (50+ queries) for
        percentile calibration. Do not treat effective_score as a probability
        or absolute quality measure — use it only for ranking and threshold
        checks within the same retrieval call.
    """
    text: str
    source: str
    page: int
    chunk_id: str
    score: float
    strategy: str
    query: str
    rerank_score: float | None = None
    calibrated_score: float | None = None
    is_reranked: bool = False

    @property
    def effective_score(self) -> float:
        """Best available score, preferring calibrated > rerank > raw."""
        if self.calibrated_score is not None:
            return self.calibrated_score
        if self.is_reranked and self.rerank_score is not None:
            return self.rerank_score
        return self.score

    def to_dict(self) -> dict:
        result = {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "effective_score": round(self.effective_score, 4),
            "strategy": self.strategy,
            "query": self.query,
        }
        if self.is_reranked:
            result["rerank_score"] = round(self.rerank_score, 4) if self.rerank_score else None
        if self.calibrated_score is not None:
            result["calibrated_score"] = round(self.calibrated_score, 4)
        return result


@dataclass
class RetrievalResponse:
    """
    Complete retrieval output with pipeline metadata.
    """
    query: str
    results: list[RetrievalResult]
    strategy: str
    sub_queries: list[str] = field(default_factory=list)
    total_candidates: int = 0
    reranked: bool = False
    iteration: int = 1
    feedback: str = ""
    excluded_chunks: int = 0

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    @property
    def top_score(self) -> float:
        if not self.results:
            return 0.0
        return self.results[0].effective_score

    @property
    def chunk_ids(self) -> set[str]:
        """All chunk IDs in this response (for feedback exclusion)."""
        return {r.chunk_id for r in self.results}

    def to_context_string(self) -> str:
        """Format results for LLM prompt with numbered citations."""
        if not self.results:
            return "No relevant documents found."

        parts = []
        for i, r in enumerate(self.results, 1):
            parts.append(
                f"[{i}] (Source: {r.source}, Page {r.page}, "
                f"relevance={r.effective_score:.3f})\n{r.text}"
            )
        return "\n\n".join(parts)

    def summary(self) -> str:
        stages = [self.strategy]
        if self.reranked:
            stages.append("reranked")
        if self.iteration > 1:
            stages.append(f"iter={self.iteration}")
        if self.excluded_chunks:
            stages.append(f"excluded={self.excluded_chunks}")
        pipeline = " -> ".join(stages)
        return (
            f"[{pipeline}] "
            f"{len(self.results)} results, "
            f"top={self.top_score:.4f}, "
            f"candidates={self.total_candidates}"
        )


# Cross-encoder reranker

class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
        - 22M parameters, ~80MB, CPU-only
        - Trained on MS MARCO passage ranking
        - No API key needed

    Lazy-loaded on first use. If sentence-transformers is not
    installed, reranking is silently skipped.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._available: bool | None = None

    def _load_model(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            self._available = True
            logger.info("Cross-encoder loaded: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, reranking disabled"
            )
            self._available = False
        except Exception as e:
            logger.warning("Failed to load cross-encoder: %s", e)
            self._available = False

        return self._available

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results or not self._load_model():
            return results

        pairs = [(query, r.text) for r in results]
        raw_scores = self._model.predict(pairs)

        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max_s - min_s

        for result, raw in zip(results, raw_scores):
            if score_range > 0:
                normalized = (raw - min_s) / score_range
            else:
                normalized = 0.5
            result.rerank_score = round(float(normalized), 4)
            result.is_reranked = True

        results.sort(key=lambda r: r.rerank_score, reverse=True)

        if top_k:
            results = results[:top_k]

        logger.info(
            "Reranked %d results, top=%.4f",
            len(results), results[0].rerank_score if results else 0.0,
        )

        return results


# BM25 keyword search

class BM25Index:
    """
    Lightweight keyword index built from VectorStore documents.

    Uses store.get_all_documents() (public API) instead of
    accessing internal collection objects directly.
    """

    def __init__(self, documents: list[dict]):
        self._documents = documents
        self._tokenized = [
            self._tokenize(doc["text"]) for doc in documents
        ]
        self._index = BM25Okapi(self._tokenized) if self._tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._index or not self._documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        scored = sorted(
            zip(scores, self._documents),
            key=lambda x: x[0],
            reverse=True,
        )

        max_score = scored[0][0] if scored and scored[0][0] > 0 else 1.0

        results = []
        for score, doc in scored[:top_k]:
            if score <= 0:
                break
            results.append({
                "text": doc["text"],
                "source": doc.get("source", ""),
                "page": doc.get("page", 0),
                "chunk_id": doc.get("chunk_id", ""),
                "score": round(score / max_score, 4),
            })

        return results


# Query expansion

def expand_query(query: str) -> list[str]:
    """
    Split a complex query into focused sub-queries.

    Rule-based approach for comparison and multi-topic patterns.
    The planner agent provides better decomposition via LLM when
    available, so this is the fallback.
    """
    sub_queries = [query]

    comparison_patterns = [
        r"(?:compare|comparing)\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)",
        r"how\s+(?:does|do|is|are)\s+(.+?)\s+differ(?:s|ent)?\s+from\s+(.+)",
        r"(?:differ(?:s|ence)?|distinction)\s+(?:between\s+)?(.+?)\s+(?:and|from|vs\.?)\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
    ]

    for pattern in comparison_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            a, b = match.group(1).strip(), match.group(2).strip()
            a = re.sub(r"[?.!,]+$", "", a).strip()
            b = re.sub(r"[?.!,]+$", "", b).strip()
            if a and b:
                sub_queries.extend([a, b])
            break

    if len(sub_queries) == 1:
        match = re.search(
            r"(?:explain|describe|summarize|what (?:is|are))\s+(.+?)\s+and\s+(.+)",
            query,
            re.IGNORECASE,
        )
        if match:
            a, b = match.group(1).strip(), match.group(2).strip()
            a = re.sub(r"[?.!,]+$", "", a).strip()
            b = re.sub(r"[?.!,]+$", "", b).strip()
            if a and b and len(a) < 60 and len(b) < 60:
                sub_queries.extend([a, b])

    return sub_queries


# Result merging

def _merge_results(
    result_groups: list[list[RetrievalResult]],
    top_k: int,
    exclude_ids: set[str] | None = None,
) -> list[RetrievalResult]:
    """
    Merge, deduplicate, optionally exclude, and rank results.

    The exclude_ids parameter supports the feedback loop:
    on re-retrieval, chunks the critic already rejected are
    excluded so the retriever finds new evidence.
    """
    exclude = exclude_ids or set()
    best: dict[str, RetrievalResult] = {}

    for group in result_groups:
        for result in group:
            if result.chunk_id in exclude:
                continue
            existing = best.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                best[result.chunk_id] = result

    ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
    return ranked[:top_k]


# Adaptive router (fallback when no Plan provided)

class AdaptiveRouter:
    """
    Picks retrieval strategy based on query patterns.

    Used ONLY when no Plan is provided. When the planner agent
    produces a Plan, its strategy takes precedence and the
    router is not consulted. This eliminates the duplication
    between planner and retriever routing logic.
    """

    @dataclass
    class RouteDecision:
        strategy: str
        expand: bool
        reason: str

    FACTUAL_PATTERNS = [
        r"^(?:who|when|where|which)\b",
        r"^(?:what (?:is|was|are|were) the)\b",
        r"^(?:name|list|define)\b",
        r"\b(?:in what year|how many|what date)\b",
    ]

    CONCEPTUAL_PATTERNS = [
        r"^(?:how does|how do|how is|how are)\b",
        r"^(?:why does|why do|why is|why are)\b",
        r"^(?:explain|describe|elaborate|what role)\b",
        r"\b(?:mechanism|theory|framework|model|principle|process)\b",
    ]

    COMPARISON_PATTERNS = [
        r"\b(?:compare|comparing|comparison)\b",
        r"\b(?:differ|difference|distinction|versus|vs\.?)\b",
        r"\b(?:better|worse|advantage|disadvantage)\b",
        r"\b(?:contrast|similarities|similar to)\b",
    ]

    def route(self, query: str) -> RouteDecision:
        q_lower = query.lower().strip()

        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, q_lower):
                return self.RouteDecision("hybrid", True, "comparison")

        has_acronym = bool(re.search(r"\b[A-Z]{2,}\b", query))
        has_number = bool(re.search(r"\b\d+\.?\d*\b", query))

        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, q_lower):
                return self.RouteDecision("keyword", False, "factual")

        word_count = len(q_lower.split())
        if word_count <= 5 and (has_acronym or has_number):
            return self.RouteDecision("keyword", False, "short+specific")

        for pattern in self.CONCEPTUAL_PATTERNS:
            if re.search(pattern, q_lower):
                return self.RouteDecision("vector", False, "conceptual")

        return self.RouteDecision("hybrid", False, "default")


# Main retriever

class Retriever:
    """
    Single retrieval interface for all agents.

    Three entry points:
        search()                — raw query, router decides strategy
        search_with_plan()      — planner Plan controls everything
        search_with_feedback()  — critic feedback loop for re-retrieval

    The planner is the brain, the retriever executes. When a Plan
    is provided, the retriever follows it without running its own
    router. The AdaptiveRouter is only consulted when no Plan exists.
    """

    STRATEGIES = ("vector", "keyword", "hybrid")

    def __init__(
        self,
        store: VectorStore | None = None,
        default_top_k: int = settings.top_k,
        rerank_by_default: bool = True,
    ):
        self._store = store or get_store()
        self._default_top_k = default_top_k
        self._rerank_by_default = rerank_by_default

        # Components (lazy init where needed)
        self._router = AdaptiveRouter()
        self._reranker: CrossEncoderReranker | None = None
        self._normalizer = ScoreNormalizer()
        self._bm25_cache: BM25Index | None = None
        self._bm25_doc_count: int = -1

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _get_bm25(self) -> BM25Index:
        """
        Build or return cached BM25 index.

        Uses store.get_all_documents() (public API) instead of
        accessing internal _collection. This is the fix for
        tight coupling.
        """
        current_count = self._store.count

        if self._bm25_cache is None or self._bm25_doc_count != current_count:
            if current_count == 0:
                self._bm25_cache = BM25Index([])
                self._bm25_doc_count = 0
                return self._bm25_cache

            # PUBLIC API — no more store._collection.get()
            documents = self._store.get_all_documents()

            self._bm25_cache = BM25Index(documents)
            self._bm25_doc_count = current_count
            logger.info("BM25 index built: %d documents", current_count)

        return self._bm25_cache

    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        raw = self._store.query(query, top_k=top_k)

        results = [
            RetrievalResult(
                text=r["text"],
                source=r["source"],
                page=r["page"],
                chunk_id=r["chunk_id"],
                score=r["score"],
                strategy="vector",
                query=query,
            )
            for r in raw
        ]

        # Record scores for cross-query calibration
        self._normalizer.record("vector", [r.score for r in results])

        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        bm25 = self._get_bm25()
        raw = bm25.search(query, top_k=top_k)

        results = [
            RetrievalResult(
                text=r["text"],
                source=r["source"],
                page=r["page"],
                chunk_id=r["chunk_id"],
                score=r["score"],
                strategy="keyword",
                query=query,
            )
            for r in raw
        ]

        self._normalizer.record("keyword", [r.score for r in results])

        return results

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        vector_results = self._vector_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)
        return _merge_results([vector_results, keyword_results], top_k)

    def _execute_search(
        self,
        strategy: str,
        queries: list[str],
        top_k: int,
        rerank: bool,
        exclude_ids: set[str] | None = None,
    ) -> RetrievalResponse:
        """
        Internal search executor. All entry points funnel through here.

        Pipeline:
            1. Run strategy on each sub-query
            2. Merge and deduplicate (excluding already-seen chunks)
            3. Rerank with cross-encoder (optional)
            4. Apply score calibration
        """
        search_fn = {
            "vector": self._vector_search,
            "keyword": self._keyword_search,
            "hybrid": self._hybrid_search,
        }[strategy]

        # Over-fetch when reranking
        fetch_k = top_k * 3 if rerank else top_k

        all_groups: list[list[RetrievalResult]] = []
        total_candidates = 0

        for q in queries:
            results = search_fn(q, fetch_k)
            all_groups.append(results)
            total_candidates += len(results)

        # Merge with exclusion
        excluded_count = 0
        if exclude_ids:
            flat_before = sum(len(g) for g in all_groups)
            merged = _merge_results(all_groups, fetch_k, exclude_ids)
            excluded_count = flat_before - len(merged) - (flat_before - total_candidates)
        else:
            merged = _merge_results(all_groups, fetch_k)

        # Rerank
        was_reranked = False
        if rerank and merged:
            reranker = self._get_reranker()
            merged = reranker.rerank(queries[0], merged, top_k=None)
            was_reranked = any(r.is_reranked for r in merged)

            if was_reranked:
                self._normalizer.record(
                    "rerank", [r.rerank_score for r in merged if r.rerank_score]
                )

        # Apply calibration if available
        for r in merged:
            method = "rerank" if r.is_reranked else r.strategy
            if self._normalizer.is_calibrated(method):
                raw = r.rerank_score if r.is_reranked else r.score
                r.calibrated_score = self._normalizer.normalize(method, raw)

        final = merged[:top_k]

        return RetrievalResponse(
            query=queries[0],
            results=final,
            strategy=strategy,
            sub_queries=queries if len(queries) > 1 else [],
            total_candidates=total_candidates,
            reranked=was_reranked,
            excluded_chunks=excluded_count,
        )

    def search(
        self,
        query: str,
        strategy: str | None = None,
        top_k: int | None = None,
        expand: bool | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResponse:
        """
        Search with raw query. Router decides strategy.

        Use this when no planner Plan is available.
        For planner-controlled search, use search_with_plan().
        """
        top_k = top_k or self._default_top_k
        rerank = rerank if rerank is not None else self._rerank_by_default

        # Route if no strategy specified
        if strategy is None:
            decision = self._router.route(query)
            strategy = decision.strategy
            if expand is None:
                expand = decision.expand
            logger.info(
                "Router: strategy=%s, expand=%s, reason='%s'",
                strategy, expand, decision.reason,
            )
        else:
            expand = expand if expand is not None else False

        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {self.STRATEGIES}")

        queries = expand_query(query) if expand else [query]

        response = self._execute_search(strategy, queries, top_k, rerank)
        logger.info("Search: %s", response.summary())

        return response

    def search_with_plan(
        self,
        plan: Plan,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResponse:
        """
        Search controlled by a planner Plan.

        The Plan's strategy and sub-queries take precedence.
        The router is NOT consulted. This eliminates the
        duplication between planner and retriever routing.

        Args:
            plan: structured Plan from the planner agent
            top_k: override plan's implied top_k
            rerank: override default reranking behavior

        Returns:
            RetrievalResponse following the plan's instructions
        """
        top_k = top_k or self._default_top_k
        rerank = rerank if rerank is not None else self._rerank_by_default

        strategy = plan.retrieval_strategy
        if strategy not in self.STRATEGIES:
            logger.warning(
                "Plan has invalid strategy '%s', defaulting to hybrid",
                strategy,
            )
            strategy = "hybrid"

        # Build query list from plan
        queries = [plan.query]
        if plan.sub_queries:
            queries.extend(plan.sub_queries)
        elif plan.expand:
            queries = expand_query(plan.query)

        response = self._execute_search(strategy, queries, top_k, rerank)

        logger.info(
            "Search with plan [%s, type=%s]: %s",
            plan.planning_mode, plan.query_type, response.summary(),
        )

        return response

    def search_with_feedback(
        self,
        query: str,
        previous_results: list[RetrievalResult],
        feedback: str = "",
        strategy: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        iteration: int = 2,
    ) -> RetrievalResponse:
        """
        Re-retrieval with critic feedback. Finds NEW evidence.

        The critic agent evaluates retrieved evidence and may
        decide it is insufficient. This method:
            1. Excludes chunks already seen (by chunk_id)
            2. Uses the feedback to refine the query
            3. Tries alternative strategies if the original failed
            4. Returns new results that complement the previous ones

        The feedback string can be a refined query from the critic
        (e.g., "Need more detail on free energy minimization")
        or a rejection reason that guides strategy selection.

        Args:
            query: original user question
            previous_results: results the critic already evaluated
            feedback: critic's guidance for re-retrieval
            strategy: override strategy (None = try alternative)
            top_k: how many new results to find
            rerank: whether to rerank new results
            iteration: which retry round this is (for logging)

        Returns:
            RetrievalResponse with NEW chunks not in previous_results
        """
        top_k = top_k or self._default_top_k
        rerank = rerank if rerank is not None else self._rerank_by_default

        # Collect chunk IDs to exclude
        exclude_ids = {r.chunk_id for r in previous_results}

        # Determine search query: use feedback as refined query if provided
        search_query = feedback if feedback.strip() else query

        # Determine strategy: if not specified, try a different one
        if strategy is None:
            # Find what strategy was used before
            previous_strategies = {r.strategy for r in previous_results}

            if "hybrid" in previous_strategies:
                # Already tried hybrid, try vector with expanded query
                strategy = "vector"
            elif "vector" in previous_strategies:
                strategy = "keyword"
            elif "keyword" in previous_strategies:
                strategy = "vector"
            else:
                strategy = "hybrid"

            logger.info(
                "Feedback loop: previous=%s, trying=%s",
                previous_strategies, strategy,
            )

        queries = [search_query]
        if search_query != query:
            queries.append(query)

        response = self._execute_search(
            strategy, queries, top_k, rerank, exclude_ids
        )

        response.iteration = iteration
        response.feedback = feedback
        response.excluded_chunks = len(exclude_ids)

        logger.info(
            "Feedback search (iter=%d): %s", iteration, response.summary()
        )

        return response