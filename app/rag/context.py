"""
Context optimizer — app/rag/context.py

Sits between the retriever and the LLM. Takes raw retrieval
results and produces an optimized context string that fits
within token budgets and maximizes information density.

Pipeline position:
    retriever -> [CONTEXT OPTIMIZER] -> LLM prompt

Without optimization:
    5 chunks at 500 chars each = 2500 chars to the LLM
    Problem: 3 chunks might repeat the same point, 1 chunk
    might be marginally relevant, wasting context window

With optimization:
    5 chunks -> relevance filter -> redundancy removal ->
    sentence extraction -> token budget -> 3 focused passages
    that directly address the query

Four stages:
    1. RelevanceGate    — drop chunks below a score threshold
    2. RedundancyFilter — remove near-duplicate chunks (Jaccard)
    3. SentenceExtractor — keep only query-relevant sentences
    4. TokenBudget      — trim to fit the LLM context window

Each stage is independent and optional. The ContextOptimizer
composes them into a pipeline that runs in order.

Usage:
    from app.rag.context import ContextOptimizer

    optimizer = ContextOptimizer()
    optimized = optimizer.optimize(
        query="How does active inference work?",
        results=retrieval_response.results,
        max_tokens=2048,
    )

    print(optimized.context_string)    # ready for LLM prompt
    print(optimized.stats)             # what was removed and why
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# Approximate token counting

def _estimate_tokens(text: str) -> int:
    """
    Rough token count: ~4 characters per token for English.

    Not exact, but consistent and fast. Used for budget
    enforcement, not billing. Over-estimates slightly,
    which is safer than under-estimating (avoids context
    window overflow).
    """
    return max(1, len(text) // 4)


# Stage 1: Relevance gate

@dataclass
class RelevanceGate:
    """
    Drop chunks below a minimum relevance score.

    Chunks with low retrieval scores are noise. They matched
    the query weakly and are unlikely to contain useful evidence.
    Removing them before the LLM sees them prevents confusion
    and hallucination grounded in irrelevant passages.

    The threshold is adaptive: if all chunks score low (bad
    retrieval), the gate relaxes to avoid returning nothing.
    Better to give the LLM weak evidence than no evidence,
    so the critic agent can trigger re-retrieval.

    Score semantics:
        effective_score is used (rerank score if available,
        else retrieval score). Both are 0-1, higher = better.
    """
    min_score: float = 0.3
    adaptive: bool = True

    def filter(self, results: list, query: str = "") -> tuple[list, int]:
        """
        Remove chunks below the relevance threshold.

        If adaptive=True and all chunks are below threshold,
        keeps the top 2 anyway (so downstream has something).

        Returns:
            (filtered_results, number_removed)
        """
        if not results:
            return results, 0

        threshold = self.min_score

        # Adaptive: if the best score is below threshold,
        # lower the gate to keep at least the top chunks
        if self.adaptive:
            best = max(r.effective_score for r in results)
            if best < threshold:
                # Keep top 2 minimum, even if scores are low
                logger.info(
                    "RelevanceGate: best score %.3f < threshold %.3f, "
                    "adaptive mode keeping top results",
                    best, threshold,
                )
                if len(results) <= 2:
                    return results, 0
                # Use a threshold that keeps at least top 2
                sorted_scores = sorted(
                    [r.effective_score for r in results], reverse=True
                )
                threshold = sorted_scores[1] - 0.001

        passed = []
        removed = 0

        for r in results:
            if r.effective_score >= threshold:
                passed.append(r)
            else:
                removed += 1
                logger.debug(
                    "RelevanceGate: dropped %s (score=%.3f < %.3f)",
                    r.chunk_id, r.effective_score, threshold,
                )

        if removed:
            logger.info(
                "RelevanceGate: kept %d, dropped %d chunks",
                len(passed), removed,
            )

        return passed, removed


# Stage 2: Redundancy filter (upgraded from retriever.py)

@dataclass
class RedundancyFilter:
    """
    Remove near-duplicate chunks using Jaccard similarity.

    Moved here from retriever.py to be part of the unified
    context optimization pipeline. Same algorithm: compare
    token sets, drop chunks that overlap above threshold.

    Jaccard similarity = |A ∩ B| / |A ∪ B|
        0.0 = completely different token sets
        1.0 = identical token sets
        0.7 = default threshold (70% overlap triggers removal)
    """
    threshold: float = 0.7

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def filter(self, results: list) -> tuple[list, int]:
        """
        Remove near-duplicates, keeping the highest-scored chunk.

        Assumes results are sorted by score descending. Each chunk
        is compared against all survivors. If too similar to any
        survivor, it is dropped.
        """
        if len(results) <= 1:
            return results, 0

        survivors = []
        survivor_tokens = []
        removed = 0

        for r in results:
            tokens = self._tokenize(r.text)
            is_dup = False

            for existing in survivor_tokens:
                if self._jaccard(tokens, existing) >= self.threshold:
                    is_dup = True
                    break

            if not is_dup:
                survivors.append(r)
                survivor_tokens.append(tokens)
            else:
                removed += 1

        if removed:
            logger.info(
                "RedundancyFilter: kept %d, removed %d chunks",
                len(survivors), removed,
            )

        return survivors, removed


# Stage 3: Sentence extractor

@dataclass
class SentenceExtractor:
    """
    Extract only query-relevant sentences from each chunk.

    A 500-char chunk might have 4 sentences, but only 2 are
    relevant to the query. Extracting those 2 gives the LLM
    focused evidence without noise.

    Method:
        1. Split chunk into sentences
        2. Tokenize query into keywords
        3. Score each sentence by keyword overlap
        4. Keep sentences above a relevance threshold
        5. Preserve original sentence order

    This is extractive compression (no LLM needed). It reduces
    context size by 30-50% on average while keeping the signal.

    Why not LLM-based summarization?
        - Adds latency (another LLM call per chunk)
        - Costs tokens (or Ollama compute)
        - Can introduce hallucination in the summary
        - Extractive is deterministic and transparent
    """
    min_keyword_overlap: float = 0.15
    min_sentences_per_chunk: int = 1

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Handles common abbreviations (Dr., Mr., etc.) and
        decimal numbers (0.31) to avoid false splits.
        """
        # Protect common abbreviations and decimals
        protected = text
        protected = re.sub(r"(\b(?:Dr|Mr|Mrs|Ms|Prof|Fig|Eq|al|vs|etc|approx))\.", r"\1<DOT>", protected)
        protected = re.sub(r"(\d)\.", r"\1<DOT>", protected)

        # Split on sentence-ending punctuation followed by space + uppercase
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

        # Restore protected dots
        sentences = [p.replace("<DOT>", ".").strip() for p in parts]
        return [s for s in sentences if s]

    @staticmethod
    def _query_keywords(query: str) -> set[str]:
        """
        Extract meaningful keywords from the query.

        Strips common stop words so that "How does active inference
        differ from GWT?" becomes {"active", "inference", "differ", "gwt"}.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "to", "for", "with", "on", "at", "from", "by",
            "about", "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "how", "what", "when", "where", "which", "who", "whom",
            "this", "that", "these", "those", "it", "its",
        }
        tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        return tokens - stop_words

    def extract(self, text: str, query: str) -> str:
        """
        Extract query-relevant sentences from a chunk.

        Returns the filtered text with only relevant sentences,
        preserving original order. If no sentences meet the
        threshold, returns the top N by keyword overlap.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= self.min_sentences_per_chunk:
            return text

        keywords = self._query_keywords(query)
        if not keywords:
            return text

        # Score each sentence by keyword overlap fraction
        scored = []
        for sent in sentences:
            sent_tokens = set(re.findall(r"[a-zA-Z0-9]+", sent.lower()))
            if not sent_tokens:
                scored.append((sent, 0.0))
                continue
            overlap = len(keywords & sent_tokens) / len(keywords)
            scored.append((sent, overlap))

        # Keep sentences above threshold
        relevant = [(s, score) for s, score in scored if score >= self.min_keyword_overlap]

        # If nothing passes, keep the top N by score
        if len(relevant) < self.min_sentences_per_chunk:
            scored.sort(key=lambda x: x[1], reverse=True)
            relevant = scored[:self.min_sentences_per_chunk]

        # Restore original order
        ordered = [s for s, _ in scored if any(s == r for r, _ in relevant)]

        return " ".join(ordered)

    def compress_results(self, results: list, query: str) -> list:
        """
        Apply sentence extraction to all results.

        Modifies result.text in place (the original text is in
        the vector store if needed). Returns the same list with
        compressed text.
        """
        total_before = 0
        total_after = 0

        for r in results:
            original_len = len(r.text)
            r.text = self.extract(r.text, query)
            compressed_len = len(r.text)

            total_before += original_len
            total_after += compressed_len

        if total_before > 0:
            ratio = 1.0 - (total_after / total_before)
            logger.info(
                "SentenceExtractor: compressed %d -> %d chars (%.0f%% reduction)",
                total_before, total_after, ratio * 100,
            )

        return results


# Stage 4: Token budget

@dataclass
class TokenBudget:
    """
    Enforce a maximum token limit on the combined context.

    The LLM has a finite context window. The system prompt,
    planner output, and writer instructions also consume tokens.
    The context budget is what remains for retrieved evidence.

    Strategy: keep chunks in score order (best first) until
    the budget is exhausted. The last chunk may be truncated
    to fit exactly.
    """
    max_tokens: int = 2048

    def enforce(self, results: list) -> tuple[list, int]:
        """
        Trim results to fit within token budget.

        Returns:
            (trimmed_results, number_of_chunks_dropped)
        """
        if not results:
            return results, 0

        kept = []
        tokens_used = 0
        dropped = 0

        for r in results:
            chunk_tokens = _estimate_tokens(r.text)

            if tokens_used + chunk_tokens <= self.max_tokens:
                kept.append(r)
                tokens_used += chunk_tokens
            elif tokens_used < self.max_tokens:
                # Partial fit: truncate this chunk to fill remaining budget
                remaining_chars = (self.max_tokens - tokens_used) * 4
                r.text = r.text[:remaining_chars].rsplit(" ", 1)[0] + "..."
                kept.append(r)
                tokens_used = self.max_tokens
                dropped += len(results) - len(kept)
                break
            else:
                dropped += 1

        if dropped:
            logger.info(
                "TokenBudget: kept %d chunks (%d tokens), dropped %d",
                len(kept), tokens_used, dropped,
            )

        return kept, dropped


# Optimization stats

@dataclass
class OptimizationStats:
    """Tracks what each pipeline stage did."""
    input_chunks: int = 0
    output_chunks: int = 0
    relevance_dropped: int = 0
    redundancy_dropped: int = 0
    budget_dropped: int = 0
    compression_ratio: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"chunks: {self.input_chunks} -> {self.output_chunks} "
            f"(relevance=-{self.relevance_dropped}, "
            f"redundancy=-{self.redundancy_dropped}, "
            f"budget=-{self.budget_dropped}) "
            f"tokens: {self.input_tokens} -> {self.output_tokens}"
        )


# Optimized context output

@dataclass
class OptimizedContext:
    """
    Final output of the context optimization pipeline.

    Contains the optimized results, a formatted context string
    ready for LLM prompts, and stats about what was changed.
    """
    query: str
    results: list
    stats: OptimizationStats

    @property
    def context_string(self) -> str:
        """Format results for LLM prompt with numbered citations."""
        if not self.results:
            return "No relevant evidence found."

        parts = []
        for i, r in enumerate(self.results, 1):
            score = r.effective_score
            parts.append(
                f"[{i}] (Source: {r.source}, Page {r.page}, "
                f"relevance={score:.3f})\n{r.text}"
            )
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    @property
    def source_list(self) -> list[str]:
        """Unique sources referenced in the optimized context."""
        seen = set()
        sources = []
        for r in self.results:
            if r.source not in seen:
                seen.add(r.source)
                sources.append(r.source)
        return sources


# Main optimizer

class ContextOptimizer:
    """
    Full context optimization pipeline.

    Composes four independent stages into a single optimize() call.
    Each stage can be enabled/disabled and configured independently.

    Pipeline:
        results -> relevance gate -> redundancy filter ->
        sentence extraction -> token budget -> OptimizedContext

    Usage:
        optimizer = ContextOptimizer()
        context = optimizer.optimize(
            query="How does active inference work?",
            results=retriever_results,
            max_tokens=2048,
        )
        # context.context_string is ready for the LLM prompt
        # context.stats shows what was removed and why
    """

    def __init__(
        self,
        relevance_threshold: float = 0.3,
        redundancy_threshold: float = 0.7,
        keyword_overlap: float = 0.15,
        default_token_budget: int = 2048,
        enable_relevance: bool = True,
        enable_redundancy: bool = True,
        enable_compression: bool = True,
        enable_budget: bool = True,
    ):
        self._relevance_gate = RelevanceGate(min_score=relevance_threshold)
        self._redundancy_filter = RedundancyFilter(threshold=redundancy_threshold)
        self._sentence_extractor = SentenceExtractor(min_keyword_overlap=keyword_overlap)
        self._token_budget = TokenBudget(max_tokens=default_token_budget)

        self._enable_relevance = enable_relevance
        self._enable_redundancy = enable_redundancy
        self._enable_compression = enable_compression
        self._enable_budget = enable_budget

    def optimize(
        self,
        query: str,
        results: list,
        max_tokens: int | None = None,
    ) -> OptimizedContext:
        """
        Run the full optimization pipeline.

        Args:
            query: the user's question (for sentence extraction)
            results: RetrievalResult list from the retriever
            max_tokens: override default token budget

        Returns:
            OptimizedContext with filtered, compressed results
            and pipeline statistics
        """
        if not results:
            return OptimizedContext(
                query=query,
                results=[],
                stats=OptimizationStats(),
            )

        stats = OptimizationStats()
        stats.input_chunks = len(results)
        stats.input_tokens = sum(_estimate_tokens(r.text) for r in results)

        current = list(results)

        # Stage 1: Relevance gate
        if self._enable_relevance:
            current, dropped = self._relevance_gate.filter(current, query)
            stats.relevance_dropped = dropped

        # Stage 2: Redundancy filter
        if self._enable_redundancy:
            current, dropped = self._redundancy_filter.filter(current)
            stats.redundancy_dropped = dropped

        # Stage 3: Sentence extraction (compression)
        if self._enable_compression and current:
            before_chars = sum(len(r.text) for r in current)
            current = self._sentence_extractor.compress_results(current, query)
            after_chars = sum(len(r.text) for r in current)
            if before_chars > 0:
                stats.compression_ratio = 1.0 - (after_chars / before_chars)

        # Stage 4: Token budget enforcement
        if self._enable_budget and current:
            if max_tokens:
                self._token_budget.max_tokens = max_tokens
            current, dropped = self._token_budget.enforce(current)
            stats.budget_dropped = dropped

        stats.output_chunks = len(current)
        stats.output_tokens = sum(_estimate_tokens(r.text) for r in current)

        logger.info("Context optimization: %s", stats.summary())

        return OptimizedContext(
            query=query,
            results=current,
            stats=stats,
        )