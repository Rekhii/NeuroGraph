"""
Pipeline orchestrator — app/pipeline.py

Wires all agents and modules into a single entry point.
One call does everything: plan, retrieve, critique, optimize, write.

Pipeline:
    query
      -> Planner (classify, decompose, set strategy)
      -> Retriever (search with plan)
      -> Critic (evaluate evidence, retry if needed)
      -> ContextOptimizer (compress, filter, budget)
      -> Writer (generate cited answer)
      -> PipelineResult

Error handling:
    Each stage has a fallback. If the planner fails, the retriever
    uses its own router. If the critic fails, evidence passes through
    unchecked. If the writer fails, raw evidence is returned.
    The pipeline never crashes — it degrades gracefully.

Usage:
    from app.pipeline import Pipeline

    pipeline = Pipeline()
    result = pipeline.run("How does active inference work?")

    print(result.answer)
    print(result.sources)
    print(result.trace)     # full pipeline trace for debugging
    print(result.timing)    # per-stage timing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.core.config import settings
from app.agents.planner import Planner, Plan
from app.agents.critic import Critic, CriticVerdict
from app.agents.writer import Writer, WriterResponse
from app.rag.retriever import Retriever, RetrievalResponse
from app.rag.context import ContextOptimizer, OptimizedContext

logger = logging.getLogger(__name__)


# Pipeline result

@dataclass
class PipelineResult:
    """
    Complete output of one pipeline run.

    Contains the final answer plus full trace data for
    debugging, evaluation, and observability.
    """
    # Final output
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    # Component outputs (for debugging and eval)
    plan: Plan | None = None
    retrieval: RetrievalResponse | None = None
    verdict: CriticVerdict | None = None
    context: OptimizedContext | None = None
    writer_response: WriterResponse | None = None

    # Metadata
    query: str = ""
    timing: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    pipeline_mode: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 3),
            "query": self.query,
            "timing": self.timing,
            "errors": self.errors,
            "plan": self.plan.to_dict() if self.plan else None,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "writer": self.writer_response.to_dict() if self.writer_response else None,
        }

    @property
    def total_time_ms(self) -> float:
        return sum(self.timing.values())

    def summary(self) -> str:
        errors = f", errors={len(self.errors)}" if self.errors else ""
        return (
            f"confidence={self.confidence:.2f}, "
            f"sources={len(self.sources)}, "
            f"time={self.total_time_ms:.0f}ms"
            f"{errors}"
        )


# Pipeline

class Pipeline:
    """
    Main orchestrator for NeuroGraph.

    Composes all agents and modules into a linear pipeline
    with feedback loops (critic retry) and graceful degradation.

    Configuration:
        planner_mode: "auto", "llm", "rules"
        critic_mode: "auto", "llm", "rules"
        writer_mode: "auto", "llm", "rules"
        enable_rerank: whether to use cross-encoder reranking
        enable_context_optimization: whether to compress context
        max_tokens: context window budget for evidence
    """

    def __init__(
        self,
        planner_mode: str = "auto",
        critic_mode: str = "auto",
        writer_mode: str = "auto",
        enable_rerank: bool = True,
        enable_context_optimization: bool = True,
        max_tokens: int = 2048,
    ):
        self._planner = Planner()
        self._retriever = Retriever(rerank_by_default=enable_rerank)
        self._critic = Critic(mode=critic_mode)
        self._writer = Writer(mode=writer_mode)
        self._context_optimizer = ContextOptimizer(
            default_token_budget=max_tokens,
        ) if enable_context_optimization else None

        self._planner_mode = planner_mode
        self._enable_context_optimization = enable_context_optimization

    def run(
        self,
        query: str,
        strategy: str | None = None,
        top_k: int | None = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on a query.

        Stages:
            1. Plan: classify query, set strategy
            2. Retrieve: search with plan
            3. Critique: evaluate + retry loop
            4. Optimize: compress context for LLM
            5. Write: generate cited answer

        Each stage is timed. Errors at any stage are caught
        and the pipeline continues with degraded output.

        Args:
            query: the user's question
            strategy: override planner's strategy choice
            top_k: override default number of results

        Returns:
            PipelineResult with answer, sources, and full trace
        """
        result = PipelineResult(query=query)

        if not query or not query.strip():
            result.answer = "Please provide a question."
            return result

        # Stage 1: Plan
        plan = self._run_planner(query, result)

        # Override strategy if provided
        if strategy:
            plan.retrieval_strategy = strategy

        # Stage 2: Retrieve
        response = self._run_retriever(plan, top_k, result)

        # Stage 3: Critique (with retry loop)
        verdict = self._run_critic(plan, response, result)

        # Stage 4: Optimize context
        if self._enable_context_optimization and verdict:
            self._run_optimizer(plan.query, verdict, result)

        # Stage 5: Write
        writer_response = self._run_writer(plan, verdict, result)

        # Assemble final result
        if writer_response:
            result.answer = writer_response.answer
            result.sources = writer_response.sources
            result.confidence = writer_response.confidence
            result.writer_response = writer_response

        result.pipeline_mode = self._describe_mode(plan, verdict, writer_response)

        logger.info("Pipeline complete: %s", result.summary())

        return result

    def _run_planner(self, query: str, result: PipelineResult) -> Plan:
        """Stage 1: Generate a plan."""
        start = time.monotonic()

        try:
            plan = self._planner.plan(query, mode=self._planner_mode)
        except Exception as e:
            logger.error("Planner failed: %s", e)
            result.errors.append(f"Planner: {e}")
            plan = Plan(
                query=query,
                planning_mode="fallback",
                reasoning=f"Planner error: {e}",
            )

        result.plan = plan
        result.timing["planner"] = round((time.monotonic() - start) * 1000, 1)

        return plan

    def _run_retriever(
        self,
        plan: Plan,
        top_k: int | None,
        result: PipelineResult,
    ) -> RetrievalResponse | None:
        """Stage 2: Retrieve evidence."""
        start = time.monotonic()

        try:
            response = self._retriever.search_with_plan(plan, top_k=top_k)
        except Exception as e:
            logger.error("Retriever failed: %s", e)
            result.errors.append(f"Retriever: {e}")
            response = None

        result.retrieval = response
        result.timing["retriever"] = round((time.monotonic() - start) * 1000, 1)

        return response

    def _run_critic(
        self,
        plan: Plan,
        response: RetrievalResponse | None,
        result: PipelineResult,
    ) -> CriticVerdict | None:
        """Stage 3: Evaluate evidence with retry loop."""
        if response is None or not response.has_results:
            verdict = CriticVerdict(
                verdict="insufficient",
                confidence=0.0,
                reasoning="No retrieval results to evaluate",
                gaps=["No evidence retrieved"],
                is_final=True,
                writer_caveats=["No evidence was found. Answer based on general knowledge if possible."],
            )
            result.verdict = verdict
            result.timing["critic"] = 0.0
            return verdict

        start = time.monotonic()

        try:
            verdict = self._critic.evaluate_with_retry(
                plan=plan,
                retriever=self._retriever,
                initial_response=response,
            )
        except Exception as e:
            logger.error("Critic failed: %s", e)
            result.errors.append(f"Critic: {e}")
            # Pass through unchecked
            verdict = CriticVerdict(
                verdict="sufficient",
                confidence=0.5,
                reasoning=f"Critic error: {e}, passing evidence through",
                approved_results=response.results,
                evaluation_mode="error_passthrough",
            )

        result.verdict = verdict
        result.timing["critic"] = round((time.monotonic() - start) * 1000, 1)

        return verdict

    def _run_optimizer(
        self,
        query: str,
        verdict: CriticVerdict,
        result: PipelineResult,
    ) -> None:
        """Stage 4: Optimize context before writing."""
        if not self._context_optimizer or not verdict.approved_results:
            return

        start = time.monotonic()

        try:
            optimized = self._context_optimizer.optimize(
                query=query,
                results=verdict.approved_results,
            )
            result.context = optimized

            # Replace approved results with optimized ones
            verdict.approved_results = optimized.results

        except Exception as e:
            logger.error("Context optimizer failed: %s", e)
            result.errors.append(f"Optimizer: {e}")

        result.timing["optimizer"] = round((time.monotonic() - start) * 1000, 1)

    def _run_writer(
        self,
        plan: Plan,
        verdict: CriticVerdict | None,
        result: PipelineResult,
    ) -> WriterResponse | None:
        """Stage 5: Generate the final answer."""
        if verdict is None:
            return WriterResponse(
                answer="Unable to retrieve evidence for this question.",
                confidence=0.0,
                query=plan.query,
            )

        start = time.monotonic()

        try:
            writer_response = self._writer.write(plan=plan, verdict=verdict)
        except Exception as e:
            logger.error("Writer failed: %s", e)
            result.errors.append(f"Writer: {e}")
            # Emergency fallback: return raw evidence
            if verdict.approved_results:
                chunks = "\n\n".join(
                    f"[{i}] {r.text} (Source: {r.source}, Page {r.page})"
                    for i, r in enumerate(verdict.approved_results, 1)
                )
                writer_response = WriterResponse(
                    answer=f"Raw evidence (writer error):\n\n{chunks}",
                    confidence=0.1,
                    query=plan.query,
                )
            else:
                writer_response = WriterResponse(
                    answer="Unable to generate an answer.",
                    confidence=0.0,
                    query=plan.query,
                )

        result.timing["writer"] = round((time.monotonic() - start) * 1000, 1)

        return writer_response

    def _describe_mode(
        self,
        plan: Plan,
        verdict: CriticVerdict | None,
        writer_response: WriterResponse | None,
    ) -> str:
        """Describe which mode each component used."""
        parts = [
            f"planner={plan.planning_mode}",
            f"critic={verdict.evaluation_mode if verdict else 'none'}",
            f"writer={writer_response.writing_mode if writer_response else 'none'}",
        ]
        return ", ".join(parts)


# Convenience function

def run_query(
    query: str,
    strategy: str | None = None,
    top_k: int | None = None,
) -> PipelineResult:
    """
    One-call function for quick usage.

    Creates a pipeline with default settings and runs the query.
    For repeated queries, create a Pipeline instance and reuse it.
    """
    pipeline = Pipeline()
    return pipeline.run(query, strategy=strategy, top_k=top_k)