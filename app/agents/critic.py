"""
Critic agent — app/agents/critic.py

The critic is the quality gate between retrieval and writing.
It evaluates whether retrieved evidence is sufficient to answer
the user's query, and triggers re-retrieval when it is not.

Pipeline position:
    planner -> retriever -> [CRITIC] -> context optimizer -> writer
                  ^             |
                  |_____________|  (feedback loop)

What the critic evaluates:
    1. Coverage: do the chunks actually address the query?
    2. Sufficiency: is there enough evidence for a complete answer?
    3. Comparison balance: for comparison queries, do we have
       evidence for BOTH sides?
    4. Score quality: are retrieval scores above acceptable thresholds?

Two evaluation modes:
    - LLM-based (primary): asks the LLM to judge evidence quality
    - Rule-based (fallback): heuristic scoring using keyword overlap,
      score thresholds, and comparison balance checks

Feedback loop:
    The critic drives the retriever feedback loop:
        iteration 1: evaluate -> insufficient -> generate refined query
        iteration 2: re-retrieve (excluding seen chunks) -> evaluate
        iteration 3: if still insufficient, accept best available
                     and add caveats for the writer

    The critic is STRICT on early iterations and LENIENT on the
    final iteration. Better to give the writer weak evidence with
    caveats than to return nothing.

Usage:
    from app.agents.critic import Critic

    critic = Critic()

    # Evaluate evidence
    verdict = critic.evaluate(plan, retrieval_response)

    # Full loop: evaluate and re-retrieve until satisfied
    final_verdict = critic.evaluate_with_retry(
        plan=plan,
        retriever=retriever,
        initial_response=response,
    )

    if final_verdict.is_sufficient:
        writer.write(final_verdict.approved_results, plan)
    else:
        writer.write_with_caveats(final_verdict.approved_results, final_verdict.gaps)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.agents.planner import Plan
    from app.rag.retriever import Retriever, RetrievalResponse, RetrievalResult

logger = logging.getLogger(__name__)


# Verdict data structure

@dataclass
class CriticVerdict:
    """
    The critic's judgment on retrieved evidence.

    Downstream consumers:
        - Writer reads: approved_results, verdict, gaps, writer_caveats
        - Retriever reads: feedback_query (for re-retrieval)
        - Orchestrator reads: is_sufficient, iteration
    """
    # Core judgment
    verdict: str = "insufficient"           # "sufficient" or "insufficient"
    confidence: float = 0.0                 # 0-1, critic's confidence in the evidence
    reasoning: str = ""                     # why the critic made this decision

    # Evidence assessment
    coverage_score: float = 0.0             # how well chunks address the query (0-1)
    sufficiency_score: float = 0.0          # enough evidence for a complete answer (0-1)
    comparison_balance: float | None = None # for comparisons: balance between sides (0-1)

    # Approved evidence
    approved_results: list = field(default_factory=list)
    rejected_results: list = field(default_factory=list)

    # Feedback for re-retrieval
    gaps: list[str] = field(default_factory=list)
    feedback_query: str = ""                # refined query for retriever
    suggested_strategy: str = ""            # strategy hint for re-retrieval

    # Metadata
    evaluation_mode: str = "fallback"       # "llm" or "fallback"
    iteration: int = 1
    is_final: bool = False                  # True on last allowed iteration

    # Writer guidance
    writer_caveats: list[str] = field(default_factory=list)

    @property
    def is_sufficient(self) -> bool:
        return self.verdict == "sufficient"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "coverage_score": round(self.coverage_score, 3),
            "sufficiency_score": round(self.sufficiency_score, 3),
            "comparison_balance": round(self.comparison_balance, 3) if self.comparison_balance is not None else None,
            "gaps": self.gaps,
            "feedback_query": self.feedback_query,
            "suggested_strategy": self.suggested_strategy,
            "evaluation_mode": self.evaluation_mode,
            "iteration": self.iteration,
            "is_final": self.is_final,
            "approved_count": len(self.approved_results),
            "rejected_count": len(self.rejected_results),
            "writer_caveats": self.writer_caveats,
        }

    def summary(self) -> str:
        bal = f", balance={self.comparison_balance:.2f}" if self.comparison_balance is not None else ""
        caveats = f", caveats={len(self.writer_caveats)}" if self.writer_caveats else ""
        return (
            f"[{self.evaluation_mode}] "
            f"verdict={self.verdict}, "
            f"confidence={self.confidence:.2f}, "
            f"coverage={self.coverage_score:.2f}, "
            f"sufficiency={self.sufficiency_score:.2f}, "
            f"approved={len(self.approved_results)}/{len(self.approved_results) + len(self.rejected_results)}, "
            f"iter={self.iteration}"
            f"{bal}{caveats}"
        )


# Critic system prompt

CRITIC_SYSTEM_PROMPT = """\
You are the evidence critic in a neuroscience research assistant.

Your job: evaluate whether retrieved evidence is sufficient to
answer the user's query.

You receive:
- The user's query
- The query type (factual/conceptual/comparison/multi_step)
- Retrieved evidence chunks with source citations

You must respond with ONLY a JSON object (no markdown, no explanation).

JSON schema:
{
    "verdict": "sufficient" | "insufficient",
    "confidence": 0.0 to 1.0,
    "coverage_score": 0.0 to 1.0,
    "sufficiency_score": 0.0 to 1.0,
    "reasoning": "brief explanation of your assessment",
    "gaps": ["list of missing information"],
    "feedback_query": "refined search query to fill gaps",
    "suggested_strategy": "vector" | "keyword" | "hybrid"
}

Evaluation rules:
- coverage_score: what fraction of the query is addressed by the evidence?
  1.0 = every aspect covered, 0.0 = completely off-topic
- sufficiency_score: is there enough detail to write a complete answer?
  1.0 = comprehensive evidence, 0.0 = almost nothing useful
- For comparisons: check that BOTH sides have evidence. If only one
  concept is covered, verdict must be "insufficient".
- For factual queries: the specific fact must be present. Close matches
  are not sufficient.
- feedback_query: if insufficient, write a specific search query
  that would find the missing information.
- Be strict but fair. If evidence covers the core of the question
  even if not exhaustive, mark as "sufficient" with appropriate confidence.
"""


# Rule-based evaluation

class RuleBasedCritic:
    """
    Evaluates evidence quality using heuristic scoring.

    Four checks, each producing a 0-1 score:
        1. Score quality: are retrieval scores high enough?
        2. Keyword coverage: do chunks contain query keywords?
        3. Volume: are there enough chunks?
        4. Comparison balance: for comparisons, both sides covered?

    The combined score determines the verdict. Thresholds are
    tuned to be strict on first iterations and lenient on final.
    """

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "about", "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "so", "yet", "both",
        "how", "what", "when", "where", "which", "who", "whom",
        "this", "that", "these", "those", "it", "its",
    }

    def evaluate(
        self,
        query: str,
        query_type: str,
        results: list,
        min_confidence: float = 0.4,
        is_final: bool = False,
    ) -> CriticVerdict:
        """
        Evaluate evidence quality using heuristics.

        Args:
            query: the user's question
            query_type: from the planner (factual/conceptual/comparison/multi_step)
            results: RetrievalResult list
            min_confidence: threshold for "sufficient" verdict
            is_final: if True, lower thresholds (last chance)
        """
        if not results:
            return CriticVerdict(
                verdict="insufficient",
                confidence=0.0,
                reasoning="No evidence retrieved",
                gaps=["No relevant documents found for the query"],
                feedback_query=query,
                suggested_strategy="hybrid",
                evaluation_mode="fallback",
            )

        # Check 1: Score quality
        scores = [r.effective_score for r in results]
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)
        score_quality = min(1.0, (avg_score + top_score) / 2.0 * 1.2)

        # Check 2: Keyword coverage
        query_keywords = self._extract_keywords(query)
        evidence_text = " ".join(r.text.lower() for r in results)
        evidence_tokens = set(re.findall(r"[a-zA-Z0-9]+", evidence_text))

        if query_keywords:
            covered = sum(1 for kw in query_keywords if kw in evidence_tokens)
            coverage = covered / len(query_keywords)
        else:
            coverage = 0.5

        # Check 3: Volume
        min_chunks = {"factual": 1, "conceptual": 2, "comparison": 3, "multi_step": 3}
        required = min_chunks.get(query_type, 2)
        volume = min(1.0, len(results) / required)

        # Check 4: Comparison balance
        comparison_balance = None
        comparison_gaps = []
        if query_type == "comparison":
            comparison_balance, comparison_gaps = self._check_comparison_balance(
                query, results
            )

        # Combined score
        weights = {"score": 0.3, "coverage": 0.35, "volume": 0.15, "balance": 0.2}

        combined = (
            score_quality * weights["score"]
            + coverage * weights["coverage"]
            + volume * weights["volume"]
        )

        if comparison_balance is not None:
            combined += comparison_balance * weights["balance"]
        else:
            # Redistribute balance weight to coverage
            combined += coverage * weights["balance"]

        # Threshold adjustment
        threshold = min_confidence
        if is_final:
            threshold *= 0.6  # More lenient on last iteration

        # Build verdict
        is_sufficient = combined >= threshold
        gaps = []
        feedback_query = ""
        suggested_strategy = ""

        if not is_sufficient:
            gaps = self._identify_gaps(query, query_keywords, evidence_tokens, query_type)
            gaps.extend(comparison_gaps)
            feedback_query = self._build_feedback_query(query, gaps)
            suggested_strategy = self._suggest_strategy(query_type, scores)

        # Separate approved vs rejected results
        score_threshold = min_confidence * 0.5 if is_final else min_confidence * 0.7
        approved = [r for r in results if r.effective_score >= score_threshold]
        rejected = [r for r in results if r.effective_score < score_threshold]

        # If all rejected, keep at least the best ones
        if not approved and results:
            approved = sorted(results, key=lambda r: r.effective_score, reverse=True)[:2]
            rejected = [r for r in results if r not in approved]

        # Writer caveats
        caveats = []
        if is_final and not is_sufficient:
            caveats.append("Evidence may be incomplete. Answer with available information and note gaps.")
            if comparison_gaps:
                caveats.append(f"Limited evidence for: {', '.join(comparison_gaps)}")

        return CriticVerdict(
            verdict="sufficient" if is_sufficient else "insufficient",
            confidence=round(combined, 3),
            reasoning=self._build_reasoning(
                score_quality, coverage, volume, comparison_balance, combined, threshold
            ),
            coverage_score=round(coverage, 3),
            sufficiency_score=round(score_quality * volume, 3),
            comparison_balance=round(comparison_balance, 3) if comparison_balance is not None else None,
            approved_results=approved,
            rejected_results=rejected,
            gaps=gaps,
            feedback_query=feedback_query,
            suggested_strategy=suggested_strategy,
            evaluation_mode="fallback",
            is_final=is_final,
            writer_caveats=caveats,
        )

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords, removing stop words."""
        tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return tokens - self.STOP_WORDS

    def _check_comparison_balance(
        self,
        query: str,
        results: list,
    ) -> tuple[float, list[str]]:
        """
        Check if both sides of a comparison have evidence.

        Extracts the two concepts being compared, then checks
        how many chunks mention each one. Returns a balance
        score (1.0 = equal coverage, 0.0 = only one side).
        """
        # Extract comparison terms
        terms = self._extract_comparison_terms(query)
        if len(terms) < 2:
            return 0.5, []

        term_a, term_b = terms[0].lower(), terms[1].lower()
        a_keywords = set(re.findall(r"[a-zA-Z0-9]+", term_a))
        b_keywords = set(re.findall(r"[a-zA-Z0-9]+", term_b))

        a_count = 0
        b_count = 0

        for r in results:
            text_lower = r.text.lower()
            text_tokens = set(re.findall(r"[a-zA-Z0-9]+", text_lower))

            if a_keywords & text_tokens:
                a_count += 1
            if b_keywords & text_tokens:
                b_count += 1

        total = a_count + b_count
        if total == 0:
            return 0.0, [f"No evidence found for '{terms[0]}' or '{terms[1]}'"]

        # Balance: ratio of minority to majority
        minority = min(a_count, b_count)
        majority = max(a_count, b_count)
        balance = minority / majority if majority > 0 else 0.0

        gaps = []
        if a_count == 0:
            gaps.append(f"No evidence found for '{terms[0]}'")
        if b_count == 0:
            gaps.append(f"No evidence found for '{terms[1]}'")
        if balance < 0.3 and a_count > 0 and b_count > 0:
            weaker = terms[0] if a_count < b_count else terms[1]
            gaps.append(f"Weak evidence for '{weaker}' ({minority} vs {majority} chunks)")

        return balance, gaps

    def _extract_comparison_terms(self, query: str) -> list[str]:
        """Extract the two concepts being compared."""
        patterns = [
            r"(?:compare|comparing)\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)",
            r"how\s+(?:does|do|is|are)\s+(.+?)\s+differ(?:s|ent)?\s+from\s+(.+)",
            r"(?:differ(?:s|ence)?|distinction)\s+(?:between\s+)?(.+?)\s+(?:and|from|vs\.?)\s+(.+)",
            r"(.+?)\s+vs\.?\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                a = re.sub(r"[?.!,]+$", "", match.group(1).strip()).strip()
                b = re.sub(r"[?.!,]+$", "", match.group(2).strip()).strip()
                if a and b:
                    return [a, b]
        return []

    def _identify_gaps(
        self,
        query: str,
        query_keywords: set[str],
        evidence_tokens: set[str],
        query_type: str,
    ) -> list[str]:
        """Identify what's missing from the evidence."""
        gaps = []

        # Find query keywords not in evidence
        missing_keywords = query_keywords - evidence_tokens
        if missing_keywords and len(missing_keywords) <= 5:
            gaps.append(f"Key terms not found in evidence: {', '.join(sorted(missing_keywords))}")

        if query_type == "factual":
            gaps.append("Specific fact requested may not be in the retrieved passages")
        elif query_type == "multi_step":
            gaps.append("May need evidence from additional sources for a complete multi-step answer")

        return gaps

    def _build_feedback_query(self, query: str, gaps: list[str]) -> str:
        """
        Build a refined query for re-retrieval based on gaps.

        Combines the original query with gap information to
        guide the retriever toward missing evidence.
        """
        if not gaps:
            return query

        # Extract key terms from gaps
        gap_text = " ".join(gaps)
        gap_keywords = set(re.findall(r"'([^']+)'", gap_text))

        if gap_keywords:
            return f"{query} {' '.join(gap_keywords)}"

        return query

    def _suggest_strategy(self, query_type: str, scores: list[float]) -> str:
        """Suggest a retrieval strategy for re-retrieval."""
        avg = sum(scores) / len(scores) if scores else 0

        if avg < 0.3:
            # Scores are very low, try a different approach
            return "hybrid"
        if query_type == "factual":
            return "keyword"
        return "vector"

    def _build_reasoning(
        self,
        score_quality: float,
        coverage: float,
        volume: float,
        balance: float | None,
        combined: float,
        threshold: float,
    ) -> str:
        """Build a human-readable explanation of the evaluation."""
        parts = [
            f"score_quality={score_quality:.2f}",
            f"keyword_coverage={coverage:.2f}",
            f"volume={volume:.2f}",
        ]
        if balance is not None:
            parts.append(f"comparison_balance={balance:.2f}")
        parts.append(f"combined={combined:.2f}")
        parts.append(f"threshold={threshold:.2f}")
        return ", ".join(parts)


# LLM-based evaluation

class LLMCritic:
    """
    Evaluates evidence quality using the LLM.

    Sends the query, query type, and evidence to the LLM and
    parses a structured judgment. Falls back to rule-based
    evaluation if the LLM call fails.

    More accurate than rule-based for:
        - Nuanced relevance (chunk is about the right topic but
          doesn't answer the specific question)
        - Semantic gap detection (evidence talks around the answer
          without directly providing it)
        - Quality assessment of explanations vs raw facts
    """

    def __init__(self):
        self._fallback = RuleBasedCritic()

    def evaluate(
        self,
        query: str,
        query_type: str,
        results: list,
        min_confidence: float = 0.4,
        is_final: bool = False,
    ) -> CriticVerdict:
        """
        Evaluate evidence using the LLM.

        Builds a prompt with the query, evidence, and evaluation
        criteria. Parses the JSON response into a CriticVerdict.
        Falls back to rule-based evaluation on failure.
        """
        from app.core.llm import get_client

        client = get_client()

        # Format evidence for the prompt
        evidence_text = self._format_evidence(results)

        prompt = (
            f"Query: {query}\n"
            f"Query type: {query_type}\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            f"Evaluate whether this evidence is sufficient to answer the query."
        )

        try:
            response = client.complete_json(
                prompt=prompt,
                system=CRITIC_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )

            if not response.has_parsed:
                logger.warning("Critic LLM returned unparseable output, falling back to rules")
                return self._fallback.evaluate(query, query_type, results, min_confidence, is_final)

            return self._parse_llm_verdict(
                query, query_type, results, response.parsed, min_confidence, is_final
            )

        except Exception as e:
            logger.warning("Critic LLM call failed (%s), falling back to rules", e)
            return self._fallback.evaluate(query, query_type, results, min_confidence, is_final)

    def _format_evidence(self, results: list) -> str:
        """Format results for the LLM prompt."""
        if not results:
            return "No evidence retrieved."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] (Source: {r.source}, Page {r.page}, "
                f"score={r.effective_score:.3f})\n{r.text}"
            )
        return "\n\n".join(parts)

    def _parse_llm_verdict(
        self,
        query: str,
        query_type: str,
        results: list,
        data: dict | list,
        min_confidence: float,
        is_final: bool,
    ) -> CriticVerdict:
        """
        Convert LLM JSON into a validated CriticVerdict.

        Every field is validated and clamped to safe ranges.
        The LLM is treated as unreliable — all outputs are checked.
        """
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return self._fallback.evaluate(query, query_type, results, min_confidence, is_final)

        # Validate verdict
        verdict = data.get("verdict", "insufficient")
        if verdict not in ("sufficient", "insufficient"):
            verdict = "insufficient"

        # Validate scores (clamp to 0-1)
        def clamp(val, default=0.0):
            try:
                return max(0.0, min(1.0, float(val)))
            except (ValueError, TypeError):
                return default

        confidence = clamp(data.get("confidence", 0.0))
        coverage_score = clamp(data.get("coverage_score", 0.0))
        sufficiency_score = clamp(data.get("sufficiency_score", 0.0))

        # Validate gaps
        gaps = data.get("gaps", [])
        if not isinstance(gaps, list):
            gaps = [str(gaps)] if gaps else []
        gaps = [str(g).strip() for g in gaps if g]

        # Validate feedback query
        feedback_query = str(data.get("feedback_query", "")).strip()
        suggested_strategy = str(data.get("suggested_strategy", "")).strip()
        if suggested_strategy not in ("vector", "keyword", "hybrid", ""):
            suggested_strategy = ""

        reasoning = str(data.get("reasoning", ""))

        # Apply threshold check (override LLM if needed)
        threshold = min_confidence * (0.6 if is_final else 1.0)
        if confidence < threshold and verdict == "sufficient":
            verdict = "insufficient"
            reasoning += f" [Overridden: confidence {confidence:.2f} < threshold {threshold:.2f}]"

        # Separate approved vs rejected
        score_threshold = min_confidence * 0.5 if is_final else min_confidence * 0.7
        approved = [r for r in results if r.effective_score >= score_threshold]
        rejected = [r for r in results if r.effective_score < score_threshold]

        if not approved and results:
            approved = sorted(results, key=lambda r: r.effective_score, reverse=True)[:2]
            rejected = [r for r in results if r not in approved]

        # Writer caveats
        caveats = []
        if is_final and verdict == "insufficient":
            caveats.append("Evidence may be incomplete. Answer with available information and note gaps.")
            if gaps:
                caveats.append(f"Known gaps: {'; '.join(gaps[:3])}")

        return CriticVerdict(
            verdict=verdict,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            coverage_score=round(coverage_score, 3),
            sufficiency_score=round(sufficiency_score, 3),
            approved_results=approved,
            rejected_results=rejected,
            gaps=gaps,
            feedback_query=feedback_query,
            suggested_strategy=suggested_strategy,
            evaluation_mode="llm",
            is_final=is_final,
            writer_caveats=caveats,
        )


# Main critic interface

class Critic:
    """
    Main critic interface for NeuroGraph.

    Two usage patterns:

    1. Single evaluation:
        verdict = critic.evaluate(plan, response)

    2. Full feedback loop (recommended):
        verdict = critic.evaluate_with_retry(plan, retriever, response)
        This handles re-retrieval automatically until evidence
        is sufficient or max iterations are reached.

    The critic adapts its strictness based on iteration:
        - Iteration 1: strict (require high coverage and score)
        - Iteration 2: moderate (accept partial coverage)
        - Final iteration: lenient (accept whatever we have, add caveats)
    """

    def __init__(self, mode: str = "auto"):
        """
        Args:
            mode: "auto" (try LLM, fall back to rules),
                  "llm" (LLM only), "rules" (rule-based only)
        """
        self._mode = mode
        self._llm_critic = LLMCritic()
        self._rule_critic = RuleBasedCritic()
        self._llm_available: bool | None = None

    def _check_llm(self) -> bool:
        """Check if LLM is reachable. Cached after first check."""
        if self._llm_available is None:
            try:
                from app.core.llm import get_client
                self._llm_available = get_client().is_available()
            except Exception:
                self._llm_available = False
        return self._llm_available

    def _get_evaluator(self) -> RuleBasedCritic | LLMCritic:
        """Return the appropriate evaluator based on mode and availability."""
        if self._mode == "rules":
            return self._rule_critic
        if self._mode == "llm":
            return self._llm_critic
        # Auto mode
        if self._check_llm():
            return self._llm_critic
        return self._rule_critic

    def evaluate(
        self,
        plan: Plan,
        response: RetrievalResponse,
        is_final: bool = False,
    ) -> CriticVerdict:
        """
        Evaluate a single retrieval response against the plan.

        Args:
            plan: from the planner (provides query_type, min_confidence)
            response: from the retriever
            is_final: if True, lower thresholds (last allowed iteration)

        Returns:
            CriticVerdict with judgment, gaps, and feedback
        """
        evaluator = self._get_evaluator()

        verdict = evaluator.evaluate(
            query=plan.query,
            query_type=plan.query_type,
            results=response.results,
            min_confidence=plan.min_confidence_for_answer,
            is_final=is_final,
        )

        verdict.iteration = plan.current_iteration

        logger.info("Critic: %s", verdict.summary())

        return verdict

    def evaluate_with_retry(
        self,
        plan: Plan,
        retriever: Retriever,
        initial_response: RetrievalResponse,
    ) -> CriticVerdict:
        """
        Full feedback loop: evaluate, re-retrieve if needed, repeat.

        This is the main entry point for the critic in the pipeline.
        It handles the entire feedback loop:

            1. Evaluate initial evidence
            2. If insufficient and iterations remain:
               a. Use critic feedback to build a refined query
               b. Call retriever.search_with_feedback() to find new chunks
               c. Merge new results with previous best
               d. Re-evaluate
            3. On final iteration: accept best available, add caveats

        The plan's max_iterations and current_iteration fields
        control loop behavior. Each iteration is logged.

        Args:
            plan: from the planner (controls iteration limits)
            retriever: for re-retrieval calls
            initial_response: first retrieval attempt

        Returns:
            Final CriticVerdict (sufficient or best-effort with caveats)
        """
        current_response = initial_response
        all_seen_results: list = list(initial_response.results)
        best_verdict: CriticVerdict | None = None

        for iteration in range(1, plan.max_iterations + 1):
            plan.current_iteration = iteration
            is_final = iteration == plan.max_iterations

            # Evaluate current evidence
            verdict = self.evaluate(plan, current_response, is_final=is_final)
            verdict.iteration = iteration
            verdict.is_final = is_final

            # Track best verdict (highest confidence)
            if best_verdict is None or verdict.confidence > best_verdict.confidence:
                best_verdict = verdict

            # If sufficient, we are done
            if verdict.is_sufficient:
                logger.info(
                    "Critic: evidence sufficient at iteration %d/%d",
                    iteration, plan.max_iterations,
                )
                return verdict

            # If final iteration, accept best available
            if is_final:
                logger.info(
                    "Critic: max iterations reached (%d), accepting best available",
                    plan.max_iterations,
                )
                # Use the best verdict we got, but with final caveats
                best_verdict.is_final = True
                if not best_verdict.writer_caveats:
                    best_verdict.writer_caveats = [
                        "Evidence may be incomplete after exhausting retrieval attempts.",
                    ]
                    if best_verdict.gaps:
                        best_verdict.writer_caveats.append(
                            f"Known gaps: {'; '.join(best_verdict.gaps[:3])}"
                        )
                # Even though insufficient, mark as "best_effort"
                # so the writer proceeds instead of returning nothing
                best_verdict.verdict = "best_effort"
                return best_verdict

            # Not sufficient and not final: re-retrieve
            logger.info(
                "Critic: insufficient at iteration %d/%d, re-retrieving. "
                "Feedback: '%s'",
                iteration, plan.max_iterations,
                verdict.feedback_query[:100] if verdict.feedback_query else "(none)",
            )

            feedback_response = retriever.search_with_feedback(
                query=plan.query,
                previous_results=all_seen_results,
                feedback=verdict.feedback_query,
                strategy=verdict.suggested_strategy or None,
                iteration=iteration + 1,
            )

            # Merge new results into our pool
            new_ids = {r.chunk_id for r in feedback_response.results}
            seen_ids = {r.chunk_id for r in all_seen_results}
            new_chunks = [r for r in feedback_response.results if r.chunk_id not in seen_ids]

            all_seen_results.extend(new_chunks)

            # Build a combined response with best results from all iterations
            combined_results = sorted(
                all_seen_results,
                key=lambda r: r.effective_score,
                reverse=True,
            )[:plan.max_iterations * settings.top_k]

            # Create a synthetic response for re-evaluation
            current_response = type(initial_response)(
                query=plan.query,
                results=combined_results,
                strategy=feedback_response.strategy,
                total_candidates=len(all_seen_results),
                iteration=iteration + 1,
            )

            logger.info(
                "Critic: re-retrieval found %d new chunks, total pool=%d",
                len(new_chunks), len(all_seen_results),
            )

        # Should not reach here, but safety fallback
        return best_verdict or verdict