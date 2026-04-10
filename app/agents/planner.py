"""
Planner agent — app/agents/planner.py

The planner is the first agent in the NeuroGraph pipeline.
It receives a user query and produces a structured Plan that
tells the retriever how to search and the writer how to respond.

Pipeline position:
    user query -> [PLANNER] -> Plan -> retriever -> critic -> writer

What the planner decides:
    1. Query type: factual / conceptual / comparison / multi_step
    2. Retrieval strategy: vector / keyword / hybrid
    3. Sub-queries: decomposed parts for complex questions
    4. Writer instructions: how the final answer should be structured
    5. Expected evidence: what kind of chunks would answer this well

Two planning modes:
    - LLM-based (primary): uses Ollama/Groq to analyze the query
    - Rule-based (fallback): uses regex patterns when LLM is unavailable

Graceful degradation:
    LLM available   -> full LLM planning
    LLM unavailable -> rule-based fallback (still functional)
    Both fail       -> minimal plan with safe defaults

Usage:
    from app.agents.planner import Planner

    planner = Planner()
    plan = planner.plan("Derive the free energy bound in variational inference")

    print(plan.query_type)          # "conceptual"
    print(plan.retrieval_strategy)  # "vector"
    print(plan.sub_queries)         # ["free energy bound", "variational inference"]
    print(plan.writer_instructions) # "Explain step by step with full LaTeX..."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# Plan data structure

QUERY_TYPES = ("factual", "conceptual", "comparison", "multi_step")
STRATEGIES = ("vector", "keyword", "hybrid")


@dataclass
class Plan:
    """
    Structured output from the planner agent.

    Every downstream component reads this plan:
        - Retriever uses: retrieval_strategy, sub_queries, expand
        - Critic uses: expected_evidence, confidence
        - Writer uses: writer_instructions, query_type
    """
    query: str

    query_type: str = "conceptual"

    retrieval_strategy: str = "hybrid"
    sub_queries: list[str] = field(default_factory=list)
    expand: bool = False

    writer_instructions: str = ""

    expected_evidence: str = ""
    confidence: float = 0.5
    planning_mode: str = "fallback"
    reasoning: str = ""
    max_iterations: int = 3
    current_iteration: int = 1
    min_confidence_for_answer: float = 0.4

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "retrieval_strategy": self.retrieval_strategy,
            "sub_queries": self.sub_queries,
            "expand": self.expand,
            "writer_instructions": self.writer_instructions,
            "expected_evidence": self.expected_evidence,
            "confidence": round(self.confidence, 2),
            "planning_mode": self.planning_mode,
            "reasoning": self.reasoning,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "min_confidence_for_answer": self.min_confidence_for_answer,
        }

    def summary(self) -> str:
        sub_q = f", sub_queries={self.sub_queries}" if self.sub_queries else ""
        return (
            f"[{self.planning_mode}] "
            f"type={self.query_type}, "
            f"strategy={self.retrieval_strategy}, "
            f"confidence={self.confidence:.2f}, "
            f"iter={self.current_iteration}/{self.max_iterations}"
            f"{sub_q}"
        )


# Planner system prompt

PLANNER_SYSTEM_PROMPT = """\
You are the planning agent in a research paper study assistant.

Your job: analyze the user's query and produce a retrieval plan.

You must respond with ONLY a JSON object (no markdown, no explanation).

JSON schema:
{
    "query_type": "factual" | "conceptual" | "comparison" | "multi_step",
    "retrieval_strategy": "vector" | "keyword" | "hybrid",
    "sub_queries": ["list", "of", "focused", "searches"],
    "expand": true | false,
    "writer_instructions": "how the writer should structure the answer",
    "expected_evidence": "what kind of passages would answer this well",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation of your decisions"
}

Classification rules:
- "factual": who/what/when/where, specific names, numbers, dates, definitions
- "conceptual": how/why, mechanisms, theories, derivations, proofs, explanations
- "comparison": compare X and Y, differences, similarities, vs, advantages
- "multi_step": requires answering sub-questions first, then synthesizing

Strategy rules:
- "keyword": best for factual queries with specific terms (names, acronyms, numbers, equation references like Eq. 3.46)
- "vector": best for conceptual queries about meaning, mechanisms, and relationships
- "hybrid": best for comparisons, multi-step, math-heavy queries, or when unsure

Sub-query rules:
- For comparisons: extract each concept as a separate sub-query
- For multi-step: break into sequential sub-questions
- For math queries: include equation numbers, variable names, theorem names, and key notation as sub-queries
- For derivations: include both the starting point and the result as sub-queries
- For simple queries: leave empty or include reformulations
- Always keep the original query as an implicit first search

Writer instruction rules:
- For factual: "Provide a direct answer with the specific fact and its source"
- For conceptual: "Explain the concept step by step with full mathematical formalism in LaTeX, define all notation, break down equations term by term, and give examples where possible"
- For comparison: "Compare and contrast using structured sections, show mathematical differences in LaTeX, define notation for both frameworks, cite both sides"
- For multi_step: "Answer each sub-question with full derivations in LaTeX, show all intermediate steps, then synthesize into a final answer"

The system handles research papers from ANY domain: mathematics, physics,
computer science, neuroscience, biology, engineering, economics, and more.
Queries may involve complex mathematical notation, proofs, derivations,
algorithms, theorems, and formal definitions.
"""


# Rule-based fallback planner

class RuleBasedPlanner:
    """
    Fallback planner using regex patterns.

    Used when the LLM is unavailable. Handles factual, conceptual,
    comparison, and multi-step queries with regex classification
    and template-based writer instructions.
    """

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
        r"\b(?:mechanism|theory|framework|principle|process)\b",
        r"\b(?:derive|derivation|proof|prove|theorem)\b",
        r"\b(?:equation|formula|formulation|formalism)\b",
    ]

    COMPARISON_PATTERNS = [
        r"\b(?:compare|comparing|comparison)\b",
        r"\b(?:differ|difference|distinction|versus|vs\.?)\b",
        r"\b(?:better|worse|advantage|disadvantage)\b",
        r"\b(?:contrast|similarities|similar to)\b",
    ]

    MULTI_STEP_PATTERNS = [
        r"\b(?:and then|first.*then|step by step)\b",
        r"\b(?:how would you|design|build|create|implement)\b",
        r"\b(?:what are the implications|what follows from)\b",
    ]

    WRITER_TEMPLATES = {
        "factual": (
            "Provide a direct answer with the specific fact requested. "
            "Cite the source document and page number. "
            "If the fact involves a number or equation, format it in LaTeX."
        ),
        "conceptual": (
            "Explain the concept step by step with full mathematical formalism. "
            "Use LaTeX for ALL equations ($$...$$ for block, $...$ for inline). "
            "Define all notation before using it. Break down every equation term by term. "
            "Give examples where possible. Cite each claim with [N] references."
        ),
        "comparison": (
            "Compare and contrast the concepts using structured sections. "
            "Show mathematical differences in proper LaTeX format. "
            "Define notation for both frameworks. Present evidence for each side "
            "and highlight key differences and similarities. Cite sources for each point."
        ),
        "multi_step": (
            "Break the answer into logical steps. For each step, show the full "
            "mathematical derivation in LaTeX with all intermediate results. "
            "Define notation at each stage. Address each sub-question before "
            "synthesizing a final answer. Cite sources throughout."
        ),
    }

    def plan(self, query: str) -> Plan:
        q_lower = query.lower().strip()
        query_type = "conceptual"
        strategy = "hybrid"
        expand = False
        sub_queries: list[str] = []
        confidence = 0.6

        # Check comparisons first
        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, q_lower):
                query_type = "comparison"
                strategy = "hybrid"
                expand = True
                confidence = 0.7
                sub_queries = self._extract_comparison_terms(query)
                break

        # Check multi-step
        if query_type == "conceptual":
            for pattern in self.MULTI_STEP_PATTERNS:
                if re.search(pattern, q_lower):
                    query_type = "multi_step"
                    strategy = "hybrid"
                    confidence = 0.5
                    break

        # Check factual
        if query_type == "conceptual":
            has_acronym = bool(re.search(r"\b[A-Z]{2,}\b", query))
            has_number = bool(re.search(r"\b\d+\.?\d*\b", query))
            word_count = len(q_lower.split())

            for pattern in self.FACTUAL_PATTERNS:
                if re.search(pattern, q_lower):
                    query_type = "factual"
                    strategy = "keyword"
                    confidence = 0.8
                    break

            if query_type == "conceptual" and word_count <= 5 and (has_acronym or has_number):
                query_type = "factual"
                strategy = "keyword"
                confidence = 0.7

        # Check conceptual explicitly
        if query_type == "conceptual":
            for pattern in self.CONCEPTUAL_PATTERNS:
                if re.search(pattern, q_lower):
                    confidence = 0.75
                    strategy = "vector"
                    break

        # Detect math-heavy queries and boost to hybrid
        has_math_terms = bool(re.search(
            r"\b(?:equation|formula|derivat|proof|theorem|lemma|"
            r"integra|gradient|matrix|vector|eigenvalue|convergence|"
            r"optimization|objective function|loss function|"
            r"likelihood|posterior|prior|bayesian|laplacian|"
            r"differential|partial|stochastic)\b",
            q_lower,
        ))
        if has_math_terms and strategy == "vector":
            strategy = "hybrid"

        writer_instructions = self.WRITER_TEMPLATES.get(
            query_type, self.WRITER_TEMPLATES["conceptual"]
        )

        return Plan(
            query=query,
            query_type=query_type,
            retrieval_strategy=strategy,
            sub_queries=sub_queries,
            expand=expand,
            writer_instructions=writer_instructions,
            expected_evidence=self._expected_evidence(query_type),
            confidence=confidence,
            planning_mode="fallback",
            reasoning=f"Rule-based classification: {query_type}",
        )

    def _extract_comparison_terms(self, query: str) -> list[str]:
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

    def _expected_evidence(self, query_type: str) -> str:
        templates = {
            "factual": "Specific passages containing the requested fact, name, date, number, or definition",
            "conceptual": "Explanatory passages with mathematical formulations, derivations, definitions, and notation",
            "comparison": "Passages about each concept being compared, with mathematical frameworks from both sides",
            "multi_step": "Evidence addressing each sub-question, including derivation steps and intermediate results",
        }
        return templates.get(query_type, templates["conceptual"])


# LLM-based planner

class LLMPlanner:
    """
    Plans queries using the LLM for deeper analysis.

    Falls back to rule-based planning if the LLM call fails
    or returns unparseable output.
    """

    def __init__(self):
        self._fallback = RuleBasedPlanner()

    def plan(self, query: str) -> Plan:
        from app.core.llm import get_client

        client = get_client()

        try:
            response = client.complete_json(
                prompt=f"Analyze this query and create a retrieval plan:\n\n{query}",
                system=PLANNER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )

            if not response.has_parsed:
                logger.warning(
                    "Planner LLM returned unparseable output, falling back to rules"
                )
                return self._fallback.plan(query)

            return self._parse_llm_plan(query, response.parsed)

        except Exception as e:
            logger.warning("Planner LLM call failed (%s), falling back to rules", e)
            return self._fallback.plan(query)

    def _parse_llm_plan(self, query: str, data: dict | list) -> Plan:
        if isinstance(data, list):
            data = data[0] if data else {}

        if not isinstance(data, dict):
            logger.warning("LLM plan is not a dict, falling back")
            return RuleBasedPlanner().plan(query)

        query_type = data.get("query_type", "conceptual")
        if query_type not in QUERY_TYPES:
            query_type = "conceptual"

        strategy = data.get("retrieval_strategy", "hybrid")
        if strategy not in STRATEGIES:
            strategy = "hybrid"

        sub_queries = data.get("sub_queries", [])
        if not isinstance(sub_queries, list):
            sub_queries = []
        sub_queries = [str(q).strip() for q in sub_queries if q]

        expand = bool(data.get("expand", False))

        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        writer_instructions = str(data.get("writer_instructions", ""))
        expected_evidence = str(data.get("expected_evidence", ""))
        reasoning = str(data.get("reasoning", ""))

        if not writer_instructions.strip():
            writer_instructions = RuleBasedPlanner.WRITER_TEMPLATES.get(
                query_type,
                RuleBasedPlanner.WRITER_TEMPLATES["conceptual"],
            )

        return Plan(
            query=query,
            query_type=query_type,
            retrieval_strategy=strategy,
            sub_queries=sub_queries,
            expand=expand,
            writer_instructions=writer_instructions,
            expected_evidence=expected_evidence,
            confidence=confidence,
            planning_mode="llm",
            reasoning=reasoning,
        )


# Main planner interface

class Planner:
    """
    Main planner interface for NeuroGraph.

    Usage:
        planner = Planner()
        plan = planner.plan("Derive the ELBO in variational inference")
        plan = planner.plan("Compare SGD and Adam optimizers", mode="rules")
    """

    def __init__(self):
        self._llm_planner = LLMPlanner()
        self._rule_planner = RuleBasedPlanner()
        self._llm_available: bool | None = None

    def _check_llm(self) -> bool:
        if self._llm_available is None:
            try:
                from app.core.llm import get_client
                self._llm_available = get_client().is_available()
            except Exception:
                self._llm_available = False

            if self._llm_available:
                logger.info("Planner: LLM available, using LLM-based planning")
            else:
                logger.info("Planner: LLM unavailable, using rule-based fallback")

        return self._llm_available

    def plan(
        self,
        query: str,
        mode: str = "auto",
    ) -> Plan:
        if not query or not query.strip():
            return Plan(
                query=query,
                query_type="conceptual",
                retrieval_strategy="hybrid",
                writer_instructions="No query provided.",
                planning_mode="fallback",
                reasoning="Empty query",
            )

        if mode == "rules":
            plan = self._rule_planner.plan(query)
            logger.info("Plan (rules): %s", plan.summary())
            return plan

        if mode == "llm":
            plan = self._llm_planner.plan(query)
            logger.info("Plan (llm): %s", plan.summary())
            return plan

        if self._check_llm():
            plan = self._llm_planner.plan(query)
        else:
            plan = self._rule_planner.plan(query)

        logger.info("Plan: %s", plan.summary())
        return plan

    def reset(self) -> None:
        self._llm_available = None