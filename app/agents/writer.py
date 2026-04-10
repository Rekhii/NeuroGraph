"""
Writer agent — app/agents/writer.py

The writer is the final agent in the NeuroGraph pipeline.
It takes critic-approved evidence and produces a detailed,
pedagogical, well-cited answer with proper LaTeX mathematics.

Pipeline position:
    planner -> retriever -> critic -> context optimizer -> [WRITER] -> answer

Design philosophy:
    NeuroGraph is a deep research paper study tool. Users upload
    complex mathematical papers and expect the system to:
    - Explain concepts thoroughly, not just summarize
    - Reproduce ALL mathematics in proper LaTeX format
    - Break down equations term by term
    - Provide intuition, examples, and connections
    - Teach, not just retrieve

    The writer is instructed to be a patient, rigorous tutor
    who assumes the reader is intelligent but may not know
    the specific notation or domain conventions.

Two writing modes:
    - LLM-based (primary): prompts the LLM to write a detailed cited answer
    - Rule-based (fallback): assembles an answer from chunk text directly

Grounding contract:
    The writer MUST NOT add information beyond what is in the
    approved evidence. Every claim must trace to a numbered
    citation. This is the anti-hallucination guarantee.

Usage:
    from app.agents.writer import Writer

    writer = Writer()
    response = writer.write(plan=plan, verdict=critic_verdict)
    print(response.answer)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.agents.planner import Plan
    from app.agents.critic import CriticVerdict

logger = logging.getLogger(__name__)


# Writer response

@dataclass
class WriterResponse:
    """The writer's final output."""
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    citation_count: int = 0
    has_caveats: bool = False
    caveats: list[str] = field(default_factory=list)
    writing_mode: str = "fallback"
    query: str = ""
    query_type: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": round(self.confidence, 3),
            "citation_count": self.citation_count,
            "has_caveats": self.has_caveats,
            "caveats": self.caveats,
            "writing_mode": self.writing_mode,
            "query": self.query,
            "query_type": self.query_type,
        }

    def summary(self) -> str:
        caveat_info = f", caveats={len(self.caveats)}" if self.caveats else ""
        return (
            f"[{self.writing_mode}] "
            f"citations={self.citation_count}, "
            f"sources={len(self.sources)}, "
            f"confidence={self.confidence:.2f}, "
            f"length={len(self.answer)}"
            f"{caveat_info}"
        )


# Writer system prompt

WRITER_SYSTEM_PROMPT = """\
You are an expert research tutor embedded in a paper study system.
Users upload complex research papers and ask you to explain them deeply.

Your job: write a detailed, pedagogical, well-cited answer using ONLY the provided evidence.

STRICT RULES:
1. Every factual claim MUST have an inline citation like [1], [2], etc.
2. Do NOT add information that is not in the provided evidence.
3. If the evidence is insufficient, say so explicitly.

MATHEMATICS — THIS IS CRITICAL:
- Reproduce ALL mathematical expressions in proper LaTeX format.
- Use $...$ for inline math: "the learning rate $\\alpha$"
- Use $$...$$ for block equations on their own line:

$$\\Delta Q(s, a) = \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$

- After every equation, explain it term by term:
  - What each variable represents
  - What the equation computes
  - Why it matters in context
- Preserve ALL subscripts, superscripts, Greek letters, operators, summations, integrals, and special notation from the source.
- If the source uses numbered equations (like Eq. 3.46), reference them by number.
- Never skip or simplify equations. Reproduce them exactly as they appear in the evidence.

ANSWER STRUCTURE:
- Start with a clear conceptual overview (what is this about, why does it matter)
- Present the formal framework with proper definitions
- Show the key equations in LaTeX with full breakdowns
- Explain the intuition behind each mathematical result
- If applicable, give a concrete example or worked-through case
- Note connections to related concepts mentioned in the evidence
- End with limitations or caveats if the evidence is incomplete

DEPTH AND DETAIL:
- Write thorough, multi-paragraph explanations — not summaries.
- Assume the reader is intelligent but may not know this specific domain.
- Define all notation before using it.
- When comparing concepts, create clear structured sections for each.
- For multi-step derivations, show each step explicitly.

CITATION FORMAT:
- Inline: "The model uses temporal difference learning [1], where the update rule is... [3]"
- Multiple sources: [1, 3]
- Each number corresponds to an evidence chunk.

If writer caveats are provided, include a note at the end acknowledging limitations.
"""


# Source extraction

def _build_sources(results: list) -> list[dict]:
    """Build a deduplicated source bibliography from results."""
    seen: dict[str, dict] = {}

    for i, r in enumerate(results, 1):
        key = r.source
        if key not in seen:
            seen[key] = {
                "source": r.source,
                "pages": set(),
                "citation_numbers": [],
            }
        seen[key]["pages"].add(r.page)
        seen[key]["citation_numbers"].append(i)

    sources = []
    for data in seen.values():
        pages = sorted(data["pages"])
        sources.append({
            "source": data["source"],
            "pages": pages,
            "cited_as": data["citation_numbers"],
        })

    return sources


def _count_citations(text: str) -> int:
    """
    Count unique citation numbers in the answer text.

    Handles both single citations [1] and comma-separated
    citations [1, 2, 3].
    """
    bracket_groups = re.findall(r"\[[\d,\s]+\]", text)
    all_refs = set()
    for group in bracket_groups:
        nums = re.findall(r"\d+", group)
        all_refs.update(nums)
    return len(all_refs)


# Rule-based writer

class RuleBasedWriter:
    """
    Assembles an answer directly from evidence chunks.

    Used when the LLM is unavailable. Maintains the citation
    contract: every piece of information traces to a source.
    """

    TEMPLATES = {
        "factual": (
            "Based on the available evidence:\n\n"
            "{evidence}\n\n"
            "{caveats}"
            "Sources:\n{bibliography}"
        ),
        "conceptual": (
            "{evidence}\n\n"
            "{caveats}"
            "Sources:\n{bibliography}"
        ),
        "comparison": (
            "Comparing the concepts based on available evidence:\n\n"
            "{evidence}\n\n"
            "{caveats}"
            "Sources:\n{bibliography}"
        ),
        "multi_step": (
            "Addressing the question step by step:\n\n"
            "{evidence}\n\n"
            "{caveats}"
            "Sources:\n{bibliography}"
        ),
    }

    def write(
        self,
        query: str,
        query_type: str,
        results: list,
        writer_instructions: str = "",
        caveats: list[str] | None = None,
    ) -> WriterResponse:
        if not results:
            return WriterResponse(
                answer="No evidence was found to answer this question.",
                confidence=0.0,
                writing_mode="fallback",
                query=query,
                query_type=query_type,
            )

        evidence_parts = []
        for i, r in enumerate(results, 1):
            evidence_parts.append(f"{r.text} [{i}]")

        evidence_text = "\n\n".join(evidence_parts)

        caveat_text = ""
        if caveats:
            caveat_text = "Note: " + " ".join(caveats) + "\n\n"

        sources = _build_sources(results)
        bib_parts = []
        for s in sources:
            pages = ", ".join(str(p) for p in s["pages"])
            refs = ", ".join(str(n) for n in s["cited_as"])
            bib_parts.append(f"- {s['source']} (Pages: {pages}) [cited as: {refs}]")

        bibliography = "\n".join(bib_parts)

        template = self.TEMPLATES.get(query_type, self.TEMPLATES["conceptual"])
        answer = template.format(
            evidence=evidence_text,
            caveats=caveat_text,
            bibliography=bibliography,
        )

        return WriterResponse(
            answer=answer.strip(),
            sources=sources,
            confidence=round(sum(r.effective_score for r in results) / len(results), 3),
            citation_count=len(results),
            has_caveats=bool(caveats),
            caveats=caveats or [],
            writing_mode="fallback",
            query=query,
            query_type=query_type,
        )


# LLM-based writer

class LLMWriter:
    """
    Generates detailed, pedagogical answers using the LLM.

    The prompt is designed to produce research-tutor quality
    explanations with proper LaTeX mathematics, term-by-term
    breakdowns, examples, and thorough citations.
    """

    def __init__(self):
        self._fallback = RuleBasedWriter()

    def write(
        self,
        query: str,
        query_type: str,
        results: list,
        writer_instructions: str = "",
        caveats: list[str] | None = None,
    ) -> WriterResponse:
        from app.core.llm import get_client

        if not results:
            return WriterResponse(
                answer="No evidence was found to answer this question.",
                confidence=0.0,
                writing_mode="llm",
                query=query,
                query_type=query_type,
            )

        client = get_client()

        # Build evidence section
        evidence_parts = []
        for i, r in enumerate(results, 1):
            evidence_parts.append(
                f"[{i}] (Source: {r.source}, Page {r.page})\n{r.text}"
            )
        evidence_text = "\n\n".join(evidence_parts)

        # Build prompt
        prompt_parts = [
            f"User question: {query}",
            f"Query type: {query_type}",
        ]

        if writer_instructions:
            prompt_parts.append(f"Writing instructions: {writer_instructions}")

        prompt_parts.append(f"\nEvidence ({len(results)} chunks):\n{evidence_text}")

        if caveats:
            prompt_parts.append(
                "\nIMPORTANT caveats to mention:\n" + "\n".join(f"- {c}" for c in caveats)
            )

        prompt_parts.append(
            "\nWrite a detailed, pedagogical answer. "
            "Reproduce ALL mathematics in proper LaTeX ($$...$$ for block, $...$ for inline). "
            "Break down every equation term by term. "
            "Define all notation. Give examples where possible. "
            "Every claim must have an inline citation [N]."
        )

        prompt = "\n\n".join(prompt_parts)

        try:
            response = client.complete(
                prompt=prompt,
                system=WRITER_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=12048,
            )

            answer = response.text.strip()

            # Validate citations
            citation_count = _count_citations(answer)
            max_valid_ref = len(results)
            invalid_refs = set()
            for group in re.findall(r"\[[\d,\s]+\]", answer):
                for ref in re.findall(r"\d+", group):
                    if int(ref) > max_valid_ref or int(ref) < 1:
                        invalid_refs.add(ref)

            if invalid_refs:
                logger.warning(
                    "Writer produced invalid citation refs: %s (max=%d)",
                    invalid_refs, max_valid_ref,
                )

            sources = _build_sources(results)

            avg_score = sum(r.effective_score for r in results) / len(results)
            citation_ratio = min(1.0, citation_count / max(1, len(results)))
            confidence = avg_score * 0.7 + citation_ratio * 0.3

            return WriterResponse(
                answer=answer,
                sources=sources,
                confidence=round(confidence, 3),
                citation_count=citation_count,
                has_caveats=bool(caveats),
                caveats=caveats or [],
                writing_mode="llm",
                query=query,
                query_type=query_type,
            )

        except Exception as e:
            logger.warning("Writer LLM call failed (%s), falling back to rules", e)
            return self._fallback.write(query, query_type, results, writer_instructions, caveats)


# Main writer interface

class Writer:
    """
    Main writer interface for NeuroGraph.

    Usage:
        writer = Writer()
        response = writer.write(plan=plan, verdict=verdict)
        print(response.answer)
    """

    def __init__(self, mode: str = "auto"):
        self._mode = mode
        self._llm_writer = LLMWriter()
        self._rule_writer = RuleBasedWriter()
        self._llm_available: bool | None = None

    def _check_llm(self) -> bool:
        if self._llm_available is None:
            try:
                from app.core.llm import get_client
                self._llm_available = get_client().is_available()
            except Exception:
                self._llm_available = False
        return self._llm_available

    def write(
        self,
        plan: Plan,
        verdict: CriticVerdict,
    ) -> WriterResponse:
        results = verdict.approved_results
        caveats = verdict.writer_caveats if verdict.writer_caveats else None

        if self._mode == "rules":
            writer = self._rule_writer
        elif self._mode == "llm":
            writer = self._llm_writer
        elif self._check_llm():
            writer = self._llm_writer
        else:
            writer = self._rule_writer

        response = writer.write(
            query=plan.query,
            query_type=plan.query_type,
            results=results,
            writer_instructions=plan.writer_instructions,
            caveats=caveats,
        )

        logger.info("Writer: %s", response.summary())

        return response