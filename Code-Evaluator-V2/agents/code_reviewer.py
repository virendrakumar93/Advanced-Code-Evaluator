"""Code Reviewer Agent — evaluates style, readability, maintainability, best practices."""

from __future__ import annotations

import logging
from typing import List

from core.schema import AgentJudgment, ExecutionArtifact, RubricScores
from llm.model_router import query_models
from llm.prompt_templates import CODE_REVIEWER_SYSTEM, CODE_REVIEWER_USER

logger = logging.getLogger(__name__)


def _format_warnings(artifact: ExecutionArtifact) -> str:
    if not artifact.static_warnings:
        return "None."
    lines = []
    for w in artifact.static_warnings[:10]:
        lines.append(f"- L{w.line} [{w.code}]: {w.message}")
    return "\n".join(lines)


def run(
    problem_description: str,
    submission_code: str,
    artifact: ExecutionArtifact,
    det_scores: RubricScores,
    models: List[str],
    llm_config: dict,
) -> AgentJudgment:
    user_prompt = CODE_REVIEWER_USER.format(
        problem_description=problem_description,
        submission_code=submission_code,
        warning_count=len(artifact.static_warnings),
        static_warnings=_format_warnings(artifact),
        det_style=det_scores.style,
        det_clarity=det_scores.clarity,
    )

    parsed, model_used = query_models(
        models, CODE_REVIEWER_SYSTEM, user_prompt,
        max_tokens=llm_config.get("max_tokens", 2048),
        temperature=llm_config.get("temperature", 0.1),
        retries=llm_config.get("retry_attempts", 3),
    )

    if parsed and "scores" in parsed:
        s = parsed["scores"]
        return AgentJudgment(
            agent_name="code_reviewer",
            scores=RubricScores(
                correctness=_clamp(s.get("correctness", det_scores.correctness)),
                edge_cases=_clamp(s.get("edge_cases", det_scores.edge_cases)),
                complexity=_clamp(s.get("complexity", det_scores.complexity)),
                style=_clamp(s.get("style", det_scores.style)),
                clarity=_clamp(s.get("clarity", det_scores.clarity)),
            ),
            reasoning=parsed.get("reasoning", ""),
            issues=parsed.get("issues", []),
            suggestions=parsed.get("suggestions", []),
            confidence=parsed.get("confidence", 0.5),
            model_used=model_used,
            fallback=False,
        )

    logger.info("Code Reviewer falling back to deterministic scores.")
    return AgentJudgment(
        agent_name="code_reviewer",
        scores=det_scores,
        reasoning="LLM unavailable — using deterministic scores.",
        issues=["LLM evaluation unavailable"],
        confidence=0.3,
        model_used="",
        fallback=True,
    )


def _clamp(v, lo=0.0, hi=10.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 5.0
