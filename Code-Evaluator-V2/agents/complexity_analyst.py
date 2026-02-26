"""Complexity Analyst Agent — estimates time/space complexity, flags inefficiency."""

from __future__ import annotations

import logging
from typing import List

from core.schema import AgentJudgment, ExecutionArtifact, RubricScores
from core.deterministic_engine import PROBLEM_META
from llm.model_router import query_models
from llm.prompt_templates import COMPLEXITY_ANALYST_SYSTEM, COMPLEXITY_ANALYST_USER

logger = logging.getLogger(__name__)


def run(
    problem_id: str,
    problem_description: str,
    submission_code: str,
    reference_code: str,
    det_scores: RubricScores,
    models: List[str],
    llm_config: dict,
) -> AgentJudgment:
    meta = PROBLEM_META.get(problem_id, {"complexity": "O(n)"})
    expected_cx = meta["complexity"]

    user_prompt = COMPLEXITY_ANALYST_USER.format(
        problem_description=problem_description,
        submission_code=submission_code,
        reference_code=reference_code,
        expected_complexity=expected_cx,
        det_complexity=det_scores.complexity,
    )

    parsed, model_used = query_models(
        models, COMPLEXITY_ANALYST_SYSTEM, user_prompt,
        max_tokens=llm_config.get("max_tokens", 2048),
        temperature=llm_config.get("temperature", 0.1),
        retries=llm_config.get("retry_attempts", 3),
    )

    if parsed and "scores" in parsed:
        s = parsed["scores"]
        return AgentJudgment(
            agent_name="complexity_analyst",
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

    logger.info("Complexity Analyst falling back to deterministic scores.")
    return AgentJudgment(
        agent_name="complexity_analyst",
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
