"""Test Designer Agent — suggests additional edge-case tests for submissions."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from core.schema import AgentJudgment, ExecutionArtifact, RubricScores
from llm.model_router import query_models
from llm.prompt_templates import TEST_DESIGNER_SYSTEM, TEST_DESIGNER_USER

logger = logging.getLogger(__name__)


def _build_failed_details(artifact: ExecutionArtifact) -> str:
    failed = [r for r in artifact.test_results.results if not r.passed]
    if not failed:
        return "None — all tests passed."
    lines = []
    for r in failed[:5]:
        msg = r.message[:200] if r.message else "no details"
        lines.append(f"- {r.test_name}: {msg}")
    return "\n".join(lines)


def run(
    problem_description: str,
    submission_code: str,
    reference_code: str,
    artifact: ExecutionArtifact,
    det_scores: RubricScores,
    models: List[str],
    llm_config: dict,
) -> AgentJudgment:
    """Run the Test Designer agent. Returns AgentJudgment."""
    tr = artifact.test_results
    user_prompt = TEST_DESIGNER_USER.format(
        problem_description=problem_description,
        submission_code=submission_code,
        reference_code=reference_code,
        total_tests=tr.total,
        passed_tests=tr.passed,
        failed_tests=tr.failed + tr.errors,
        pass_rate=tr.pass_rate,
        failed_details=_build_failed_details(artifact),
    )

    parsed, model_used = query_models(
        models, TEST_DESIGNER_SYSTEM, user_prompt,
        max_tokens=llm_config.get("max_tokens", 2048),
        temperature=llm_config.get("temperature", 0.1),
        retries=llm_config.get("retry_attempts", 3),
    )

    if parsed and "scores" in parsed:
        s = parsed["scores"]
        return AgentJudgment(
            agent_name="test_designer",
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

    # Deterministic fallback
    logger.info("Test Designer falling back to deterministic scores.")
    return AgentJudgment(
        agent_name="test_designer",
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
