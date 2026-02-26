"""Consensus Agent — merges scores from specialist agents, resolves disagreements."""

from __future__ import annotations

import logging
from typing import Dict, List

from core.schema import AgentJudgment, ConsensusResult, RubricScores
from llm.model_router import query_models
from llm.prompt_templates import CONSENSUS_SYSTEM, CONSENSUS_USER

logger = logging.getLogger(__name__)

DIMENSIONS = ["correctness", "edge_cases", "complexity", "style", "clarity"]


def _avg(*values: float) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else 5.0


def _detect_disagreements(
    judgments: List[AgentJudgment], threshold: float = 2.0,
) -> List[str]:
    flags = []
    for dim in DIMENSIONS:
        vals = {j.agent_name: getattr(j.scores, dim) for j in judgments}
        scores = list(vals.values())
        if max(scores) - min(scores) > threshold:
            parts = [f"{name}={v}" for name, v in vals.items()]
            flags.append(f"{dim}: {', '.join(parts)}")
    return flags


def run_deterministic_consensus(
    det_scores: RubricScores,
    judgments: List[AgentJudgment],
) -> ConsensusResult:
    """Weighted average consensus without LLM (used as fallback or when all agents fell back)."""
    all_agents_fell_back = all(j.fallback for j in judgments)
    if all_agents_fell_back or not judgments:
        return ConsensusResult(
            scores=det_scores,
            agent_scores={j.agent_name: j.scores for j in judgments},
            disagreements=[],
            confidence=0.3 if judgments else 1.0,
            model_used="",
            reasoning="All agents used deterministic fallback — consensus equals deterministic scores.",
        )

    # Confidence-weighted average per dimension
    merged = {}
    for dim in DIMENSIONS:
        total_weight = 0.0
        weighted_sum = 0.0
        for j in judgments:
            w = j.confidence if not j.fallback else 0.1
            weighted_sum += getattr(j.scores, dim) * w
            total_weight += w
        merged[dim] = round(weighted_sum / total_weight, 2) if total_weight > 0 else getattr(det_scores, dim)

    return ConsensusResult(
        scores=RubricScores(**merged),
        agent_scores={j.agent_name: j.scores for j in judgments},
        disagreements=_detect_disagreements(judgments),
        confidence=round(sum(j.confidence for j in judgments if not j.fallback) / max(1, sum(1 for j in judgments if not j.fallback)), 2),
        model_used="",
        reasoning="Deterministic consensus via confidence-weighted averaging.",
    )


def run(
    det_scores: RubricScores,
    judgments: List[AgentJudgment],
    models: List[str],
    llm_config: dict,
) -> ConsensusResult:
    """Run the Consensus Agent. Falls back to deterministic if LLM unavailable."""
    all_agents_fell_back = all(j.fallback for j in judgments)
    if all_agents_fell_back or not judgments:
        return run_deterministic_consensus(det_scores, judgments)

    # Build prompt for LLM-based consensus
    td = _find(judgments, "test_designer", det_scores)
    cr = _find(judgments, "code_reviewer", det_scores)
    ca = _find(judgments, "complexity_analyst", det_scores)

    user_prompt = CONSENSUS_USER.format(
        det_correctness=det_scores.correctness,
        det_edge_cases=det_scores.edge_cases,
        det_complexity=det_scores.complexity,
        det_style=det_scores.style,
        det_clarity=det_scores.clarity,
        td_correctness=td.scores.correctness, td_edge_cases=td.scores.edge_cases,
        td_complexity=td.scores.complexity, td_style=td.scores.style,
        td_clarity=td.scores.clarity, td_confidence=td.confidence,
        td_reasoning=td.reasoning[:300],
        cr_correctness=cr.scores.correctness, cr_edge_cases=cr.scores.edge_cases,
        cr_complexity=cr.scores.complexity, cr_style=cr.scores.style,
        cr_clarity=cr.scores.clarity, cr_confidence=cr.confidence,
        cr_reasoning=cr.reasoning[:300],
        ca_correctness=ca.scores.correctness, ca_edge_cases=ca.scores.edge_cases,
        ca_complexity=ca.scores.complexity, ca_style=ca.scores.style,
        ca_clarity=ca.scores.clarity, ca_confidence=ca.confidence,
        ca_reasoning=ca.reasoning[:300],
    )

    parsed, model_used = query_models(
        models, CONSENSUS_SYSTEM, user_prompt,
        max_tokens=llm_config.get("max_tokens", 2048),
        temperature=llm_config.get("temperature", 0.1),
        retries=llm_config.get("retry_attempts", 3),
    )

    if parsed and "scores" in parsed:
        s = parsed["scores"]
        return ConsensusResult(
            scores=RubricScores(
                correctness=_clamp(s.get("correctness", det_scores.correctness)),
                edge_cases=_clamp(s.get("edge_cases", det_scores.edge_cases)),
                complexity=_clamp(s.get("complexity", det_scores.complexity)),
                style=_clamp(s.get("style", det_scores.style)),
                clarity=_clamp(s.get("clarity", det_scores.clarity)),
            ),
            agent_scores={j.agent_name: j.scores for j in judgments},
            disagreements=parsed.get("disagreements", []) or _detect_disagreements(judgments),
            confidence=parsed.get("confidence", 0.5),
            model_used=model_used,
            reasoning=parsed.get("reasoning", ""),
        )

    logger.info("Consensus Agent falling back to deterministic consensus.")
    return run_deterministic_consensus(det_scores, judgments)


def _find(judgments: List[AgentJudgment], name: str, fallback: RubricScores) -> AgentJudgment:
    for j in judgments:
        if j.agent_name == name:
            return j
    return AgentJudgment(agent_name=name, scores=fallback, confidence=0.3, fallback=True)


def _clamp(v, lo=0.0, hi=10.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 5.0
