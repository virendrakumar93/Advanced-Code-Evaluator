"""Orchestrator — coordinates the full evaluation pipeline for V2.

Pipeline:
  CLI → Orchestrator → Deterministic Engine → LLM Agents → Consensus → Hallucination Auditor → Score Blending → Reports
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.schema import (
    AgentJudgment,
    ConsensusResult,
    EvaluationResult,
    ExecutionArtifact,
    HallucinationFlag,
    RubricScores,
)
from core.deterministic_engine import (
    PROBLEM_META,
    evaluate_submission as det_evaluate,
    strip_comments,
)
from agents import test_designer, code_reviewer, complexity_analyst
from agents.consensus_agent import run as consensus_run
from safety.hallucination_auditor import audit as hallucination_audit

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_problems(problems_dir: str) -> Dict[str, Path]:
    """Return {problem_id: Path} for all problems."""
    base = Path(problems_dir)
    problems = {}
    if not base.exists():
        logger.error("Problems directory not found: %s", problems_dir)
        return problems
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("problem_"):
            problems[d.name] = d
    return problems


def discover_submissions(problem_dir: Path) -> Dict[str, Path]:
    """Return {submission_id: Path} for all submissions under a problem."""
    sub_dir = problem_dir / "submissions"
    subs = {}
    if not sub_dir.exists():
        return subs
    for f in sorted(sub_dir.iterdir()):
        if f.suffix == ".py":
            subs[f.stem] = f
    return subs


def evaluate_single(
    problem_id: str,
    submission_id: str,
    problem_dir: Path,
    submission_path: Path,
    config: dict,
    no_llm: bool = False,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate a single submission through the full pipeline."""
    logger.info("Evaluating %s / %s", problem_id, submission_id)

    # Read files
    submission_code = submission_path.read_text()
    test_path = problem_dir / "tests.py"
    test_code = test_path.read_text() if test_path.exists() else ""
    desc_path = problem_dir / "description.md"
    problem_description = desc_path.read_text() if desc_path.exists() else ""
    ref_path = problem_dir / "reference.py"
    reference_code = ref_path.read_text() if ref_path.exists() else ""

    sandbox_timeout = config.get("sandbox", {}).get("timeout", 3.0)

    # ── Step 1: Deterministic evaluation ──
    artifact, det_scores = det_evaluate(problem_id, submission_code, test_code, sandbox_timeout)
    det_overall = det_scores.overall(config.get("scoring", {}).get("weights"))
    logger.info("  Deterministic score: %.2f", det_overall)

    result = EvaluationResult(
        problem_id=problem_id,
        submission_id=submission_id,
        deterministic_score=det_overall,
        execution_artifact=artifact,
        deterministic_rubric=det_scores,
    )

    # ── Step 2: LLM agent layer (skip if --no-llm) ──
    llm_enabled = config.get("llm", {}).get("enabled", True) and not no_llm
    models = config.get("models", [])
    llm_config = config.get("llm", {})

    if not llm_enabled or not models:
        result.llm_adjusted_score = det_overall
        result.final_score = det_overall
        result.fallback_info = "LLM disabled" if not llm_enabled else "No models configured"
        result.confidence = 1.0
        logger.info("  LLM disabled — final score = deterministic: %.2f", det_overall)
        return result

    sanitized_code = strip_comments(submission_code)

    # Run agents
    judgments: List[AgentJudgment] = []

    td_judgment = test_designer.run(
        problem_description, sanitized_code, reference_code,
        artifact, det_scores, models, llm_config,
    )
    judgments.append(td_judgment)
    logger.info("  Test Designer: confidence=%.2f, fallback=%s", td_judgment.confidence, td_judgment.fallback)

    cr_judgment = code_reviewer.run(
        problem_description, sanitized_code, artifact, det_scores, models, llm_config,
    )
    judgments.append(cr_judgment)
    logger.info("  Code Reviewer: confidence=%.2f, fallback=%s", cr_judgment.confidence, cr_judgment.fallback)

    ca_judgment = complexity_analyst.run(
        problem_id, problem_description, sanitized_code, reference_code,
        det_scores, models, llm_config,
    )
    judgments.append(ca_judgment)
    logger.info("  Complexity Analyst: confidence=%.2f, fallback=%s", ca_judgment.confidence, ca_judgment.fallback)

    result.agent_judgments = judgments

    # ── Step 3: Consensus ──
    consensus = consensus_run(det_scores, judgments, models, llm_config)
    result.consensus = consensus
    result.model_used = consensus.model_used or _first_model_used(judgments)
    logger.info("  Consensus confidence=%.2f", consensus.confidence)

    # ── Step 4: Hallucination audit ──
    h_flags = hallucination_audit(det_scores, consensus, artifact)
    result.hallucination_flags = h_flags
    if h_flags:
        logger.warning("  Hallucination flags: %d", len(h_flags))

    # ── Step 5: Score blending ──
    scoring_cfg = config.get("scoring", {})
    llm_weight = scoring_cfg.get("llm_weight", 0.4)
    weights = scoring_cfg.get("weights")

    consensus_overall = consensus.scores.overall(weights)

    # Reduce LLM weight if hallucination flags triggered
    if h_flags:
        reduction = scoring_cfg.get("hallucination_llm_weight_reduction", 0.5)
        llm_weight *= reduction
        result.fallback_info = f"LLM weight reduced from {scoring_cfg.get('llm_weight', 0.4)} to {llm_weight:.2f} due to {len(h_flags)} hallucination flag(s)"

    # Only blend if consensus has reasonable confidence
    if consensus.confidence >= 0.4 and not all(j.fallback for j in judgments):
        blended = det_overall * (1 - llm_weight) + consensus_overall * llm_weight
        result.llm_adjusted_score = round(consensus_overall, 4)
        result.final_score = round(blended, 4)
        result.confidence = round(consensus.confidence, 4)
    else:
        result.llm_adjusted_score = det_overall
        result.final_score = det_overall
        result.confidence = 1.0
        if not result.fallback_info:
            result.fallback_info = "Low consensus confidence — using deterministic only"

    logger.info("  Final score: %.4f (det=%.4f, llm=%.4f)", result.final_score, det_overall, result.llm_adjusted_score)
    return result


def evaluate_all(
    config: dict,
    problem_filter: Optional[str] = None,
    submission_filter: Optional[str] = None,
    no_llm: bool = False,
    verbose: bool = False,
) -> List[EvaluationResult]:
    """Evaluate all (or filtered) submissions."""
    problems_dir = config.get("paths", {}).get("problems", "problems")
    problems = discover_problems(problems_dir)

    if problem_filter and problem_filter != "all":
        problems = {k: v for k, v in problems.items() if k == problem_filter}

    results = []
    for pid, pdir in problems.items():
        subs = discover_submissions(pdir)
        if submission_filter:
            subs = {k: v for k, v in subs.items() if k == submission_filter}
        for sid, spath in subs.items():
            r = evaluate_single(pid, sid, pdir, spath, config, no_llm, verbose)
            results.append(r)
    return results


def _first_model_used(judgments: List[AgentJudgment]) -> str:
    for j in judgments:
        if j.model_used:
            return j.model_used
    return ""
