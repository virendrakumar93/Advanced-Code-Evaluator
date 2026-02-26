"""Hallucination Auditor — detects LLM claims inconsistent with deterministic evidence."""

from __future__ import annotations

from typing import List

from core.schema import (
    ConsensusResult,
    ExecutionArtifact,
    HallucinationFlag,
    RubricScores,
)


def audit(
    det_scores: RubricScores,
    consensus: ConsensusResult,
    artifact: ExecutionArtifact,
) -> List[HallucinationFlag]:
    """Compare consensus scores against deterministic evidence. Return flags."""
    flags: List[HallucinationFlag] = []
    cs = consensus.scores
    tr = artifact.test_results

    # 1. High correctness but low test pass rate
    if cs.correctness > 7 and tr.pass_rate < 0.7:
        flags.append(HallucinationFlag(
            dimension="correctness",
            description=f"LLM correctness={cs.correctness} but test pass rate={tr.pass_rate:.0%}",
            severity="high",
        ))

    # 2. Low correctness but high test pass rate
    if cs.correctness < 3 and tr.pass_rate > 0.9:
        flags.append(HallucinationFlag(
            dimension="correctness",
            description=f"LLM correctness={cs.correctness} but tests pass at {tr.pass_rate:.0%}",
            severity="high",
        ))

    # 3. High edge-case score but low pass rate
    if cs.edge_cases > 8 and tr.pass_rate < 0.8:
        flags.append(HallucinationFlag(
            dimension="edge_cases",
            description=f"LLM edge_cases={cs.edge_cases} but pass rate={tr.pass_rate:.0%}",
            severity="medium",
        ))

    # 4. High style but many static warnings
    warning_count = len(artifact.static_warnings)
    if cs.style > 8 and warning_count > 5:
        flags.append(HallucinationFlag(
            dimension="style",
            description=f"LLM style={cs.style} but {warning_count} static warnings found",
            severity="medium",
        ))

    # 5. Timeout/sandbox violation yet positive scores
    if artifact.timeout and cs.correctness > 3:
        flags.append(HallucinationFlag(
            dimension="correctness",
            description=f"Submission timed out but LLM correctness={cs.correctness}",
            severity="high",
        ))
    if artifact.sandbox_violation and cs.correctness > 0:
        flags.append(HallucinationFlag(
            dimension="correctness",
            description=f"Sandbox violation but LLM correctness={cs.correctness}",
            severity="high",
        ))

    # 6. Large deviation from deterministic (> 3 points)
    for dim in ("correctness", "edge_cases", "complexity", "style", "clarity"):
        det_val = getattr(det_scores, dim)
        llm_val = getattr(cs, dim)
        if abs(det_val - llm_val) > 3:
            flags.append(HallucinationFlag(
                dimension=dim,
                description=f"{dim}: deterministic={det_val}, consensus={llm_val} (diff={abs(det_val-llm_val):.1f})",
                severity="medium",
            ))

    return flags
