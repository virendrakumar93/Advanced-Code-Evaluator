#!/usr/bin/env python3
"""CLI entry point for Code Evaluator V2."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import evaluate_all, load_config
from core.schema import EvaluationResult

logger = logging.getLogger("code_evaluator_v2")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[EvaluationResult], output_dir: str) -> str:
    """Generate evaluation_report.md and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "evaluation_report.md")

    lines = ["# Evaluation Report — Code Evaluator V2\n"]
    lines.append(f"**Total submissions evaluated:** {len(results)}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Problem | Submission | Correctness | Edge Cases | Complexity | Style | Clarity | Det Score | LLM Score | Final Score | Model |")
    lines.append("|---------|-----------|-------------|------------|------------|-------|---------|-----------|-----------|-------------|-------|")
    for r in results:
        dr = r.deterministic_rubric
        lines.append(
            f"| {r.problem_id} | {r.submission_id} | {dr.correctness} | {dr.edge_cases} | "
            f"{dr.complexity} | {dr.style} | {dr.clarity} | "
            f"{r.deterministic_score:.4f} | {r.llm_adjusted_score:.4f} | **{r.final_score:.4f}** | "
            f"{r.model_used or 'deterministic'} |"
        )

    # Agent details
    lines.append("\n## Agent Details\n")
    for r in results:
        lines.append(f"### {r.problem_id} / {r.submission_id}\n")

        # Test results
        tr = r.execution_artifact.test_results
        lines.append(f"- **Tests:** {tr.passed}/{tr.total} passed ({tr.pass_rate:.0%})")
        lines.append(f"- **Static Warnings:** {len(r.execution_artifact.static_warnings)}")
        lines.append(f"- **Deterministic Score:** {r.deterministic_score:.4f}")
        lines.append(f"- **Final Score:** {r.final_score:.4f}")

        if r.agent_judgments:
            lines.append("\n**Agent Scores:**\n")
            lines.append("| Agent | Correctness | Edge Cases | Complexity | Style | Clarity | Confidence | Fallback |")
            lines.append("|-------|-------------|------------|------------|-------|---------|------------|----------|")
            for aj in r.agent_judgments:
                s = aj.scores
                lines.append(
                    f"| {aj.agent_name} | {s.correctness} | {s.edge_cases} | "
                    f"{s.complexity} | {s.style} | {s.clarity} | "
                    f"{aj.confidence:.2f} | {'Yes' if aj.fallback else 'No'} |"
                )

        if r.consensus:
            cs = r.consensus.scores
            lines.append(f"\n**Consensus Score:** correctness={cs.correctness}, "
                         f"edge_cases={cs.edge_cases}, complexity={cs.complexity}, "
                         f"style={cs.style}, clarity={cs.clarity} "
                         f"(confidence={r.consensus.confidence:.2f})")
            if r.consensus.disagreements:
                lines.append(f"\n**Disagreements:** {', '.join(r.consensus.disagreements)}")
            if r.consensus.reasoning:
                lines.append(f"\n**Consensus Reasoning:** {r.consensus.reasoning[:300]}")

        if r.hallucination_flags:
            lines.append("\n**Hallucination Flags:**")
            for hf in r.hallucination_flags:
                lines.append(f"- [{hf.severity.upper()}] {hf.dimension}: {hf.description}")

        if r.fallback_info:
            lines.append(f"\n**Fallback Info:** {r.fallback_info}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def save_results_json(results: List[EvaluationResult], output_dir: str) -> str:
    """Save results.json and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Code Evaluator V2 — Multi-Agent Evaluation Framework",
    )
    parser.add_argument(
        "--evaluate", type=str, default=None,
        help="What to evaluate: 'all' or a specific problem_id (e.g. problem_1)",
    )
    parser.add_argument(
        "--submission", type=str, default=None,
        help="Specific submission to evaluate (requires --evaluate <problem_id>)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM evaluation, use deterministic scoring only",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Directory for output reports (default: reports)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.evaluate:
        parser.print_help()
        sys.exit(0)

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    config = load_config(config_path)

    # Resolve relative paths in config against V2 root
    v2_root = os.path.dirname(os.path.abspath(__file__))
    problems_path = config.get("paths", {}).get("problems", "problems")
    if not os.path.isabs(problems_path):
        config.setdefault("paths", {})["problems"] = os.path.join(v2_root, problems_path)

    # Run evaluation
    results = evaluate_all(
        config,
        problem_filter=args.evaluate,
        submission_filter=args.submission,
        no_llm=args.no_llm,
        verbose=args.verbose,
    )

    if not results:
        print("No submissions found to evaluate.")
        sys.exit(1)

    # Output
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(v2_root, output_dir)

    report_path = generate_report(results, output_dir)
    json_path = save_results_json(results, output_dir)

    print(f"\nEvaluation complete: {len(results)} submission(s)")
    print(f"  Report: {report_path}")
    print(f"  JSON:   {json_path}")

    # Print summary
    print(f"\n{'Problem':<12} {'Submission':<20} {'Det':>6} {'LLM':>6} {'Final':>7}")
    print("-" * 55)
    for r in results:
        print(f"{r.problem_id:<12} {r.submission_id:<20} {r.deterministic_score:>6.2f} "
              f"{r.llm_adjusted_score:>6.2f} {r.final_score:>7.2f}")


if __name__ == "__main__":
    main()
