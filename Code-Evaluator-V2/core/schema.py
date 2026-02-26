"""Data models for Code Evaluator V2."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestResult:
    test_name: str
    passed: bool
    message: str = ""
    execution_time: float = 0.0


@dataclass
class TestSuiteResult:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: List[TestResult] = field(default_factory=list)
    pass_rate: float = 0.0


@dataclass
class StaticWarning:
    file: str
    line: int
    code: str
    message: str


@dataclass
class ExecutionArtifact:
    test_results: TestSuiteResult = field(default_factory=TestSuiteResult)
    static_warnings: List[StaticWarning] = field(default_factory=list)
    runtime_errors: str = ""
    execution_time: float = 0.0
    sandbox_violation: bool = False
    timeout: bool = False


@dataclass
class RubricScores:
    correctness: float = 0.0
    edge_cases: float = 0.0
    complexity: float = 0.0
    style: float = 0.0
    clarity: float = 0.0

    def overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or {
            "correctness": 0.35,
            "edge_cases": 0.20,
            "complexity": 0.15,
            "style": 0.15,
            "clarity": 0.15,
        }
        return (
            w["correctness"] * self.correctness
            + w["edge_cases"] * self.edge_cases
            + w["complexity"] * self.complexity
            + w["style"] * self.style
            + w["clarity"] * self.clarity
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "edge_cases": self.edge_cases,
            "complexity": self.complexity,
            "style": self.style,
            "clarity": self.clarity,
        }


@dataclass
class AgentJudgment:
    agent_name: str
    scores: RubricScores = field(default_factory=RubricScores)
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    fallback: bool = False


@dataclass
class ConsensusResult:
    scores: RubricScores = field(default_factory=RubricScores)
    agent_scores: Dict[str, RubricScores] = field(default_factory=dict)
    disagreements: List[str] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    reasoning: str = ""


@dataclass
class HallucinationFlag:
    dimension: str
    description: str
    severity: str  # "low", "medium", "high"


@dataclass
class EvaluationResult:
    problem_id: str
    submission_id: str
    deterministic_score: float = 0.0
    llm_adjusted_score: float = 0.0
    final_score: float = 0.0
    execution_artifact: ExecutionArtifact = field(default_factory=ExecutionArtifact)
    deterministic_rubric: RubricScores = field(default_factory=RubricScores)
    agent_judgments: List[AgentJudgment] = field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    hallucination_flags: List[HallucinationFlag] = field(default_factory=list)
    model_used: str = ""
    fallback_info: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "submission_id": self.submission_id,
            "deterministic_score": round(self.deterministic_score, 4),
            "llm_adjusted_score": round(self.llm_adjusted_score, 4),
            "final_score": round(self.final_score, 4),
            "execution_artifact": {
                "test_results": {
                    "total": self.execution_artifact.test_results.total,
                    "passed": self.execution_artifact.test_results.passed,
                    "failed": self.execution_artifact.test_results.failed,
                    "errors": self.execution_artifact.test_results.errors,
                    "results": [
                        {
                            "test_name": r.test_name,
                            "passed": r.passed,
                            "message": r.message,
                            "execution_time": r.execution_time,
                        }
                        for r in self.execution_artifact.test_results.results
                    ],
                    "pass_rate": self.execution_artifact.test_results.pass_rate,
                },
                "static_warnings": [
                    {
                        "file": w.file,
                        "line": w.line,
                        "code": w.code,
                        "message": w.message,
                    }
                    for w in self.execution_artifact.static_warnings
                ],
                "runtime_errors": self.execution_artifact.runtime_errors,
                "execution_time": self.execution_artifact.execution_time,
                "sandbox_violation": self.execution_artifact.sandbox_violation,
                "timeout": self.execution_artifact.timeout,
            },
            "deterministic_rubric": self.deterministic_rubric.to_dict(),
            "agent_judgments": [
                {
                    "agent_name": aj.agent_name,
                    "scores": aj.scores.to_dict(),
                    "reasoning": aj.reasoning,
                    "issues": aj.issues,
                    "suggestions": aj.suggestions,
                    "confidence": aj.confidence,
                    "model_used": aj.model_used,
                    "fallback": aj.fallback,
                }
                for aj in self.agent_judgments
            ],
            "consensus": {
                "scores": self.consensus.scores.to_dict(),
                "agent_scores": {
                    k: v.to_dict() for k, v in self.consensus.agent_scores.items()
                },
                "disagreements": self.consensus.disagreements,
                "confidence": self.consensus.confidence,
                "model_used": self.consensus.model_used,
                "reasoning": self.consensus.reasoning,
            }
            if self.consensus
            else None,
            "hallucination_flags": [
                {
                    "dimension": hf.dimension,
                    "description": hf.description,
                    "severity": hf.severity,
                }
                for hf in self.hallucination_flags
            ],
            "model_used": self.model_used,
            "fallback_info": self.fallback_info,
            "confidence": round(self.confidence, 4),
        }
