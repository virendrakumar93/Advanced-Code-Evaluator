"""Deterministic evaluation engine: sandbox, test runner, static analysis, rubric scoring."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List

from core.schema import (
    ExecutionArtifact,
    RubricScores,
    StaticWarning,
    TestResult,
    TestSuiteResult,
)

# ---------------------------------------------------------------------------
# Sandbox constants
# ---------------------------------------------------------------------------
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "ftplib", "smtplib", "ctypes", "importlib", "code",
    "codeop", "compile", "compileall", "py_compile", "zipimport",
    "pkgutil", "multiprocessing", "threading", "signal", "resource",
    "webbrowser", "antigravity", "turtle", "tkinter", "pathlib",
}
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "breakpoint", "exit", "quit",
}
SAFE_IMPORTS = {
    "math", "collections", "itertools", "functools", "heapq",
    "bisect", "string", "re", "typing", "dataclasses", "enum",
    "copy", "operator", "statistics",
}

# Problem → expected function name / complexity
PROBLEM_META = {
    "problem_1": {"function": "two_sum", "complexity": "O(n)"},
    "problem_2": {"function": "generate_parentheses", "complexity": "O(4^n/sqrt(n))"},
    "problem_3": {"function": "climb_stairs", "complexity": "O(n)"},
    "problem_4": {"function": "evaluate_expression", "complexity": "O(n)"},
    "problem_5": {"function": "shortest_path", "complexity": "O(V+E)"},
}

# ---------------------------------------------------------------------------
# Code safety
# ---------------------------------------------------------------------------

def check_code_safety(code: str) -> List[str]:
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod not in SAFE_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod not in SAFE_IMPORTS:
                    violations.append(f"Blocked import from: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                violations.append(f"Blocked builtin call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in (
                "system", "popen", "exec", "spawn",
            ):
                violations.append(f"Blocked method call: {node.func.attr}")
    return violations


def strip_comments(code: str) -> str:
    lines = code.split("\n")
    cleaned = []
    in_docstring = False
    docstring_char = None
    for line in lines:
        stripped = line.strip()
        if in_docstring:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            docstring_char = stripped[:3]
            if stripped.count(docstring_char) >= 2:
                continue
            in_docstring = True
            continue
        if "#" in line:
            in_quote = False
            quote_char = None
            comment_pos = -1
            for i, ch in enumerate(line):
                if ch in ('"', "'") and not in_quote:
                    in_quote = True
                    quote_char = ch
                elif ch == quote_char and in_quote:
                    in_quote = False
                elif ch == "#" and not in_quote:
                    comment_pos = i
                    break
            if comment_pos >= 0:
                line = line[:comment_pos].rstrip()
                if not line.strip():
                    continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

def run_in_sandbox(code: str, test_code: str, function_name: str, timeout: float = 3.0) -> dict:
    violations = check_code_safety(code)
    if violations:
        return {
            "stdout": "", "stderr": f"Sandbox violations: {'; '.join(violations)}",
            "returncode": -1, "execution_time": 0.0, "timeout": False,
            "sandbox_violation": True, "violations": violations,
        }

    runner_script = textwrap.dedent(f"""\
        import sys, time, unittest, io, json
        _safe_exec = exec
        _safe_compile = compile
        _safe_open = open
        submission_code = _safe_open(sys.argv[1]).read()
        test_code = _safe_open(sys.argv[2]).read()
        exec_globals = {{}}
        _safe_exec(_safe_compile(submission_code, "submission.py", "exec"), exec_globals)
        test_globals = {{"__name__": "__test_runner__", "unittest": unittest}}
        for k, v in exec_globals.items():
            if not k.startswith("_"):
                test_globals[k] = v
        func = exec_globals.get("{function_name}")
        if func:
            test_globals["{function_name}"] = func
        _safe_exec(_safe_compile(test_code, "tests.py", "exec"), test_globals)
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for name, obj in test_globals.items():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
                for k, v in exec_globals.items():
                    if not k.startswith("_"):
                        setattr(obj, k, staticmethod(v) if callable(v) else v)
                suite.addTests(loader.loadTestsFromTestCase(obj))
        start = time.time()
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        elapsed = time.time() - start
        test_results = []
        for test, tb in result.failures:
            test_results.append({{"name": str(test), "status": "FAIL", "message": tb}})
        for test, tb in result.errors:
            test_results.append({{"name": str(test), "status": "ERROR", "message": tb}})
        failed_tests = set()
        for test, _ in result.failures + result.errors:
            failed_tests.add(str(test))
        for suite_item in suite:
            tname = str(suite_item)
            if tname not in failed_tests:
                test_results.append({{"name": tname, "status": "PASS", "message": ""}})
        output = {{
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "elapsed": round(elapsed, 4),
            "details": test_results,
        }}
        print("===JSON_RESULT===")
        print(json.dumps(output))
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        sub_path = os.path.join(tmpdir, "submission.py")
        test_path = os.path.join(tmpdir, "tests.py")
        runner_path = os.path.join(tmpdir, "runner.py")
        with open(sub_path, "w") as f:
            f.write(code)
        with open(test_path, "w") as f:
            f.write(test_code)
        with open(runner_path, "w") as f:
            f.write(runner_script)
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, runner_path, sub_path, test_path],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir, env=env,
            )
            return {
                "stdout": proc.stdout, "stderr": proc.stderr,
                "returncode": proc.returncode, "execution_time": 0.0,
                "timeout": False, "sandbox_violation": False, "violations": [],
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "", "stderr": f"Execution timed out after {timeout}s",
                "returncode": -1, "execution_time": timeout,
                "timeout": True, "sandbox_violation": False, "violations": [],
            }
        except Exception as e:
            return {
                "stdout": "", "stderr": str(e), "returncode": -1,
                "execution_time": 0.0, "timeout": False, "sandbox_violation": False,
                "violations": [],
            }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def parse_sandbox_output(sandbox_result: dict) -> TestSuiteResult:
    stdout = sandbox_result.get("stdout", "")
    if "===JSON_RESULT===" in stdout:
        json_str = stdout.split("===JSON_RESULT===")[1].strip()
        try:
            data = json.loads(json_str)
            results = []
            for d in data.get("details", []):
                results.append(TestResult(
                    test_name=d.get("name", ""),
                    passed=d.get("status") == "PASS",
                    message=d.get("message", ""),
                    execution_time=data.get("elapsed", 0.0),
                ))
            total = data.get("total", 0)
            passed = data.get("passed", 0)
            return TestSuiteResult(
                total=total, passed=passed,
                failed=data.get("failed", 0), errors=data.get("errors", 0),
                results=results, pass_rate=passed / total if total > 0 else 0.0,
            )
        except json.JSONDecodeError:
            pass
    return TestSuiteResult()


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

def run_ruff(code: str) -> List[StaticWarning]:
    warnings = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--output-format", "json",
             "--select", "E,W,F,C,N,B", tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            try:
                issues = json.loads(result.stdout)
                for issue in issues:
                    loc = issue.get("location", {})
                    warnings.append(StaticWarning(
                        file=tmp_path, line=loc.get("row", 0),
                        code=issue.get("code", "unknown"),
                        message=issue.get("message", ""),
                    ))
            except json.JSONDecodeError:
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    finally:
        os.unlink(tmp_path)
    return warnings


def style_score_from_warnings(warnings: List[StaticWarning], code_lines: int) -> float:
    if code_lines == 0:
        return 5.0
    warning_count = len(warnings)
    density = warning_count / max(code_lines, 1)
    if density == 0:
        return 10.0
    elif density < 0.05:
        return 9.0
    elif density < 0.1:
        return 8.0
    elif density < 0.2:
        return 6.0
    elif density < 0.3:
        return 4.0
    elif density < 0.5:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------

def compute_correctness(test_result: TestSuiteResult) -> float:
    return round(test_result.pass_rate * 10, 2)


def compute_edge_cases(test_result: TestSuiteResult) -> float:
    if test_result.total == 0:
        return 0.0
    pr = test_result.pass_rate
    if pr >= 1.0:
        return 10.0
    elif pr >= 0.8:
        return 7.0 + (pr - 0.8) * 15
    elif pr >= 0.5:
        return 4.0 + (pr - 0.5) * 10
    return pr * 8


def _count_max_loop_nesting(tree: ast.AST) -> int:
    max_depth = 0
    def walk(node, depth=0):
        nonlocal max_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                nd = depth + 1
                max_depth = max(max_depth, nd)
                walk(child, nd)
            else:
                walk(child, depth)
    walk(tree)
    return max_depth


def _has_recursion(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    func_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.add(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in func_names:
                return True
    return False


def compute_complexity(code: str, expected: str = "O(n)") -> float:
    score = 10.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 3.0
    depth = _count_max_loop_nesting(tree)
    if "O(n)" in expected or "O(V+E)" in expected:
        if depth >= 3:
            score = 2.0
        elif depth >= 2:
            score = 5.0
        elif depth == 1:
            score = 9.0
    elif "O(n^2)" in expected:
        if depth >= 3:
            score = 4.0
        elif depth >= 2:
            score = 8.0
    has_rec = _has_recursion(code)
    has_memo = any(k in code for k in ("memo", "cache", "lru_cache", "@cache"))
    if has_rec and not has_memo and "O(n)" in expected:
        score = min(score, 6.0)
    return round(score, 2)


def compute_style(code: str, warnings: List[StaticWarning]) -> float:
    lines = code.strip().split("\n")
    num_lines = len(lines)
    base = style_score_from_warnings(warnings, num_lines)
    deductions = 0.0
    acceptable = {"i", "j", "k", "n", "x", "y", "v", "e", "f", "s", "t", "q", "_"}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and len(node.id) == 1 and node.id not in acceptable:
                deductions += 0.3
    except SyntaxError:
        deductions += 2.0
    long_lines = sum(1 for l in lines if len(l) > 100)
    if long_lines > 0:
        deductions += min(long_lines * 0.3, 2.0)
    return round(max(0.0, min(10.0, base - deductions)), 2)


def compute_clarity(code: str) -> float:
    score = 8.0
    lines = code.strip().split("\n")
    num_lines = len(lines)
    if num_lines > 50:
        score -= 2.0
    elif num_lines > 30:
        score -= 1.0
    try:
        tree = ast.parse(code)
        fc, short = 0, 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fc += 1
                if len(node.name) <= 2:
                    short += 1
        if fc > 0 and short / fc > 0.5:
            score -= 2.0
    except SyntaxError:
        score -= 3.0
    if '"""' in code or "'''" in code:
        score += 1.0
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    if comment_lines > 0 and num_lines > 10:
        score += 0.5
    return round(max(0.0, min(10.0, score)), 2)


def compute_deterministic_scores(
    code: str,
    test_result: TestSuiteResult,
    warnings: List[StaticWarning],
    expected_complexity: str = "O(n)",
) -> RubricScores:
    return RubricScores(
        correctness=compute_correctness(test_result),
        edge_cases=compute_edge_cases(test_result),
        complexity=compute_complexity(code, expected_complexity),
        style=compute_style(code, warnings),
        clarity=compute_clarity(code),
    )


# ---------------------------------------------------------------------------
# Full deterministic evaluation of a single submission
# ---------------------------------------------------------------------------

def evaluate_submission(
    problem_id: str,
    submission_code: str,
    test_code: str,
    timeout: float = 3.0,
) -> tuple[ExecutionArtifact, RubricScores]:
    meta = PROBLEM_META.get(problem_id, {"function": "solution", "complexity": "O(n)"})
    func_name = meta["function"]
    expected_cx = meta["complexity"]

    sandbox_result = run_in_sandbox(submission_code, test_code, func_name, timeout)
    test_result = parse_sandbox_output(sandbox_result)
    warnings = run_ruff(submission_code)

    artifact = ExecutionArtifact(
        test_results=test_result,
        static_warnings=warnings,
        runtime_errors=sandbox_result.get("stderr", ""),
        execution_time=sandbox_result.get("execution_time", 0.0),
        sandbox_violation=sandbox_result.get("sandbox_violation", False),
        timeout=sandbox_result.get("timeout", False),
    )
    scores = compute_deterministic_scores(submission_code, test_result, warnings, expected_cx)
    return artifact, scores
