"""Score candidate code against a problem's tests and classify the outcome.

Verdicts:
  passed        ran, all tests ok
  syntax_error  candidate fails to compile (no sandbox run)
  runtime_error  crashed: exception other than AssertionError, or memory/OS error
  wrong_answer  ran but a test assertion failed
  timeout       hit the sandbox wall-clock limit
  no_code       nothing executable was available
failure_type: "execution" for syntax_error/runtime_error/timeout/no_code,
  "correctness" for wrong_answer, None when passed.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from .config import ExperimentCfg
from .dataset import Problem
from .sandbox import SandboxResult, execute_code


def make_test_script(problem: Problem, code: str, marker: str) -> str:
    """Candidate code + its test file, wired to run in one process.

    HumanEval+ test scripts define check(candidate); we bind candidate to
    the problem's entry point. MBPP+-style scripts contain bare asserts and
    are appended as-is.
    """
    base = code.rstrip("\n")
    test = problem.test_script or ""
    if "def check(candidate)" in test:
        wiring = (f"\n\ncandidate = {problem.entry_point}\n"
                   f"check(candidate)\nprint({marker!r})")
        return base + "\n\n" + test.rstrip("\n") + "\n" + wiring
    if not test.strip():
        return base
    return base + "\n\n" + test + f"\nprint({marker!r})"


@dataclass
class ScoreResult:
    verdict: str
    failure_type: str | None
    test_error: str = ""
    sandbox: SandboxResult | None = field(default=None)


def _classify(sb: SandboxResult) -> tuple[str, str, str]:
    if sb.timed_out:
        return "timeout", "execution", sb.stderr or "wall-clock timeout"
    if sb.exit_code == 0:
        return "passed", None, ""
    tail = sb.stderr[-2000:]
    if "AssertionError" in sb.stderr:
        return "wrong_answer", "correctness", tail
    return "runtime_error", "execution", tail


def score(problem: Problem, code: str, cfg: ExperimentCfg) -> ScoreResult:
    if code is None or not code.strip():
        return ScoreResult("no_code", "execution", "no code text available")
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        return ScoreResult("syntax_error", "execution",
                            f"line {e.lineno}: {e.msg}")
    marker = f"PASSMARK-{secrets.token_hex(8).upper()}"
    sb = execute_code(make_test_script(problem, code, marker), cfg,
                      label="run")
    if sb.exit_code == 0 and marker in sb.stdout:
        return ScoreResult("passed", None, "", sb)
    verdict, ftype, err = _classify(sb)
    if verdict == "passed":
        # exited 0 but the nonce marker is missing: treat as a harness anomaly
        verdict, ftype, err = "runtime_error", "execution", \
            "process exited 0 but test marker missing"
    return ScoreResult(verdict, ftype, err, sb)
