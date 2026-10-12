"""Deterministic offline fakes so the whole pipeline runs without network.

MockCloudClient: for a task it "writes" the dataset's canonical solution
inside a well-formed two-part message (plausible plan + full code).

MockDeviceClient:
  - solve mode (no received message): solves correctly only for tasks whose
    numeric id suffix is even (~50%), raising otherwise -> the pre-screen
    filter has work to do, as with a real weaker device model;
  - recovery mode: returns the received code if it is complete and
    compiles; otherwise it "reconstructs" the canonical solution.
"""
from __future__ import annotations

from ..config import ExperimentCfg
from .cloud_client import CloudResult
from .device_client import DeviceResult

_PROBLEMS: dict = {}


def _problems(cfg: ExperimentCfg) -> list:
    if not _PROBLEMS.get(id(cfg)):
        from ..dataset import load_problems
        _PROBLEMS[id(cfg)] = load_problems(cfg)
    return _PROBLEMS[id(cfg)]


def _find_problem(cfg: ExperimentCfg, user: str):
    for p in _problems(cfg):
        if p.prompt and p.prompt in user:
            return p
    return None


class MockCloudClient:
    def __init__(self, cfg: ExperimentCfg):
        self._cfg = cfg

    def complete(self, system: str, user: str, temperature: float,
                 seed: int | None, max_tokens: int) -> CloudResult:
        p = _find_problem(self._cfg, user)
        if p is None:  # unknown task: emit a minimal well-formed message
            code = "def solution():\n    return None\n"
        else:
            code = (p.prompt + p.canonical_solution).strip("\n") + "\n"
        plan = (f"Plan for {p.entry_point if p else 'the task'}: "
                 "read the spec, enumerate the edge cases (empty inputs, "
                 "boundary values, duplicate elements), choose the direct "
                 "algorithm, implement it without helper modules, then "
                 "self-check against the given examples. "
                 "I will keep the code compact and complete.")
        from ..message_format import build
        order = ("code_first" if "<action>" in system and
                 system.index("<action>") < system.index("<thought>")
                 else "reasoning_first")
        text = build(plan, code, order)
        return CloudResult(text=text, prompt_tokens=len(user) // 4,
                            completion_tokens=len(text) // 4,
                            latency_s=0.01, model=self._cfg.cloud.model)


class MockDeviceClient:
    def __init__(self, cfg: ExperimentCfg):
        self._cfg = cfg

    def complete(self, system: str, user: str,
                 temperature: float = 0.0) -> DeviceResult:
        p = _find_problem(self._cfg, user)
        recovery = "RECEIVED MESSAGE" in user
        if p is None or not recovery:
            # solve mode: pass only on even task numbers (deterministic ~50%)
            ok = p is not None and int(p.task_id.rsplit("/", 1)[-1]) % 2 == 0
            if ok:
                content = (p.prompt + p.canonical_solution).strip("\n") + "\n"
            else:
                stub = f"def {p.entry_point if p else 'solution'}(*a, **k):\n" \
                       "    raise NotImplementedError('mock device baseline')\n"
                content = stub
            return DeviceResult(content=content, thinking="",
                                 prompt_tokens=len(user) // 4,
                                 completion_tokens=len(content) // 4,
                                 latency_s=0.01, model=self._cfg.device.model)
        # recovery mode: keep what arrived if usable, else reconstruct
        from ..message_format import parse, extract_code
        msg = user.split("RECEIVED MESSAGE", 1)[-1]
        code = extract_code(parse(msg).action)
        if code.strip():
            try:
                compile(code, "<received>", "exec")
                content = code
            except SyntaxError:
                content = (p.prompt + p.canonical_solution).strip("\n") + "\n"
        else:
            content = (p.prompt + p.canonical_solution).strip("\n") + "\n"
        return DeviceResult(content=content, thinking="",
                            prompt_tokens=len(user) // 4,
                            completion_tokens=len(content) // 4,
                            latency_s=0.01, model=self._cfg.device.model)
