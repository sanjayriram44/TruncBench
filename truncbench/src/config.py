"""Configuration dataclasses + YAML loader.

API keys are NEVER stored: configs carry only the *name* of the env var
holding a key; clients read os.environ at call time.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields

import yaml

VALID_KINDS = ("mock", "real")
VALID_BENCHMARKS = ("humaneval_plus", "mbpp_plus")
VALID_ORDERINGS = ("reasoning_first", "code_first")
VALID_SANDBOX_KINDS = ("docker", "bwrap", "rlimit")


@dataclass
class CloudCfg:
    kind: str = "mock"
    base_url: str = "https://openrouter.ai/api/v1"
    base_url_env: str = ""      # optional: env var that overrides base_url
    model: str = "z-ai/glm-5.3-flash"
    model_env: str = ""          # optional: env var that overrides model
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    max_requests: int = 4


@dataclass
class DeviceCfg:
    kind: str = "mock"
    base_url: str = "http://127.0.0.1:8000/v1"
    base_url_env: str = ""      # optional: env var that overrides base_url
    model: str = "Qwen3.8-27B-MTP"
    model_env: str = ""          # optional: env var that overrides model
    api_key_env: str = "VLLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_context: int = 98304
    concurrency: int = 4


@dataclass
class SandboxCfg:
    timeout_s: int = 30
    cpu_seconds: int = 30      # rlimit mode only; docker mode uses --cpus + wall clock
    mem_mb: int = 2048
    kind: str = "docker"       # "docker" | "bwrap" | "rlimit"
    docker_image: str = "python:3.12-slim"
    bubblewrap_bin: str = "bwrap"


@dataclass
class RunCfg:
    benchmark: str = "humaneval_plus"
    problem_limit: int = 50
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    orderings: list = field(default_factory=lambda: list(VALID_ORDERINGS))
    cut_fractions: list = field(default_factory=lambda: [0.25, 0.5, 0.75])
    control_tail_chars: int = 512
    tiktoken_encoding: str = "o200k_base"
    results_dir: str = "results"
    bootstrap_resamples: int = 1000
    device_max_requests: int = 4


@dataclass
class ExperimentCfg:
    cloud: CloudCfg
    device: DeviceCfg
    sandbox: SandboxCfg
    run: RunCfg
    config_path: str = ""
    _raw: dict = field(default_factory=dict, repr=False)

    def api_key(self, name: str) -> str:
        """Resolve a key from the environment at call time."""
        return os.environ.get(name, "")

    def sha256(self) -> str:
        raw = dict(self._raw)
        return hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


def _fill(dc_cls, data: dict):
    names = {f.name for f in fields(dc_cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)} for {dc_cls.__name__}")
    return dc_cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str) -> ExperimentCfg:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - {"run", "cloud", "device", "sandbox"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    run = _fill(RunCfg, raw.get("run", {}))
    if run.benchmark not in VALID_BENCHMARKS:
        raise ValueError(f"benchmark must be one of {VALID_BENCHMARKS}")
    if not set(run.orderings) <= set(VALID_ORDERINGS):
        raise ValueError(f"orderings must be within {VALID_ORDERINGS}")
    for frac in run.cut_fractions:
        if not 0 < frac < 1:
            raise ValueError(f"cut fractions must be in (0,1), got {frac}")
    if not run.seeds:
        raise ValueError("run.seeds must be non-empty")
    cloud = _fill(CloudCfg, raw.get("cloud", {}))
    device = _fill(DeviceCfg, raw.get("device", {}))
    sandbox = _fill(SandboxCfg, raw.get("sandbox", {}))
    if sandbox.kind not in VALID_SANDBOX_KINDS:
        raise ValueError(f"sandbox.kind must be one of {VALID_SANDBOX_KINDS}")
    for name, cfg in (("cloud", cloud), ("device", device)):
        if cfg.kind not in VALID_KINDS:
            raise ValueError(f"{name}.kind must be one of {VALID_KINDS}")
        if cfg.model_env:
            val = os.environ.get(cfg.model_env, "")
            if val:
                cfg.model = val
                raw[name]["model"] = val
            elif cfg.kind == "real":
                raise SystemExit(f"{name}.model_env points at '{cfg.model_env}', "
                                  f"which is not set — export it before running")
        if cfg.base_url_env:
            val = os.environ.get(cfg.base_url_env, "")
            if val:
                cfg.base_url = val
                raw[name]["base_url"] = val
        if cfg.kind == "real" and not os.environ.get(cfg.api_key_env):
            raise SystemExit(f"{name} client requires env var {cfg.api_key_env}")
    return ExperimentCfg(cloud=cloud, device=device, sandbox=sandbox,
                          run=run, config_path=os.path.abspath(path), _raw=raw)
