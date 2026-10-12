"""Execution sandbox: one candidate script per run.

Three fence options (sandbox.kind):
  docker  - candidate runs in a pinned container: --network none, --memory,
            --cpus, clean env via `env -i`; orphans killed via --cidfile.
  bwrap   - bubblewrap namespaces on the host (no network).
  rlimit  - plain subprocess with RLIMIT_CPU/AS/FSIZE/NPROC.
All modes share: hard wall-clock timeout, throwaway working dir, stripped
env (no API keys). Docker is the default (user choice, 2026-07).
"""
from __future__ import annotations

import os
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED")
_ENV_SKIP = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "GITHUB", "GITLAB")


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    mem_exceeded: bool = False
    bubblewrap: bool = False
    sandbox_kind: str = "rlimit"


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k in ENV_KEEP or k.startswith("LC_")}
    # belt and braces: drop anything that looks credential-like
    return {k: v for k, v in env.items()
            if not any(s in k.upper() for s in _ENV_SKIP)}


def _make_preexec(cpu_seconds: int, mem_mb: int):
    def _pre() -> None:
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    return _pre


def _timeout_payload(e: subprocess.TimeoutExpired):
    out = e.stdout if isinstance(e.stdout, (str, bytes)) else b""
    err = e.stderr if isinstance(e.stderr, (str, bytes)) else b""
    return str(out), str(err), -1


def _run_docker(workdir, code_path, env, timeout_s, mem_mb, image, started):
    # Candidate file lives in `workdir` (host), mounted at /work.
    # `env -i` gives the candidate ONLY our stripped env — docker run would
    # otherwise pass the host environment (API keys!) into the container.
    cidfile = os.path.join(workdir, ".container_id")
    cmd = ["docker", "run", "--rm", "--network", "none",
           "--memory", f"{mem_mb}m", "--cpus", "1",
           "--cidfile", cidfile,
           "-v", f"{workdir}:/work", "--workdir", "/work",
           "--entrypoint", "sh"]
    inner = "exec env -i " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    inner += f" python -B /work/{os.path.basename(code_path)}"
    cmd += [image, "-c", inner]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        code, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as e:
        out, err, code = _timeout_payload(e)
        timed_out = True
        # Client timed out but the container may still be running: kill via cidfile.
        try:
            with open(cidfile) as fh:
                cid = fh.read().strip()
            if cid:
                subprocess.run(["docker", "kill", cid],
                                capture_output=True, timeout=15)
        except OSError:
            pass
    # 137 = 128 + SIGKILL: cgroup OOM kill inside the container (verified).
    mem_exceeded = (code == 137) or ("MemoryError" in err)
    return SandboxResult(code, str(out)[:20000], str(err)[:8000],
                          time.monotonic() - started, timed_out, mem_exceeded,
                          False, "docker")


def run_sandbox(code_path: str, timeout_s: int, mem_mb: int,
                cpu_seconds: int, kind: str = "rlimit",
                docker_image: str = "python:3.12-slim",
                bubblewrap_bin: str = "bwrap",
                cwd: str | None = None) -> SandboxResult:
    workdir = cwd or tempfile.mkdtemp(prefix="truncbench_")
    os.makedirs(workdir, exist_ok=True)
    own_dir = cwd is None
    started = time.monotonic()
    env = clean_env()
    try:
        if kind == "docker":
            return _run_docker(workdir, code_path, env, timeout_s, mem_mb,
                                docker_image, started)
        if kind == "bwrap" and shutil.which(bubblewrap_bin):
            cmd = [bubblewrap_bin, "--unshare-net", "--die-with-parent",
                   "--dev-bind", "/", "/", "--", sys.executable, code_path]
            pre, res_kind = None, "bwrap"
        else:
            if kind == "bwrap":
                print("warning: bwrap unavailable, falling back to rlimit sandbox",
                      file=sys.stderr)
            cmd = [sys.executable, "-B", code_path]
            pre = _make_preexec(cpu_seconds, mem_mb) if os.name == "posix" else None
            res_kind = "rlimit"
        try:
            proc = subprocess.run(cmd, cwd=workdir, env=env,
                                   capture_output=True, text=True,
                                   timeout=timeout_s, preexec_fn=pre)
            code, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            out, err, code = _timeout_payload(e)
            timed_out = True
        # RLIMIT_AS triggers MemoryError; SIGKILL(-9) = system OOM kill.
        mem_exceeded = (code == -9) or ("MemoryError" in err)
        return SandboxResult(code, str(out)[:20000], str(err)[:8000],
                              time.monotonic() - started, timed_out, mem_exceeded,
                              res_kind == "bwrap", res_kind)
    finally:
        if own_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def execute_code(code: str, cfg, label: str = "candidate") -> SandboxResult:
    """Write code to a throwaway file and run it under the sandbox limits."""
    workdir = tempfile.mkdtemp(prefix="truncbench_")
    try:
        path = os.path.join(workdir, f"{label}.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        return run_sandbox(path, cfg.sandbox.timeout_s, cfg.sandbox.mem_mb,
                            cfg.sandbox.cpu_seconds, cfg.sandbox.kind,
                            cfg.sandbox.docker_image, cfg.sandbox.bubblewrap_bin,
                            cwd=workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
