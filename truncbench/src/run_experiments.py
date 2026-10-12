"""Run every (kept problem, seed, condition) and log one JSONL record each.

Resumable by run_id; device-model calls are pooled up to
cfg.device.concurrency, sandbox scoring runs inline.
Usage: python -m src.run_experiments --config configs/pilot.yaml
       [--conditions case_ii] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .clients.device_client import make_device_client
from .config import load_config
from .dataset import load_problems
from .generate import prompts_dir
from .message_format import parse, extract_code, code_from_answer
from .scoring import score
from .truncation import enumerate_conditions, cut_message

SYS_RECOVERY = ("You are a careful Python engineer repairing received code. "
                 "You output only the final complete solution code.")


def load_prescreen(cfg) -> dict:
    path = Path(cfg.run.results_dir) / "prescreen.json"
    if not path.exists():
        raise SystemExit("no prescreen results; run src.prescreen first")
    return json.loads(path.read_text())


def existing_run_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {json.loads(line)["run_id"]
                for line in fh if line.strip()}


def _load_gen(cfg, order, task_id, seed):
    from .generate import cache_path
    p = cache_path(cfg, order, task_id, seed)
    if not p.exists():
        raise SystemExit(f"missing cached message {p}; run src.generate")
    return json.loads(p.read_text())


def recovery_prompt(cfg, task: str, message: str) -> str:
    return (prompts_dir(cfg) / "recovery.txt").read_text(
        encoding="utf-8").format(task=task, message=message)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--conditions", default=None,
                    help="substring filter on condition keys")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.limit:
        cfg.run.problem_limit = min(cfg.run.problem_limit, args.limit)

    prescreen = load_prescreen(cfg)
    problems = [p for p in load_problems(cfg)
                if prescreen.get(p.task_id, {}).get("kept")]
    if not problems:
        raise SystemExit("no kept problems; prescreen kept none")
    conds = enumerate_conditions(cfg)
    if args.conditions:
        conds = [c for c in conds if args.conditions in c.key]
    print(f"runs: {len(problems)} problems x {len(cfg.run.seeds)} seeds x "
          f"{len(conds)} conditions = {len(problems)*len(cfg.run.seeds)*len(conds)}")

    device = make_device_client(cfg)
    runs_path = Path(cfg.run.results_dir) / "runs" / cfg.run.benchmark / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = existing_run_ids(runs_path)
    with ThreadPoolExecutor(max_workers=max(1, cfg.device.concurrency)) as pool, \
            runs_path.open("a", encoding="utf-8") as out:
        for p in problems:
            for seed in cfg.run.seeds:
                for cond in conds:
                    run_id = hashlib.sha1(
                        f"{p.task_id}|{seed}|{cond.key}".encode()).hexdigest()[:12]
                    if run_id in done_ids:
                        continue
                    gen = _load_gen(cfg, cond.order, p.task_id, seed)
                    cut = cut_message(gen["message"], cond.order,
                                       cond.truncation, cond.cut_fraction,
                                       cfg.run.control_tail_chars,
                                       cfg.run.tiktoken_encoding)
                    sent = parse(cut.message)
                    dev_future = None
                    if cond.recovery:
                        dev_future = pool.submit(
                            device.complete, SYS_RECOVERY,
                            recovery_prompt(cfg, p.prompt, cut.message),
                            cfg.device.temperature)
                    sb = None
                    if cond.recovery:
                        dev = dev_future.result()
                        final_code = code_from_answer(dev.content)
                        device_usage = {"prompt_tokens": dev.prompt_tokens,
                                        "completion_tokens": dev.completion_tokens,
                                        "latency_s": round(dev.latency_s, 3)}
                    else:
                        final_code = extract_code(sent.action)
                        device_usage = None
                    result = score(p, final_code, cfg)
                    sb = result.sandbox
                    rec = {
                        "run_id": run_id, "benchmark": cfg.run.benchmark,
                        "task_id": p.task_id, "seed": seed, "order": cond.order,
                        "condition": {"truncation": cond.truncation,
                                      "cut_fraction": cond.cut_fraction,
                                      "recovery": cond.recovery, "key": cond.key},
                        "cut": {"cut_part": cut.cut_part,
                                "index_chars": cut.index_chars,
                                "index_tokens": cut.index_tokens,
                                "frac_of_whole": round(cut.frac_of_whole, 4),
                                "removed": cut.removed},
                        "message_sent": cut.message,
                        "original_message_sha256": hashlib.sha256(
                            gen["message"].encode()).hexdigest()[:16],
                        "recovered_code": (final_code if cond.recovery else None),
                        "final_code": final_code,
                        "extraction": {"method": "recovered" if cond.recovery else "tags",
                                       "had_open_tag": sent.action is not None,
                                       "had_close_tag": sent.action_status == "full",
                                       "stripped_fence": bool(sent.action) and sent.action.lstrip().startswith("```")},
                        "verdict": result.verdict, "failure_type": result.failure_type,
                        "test_error": result.test_error,
                        "cloud_usage": gen.get("usage"),
                        "device_usage": device_usage,
                        "sandbox": None if sb is None else {
                            "exit_code": sb.exit_code,
                            "duration_s": round(sb.duration_s, 3),
                            "timed_out": sb.timed_out,
                            "mem_exceeded": sb.mem_exceeded,
                            "sandbox_kind": sb.sandbox_kind},
                        "config": {"cloud_model": cfg.cloud.model,
                                    "device_model": cfg.device.model,
                                    "config_sha256": cfg.sha256()},
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"{p.task_id} seed={seed} {cond.key} -> {result.verdict}")
    print(f"done. runs logged: {len(existing_run_ids(runs_path))} ({runs_path})")


if __name__ == "__main__":
    sys.exit(main())
