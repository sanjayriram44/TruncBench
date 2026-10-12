"""Pre-screening: keep only problems the device cannot solve alone but can
solve when given the full, untruncated cloud message.

  (a) device-alone (task only, no cloud message) must FAIL
  (b) full cloud message, recovery=no, reasoning_first, seed 0 must PASS

Resumable: results/prescreen.json accumulates per-task verdicts.
The cloud message must already be cached (run src.generate first).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .clients.device_client import make_device_client
from .config import load_config
from .dataset import load_problems
from .generate import cache_path, system_prompt
from .message_format import parse, code_from_answer
from .scoring import score


def prescreen_problems(cfg, device) -> dict:
    out_path = Path(cfg.run.results_dir) / "prescreen.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    problems = {p.task_id: p for p in load_problems(cfg)}
    seed0 = cfg.run.seeds[0]
    missing = [t for t in problems
                if t not in results
                and not cache_path(cfg, "reasoning_first", t, seed0).exists()]
    if missing:
        raise SystemExit(f"no cached cloud message for {len(missing)} "
                         f"problems (e.g. {missing[0]}); run src.generate first")
    for task_id, p in problems.items():
        if task_id in results:
            continue
        gen_path = cache_path(cfg, "reasoning_first", task_id, seed0)
        gen = json.loads(gen_path.read_text())
        # (b) full cloud message, no recovery
        full_code = code_from_answer(gen["action"] if gen["action"] is not None
                                      else parse(gen["message"]).action)
        r_full = score(p, full_code, cfg)
        # (a) device alone: task only, no cloud message
        user_solve = system_prompt_for_solve(cfg).format(task=p.prompt)
        res = device.complete("You are a capable Python coding model.",
                              user_solve, cfg.device.temperature)
        r_alone = score(p, code_from_answer(res.content), cfg)
        kept = (r_alone.verdict != "passed") and (r_full.verdict == "passed")
        results[task_id] = {
            "device_alone": r_alone.verdict, "full_msg": r_full.verdict,
            "kept": kept,
            "reason": ("kept" if kept else
                        "device-alone solved it" if r_alone.verdict == "passed"
                        else "full cloud message failed"),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"prescreen {task_id}: alone={r_alone.verdict} "
              f"full={r_full.verdict} -> {'KEEP' if kept else 'drop'}")
    kept = sum(1 for r in results.values() if r["kept"])
    print(f"prescreen: {kept}/{len(results)} problems kept")
    if len(results) and kept == 0:
        print("RED FLAG: no problems survived the filter — check the "
              "device model / sandbox before running the grid.")
    return results


def system_prompt_for_solve(cfg) -> str:
    from .generate import prompts_dir
    return (prompts_dir(cfg) / "device_solve.txt").read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.limit:
        cfg.run.problem_limit = min(cfg.run.problem_limit, args.limit)
    prescreen_problems(cfg, make_device_client(cfg))


if __name__ == "__main__":
    sys.exit(main())
