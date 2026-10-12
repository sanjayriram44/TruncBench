"""Load EvalPlus problems (HumanEval+ primary, MBPP+ secondary).

Dataset records (confirmed by introspection, humaneval_plus):
  task_id, prompt, entry_point, canonical_solution, test,
  plus_input (list of input lists), base_input, contract, atol
The `test` field is a full test script defining `check(candidate)`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from .config import ExperimentCfg


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    entry_point: str
    test_script: str
    plus_input: list
    canonical_solution: str = ""
    atol: int = 0


def _load_raw(benchmark: str) -> dict:
    if benchmark == "humaneval_plus":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    if benchmark == "mbpp_plus":
        from evalplus.data import get_mbpp_plus
        return get_mbpp_plus()
    raise ValueError(f"unknown benchmark {benchmark!r}")


def load_problems(cfg: ExperimentCfg) -> list[Problem]:
    data = _load_raw(cfg.run.benchmark)
    tasks = sorted(data.keys())
    if cfg.run.problem_limit:
        tasks = tasks[: cfg.run.problem_limit]
    out = []
    for t in tasks:
        rec = data[t]
        out.append(Problem(
            task_id=t,
            prompt=rec["prompt"],
            entry_point=rec.get("entry_point", "solution"),
            test_script=rec.get("test", ""),
            plus_input=rec.get("plus_input", []),
            canonical_solution=rec.get("canonical_solution", ""),
            atol=int(rec.get("atol", 0) or 0),
        ))
    return out


def dump_one(cfg: ExperimentCfg) -> None:
    """Offline introspection aid: print one raw record's shape."""
    data = _load_raw(cfg.run.benchmark)
    t = sorted(data.keys())[0]
    rec = data[t]
    print(f"benchmark={cfg.run.benchmark} first_task={t}")
    print("fields:", sorted(rec.keys()))
    for f in ("prompt", "entry_point", "test", "plus_input"):
        s = str(rec.get(f, ""))
        print(f"--- {f} ({len(s)} chars) ---")
        print(s[:400])


def main(argv: list[str] | None = None) -> None:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--dump-one", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.dump_one:
        dump_one(cfg)
    else:
        probs = load_problems(cfg)
        print(f"loaded {len(probs)} problems; first={probs[0].task_id!r}")
        print(f"first entry_point={probs[0].entry_point!r} "
              f"plus_cases={len(probs[0].plus_input)}")


if __name__ == "__main__":
    sys.exit(main())
