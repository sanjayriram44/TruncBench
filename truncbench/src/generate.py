"""Generate cloud messages once per (problem, ordering, seed) and cache them.

Every truncation condition later reuses the cached ORIGINAL message, so all
cuts within a (problem, ordering, seed) are made from identical text.
Resumable: existing cache files are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .config import load_config
from .clients.cloud_client import make_cloud_client
from .dataset import load_problems
from .message_format import ORDER_CODE_FIRST, parse

PROMPT_FILES = {
    "reasoning_first": "cloud_reasoning_first.txt",
    ORDER_CODE_FIRST: "cloud_code_first.txt",
}


def prompts_dir(cfg) -> Path:
    return Path(cfg.config_path).resolve().parent.parent / "prompts"


def system_prompt(cfg, order: str) -> str:
    return (prompts_dir(cfg) / PROMPT_FILES[order]).read_text(encoding="utf-8")


def cache_path(cfg, order: str, task_id: str, seed: int) -> Path:
    safe = task_id.replace("/", "_")
    d = Path(cfg.run.results_dir) / "generations" / cfg.run.benchmark / order
    d.mkdir(parents=True, exist_ok=True)
    return d / f"task_{safe}_seed{seed}.json"


def _token_count(cfg, text: str) -> int:
    import tiktoken
    try:
        return len(tiktoken.get_encoding(cfg.run.tiktoken_encoding).encode(text))
    except Exception:
        return max(1, len(text) // 4)


def generate_all(cfg, force: bool = False) -> tuple[int, int]:
    client = make_cloud_client(cfg)
    problems = load_problems(cfg)
    done = skipped = 0
    for p in problems:
        for order in cfg.run.orderings:
            sys_prompt = system_prompt(cfg, order)
            for seed in cfg.run.seeds:
                path = cache_path(cfg, order, p.task_id, seed)
                if path.exists() and not force:
                    skipped += 1
                    continue
                res = client.complete(sys_prompt, p.prompt,
                                      cfg.cloud.temperature, seed,
                                      cfg.cloud.max_tokens)
                parsed = parse(res.text)
                rec = {
                    "task_id": p.task_id, "order": order, "seed": seed,
                    "cloud_model": res.model or cfg.cloud.model,
                    "message": res.text,
                    "thought": parsed.thought, "action": parsed.action,
                    "thought_status": parsed.thought_status,
                    "action_status": parsed.action_status,
                    "n_chars": len(res.text),
                    "n_tokens": _token_count(cfg, res.text),
                    "prompt_sha256": hashlib.sha256(
                        (sys_prompt + p.prompt).encode()).hexdigest()[:16],
                    "usage": {"prompt_tokens": res.prompt_tokens,
                               "completion_tokens": res.completion_tokens},
                    "latency_s": round(res.latency_s, 3),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                done += 1
                print(f"  cached {p.task_id} {order} seed={seed} "
                      f"({rec['n_chars']} chars)")
    return done, skipped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even when a cache file exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="override run.problem_limit (smaller only)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.limit:
        cfg.run.problem_limit = min(cfg.run.problem_limit, args.limit)
    done, skipped = generate_all(cfg, force=args.force)
    print(f"generate: {done} generated, {skipped} skipped (cached)")


if __name__ == "__main__":
    sys.exit(main())
