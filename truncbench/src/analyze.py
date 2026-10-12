"""Analysis: tables, paired bootstrap deltas, cut-position curves.

Outputs (results/analysis/): summary.csv, deltas.csv, case_split.csv,
failure_breakdown.csv, curves.csv, curves.png, prescreen_report.txt,
pilot_variance.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config
from .truncation import TRUNC_CASE_I, TRUNC_CASE_II, TRUNC_CONTROL, TRUNC_NONE

VERDICTS = ["passed", "syntax_error", "runtime_error", "wrong_answer",
             "timeout", "no_code"]
EXEC = ("syntax_error", "runtime_error", "timeout", "no_code")


def load_rows(cfg) -> list[dict]:
    path = Path(cfg.run.results_dir) / "runs" / cfg.run.benchmark / "runs.jsonl"
    if not path.exists():
        raise SystemExit(f"no runs found at {path}; run src.run_experiments")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_index(rows: list[dict]) -> dict:
    idx = defaultdict(list)
    for r in rows:
        c = r["condition"]
        key = (r["order"], c["truncation"],
               c["cut_fraction"] if c["truncation"] in (TRUNC_CASE_I, TRUNC_CASE_II) else None,
               c["recovery"])
        idx[key].append(r)
    return idx


def pass_rate(rows: list[dict]) -> float:
    return sum(1 for r in rows if r["verdict"] == "passed") / len(rows) if rows else 0.0


def write_summary(cfg, idx: dict) -> None:
    out = Path(cfg.run.results_dir) / "analysis"
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "truncation", "case", "cut_fraction", "recovery",
                    "n_runs", "n_problems", "pass_rate"] + VERDICTS + ["mean_device_latency_s"])
        for (order, trunc, frac, rec), rows in sorted(idx.items()):
            counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
            lats = [r["device_usage"]["latency_s"] for r in rows
                    if r["device_usage"]] or [0.0]
            w.writerow([order, trunc, TRUNC_CASE_I if frac is not None and trunc == TRUNC_CASE_I else
                        (TRUNC_CASE_II if trunc == TRUNC_CASE_II else trunc),
                        frac if frac is not None else "", rec, len(rows),
                        len({r["task_id"] for r in rows}),
                        f"{pass_rate(rows):.4f}"] + [counts[v] for v in VERDICTS] +
                       [f"{sum(lats)/len(lats):.3f}"])


def per_problem_rates(rows: list[dict]) -> dict:
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        frac = c["cut_fraction"] if c["truncation"] in (TRUNC_CASE_I, TRUNC_CASE_II) else None
        per[(r["order"], c["truncation"], frac, c["recovery"])][r["task_id"]].append(
            r["verdict"] == "passed")
    return per


def bootstrap_delta(cfg, idx: dict, base_key: tuple, cell_key: tuple) -> tuple[float, float, float]:
    per = per_problem_rates(idx[base_key] + idx[cell_key])
    probs = sorted({r["task_id"] for r in idx[base_key] + idx[cell_key]})
    rng = random.Random(12345)
    n = cfg.run.bootstrap_resamples
    deltas = []
    for _ in range(n):
        sample = [rng.choice(probs) for _ in range(len(probs))]
        ds = []
        for t in set(sample):
            b, c = per[base_key].get(t), per[cell_key].get(t)
            if b and c:
                ds.append(sum(b) / len(b) - sum(c) / len(c))
        if ds:
            deltas.append(sum(ds) / len(ds))
    deltas.sort()
    if not deltas:
        return (0.0, 0.0, 0.0)
    m = sum(deltas) / len(deltas)
    lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
    return (m, lo, hi)


def write_deltas(cfg, idx: dict) -> None:
    out = Path(cfg.run.results_dir) / "analysis"
    with (out / "deltas.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["effect", "order", "truncation", "cut_fraction", "recovery",
                    "mean_delta", "ci_low", "ci_high", "n_runs_cell"])
        for order in {k[0] for k in idx}:
            for rec in (False, True):
                base = (order, TRUNC_NONE, None, rec)
                for (o, trunc, frac, r) in idx:
                    if o == order and r == rec and trunc != TRUNC_NONE and trunc != TRUNC_CONTROL:
                        m, lo, hi = bootstrap_delta(cfg, idx, base, (o, trunc, frac, r))
                        w.writerow(["truncation_drop", order, trunc, frac, r,
                                    f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(idx[(o, trunc, frac, r)])])
                if rec is False:
                    for (o, trunc, frac, r) in idx:
                        if o == order and trunc in (TRUNC_CASE_I, TRUNC_CASE_II) and not r:
                            m, lo, hi = bootstrap_delta(cfg, idx, (o, trunc, frac, True), (o, trunc, frac, False))
                            w.writerow(["recovery_gain", order, trunc, frac, r,
                                        f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(idx[(o, trunc, frac, False)])])
                    m, lo, hi = bootstrap_delta(cfg, idx, (order, TRUNC_NONE, None, True),
                                                 (order, TRUNC_NONE, None, False))
                    w.writerow(["recovery_prompt_damage_at_no_trunc", order, TRUNC_NONE, "", False,
                                f"{-m:.4f}", f"{-hi:.4f}", f"{-lo:.4f}", len(idx[(order, TRUNC_NONE, None, True)])])
                m, lo, hi = bootstrap_delta(cfg, idx, (order, TRUNC_CONTROL, None, rec),
                                             (order, TRUNC_NONE, None, rec))
                w.writerow(["control_check", order, TRUNC_CONTROL, "", rec,
                            f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(idx[(order, TRUNC_CONTROL, None, rec)])])
        for rec in (False, True):
            a, b = ("reasoning_first", TRUNC_NONE, None, rec), ("code_first", TRUNC_NONE, None, rec)
            if a in idx and b in idx:
                m, lo, hi = bootstrap_delta(cfg, idx, a, b)
                w.writerow(["ordering_effect_at_no_trunc", "reasoning_first-vs-code_first", TRUNC_NONE, "", rec,
                            f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", 0])


def write_case_split(cfg, idx: dict) -> None:
    out = Path(cfg.run.results_dir) / "analysis"
    with (out / "case_split.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "case", "cut_fraction", "recovery", "n_runs",
                    "pass_rate", "execution_failures", "correctness_failures"])
        for (order, trunc, frac, rec), rows in sorted(idx.items()):
            if trunc not in (TRUNC_CASE_I, TRUNC_CASE_II):
                continue
            n = len(rows)
            w.writerow([order, trunc, frac, rec, n, f"{pass_rate(rows):.4f}",
                        sum(1 for r in rows if r["failure_type"] == "execution"),
                        sum(1 for r in rows if r["failure_type"] == "correctness")])


def write_failure_breakdown(cfg, idx: dict) -> None:
    out = Path(cfg.run.results_dir) / "analysis"
    with (out / "failure_breakdown.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "truncation", "cut_fraction", "recovery",
                    "execution_share", "correctness_share", "n_failures"])
        for (order, trunc, frac, rec), rows in sorted(idx.items()):
            fails = [r for r in rows if r["verdict"] != "passed"]
            n = len(fails)
            ex = sum(1 for r in fails if r["failure_type"] == "execution")
            co = sum(1 for r in fails if r["failure_type"] == "correctness")
            w.writerow([order, trunc, frac if frac is not None else "", rec,
                        f"{ex/n:.3f}" if n else "", f"{co/n:.3f}" if n else "", n])


def write_curves(cfg, idx: dict) -> None:
    out = Path(cfg.run.results_dir) / "analysis"
    rows_out, fig_ax = [], {}
    for order in {k[0] for k in idx}:
        for case in (TRUNC_CASE_I, TRUNC_CASE_II):
            for rec in (False, True):
                xs, ys = [0.0], [pass_rate(idx.get((order, TRUNC_NONE, None, rec), []))]
                for frac in sorted({f for (o, t, f, r) in idx if o == order and t == case and r == rec}):
                    xs.append(frac)
                    ys.append(pass_rate(idx[(order, case, frac, rec)]))
                for x, y in zip(xs, ys):
                    rows_out.append([order, case, rec, x, f"{y:.4f}"])
                if (order, case) not in fig_ax:
                    fig_ax[(order, case)] = plt.figure().add_subplot(2, 2, 1)
                ax = fig_ax[(order, case)]
                ax.plot(xs, ys, marker="o", label="recovery" if rec else "no recovery")
    with (out / "curves.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "case", "recovery", "cut_fraction", "pass_rate"])
        w.writerows(rows_out)
    for (order, case), ax in fig_ax.items():
        ax.set_title(f"{order}, {case}")
        ax.set_xlabel("fraction of part cut")
        ax.set_ylabel("pass rate")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.figure.tight_layout()
        ax.figure.savefig(out / "curves.png", dpi=110)
        plt.close(ax.figure)


def write_variance_and_prescreen(cfg, idx: dict) -> None:
    out = Path(cfg.run.results_dir) / "analysis"
    var = {}
    for key, rows in idx.items():
        n = len(rows)
        p = pass_rate(rows)
        var[f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"] = {
            "n_runs": n, "n_problems": len({r["task_id"] for r in rows}),
            "pass_rate": round(p, 4), "se": round((p * (1 - p) / n) ** 0.5, 4) if n else 0}
    (out / "pilot_variance.json").write_text(json.dumps(var, indent=1))
    pres = Path(cfg.run.results_dir) / "prescreen.json"
    if pres.exists():
        data = json.loads(pres.read_text())
        kept = sum(1 for v in data.values() if v.get("kept"))
        lines = [f"pre-screened problems: {kept} kept / {len(data)} total",
                 "criterion: device-alone FAILS and full cloud message PASSES", ""]
        for t, v in sorted(data.items()):
            lines.append(f"  {t}: alone={v['device_alone']} full_msg={v['full_msg']} -> {v['reason']}")
        if kept == 0:
            lines.append("RED FLAG: nothing survived the filter.")
        (out / "prescreen_report.txt").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(cfg.run.results_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(cfg)
    idx = cell_index(rows)
    write_summary(cfg, idx)
    write_deltas(cfg, idx)
    write_case_split(cfg, idx)
    write_failure_breakdown(cfg, idx)
    write_curves(cfg, idx)
    write_variance_and_prescreen(cfg, idx)
    print(f"analysis written to {out}/ ({len(rows)} runs, {len(idx)} cells)")


if __name__ == "__main__":
    sys.exit(main())
