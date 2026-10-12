# Implementation Plan: Measuring Truncation Damage in Device–Cloud Code Delivery

**Research question.** A cloud LLM solves a Python coding task and sends a two-part message (reasoning + code) to a device. Transmission can truncate the message silently. We measure (a) how much truncation hurts test pass rate, (b) how much a local recovery LLM recovers, (c) whether message ordering (reasoning-first vs code-first) changes the damage, and (d) whether the recovery prompt itself damages un-truncated messages.

**Executor contract.** The implementing agent runs one step at a time, sees only this file, and must never make real API calls — only unit tests and offline smoke runs with the mock clients. Every step below names one file, its purpose, required interfaces, and one verify command. Every file stays under ~150 lines. All endpoints, model ids, keys (env-var names only), seeds, and limits live in `configs/*.yaml` — never hard-coded, never logged.

**Current status.** Nothing exists yet in the repo except `AGENTS.md`. All code goes under a new project directory `truncbench/`.

## 1. Recommended benchmark: HumanEval+ via EvalPlus (MBPP+ as secondary)

**Why HumanEval+.** Checked current state: EvalPlus (github.com/evalplus/evalplus, latest v0.3.x) is actively maintained, installable with `pip`, and ships HumanEval+ (164 problems, ~80× more tests than HumanEval) and MBPP+ (399 problems) as plain JSON datasets. Test inputs/outputs are JSON-serializable, so scoring is just "call the entry point with inputs, compare to outputs" — no Docker, no stdin/stdout harness, works in a plain Ubuntu subprocess. LiveCodeBench is more contamination-free but needs a stdin/stdout harness and contest-style I/O parsing — more setup for the same pass/fail signal; recommend it only as a later extension. MBPP+ is kept as a config-selectable secondary dataset because it shares the identical harness (MBPP solutions are full functions, same entry-point style).

**Setup steps (run by the human or in pilot step 2):**
1. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` with `evalplus==0.3.1` pinned.
2. `DatasetLoader` calls `from evalplus.data import get_human_eval_plus, get_mbpp_plus` (auto-downloads on first call; cache lives in `~/.cache`).
3. The executor must **introspect the returned dict** (do not trust this spec blindly): each value has at least `prompt`, `entry_point`, `canonical_solution`, and test cases keyed like `plus_input` (list of `(input_tuple, expected_output)`). Print one record offline to confirm field names before writing the test generator.

**Problem text.** The cloud/device prompts use the EvalPlus `prompt` (function signature + docstring) as the task description, verbatim.

**Test generation.** For problem *p*, `Scoring` writes one test script per run:
```python
# candidate code is inserted verbatim, then:
CASES = [...]  # (input, expected) pairs from plus_input
for inp, exp in CASES:
    got = <entry_point>(*inp)
    assert repr(got) == repr(exp), f"FAIL input={inp!r} got={got!r} exp={exp!r}"
print("ALL TESTS PASSED")
```
If introspection shows inputs are flat (not tuples), call with the input directly. `repr` comparison sidesteps float/np-type quirks; HumanEval+ inputs are plain Python literals.
## 2. Repository and module structure

All under `truncbench/`, run with the project-local venv (`truncbench/.venv`):

```
truncbench/
  requirements.txt            # pinned deps: evalplus, openai, pyyaml, tiktoken, matplotlib
  configs/
    pilot.yaml                # all knobs for the pilot (see §5)
    full.yaml                 # copy of pilot with larger n
  prompts/                    # one short file per prompt template (plain text, {placeholders})
    cloud_reasoning_first.txt
    cloud_code_first.txt
    recovery.txt
    device_solve.txt
  src/
    config.py                 # dataclasses + YAML loader, env-var key lookup
    dataset.py                # EvalPlus loading, offline dump for inspection
    message_format.py         # build + robustly parse the 2-part message
    truncation.py             # cut engine, condition spec, lost-part bookkeeping
    sandbox.py                # subprocess runner with limits, no network
    scoring.py                # test-script generation, failure-type classification
    clients/
      cloud_client.py         # real OpenAI-compatible cloud client + factory
      device_client.py        # real vLLM client + factory
      mock_clients.py         # deterministic offline fakes for both
    recovery.py               # device recovery call, uses prompts/recovery.txt
    generate.py               # CLI: create + cache cloud messages
    prescreen.py              # CLI: problem filter (§ below)
    run_experiments.py        # CLI: run all conditions, write runs.jsonl
    analyze.py                # CLI: tables, deltas, curves
  tests/
    test_all.py               # pure-unit tests, no network
  results/                    # created at runtime; gitignored
    generations/              # cached cloud messages (§4)
    prescreen.json
    runs/                     # runs.jsonl per benchmark
    analysis/
```

Import style: plain modules, no package install needed — scripts run from `truncbench/` as `python -m src.run_experiments --config configs/pilot.yaml` with an empty `src/__init__.py` (created in step 1).
## 3. Message format and prompts

### 3.1 Wire format (both orderings)

Tags only, no code fences inside the message (fences complicate truncation):

```
<thought>
{reasoning}
</thought>
<action>
{code}
</action>
```

Ordering `code_first` swaps the two blocks. `MessageBuilder.build(thought, action, order)` and `MessageParser`:

```python
def parse(message: str) -> ParsedMessage:
    # returns ParsedMessage(thought: str|None, action: str|None,
    #   thought_status/action_status: "full"|"partial"|"missing", raw: str)
```

Robust parsing rules (truncation-aware — **must never raise**):
- A part's opening tag present → text after the tag is that part's content (may be empty → `partial`).
- Opening tag absent → part `missing` (content `None`).
- Content ends at the closing tag, the *next* opening tag, or end-of-string; missing closing tag ⇒ `partial` if the other part's opening tag is also absent (we can't be sure it's complete — treat cut-adjacent parts as partial).
- `extract_code(action_text)`: strip one leading/trailing ```` ``` ```` or ```` ```python ```` line if present; otherwise return text as-is.

### 3.2 Cloud prompts (`prompts/cloud_reasoning_first.txt`)

System:

```
You are an expert Python programmer. Solve the task you are given.
Respond with EXACTLY two sections and nothing else:
<thought>
A concise step-by-step plan for solving the task (key ideas, edge cases).
</thought>
<action>
Complete, runnable Python code implementing the task. No explanations,
no markdown fences, no text after the code.
</action>
```

User: the EvalPlus `prompt` verbatim. `cloud_code_first.txt` is identical except the `<action>` block comes first and the `<thought>` block last. Same temperature/seed must be used per (task, seed) across both orderings for symmetry.

### 3.3 Device prompts

`prompts/recovery.txt` — the recovery condition:

```
You will receive a coding task and a message from a remote assistant that
contains the task plan and the solution code. The message may have been
cut off during transmission. Your job: output the COMPLETE solution code.
- If the code looks complete, fix only obvious truncation artifacts
  (e.g. an unfinished last line) and otherwise return it unchanged.
- If the code is cut off or missing, reconstruct it from the task and
  whatever plan text arrived.
- Output ONLY a fenced Python code block containing the full solution.

TASK:
{task}

RECEIVED MESSAGE (may be truncated):
{message}
```

`prompts/device_solve.txt` — the device-alone baseline (for pre-screening only):

```
Solve this Python task. Output ONLY a fenced Python code block with the
full solution.

TASK:
{task}
```

Device clients use the vLLM endpoint's reasoning parser: thinking arrives in a separate field — take **only the final `content`**, never the reasoning field.
## 4. Cloud generation cache: generate once, cut many

Every truncation condition reuses the identical original message. `generate.py` CLI:

```
python -m src.generate --config configs/pilot.yaml   # resumable, skips existing files
```

One JSON per (problem, ordering, seed) at
`results/generations/{benchmark}/{order}/task_{id}_seed{k}.json`:

```json
{"task_id": "HumanEvalPlus/0", "order": "reasoning_first", "seed": 0,
 "cloud_model": "...", "message": "<thought>...", "thought": "...", "action": "...",
 "n_chars": 0, "n_tokens": 0, "created_at": "...", "prompt_sha256": "..."}
```

`n_tokens` is a reference count via `tiktoken` (`o200k_base`, configured). Cache validity: file exists ⇒ reuse. A `--force` flag regenerates. Mock client writes the same schema so the offline smoke run exercises the full pipeline.

## 5. Conditions, seeds, sampling, and variance

### 5.1 Condition enumeration (in `truncation.py`, config-driven)

Grid cells (no truncation): `(order ∈ {reasoning_first, code_first}) × (recovery ∈ {no, yes}) × truncation=none` → 4 cells. These answer "does code-first hurt on its own" and "does the recovery prompt damage uncut messages" (recovery=yes vs recovery=no at truncation=none).

Truncated cells: `(case ∈ {i: reasoning, ii: code}) × (frac ∈ {0.25, 0.50, 0.75}) × (order ×2) × (recovery ×2)` → 24 cells.

Control cut: truncation=control (chars removed ≈ what a 50% case-i cut removes, taken from after `</action>` — e.g. a trailing comment the cloud prompt is *not* asked to produce, so the control appends a synthetic `# Footnote: ...` tail before cutting it). `(order ×2) × (recovery ×2)` → 4 cells. Total **32 conditions**.

### 5.2 Seeds and runs per condition

One **run** = one (problem, seed) under one condition. Cloud: seeds {0,1,2}, temperature 0.7, top_p 0.95 (configurable). Device recovery: temperature 0.0 (greedy), so device-side seed variance is 0; run-to-run variance in truncated cells comes from the cloud message and cut interaction only.

### 5.3 Pilot-first variance estimate

Run the pilot (50 problems × 3 seeds = 150 runs/cell). Per-cell pass rate p has SE ≈ `sqrt(p(1-p)/150)` ≈ 4 pp at p=0.5. For the **delta** between two cells (paired, same problems/messages) the SE is smaller; `analyze.py` computes a problem-level bootstrap (1000 resamples) CI on each delta. Rule: if pilot CIs are too wide to separate the main effects, scale seeds to 5 for the full run (SE ≈ 2.5 pp); decide from the pilot's actual numbers, recorded in `results/analysis/pilot_variance.json`.

### 5.4 Pre-screening (`prescreen.py`)

Keep a problem iff: (a) device-alone (device model, `device_solve.txt`) fails its tests, and (b) full untruncated cloud message, recovery=no, reasoning_first passes. Results cached in `results/prescreen.json` (`{task_id: {device_alone: pass/fail, full_msg: pass/fail, kept: bool}}`), resumable. Report `n_kept / n_total` in the analysis. If device-alone fails for 0 problems (too easy) or everything fails (sandbox broken), that is a red flag surfaced by the report, not silently ignored.
## 6. Logging: what is saved per run

All runs append one JSON object per line to `results/runs/{benchmark}/runs.jsonl` (resumable: runs whose `run_id` already appears are skipped). Required fields:

```json
{"run_id": "<task|seed|order|condition-key hash>",
 "benchmark": "humaneval_plus",
 "task_id": "HumanEvalPlus/0",
 "seed": 0,
 "order": "reasoning_first|code_first",
 "condition": {"truncation": "none|case_i|case_ii|control",
               "cut_fraction": 0.5|null, "recovery": true|false},
 "cut": {"cut_part": "thought|action|tail|null",
         "index_chars": 0, "index_tokens": 0, "frac_of_whole": 0.0,
         "removed": {"partial": ["thought"], "missing": ["action"], "full": []}},
 "message_sent": "<truncated message string, may be empty>",
 "original_message_sha256": "...",
 "recovered_code": "<device output or null>",
 "final_code": "<code actually executed>",
 "extraction": {"method": "tags|recovered|raw", "had_open_tag": true,
                "had_close_tag": true, "stripped_fence": false},
 "verdict": "passed|syntax_error|runtime_error|wrong_answer|timeout|no_code",
 "failure_type": "execution|correctness|null",
 "test_error": "<first failing assert / stderr tail, 2000 chars max>",
 "cloud_usage": {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0},
 "device_usage": {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}|null,
 "sandbox": {"exit_code": 0, "duration_s": 0.0, "timed_out": false, "mem_exceeded": false},
 "config": {"cloud_model": "...", "device_model": "...", "config_sha256": "..."},
 "created_at": "..."}
```

Rules: API keys never enter any file or log (they live only in env vars named in the config); messages and code are content, everything numeric is provenance. `failure_type` is `"execution"` for verdicts `syntax_error|runtime_error|timeout|no_code` and `"correctness"` for `wrong_answer`.
## 7. Analysis outputs (`analyze.py` CLI → `results/analysis/`)

1. **`summary.csv`** — one row per condition: order, truncation, case, cut_fraction, recovery, n, n_problems, pass_rate, syntax/runtime/wrong/timeout/no_code counts, mean device latency. Plus the 4 no-truncation cells and the control cells.
2. **`deltas.csv`** — headline numbers, each with bootstrap 95% CI (problem-level, 1000 resamples, paired):
   - *Truncation drop*: pass_rate(truncation=none, order=o, recovery=r) − pass_rate(cell), for each of the 24 truncated cells.
   - *Recovery gain*: pass_rate(recovery=yes) − pass_rate(recovery=no), computed at truncation=none (tests prompt damage), per case/frac, and averaged over truncated cells, per ordering.
   - *Ordering effect*: reasoning_first − code_first at truncation=none (recovery=no), and within case ii.
   - *Control check*: pass_rate(control) − pass_rate(truncation=none) ≈ 0 validates the harness.
3. **`case_split.csv`** — all results split by case i (reasoning cut) vs case ii (code cut), per ordering and recovery.
4. **`curves_*.csv` / `curves.png`** — pass rate vs cut fraction {0.25, 0.5, 0.75} (+0% reference point from truncation=none), one curve per (order, case, recovery) = 8 curves; matplotlib saved offline (Agg backend).
5. **`failure_breakdown.csv`** — execution vs correctness failure share per condition (does truncation produce *broken* code or *plausible but wrong* code?).
6. **`prescreen_report.txt`** — kept/total problems, red-flag warnings from §5.4.

Everything is plain CSV + one PNG; no notebook needed.
## 8. Cost and compute estimate

Assumptions: cloud message ≈ 1.4K tokens (0.7K prompt echo + 0.7K completion incl. reasoning); device recovery call ≈ 1.6K tokens in + 0.7K out at ~85 tok/s single-stream, ~4× with concurrency ≈ 300+ tok/s aggregate; device-alone prescreen ≈ 50–164 calls.

**Pilot: 50 problems × 3 seeds, 32 conditions**
- Cloud generations: 50 × 3 seeds × 2 orderings = **300 calls** (cached once, reused by all 32 conditions) ≈ 0.4M tokens. With a flash-class model (~$0.1–0.4/M in, $0.4–2/M out): **well under $1**.
- Device calls: prescreen 50–100 + recovery=yes cells (4 no-trunc + 4 control + 24 truncated = 32... of which 16 need recovery: 2×2 no-trunc/control + 2×3×2 truncated = 16 conditions × 150 runs = 2400) ≈ **2,500 calls** ≈ 4.8M tokens ≈ **1.5–2.5 h** on the local vLLM box. Recovery=no cells run sandbox only — free.
- Wall clock total ≈ **2–3 h** device compute + negligible cloud cost.

**Full run: 164 problems × 5 seeds** — ~5.5× pilot ⇒ **9–14 h** device compute (one overnight run; resumable at every level), cloud **<$5**.

Sanity valves: `problem_limit`, `seeds`, and a `--conditions` filter in `run_experiments.py` let the human run any subset (e.g. only case ii) without code changes.
## 9. Risks, design issues, and questions before implementation

**Risks / design decisions taken:**
- *Recovery at truncation=none is not a no-op*: recovery=yes must call the device model even on intact messages (measures prompt damage). Cheap: 4 extra cells.
- *Greedy device decoding* removes device-side variance; variance attribution stays with the cloud message + cut, which is the object of study.
- *Extraction robustness*: every path through `message_format.parse` returns something; `run_experiments.py` wraps scoring so one bad run logs a verdict (`no_code`) and never kills the sweep. Fenced markdown in cloud output is stripped on extraction.
- *Sandbox limits*: `resource.setrlimit` (AS + CPU), `subprocess` timeout, throwaway `tempfile.mkdtemp`, cwd = scratch dir, env stripped of secrets; network isolation via optional bubblewrap flag (`use_bubblewrap: false` default) — plain-Ubuntu safe.
- *Tokenizer*: `n_tokens` uses tiktoken as a reference only; the device vLLM tokenizer differs, which is fine because cuts are defined in characters of the original message and tokens are recorded for reporting.
- *Prescreen leakage*: prescreen uses seed 0's reasoning_first message only; kept problems are re-used across all conditions and all seeds.
- *Contamination*: HumanEval+ is publicly known — acceptable for a relative-effect (truncation) study, and pre-screening removes problems the device already solves.
- *`plus_input` schema drift*: step 3 verifies field names by introspection before writing the test generator.

**Questions for the human (answers change config only, not code):**
1. Cloud model id + temperature/top_p for the pilot — default plan: OpenRouter `z-ai/glm-5.3-flash`, T=0.7.
2. Is `z-ai/glm-5.3-flash` known to emit `<thought>`-style answers of its own? If it wraps reasoning in its own tags, the cloud prompt may need one extra line telling it to use our tags only.
3. Full-run scale: 164 problems × 5 seeds OK given ~9–14 h device compute, or cap problems?
4. Do you want the reference "re-request the missing part" condition implemented in this repo at all, or deferred entirely?
## 10. Build checklist (executor: one file per step, ≤150 lines each, stop on first failing verify)

Work inside `truncbench/`. Run verifies with `.venv/bin/python` after step 2. **Pilot-ready marker:** after step 14 the whole pipeline runs end-to-end offline with mocks.

1. **`requirements.txt` + `src/__init__.py` + `.gitignore`** — pin `evalplus==0.3.1`, `openai>=1.40`, `pyyaml`, `tiktoken`, `matplotlib`; gitignore `.venv/ results/ *.env`. Verify: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip check`.
2. **`src/config.py`** — dataclasses `CloudCfg, DeviceCfg, SandboxCfg, RunCfg, ExperimentCfg` + `load_config(path) -> ExperimentCfg` (YAML; API keys resolved from env-var *names* at call time, never stored). Verify: `.venv/bin/python -c "from src.config import load_config"` after step 3's YAML exists.
3. **`configs/pilot.yaml`** — every knob of §5 (endpoints `http://127.0.0.1:8000/v1`, model ids, env var names `OPENROUTER_API_KEY`, concurrency 4, seeds [0,1,2], problem_limit 50, fracs [0.25,0.5,0.75], timeouts, `use_bubblewrap: false`, benchmark, dataset `humaneval_plus`). Verify: `.venv/bin/python -c "from src.config import load_config; c=load_config('configs/pilot.yaml'); print(c.device.model)"`.
4. **`src/dataset.py`** — `load_problems(cfg) -> list[Problem]` (dataclass: task_id, prompt, entry_point, test_cases), `dump_one(cfg)` to print a record for introspection. Verify: `.venv/bin/python -m src.dataset --config configs/pilot.yaml --dump-one` (offline after first download).
5. **`src/message_format.py`** — `build(thought, action, order) -> str`; `parse(message) -> ParsedMessage` per §3.1; `extract_code(action) -> str`. Never raises. Verify: `.venv/bin/python -c "from src.message_format import parse; print(parse('<action>print(1)').action)"`.
6. **`src/truncation.py`** — `cut_message(original, thought, action, order, truncation, frac, tail_budget) -> CutResult(message, cut_info)`; `enumerate_conditions(cfg) -> list[Condition]` (32 per §5.1, key `"{order}|{trunc}|{case}|{frac}|{recovery}"`); control tail appended synthetically then cut. Verify: `.venv/bin/python -c "from src.truncation import enumerate_conditions"`.
7. **`src/sandbox.py`** — `run_sandbox(code_path, timeout_s, mem_mb, use_bubblewrap, cwd=None) -> SandboxResult(exit_code, stdout, stderr, duration_s, timed_out, mem_exceeded)`; rlimits + stripped env + throwaway dir. Verify: `.venv/bin/python -c "import tempfile,os; from src.sandbox import run_sandbox; p=tempfile.mktemp(suffix='.py'); open(p,'w').write('import os\nprint(sum(range(10)))'); r=run_sandbox(p,5,512,False); assert r.exit_code==0"`.
8. **`src/scoring.py`** — `make_test_script(problem, code) -> str` per §1; `score(problem, code, cfg) -> ScoreResult(verdict, failure_type, test_error, sandbox_stats)`; `compile()`-first classification. Verify: `.venv/bin/python -c "from src.scoring import score"`.
9. **`prompts/cloud_reasoning_first.txt`** — verbatim §3.2. Verify: `grep -c '<thought>' prompts/cloud_reasoning_first.txt`.
10. **`prompts/cloud_code_first.txt`** — §3.2 with blocks swapped. Verify: `head -1 prompts/cloud_code_first.txt | grep -q action`.
11. **`prompts/recovery.txt` + `prompts/device_solve.txt`** — verbatim §3.3 (two files, two micro-steps, same interfaces: `{task}`, `{message}` placeholders). Verify: `grep -q '{task}' prompts/device_solve.txt && grep -q '{message}' prompts/recovery.txt`.
12. **`src/clients/cloud_client.py` + `src/clients/device_client.py`** — `CloudResult(text, prompt_tokens, completion_tokens, latency_s)`; `CloudClient.complete(system, user, temperature, seed, max_tokens)`; `DeviceResult(content, thinking, usage…)`; `DeviceClient.complete(task, message)`; factories `make_cloud_client(cfg)/make_device_client(cfg)` reading `cfg.cloud.kind`/`cfg.device.kind` (`real`|`mock`). Reasoning-parser: use `content` only. Verify: `.venv/bin/python -c "from src.clients.cloud_client import make_cloud_client"`.
13. **`src/clients/mock_clients.py`** — `MockCloudClient` returns a well-formed 2-part message (correct-ish code from the problem's canonical pattern, plus trailing commentary so the control cut has something to remove); `MockDeviceClient` returns either fixed correct code (recovery of near-intact messages) or obviously corrupted code (configurable). Deterministic, no network. Verify: `.venv/bin/python -c "from src.clients.mock_clients import MockCloudClient"`.
14. **`src/generate.py`** — CLI `python -m src.generate --config ... [--force]`: writes cache JSONs (§4) via the configured cloud client, resumable. Verify (offline): `.venv/bin/python -m src.generate --config configs/pilot.yaml --limit 2 && ls results/generations | head`.
15. **`src/prescreen.py`** — CLI `python -m src.prescreen --config ...`: device-alone score + full-message score, writes `results/prescreen.json`, resumable. Verify (offline with mocks): `.venv/bin/python -m src.prescreen --config configs/pilot.yaml --limit 3`.
16. **`src/run_experiments.py`** — CLI `python -m src.run_experiments --config ... [--conditions substr] [--limit N]`: enumerates conditions, loads cached messages, applies cuts, runs recovery if `condition.recovery`, executes `final_code` via `scoring.score`, appends JSONL (§6), resumable by `run_id`. **⇐ PILOT-READY OFFLINE MARKER.** Verify (offline smoke, must complete green): `.venv/bin/python -m src.run_experiments --config configs/pilot.yaml --limit 3 && wc -l results/runs/*/runs.jsonl`.
17. **`tests/test_all.py`** — pytest, no network: message parse on truncated inputs (cut mid-tag, no closing tag, fence remnants); cut fractions land where expected; conditions count == 32; scoring classifies syntax/runtime/wrong/timeout from injected samples; mock clients well-formed. Verify: `.venv/bin/python -m pytest tests -q`.
18. **`src/analyze.py`** — CLI `python -m src.analyze --config ...`: emits all §7 artifacts (CSV + one PNG, Agg backend) incl. `pilot_variance.json` (§5.3). Verify: `.venv/bin/python -m src.analyze --config configs/pilot.yaml && ls results/analysis`.
19. **`configs/full.yaml`** — copy of pilot with `problem_limit: 164`, `seeds: [0,1,2,3,4]`, `configs/full.yaml` verified like step 3.

After step 19 the human takes over: create the venv, set API-key env vars, run `prescreen`, `generate`, `run_experiments`, `analyze` for real.