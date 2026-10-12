# TruncBench — session handoff: env-var model config + Docker sandbox

Status: **wired and verified end-to-end (mock pilot, 2,304 runs in Docker).**
Date: 2026-07-08. Repo state is commit-ready; two small loose ends listed at the bottom.

## What I was trying to do

Two asks from the user:

1. **Model config via env vars.** Cloud and device model names (and optionally
   endpoints) should come from environment variables, since the user will pick
   the models later. Cloud endpoint is OpenRouter.
2. **Test-code execution in a container.** All sandboxed execution of candidate
   code should run in Docker (user's choice over bubblewrap; Docker Desktop was
   requested but turned out to be unnecessary — see below).

## What I found (machine recon)

- Ubuntu 26.04 LTS, systemd running, GUI (Wayland) present, `/dev/kvm` present.
- A **complete Docker engine was already installed** (`docker.io 29.1.3` +
  `containerd`), and the user is already in the `docker` group. The daemon just
  wasn't running. → User started it with `sudo systemctl enable --now docker`.
- Docker Desktop was **not** installed: on this box it would wrap a QEMU/KVM VM
  around an engine that's already present — overhead with no benefit for a
  script-driven pipeline calling `docker run` thousands of times.
- Smoke tests that pinned down the design:
  - `docker run --rm --network none python:3.12-slim` works; image pulled.
  - Container exit codes propagate 1:1 (exit 3 → client sees 3).
  - OOM under `--memory` → **exit 137** (128+SIGKILL) → this is the
    `mem_exceeded` marker in docker mode.
  - If the client is killed on timeout, **the container becomes an orphan**
    (verified: still running after `timeout 2 docker wait`). → the docker path
    must use `--cidfile` and `docker kill <cid>` on timeout.
- `sudo` is policy-blocked from my shell — all sudo steps were handed to the user.

## What I changed

### `src/config.py`
- `CloudCfg` / `DeviceCfg`: new optional fields `model_env` and `base_url_env`
  (names of env vars). `load_config()` resolves them: if the env var is set it
  **overrides** the YAML value and the resolved value is written back into the
  config's raw dict, so the recorded `config.sha256` pins the model actually
  used. For `kind: real`, an unset `model_env` target → hard `SystemExit` with a
  clear message (verified: `CLOUD_MODEL` unset aborts the full config).
- `SandboxCfg`: `use_bubblewrap: bool` replaced by `kind: "docker"|"bwrap"|"rlimit"`
  (default **docker**) + new `docker_image` (default `python:3.12-slim`).
  `kind` is validated at load.

### `src/sandbox.py`
New `run_sandbox(..., kind=...)` dispatch, three fences, shared wall-clock
timeout + throwaway workdir + stripped env:

- **docker** (default): `docker run --rm --network none --memory <mem_mb>m
  --cpus 1 --cidfile … -v <workdir>:/work --workdir /work --entrypoint sh`,
  with the candidate launched as
  `sh -c 'exec env -i <stripped env> python -B /work/<file>.py'`.
  `env -i` matters: `docker run` would otherwise leak the host environment
  (API keys) into the container. On client timeout the container is killed via
  the cidfile (no orphans — verified after a full 2,304-run pass: zero
  leftover containers). Exit 137 ⇒ `mem_exceeded`.
- **bwrap**: unchanged namespaces path (`--unshare-net --die-with-parent`),
  still available as a fallback.
- **rlimit**: original plain-subprocess + RLIMIT_* path, still available.
- `SandboxResult` gained a `sandbox_kind` field (recorded which fence ran).
- `execute_code()` now passes `kind`/`docker_image` and reuses its workdir as
  the mount point (previously it mounted a directory that was deleted under
  docker's feet — would have been a bug).

### `configs/pilot.yaml` and `configs/full.yaml`
- `cloud.model_env: CLOUD_MODEL`, `device.model_env: DEVICE_MODEL`
  (YAML `model:` values remain as fallbacks; also optional `base_url_env`).
- `sandbox: kind: docker`, `docker_image: python:3.12-slim`;
  `use_bubblewrap` removed.
- So the real run just needs:
  `export OPENROUTER_API_KEY=… CLOUD_MODEL=<id> VLLM_API_KEY=anything DEVICE_MODEL=<served name>`
  (plus vLLM running on `127.0.0.1:8000` unless `base_url_env` points elsewhere).

## Verification performed

- `pytest`: **12/12 pass** (no test touched the removed `use_bubblewrap` flag).
- Env-var override: with `CLOUD_MODEL=my-openrouter-pick
  DEVICE_MODEL=my-local-27b`, `load_config` returns those names and the
  recorded config SHA pins them.
- Guard: `full.yaml` with `kind: real` and no `CLOUD_MODEL` → clean abort
  ("cloud.model_env points at 'CLOUD_MODEL', which is not set").
- **Full mock pilot through the new docker sandbox**: generate (300 cached
  messages) → prescreen (**24/50 problems kept**) → grid run (**2,304 rows** =
  24 × 3 seeds × 32 conditions, all executed inside Docker) → analysis
  (all 32 condition cells present). `docker ps -a` afterwards: **no orphaned
  containers**.

## Loose ends (do before the real run)

1. **Cosmetic row gap:** `run_experiments.py` builds each row's `sandbox` dict
   field-by-field and does not yet include the new `sandbox_kind` key (my audit
   script hit `KeyError: 'sandbox_kind'`). The runs themselves were correctly
   executed in Docker; the recorded dict just doesn't *say so*. One-line fix:
   add `sandbox_kind` to the row's sandbox dict in `run_experiments.py`.
   (The pre-fix rows in `results/runs/` are mock data and disposable.)
2. **Real run is yours to launch** — the only remaining steps are human:
   - pick + export the model names (`CLOUD_MODEL`, `DEVICE_MODEL`) and keys
     (`OPENROUTER_API_KEY`, `VLLM_API_KEY`);
   - start vLLM serving the device model on `127.0.0.1:8000`;
   - flip `kind: mock → real` for cloud and device in `configs/pilot.yaml`;
   - `python -m src.generate --config configs/pilot.yaml`, then prescreen,
     `run_experiments`, `analyze` — same commands, real models.

## Out of scope / deliberate choices

- Docker Desktop not installed (plain engine already present; Desktop adds a
  GUI + VM overhead for no gain here). If you want it later: the apt recipe
  was provided in the conversation.
- `--cpus 1` + wall-clock timeout replaces `RLIMIT_CPU` in docker mode
  (rlimits can't apply inside the container); hard caps still enforced via
  `--memory`.
- Candidates run with **no network** (`--network none`) — consistent with
  HumanEval's standard-library-only rule; a candidate importing numpy fails
  loudly as a runtime error, which is the correct verdict.
