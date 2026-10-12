"""Cloud model client (OpenAI-compatible API) + factory.

The factory returns the real client or the offline mock depending on
cfg.cloud.kind, so the rest of the pipeline never touches an API directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..config import ExperimentCfg


@dataclass
class CloudResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str = ""


class CloudClient(Protocol):
    def complete(self, system: str, user: str, temperature: float,
                 seed: int | None, max_tokens: int) -> CloudResult: ...


class OpenAICompatCloud:
    """Any OpenAI-compatible /chat/completions endpoint (OpenRouter etc.)."""

    def __init__(self, cfg: ExperimentCfg):
        from openai import OpenAI  # imported lazily; not needed for mocks
        key = cfg.api_key(cfg.cloud.api_key_env)
        self._client = OpenAI(base_url=cfg.cloud.base_url, api_key=key)
        self._model = cfg.cloud.model

    def complete(self, system: str, user: str, temperature: float,
                 seed: int | None, max_tokens: int) -> CloudResult:
        t0 = time.monotonic()
        kwargs = dict(model=self._model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      temperature=temperature, max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        resp = self._client.chat.completions.create(**kwargs)
        dt = time.monotonic() - t0
        usage = resp.usage
        return CloudResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=dt, model=getattr(resp, "model", self._model))


def make_cloud_client(cfg: ExperimentCfg) -> CloudClient:
    if cfg.cloud.kind == "mock":
        from .mock_clients import MockCloudClient
        return MockCloudClient(cfg)
    if cfg.cloud.kind == "real":
        return OpenAICompatCloud(cfg)
    raise ValueError(f"unknown cloud kind {cfg.cloud.kind!r}")
