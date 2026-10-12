"""On-device model client (local vLLM, OpenAI-compatible) + factory.

The endpoint serves a reasoning-model chat template: extended thinking
arrives in a separate field (e.g. `reasoning_content` / `reasoning`).
Per the research design we consume ONLY the final answer content.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..config import ExperimentCfg


@dataclass
class DeviceResult:
    content: str            # final answer only (never the thinking text)
    thinking: str           # separate reasoning field, kept for logging only
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str = ""


class DeviceClient(Protocol):
    def complete(self, system: str, user: str,
                 temperature: float = 0.0) -> DeviceResult: ...


class VLLMDevice:
    def __init__(self, cfg: ExperimentCfg):
        from openai import OpenAI  # lazy import; not needed for mocks
        self._client = OpenAI(base_url=cfg.device.base_url,
                               api_key=cfg.api_key(cfg.device.api_key_env) or "local")
        self._model = cfg.device.model
        self._max_tokens = cfg.device.max_tokens

    def complete(self, system: str, user: str,
                 temperature: float = 0.0) -> DeviceResult:
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=self._max_tokens)
        dt = time.monotonic() - t0
        msg = resp.choices[0].message
        content = msg.content or ""
        thinking = getattr(msg, "reasoning_content", None) \
            or getattr(msg, "reasoning", None) or ""
        usage = resp.usage
        return DeviceResult(
            content=content,
            thinking=thinking,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=dt, model=getattr(resp, "model", self._model))


def make_device_client(cfg: ExperimentCfg) -> DeviceClient:
    if cfg.device.kind == "mock":
        from .mock_clients import MockDeviceClient
        return MockDeviceClient(cfg)
    if cfg.device.kind == "real":
        return VLLMDevice(cfg)
    raise ValueError(f"unknown device kind {cfg.device.kind!r}")
