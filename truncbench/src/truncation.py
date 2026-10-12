"""Truncation engine and condition enumeration.

Cut positions are defined relative to a PART of the message, never to the
whole message: case_i cuts at frac*len(thought), case_ii at frac*len(action).
The control cut removes the same budget from a synthetic footnote appended
after the message, so nothing needed is lost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import ExperimentCfg
from .message_format import (ACTION_CLOSE, ACTION_OPEN, THOUGHT_CLOSE,
                             THOUGHT_OPEN, parse)

TRUNC_NONE = "none"
TRUNC_CASE_I = "case_i"       # reasoning (thought) truncated
TRUNC_CASE_II = "case_ii"      # code (action) truncated
TRUNC_CONTROL = "control"

PARTS = ("thought", "action")


@dataclass(frozen=True)
class Condition:
    order: str
    truncation: str          # none | case_i | case_ii | control
    cut_fraction: float | None
    recovery: bool

    @property
    def key(self) -> str:
        frac = "" if self.cut_fraction is None else f"{self.cut_fraction:g}"
        rec = "rec" if self.recovery else "norec"
        return f"{self.order}|{self.truncation}|{frac}|{rec}"


def enumerate_conditions(cfg: ExperimentCfg) -> list[Condition]:
    conds: list[Condition] = []
    for order in cfg.run.orderings:
        for recovery in (False, True):
            conds.append(Condition(order, TRUNC_NONE, None, recovery))
            conds.append(Condition(order, TRUNC_CONTROL, 0.5, recovery))
            for case in (TRUNC_CASE_I, TRUNC_CASE_II):
                for frac in cfg.run.cut_fractions:
                    conds.append(Condition(order, case, frac, recovery))
    return conds


@dataclass
class CutResult:
    message: str                       # the (possibly cut) message to send
    cut_part: str | None               # thought | action | tail | None
    index_chars: int
    index_tokens: int
    frac_of_whole: float
    removed: dict = field(default_factory=dict)   # {"full":..,"partial":..,"missing":..}


def _span(message: str, open_tag: str, close_tag: str, other_open: str):
    """Return (content_start, content_end) for a part, None if missing."""
    start = message.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    close = message.find(close_tag, start)
    other = message.find(other_open, start)
    bounds = [i for i in (close, other) if i != -1]
    return start, min(bounds) if bounds else len(message)


def cut_message(message: str, order: str, truncation: str,
                frac: float, tail_budget: int, encoding: str) -> CutResult:
    if truncation == TRUNC_NONE:
        pr = parse(message)
        return CutResult(message, None, len(message), _tok(message, encoding),
                          1.0, _bookkeep(pr))
    if truncation == TRUNC_CONTROL:
        footnote = _footnote(tail_budget)
        ext = message + footnote
        cut = len(message) + max(1, int(tail_budget * frac))
        pr = parse(ext[:cut])
        return CutResult(ext[:cut], "tail", cut, _tok(ext[:cut], encoding),
                          cut / len(ext), _bookkeep(pr))
    target = "thought" if truncation == TRUNC_CASE_I else "action"
    open_tag = THOUGHT_OPEN if target == "thought" else ACTION_OPEN
    close_tag = THOUGHT_CLOSE if target == "thought" else ACTION_CLOSE
    other_open = ACTION_OPEN if target == "thought" else THOUGHT_OPEN
    span = _span(message, open_tag, close_tag, other_open)
    if span is None:  # part never present: degenerate, cut the whole message
        pr = parse("")
        return CutResult("", target, 0, 0, 0.0, _bookkeep(pr))
    content_start, content_end = span
    content_len = max(1, content_end - content_start)
    cut = content_start + int(math.floor(frac * content_len))
    cut = max(cut, content_start)
    pr = parse(message[:cut])
    return CutResult(message[:cut], target, cut, _tok(message[:cut], encoding),
                     cut / len(message), _bookkeep(pr))


def _footnote(budget: int) -> str:
    body = "padding padding padding " * 64
    return f"\n# Footnote: {body[:budget]}"


def _tok(text: str, encoding: str) -> int:
    import tiktoken
    try:
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _bookkeep(truncated: "object") -> dict:
    """For each part, what the device actually received: which part was
    partially cut, which part is missing entirely, which arrived full."""
    out = {"full": [], "partial": [], "missing": []}
    for part in PARTS:
        status = getattr(truncated, f"{part}_status")
        out["full" if status == "full" else status].append(part)
    return out
