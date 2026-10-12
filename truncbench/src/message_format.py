"""Two-part message format: <thought>...</thought><action>...</action>.

The parser is truncation-tolerant by design: it NEVER raises, and every
part is reported as "full", "partial" (opened, not closed) or "missing"
(opening tag absent).
"""
from __future__ import annotations

from dataclasses import dataclass

THOUGHT_OPEN = "<thought>"
THOUGHT_CLOSE = "</thought>"
ACTION_OPEN = "<action>"
ACTION_CLOSE = "</action>"

ORDER_REASONING_FIRST = "reasoning_first"
ORDER_CODE_FIRST = "code_first"


@dataclass
class ParsedMessage:
    raw: str = ""
    thought: str | None = None
    action: str | None = None
    # each of: "full" (own closing tag seen before the other part started),
    # "partial" (opened; cut at end of message or by the other part's tag),
    # "missing" (opening tag absent)
    thought_status: str = "missing"
    action_status: str = "missing"


def build(thought: str, action: str, order: str) -> str:
    t = f"{THOUGHT_OPEN}\n{thought.strip()}\n{THOUGHT_CLOSE}"
    a = f"{ACTION_OPEN}\n{action.strip()}\n{ACTION_CLOSE}"
    if order == ORDER_REASONING_FIRST:
        return f"{t}\n{a}"
    if order == ORDER_CODE_FIRST:
        return f"{a}\n{t}"
    raise ValueError(f"unknown ordering {order!r}")


def _parse_part(message: str, open_tag: str, close_tag: str,
                other_open: str, start: int) -> tuple[str | None, str]:
    idx = message.find(open_tag, start)
    if idx == -1:
        return None, "missing"
    content_start = idx + len(open_tag)
    close_idx = message.find(close_tag, content_start)
    other_idx = message.find(other_open, content_start)
    boundaries = [i for i in (close_idx, other_idx) if i != -1]
    if not boundaries:
        # No closing tag and no later opening tag: cut at end of message.
        return message[content_start:], "partial"
    status = ("full" if close_idx != -1 and (other_idx == -1 or close_idx <= other_idx)
              else "partial")
    return message[content_start:min(boundaries)], status


def parse(message: str) -> ParsedMessage:
    if not message:
        return ParsedMessage(raw="")
    t, t_status = _parse_part(message, THOUGHT_OPEN, THOUGHT_CLOSE, ACTION_OPEN, 0)
    a_idx = message.find(ACTION_OPEN)
    a, a_status = (None, "missing") if a_idx == -1 else \
        _parse_part(message, ACTION_OPEN, ACTION_CLOSE, THOUGHT_OPEN, a_idx)
    return ParsedMessage(raw=message, thought=t, action=a,
                         thought_status=t_status, action_status=a_status)


_FENCE_HEAD = ("```", "```python", "```py")


def extract_code(action: str | None) -> str:
    """Return runnable code text from a (possibly truncated) action part.

    Strips at most one markdown fence pair if the model added it. Never
    raises; returns "" when no action text arrived.
    """
    if action is None:
        return ""
    lines = action.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    stripped = list(lines)
    if stripped[0].strip() in _FENCE_HEAD:
        stripped = stripped[1:]
        if stripped and stripped[-1].strip() == "```":
            stripped.pop()
    return "\n".join(stripped)


def code_from_answer(content: str | None) -> str:
    """Best-effort code extraction from a raw device-model answer.

    Handles (in order): a two-part message (action tag), a fenced code
    block (possibly with a language tag), or bare code. Never raises;
    returns "" when nothing usable is present.
    """
    if not content or not content.strip():
        return ""
    s = content.strip()
    if "<action>" in s:
        return extract_code(parse(s).action)
    if "```" in s:
        blocks = [b for b in s.split("```")[1::2]]
        if blocks:
            b = max(blocks, key=len)
            b = b.lstrip()
            if b.startswith("python") or b.startswith("py"):
                nl = b.find("\n")
                if nl != -1:
                    b = b[nl + 1:]
            return b.rstrip("\n")
    return s
