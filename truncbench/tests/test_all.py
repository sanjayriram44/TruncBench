"""Offline unit tests: no network, no API keys. Run from truncbench/:
   .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.config import load_config            # noqa: E402
from src.dataset import Problem                # noqa: E402
from src.message_format import build, parse, extract_code, code_from_answer  # noqa: E402
from src.truncation import enumerate_conditions, cut_message  # noqa: E402
from src.scoring import score                 # noqa: E402
from src.clients.cloud_client import make_cloud_client       # noqa: E402
from src.clients.device_client import make_device_client     # noqa: E402

CFG = load_config("configs/pilot.yaml")

RF = build("step one\nstep two\nstep three", "def f(x):\n    return x * 2", "reasoning_first")
CF = build("plan here", "def g(x):\n    return x", "code_first")


def test_parse_full_both_orderings():
    for m, order in ((RF, "reasoning_first"), (CF, "code_first")):
        p = parse(m)
        assert (p.thought_status, p.action_status) == ("full", "full"), order


def test_cut_mid_thought_reasoning_first_loses_code():
    p = parse(RF[:RF.index("</thought>")])
    assert p.thought_status == "partial"
    assert p.action is None and p.action_status == "missing"


def test_cut_mid_action_code_first_loses_plan():
    p = parse(CF[:CF.index("</action>")])
    assert p.action_status == "partial"
    assert p.thought_status == "missing"


def test_parse_empty_and_tag_only():
    assert parse("").thought_status == "missing"
    p = parse("<thought>")
    assert p.thought_status == "partial" and p.thought == ""
    assert parse("<action>\n") .action_status == "partial"


def test_extract_code_strips_fences_and_never_raises():
    assert extract_code("```python\ndef f(): pass\n```") == "def f(): pass"
    assert extract_code("def f(): pass") == "def f(): pass"
    assert extract_code(None) == ""


def test_code_from_answer_forms():
    assert code_from_answer("```python\ndef f(): pass\n```") == "def f(): pass"
    assert code_from_answer("<action>\ndef f(): pass\n</action>") == "def f(): pass"
    assert code_from_answer("def f(): pass") == "def f(): pass"
    assert code_from_answer("") == ""


def test_conditions_count_is_32():
    conds = enumerate_conditions(CFG)
    assert len(conds) == 32
    assert len({c.key for c in conds}) == 32


def test_cut_lands_inside_the_named_part():
    start = RF.index("<thought>") + len("<thought>")
    content_len = RF.index("</thought>") - start
    r = cut_message(RF, "reasoning_first", "case_i", 0.5, 512, "o200k_base")
    assert r.index_chars == start + int(0.5 * content_len)
    assert r.removed["partial"] == ["thought"] and r.removed["missing"] == ["action"]


def test_control_cut_loses_nothing_needed():
    r = cut_message(RF, "reasoning_first", "control", 0.5, 512, "o200k_base")
    assert r.removed == {"full": ["thought", "action"], "partial": [], "missing": []}
    assert "# Footnote" in r.message


def _prob():
    return Problem(task_id="Unit/1", prompt="def add(a: int, b: int) -> int:\n    pass",
                   entry_point="add",
                   test_script="\ndef check(candidate):\n    assert candidate(1, 2) == 3\n"
                                "    assert candidate(-1, 1) == 0",
                   plus_input=[])


def test_scoring_verdicts():
    p = _prob()
    assert score(p, "def add(a,b):\n    return a + b", CFG).verdict == "passed"
    r = score(p, "def add(a,b):\n    return a - b", CFG)
    assert (r.verdict, r.failure_type) == ("wrong_answer", "correctness")
    r = score(p, "def add(a,b):\n    return 1/0", CFG)
    assert (r.verdict, r.failure_type) == ("runtime_error", "execution")
    assert score(p, "def add(a,b)(", CFG).verdict == "syntax_error"
    assert score(p, "   ", CFG).verdict == "no_code"


def test_scoring_timeout_path():
    import copy
    cfg2 = copy.copy(CFG)
    cfg2.sandbox = copy.copy(CFG.sandbox)
    cfg2.sandbox.timeout_s = 2
    cfg2.sandbox.cpu_seconds = 30
    r = score(_prob(), "def add(a,b):\n    import time\n    time.sleep(10)\n    return a+b", cfg2)
    assert (r.verdict, r.failure_type) == ("timeout", "execution")


def test_mock_clients_well_formed():
    cloud, dev = make_cloud_client(CFG), make_device_client(CFG)
    sys_rf = open("prompts/cloud_reasoning_first.txt").read()
    res = cloud.complete(sys_rf, "def add(a: int, b: int) -> int:\n    pass", 0.7, 0, 512)
    p = parse(res.text)
    assert (p.thought_status, p.action_status) == ("full", "full")
    rec = dev.complete("sys", "TASK:\nxx\n\nRECEIVED MESSAGE (may be truncated):\n" + res.text)
    assert isinstance(rec.content, str) and rec.content


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
