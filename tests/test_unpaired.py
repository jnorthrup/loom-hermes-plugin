"""Hosts without the paired seam call select_context with only the context window."""
import json

import pytest
from agent.context_engine import ContextSelectionRejected

REQUEST = [{"role": "system", "content": "System"}, {"role": "user", "content": "Question"}]


def engine(loom, home):
    e = loom.LoomEngine(model="test/model")
    e.context_length = 8192
    e.on_session_start("s", hermes_home=str(home))
    return e


def push(e, text):
    assert json.loads(e.handle_tool_call("loom", {"action": "push", "text": text}))["ok"]


def test_unpaired_host_sends_frames_with_only_a_context_window(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "Ship the release")
    out = e.select_context(list(REQUEST), budget_tokens=8192)
    assert "Ship the release" in out[0]["content"] and out[1] == REQUEST[1]
    assert REQUEST[0]["content"] == "System"  # the caller's list is not mutated


def test_unpaired_host_needs_no_budget_contract_and_sends_nothing_without_frames(loom, tmp_path):
    assert engine(loom, tmp_path).select_context(list(REQUEST), budget_tokens=8192) is None


def test_unpaired_host_refuses_frames_over_a_tenth_of_the_window(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "x" * 1000)
    with pytest.raises(ContextSelectionRejected):
        e.select_context(list(REQUEST), budget_tokens=8192)


def test_unpaired_host_without_a_window_refuses(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "frame")
    with pytest.raises(ContextSelectionRejected):
        e.select_context(list(REQUEST), budget_tokens=0)
