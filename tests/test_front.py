"""The front stays byte-identical across recorded push/pop and catches up only when the
conversation behind it was rewritten, so prefix caches keep the whole conversation."""
import json

import pytest
from agent.context_engine import ContextSelectionRejected


def engine(loom, home, session="a"):
    e = loom.LoomEngine(model="test/model", loom_budget={"model": "test/model", "tokenizer": "utf8-byte-fallback", "prompt_overhead": 256, "message_overhead": 64, "tool_overhead": 128})
    e.context_length = 32768
    e.on_session_start(session, hermes_home=str(home))
    return e


def loom_call(e, history, call_id, **args):
    """The model calls the loom tool; the host records the call and its result."""
    result = e.handle_tool_call("loom", args)
    assert json.loads(result)["ok"]
    history += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "loom", "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def front(e, history):
    """The system content the provider receives for this conversation."""
    rows = e.select_context([{"role": "system", "content": "System"}] + history, conversation_messages=[dict(m) for m in history],
                            budget_tokens=32768, max_output_tokens=512, tool_schemas=[])
    return "System" if rows is None else rows[0]["content"]


def test_recorded_push_and_pop_keep_the_front_byte_identical(loom, tmp_path):
    e = engine(loom, tmp_path)
    history = [{"role": "user", "content": "start"}]
    loom_call(e, history, "c1", action="push", text="outer: ship the release", label="outer")
    first = front(e, history)
    assert "outer: ship the release" in first
    for n, args in enumerate([{"action": "push", "text": "inner: build", "label": "build"},
                              {"action": "pop"},
                              {"action": "push", "text": "inner: test", "label": "test"}]):
        loom_call(e, history, f"c{n + 2}", **args)
        history.append({"role": "user", "content": f"turn {n}"})
        assert front(e, history) == first
    assert [frame["label"] for frame in e.stack.frames] == ["outer", "test"]


def test_unrecorded_stack_change_moves_the_front(loom, tmp_path):
    e = engine(loom, tmp_path)
    history = [{"role": "user", "content": "start"}]
    assert front(e, history) == "System"
    assert json.loads(e.handle_tool_call("loom", {"action": "push", "text": "never recorded"}))["ok"]
    assert "never recorded" in front(e, history)


@pytest.mark.parametrize("rewrite", ["compaction", "undo"])
def test_rewritten_conversation_moves_the_front_to_the_current_stack(loom, tmp_path, rewrite):
    e = engine(loom, tmp_path)
    history = [{"role": "user", "content": "start"}]
    loom_call(e, history, "c1", action="push", text="outer frame", label="outer")
    before = front(e, history)
    loom_call(e, history, "c2", action="push", text="inner frame", label="inner")
    assert front(e, history) == before and "inner frame" not in before
    rewritten = [{"role": "user", "content": "summary of earlier turns"}] if rewrite == "compaction" else history[:3]
    after = front(e, rewritten)
    assert "outer frame" in after and "inner frame" in after
    rewritten.append({"role": "assistant", "content": "continuing"})
    assert front(e, rewritten) == after


def test_goal_change_moves_the_front(loom, tmp_path, monkeypatch):
    goal = {"text": "first goal"}
    monkeypatch.setattr(loom, "_goal_block", lambda session, home: goal["text"])
    e = engine(loom, tmp_path)
    history = [{"role": "user", "content": "start"}]
    assert "first goal" in front(e, history)
    goal["text"] = "second goal"
    history.append({"role": "assistant", "content": "ok"})
    after = front(e, history)
    assert "second goal" in after and "first goal" not in after


def test_reset_and_session_switch_move_the_front(loom, tmp_path):
    e = engine(loom, tmp_path)
    history = [{"role": "user", "content": "start"}]
    loom_call(e, history, "c1", action="push", text="private A")
    assert "private A" in front(e, history)
    e.on_session_start("b", hermes_home=str(tmp_path))
    assert front(e, history) == "System"
    e.on_session_start("a", hermes_home=str(tmp_path))
    assert "private A" in front(e, history)
    e.on_session_reset()
    assert front(e, history) == "System"


def test_final_guard_checks_the_block_that_was_sent(loom, tmp_path):
    e = engine(loom, tmp_path)
    history = [{"role": "user", "content": "start"}]
    loom_call(e, history, "c1", action="push", text="outer frame")
    front(e, history)
    loom_call(e, history, "c2", action="push", text="inner frame")
    rows = e.select_context([{"role": "system", "content": "System"}] + history, conversation_messages=list(history), request_validation=True)
    assert "outer frame" in rows[0]["content"] and "inner frame" not in rows[0]["content"]
    e.validate_request({"messages": rows, "max_tokens": 512, "tools": []}, budget_tokens=32768, api_mode="chat_completions")
    stripped = {"messages": [{"role": "system", "content": "System"}] + history, "max_tokens": 512, "tools": []}
    with pytest.raises(ContextSelectionRejected):
        e.validate_request(stripped, budget_tokens=32768, api_mode="chat_completions")
