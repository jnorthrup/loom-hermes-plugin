import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from agent.context_engine import ContextSelectionRejected


def engine(loom, home, session="a"):
    e = loom.LoomEngine(model="test/model", loom_budget={"model": "test/model", "tokenizer": "utf8-byte-fallback", "prompt_overhead": 256, "message_overhead": 64, "tool_overhead": 128})
    e.context_length = 8192
    e.on_session_start(session, hermes_home=str(home))
    return e


def push(e, text):
    assert json.loads(e.handle_tool_call("loom", {"action": "push", "text": text}))["ok"]


def selected(e, budget=8192, output=512, tools=()):
    return e.select_context([{"role": "system", "content": "System"}, {"role": "user", "content": "Question"}], budget_tokens=budget, max_output_tokens=output, tool_schemas=list(tools))


def test_session_reuse_starts_empty_and_slash_cannot_disclose(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "private A")
    assert "private A" not in loom._loom_command()
    e.on_session_start("b", hermes_home=str(tmp_path))
    assert e.stack.frames == []
    e.on_session_start("a", hermes_home=str(tmp_path))
    assert e.stack.frames == [{"label": "", "text": "private A"}]


def test_slash_shows_only_the_cli_agents_own_bound_session(loom, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.plugins import get_plugin_manager
    e = engine(loom, tmp_path)
    push(e, "private A")
    manager = get_plugin_manager()
    monkeypatch.setattr(manager, "_cli_ref", None, raising=False)  # TUI and gateways
    assert "cannot identify" in loom._loom_command()
    agent = SimpleNamespace(session_id="a", context_compressor=e)
    monkeypatch.setattr(manager, "_cli_ref", SimpleNamespace(agent=agent))
    assert "private A" in loom._loom_command()
    agent.session_id = "b"  # the CLI moved on before its engine did
    assert "private A" not in loom._loom_command()
    e.on_session_start("b", hermes_home=str(tmp_path))
    assert loom._loom_command() == "(no frames)"
    agent.session_id = ""
    assert "no frame stack" in loom._loom_command()
    agent.session_id, agent.context_compressor = "b", object()  # another engine is active
    assert "no frame stack" in loom._loom_command()
    monkeypatch.setattr(manager, "_cli_ref", SimpleNamespace(agent=None))  # before the first turn
    assert "no frame stack" in loom._loom_command()


def test_slash_reads_only_the_session_the_host_binds(loom, tmp_path, monkeypatch):
    import contextvars
    from gateway import session_context
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(get_plugin_manager(), "_cli_ref", None, raising=False)
    monkeypatch.setattr(session_context, "_session_context_engaged", session_context._session_context_engaged)
    a, b = engine(loom, tmp_path / "one", "a"), engine(loom, tmp_path / "two", "b")
    push(a, "private A")
    push(b, "private B")

    def run(**bound):  # one TUI request: a fresh context, bound and cleared like _run_plugin_command
        def call():
            tokens = session_context.set_session_vars(**bound) if bound else []
            try:
                return loom._loom_command()
            finally:
                if tokens:
                    session_context.clear_session_vars(tokens)
        return contextvars.Context().run(call)

    out = run(session_key="tui:1", session_id="a")
    assert "private A" in out and "private B" not in out
    assert "private B" in run(session_key="tui:2", session_id="b")
    assert "cannot identify" in run(session_key="agent:main:telegram:dm:1")  # messaging gateway: no id
    assert "cannot identify" in run(session_id="a")  # an id without a key is not this command's binding
    monkeypatch.setenv("HERMES_SESSION_ID", "a")  # os.environ names whichever agent started last
    assert "cannot identify" in run()
    assert "no frame stack" in run(session_key="tui:3", session_id="c")
    b.on_session_start("b2", hermes_home=str(tmp_path / "two"))
    assert "no frame stack" in run(session_key="tui:2", session_id="b")
    assert run(session_key="tui:2", session_id="b2") == "(no frames)"


def test_collision_free_sessions_and_profiles_concurrently(loom, tmp_path):
    a = engine(loom, tmp_path / "one", "a/b")
    b = engine(loom, tmp_path / "one", "a_b")
    c = engine(loom, tmp_path / "two", "a/b")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: push(*pair), [(a, "private A"), (b, "private B"), (c, "private C")]))
    assert a._state_path() != b._state_path()
    for e, text in [(a, "private A"), (b, "private B"), (c, "private C")]:
        restored = engine(loom, Path(e._loom_home), e._loom_session)
        assert restored.stack.frames == [{"label": "", "text": text}]
        assert text in str(selected(restored))


def test_reset_is_durable(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "must not return")
    e.on_session_reset()
    assert engine(loom, tmp_path).stack.frames == []


@pytest.mark.parametrize("frames", [[{"text": "x" * 4001, "label": ""}], [{"text": "x", "label": ""}] * 65, [{"text": "x", "label": "bad label"}], [{"text": 5, "label": ""}]])
def test_persisted_bounds_refuse_without_truncation(loom, frames):
    with pytest.raises(ValueError):
        loom.LoomStack.from_dict({"frames": frames})


def test_malformed_state_clears_prior_memory_and_blocks_request(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "private A")
    Path(e._state_path()).write_text('{"frames":')
    e.on_session_start("a", hermes_home=str(tmp_path))
    assert e.stack.frames == []
    with pytest.raises(ContextSelectionRejected):
        selected(e)
    assert "private A" not in e.handle_tool_call("loom", {"action": "show"})


def test_save_failure_does_not_publish_uncommitted_mutation(loom, tmp_path, monkeypatch):
    e = engine(loom, tmp_path)
    push(e, "durable")
    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")
    monkeypatch.setattr(loom.os, "replace", fail)
    result = json.loads(e.handle_tool_call("loom", {"action": "push", "text": "not durable"}))
    assert "error" in result
    assert e.stack.frames == [{"label": "", "text": "durable"}]
    assert engine(loom, tmp_path).stack.frames == e.stack.frames


def test_explicit_inheritance_is_copy_and_cross_profile_refuses(loom, tmp_path):
    parent = engine(loom, tmp_path / "one", "parent")
    push(parent, "required parent fact")
    child = engine(loom, tmp_path / "one", "child")
    assert child.stack.frames == []
    child.inherit_from_parent(parent)
    assert child.stack.frames == parent.stack.frames
    child.handle_tool_call("loom", {"action": "pop"})
    assert len(parent.stack.frames) == 1
    assert engine(loom, tmp_path / "one", "child").stack.frames == []
    wrong_profile = engine(loom, tmp_path / "two", "child")
    with pytest.raises(ContextSelectionRejected):
        wrong_profile.inherit_from_parent(parent)
    assert wrong_profile.stack.frames == []


def test_budget_counts_history_frames_tools_and_output_without_mutating(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "required fact " * 200)
    before = e.stack.to_dict()
    with pytest.raises(ContextSelectionRejected):
        selected(e, budget=256, output=128)
    with pytest.raises(ContextSelectionRejected):
        selected(e, budget=8192, output=8192)
    with pytest.raises(ContextSelectionRejected):
        selected(e, budget=8192, output=None)
    with pytest.raises(ContextSelectionRejected):
        selected(e, tools=[{"description": "x" * 40000}])
    assert e.stack.to_dict() == before
    assert "required fact" in str(selected(e))


def test_missing_system_refuses_required_frames(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "required fact")
    with pytest.raises(ContextSelectionRejected):
        e.select_context([{"role": "user", "content": "question"}], budget_tokens=8192, max_output_tokens=512, tool_schemas=[])


@pytest.mark.parametrize("node", [{"text": "x" * 4001}, {"text": "x", "label": "bad label"}, {"text": ["x"]}])
def test_precook_uses_same_frame_validation(precook, node):
    with pytest.raises(ValueError):
        precook.load_script({"outline": [node]})


def test_precook_rejects_depth_65_and_preserves_valid_render(precook):
    node = {"text": "leaf"}
    for _ in range(64):
        node = {"text": "outer", "children": [node]}
    with pytest.raises(ValueError):
        precook.load_script({"outline": [node]})
    script = {"system": "System", "goal": "Goal", "outline": [{"text": "Fact", "label": "outer"}]}
    runner = precook.Loom(script, dry_run=True)
    with runner.frame(runner.outline[0]):
        assert runner.messages(runner.outline[0])[0]["content"] == "System\n\n[loom: nested work frames, outermost first; the last frame is the current loop]\n== goal ==\nGoal\n== 1 outer ==\nFact\n[/loom]"


def test_legacy_collision_state_requires_explicit_reset(loom, tmp_path):
    folder = tmp_path / "loom"
    folder.mkdir()
    legacy = folder / "a_b.json"
    legacy.write_text(json.dumps({"frames": [{"text": "ambiguous owner", "label": ""}]}))
    e = engine(loom, tmp_path, "a/b")
    with pytest.raises(ContextSelectionRejected):
        selected(e)
    e.on_session_reset()
    assert engine(loom, tmp_path, "a/b").stack.frames == []
    assert legacy.exists(), "reset must not delete an ambiguously owned legacy file"


@pytest.mark.parametrize("fault", ["identity", "duplicate", "oversize", "symlink"])
def test_corrupt_or_redirected_state_never_loads(loom, tmp_path, fault):
    e = engine(loom, tmp_path)
    push(e, "private")
    path = Path(e._state_path())
    if fault == "identity":
        data = json.loads(path.read_text())
        data["session"] = "different"
        path.write_text(json.dumps(data))
    elif fault == "duplicate":
        path.write_text('{"frames":[],"frames":[]}')
    elif fault == "oversize":
        path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    else:
        target = tmp_path / "other.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    restored = engine(loom, tmp_path)
    assert restored.stack.frames == []
    with pytest.raises(ContextSelectionRejected):
        selected(restored)


def test_compression_rotation_requires_explicit_matching_lineage(loom, tmp_path):
    e = engine(loom, tmp_path, "old")
    push(e, "required across compaction")
    e.on_session_start("new", hermes_home=str(tmp_path), boundary_reason="compression", old_session_id="old")
    assert engine(loom, tmp_path, "new").stack.frames == e.stack.frames
    e.on_session_start("wrong", hermes_home=str(tmp_path), boundary_reason="compression", old_session_id="unrelated")
    with pytest.raises(ContextSelectionRejected):
        selected(e)


def test_clone_does_not_copy_session_state_or_lock(loom, tmp_path):
    e = engine(loom, tmp_path)
    push(e, "private")
    child = e.clone_for_agent()
    assert child.stack.frames == []
    assert child._loom_session == ""
    assert child._loom_lock is not e._loom_lock
    assert Path(e._state_path()).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("raw", ['{bad', '{"goal":123}', '{"goal":"required","subgoals":"lost"}', '{"goal":"a","goal":"b"}', ''])
def test_corrupt_goal_refuses_even_without_frames(loom, tmp_path, raw):
    from hermes_cli import goals
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    e = engine(loom, tmp_path)
    token = set_hermes_home_override(e._loom_home)
    try:
        db = goals._get_session_db()
        db.set_meta("goal:a", raw)
    finally:
        reset_hermes_home_override(token)
    with pytest.raises(ContextSelectionRejected):
        selected(e)


def test_goal_read_failure_refuses_but_missing_is_empty(loom, tmp_path, monkeypatch):
    from hermes_cli import goals
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    e = engine(loom, tmp_path)
    assert selected(e) is None
    token = set_hermes_home_override(e._loom_home)
    try:
        db = goals._get_session_db()
    finally:
        reset_hermes_home_override(token)
    def unavailable(*args):
        raise OSError("synthetic DB failure")
    monkeypatch.setattr(db, "get_meta", unavailable)
    with pytest.raises(ContextSelectionRejected):
        selected(e)
