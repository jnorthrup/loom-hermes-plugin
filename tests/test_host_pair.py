"""Real Hermes discovery/delegation and final send boundary; all provider calls are fakes."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextSelectionRejected


@pytest.fixture
def agent_factory(tmp_path, monkeypatch):
    from run_agent import AIAgent
    home = tmp_path / "profile"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "loom").symlink_to(Path(__file__).resolve().parents[1], target_is_directory=True)
    (home / "config.yaml").write_text('''context:
  engine: loom
  loom:
    model: test/model
    tokenizer: utf8-byte-fallback
    prompt_overhead: 256
    message_overhead: 64
    tool_overhead: 128
''')
    monkeypatch.setenv("HERMES_HOME", str(home))
    agents = []
    def make(session):
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch("agent.model_metadata.get_model_context_length", return_value=256_000),
            patch("agent.context_compressor.get_model_context_length", return_value=256_000),
        ):
            agent = AIAgent(api_key="synthetic", base_url="https://example.invalid/v1", model="test/model",
                            api_mode="chat_completions", quiet_mode=True, skip_context_files=True, skip_memory=True,
                            max_iterations=3, max_tokens=512, session_id=session, save_trajectories=False)
        assert agent.context_compressor.name == "loom", "fixture must use actual plugin discovery"
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Synthetic answer", reasoning_content=None,
                       reasoning=None, tool_calls=None), finish_reason="stop")], model="test/model", usage=None)
        agent._cached_system_prompt = "System"
        agent._use_prompt_caching = False
        agent._disable_streaming = True
        agent.compression_enabled = False
        agents.append(agent)
        return agent
    yield make
    for agent in agents:
        agent.close()


def push(agent, text):
    assert json.loads(agent.context_compressor.handle_tool_call("loom", {"action": "push", "text": text}))["ok"]


@pytest.mark.parametrize("fault", ["output", "tools", "system", "image", "mode", "contract", None])
def test_real_plugin_final_guard(agent_factory, fault):
    agent = agent_factory("send")
    push(agent, "Required invariant")
    engine = agent.context_compressor
    engine.context_length = 8192
    if fault == "contract":
        engine._loom_budget = None
    def build(messages, **kwargs):
        payload = {"model": "test/model", "messages": messages, "max_tokens": 512, "tools": []}
        if fault == "output":
            payload["max_tokens"] = 8192
        if fault == "tools":
            payload["tools"] = [{"type": "function", "function": {"name": "real_wire_tool", "description": "x" * 9000}}]
        if fault == "system":
            payload["messages"] = [{"role": "system", "content": "replaced"}] + messages[1:]
        if fault == "image":
            payload["messages"] = messages + [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}]}]
        return payload
    if fault == "mode":
        agent.api_mode = "anthropic_messages"
    with patch.object(agent, "_build_api_kwargs", side_effect=build), patch.object(agent, "_try_activate_fallback") as fallback:
        result = agent.run_conversation("Keep this accepted message")
    if fault:
        assert result["failure_reason"] == "context_selection_rejected"
        assert result["failure_retryable"] is False
        assert result["api_calls"] == agent._api_call_count == 0
        agent.client.chat.completions.create.assert_not_called()
        fallback.assert_not_called()
    else:
        agent.client.chat.completions.create.assert_called_once()
        sent = agent.client.chat.completions.create.call_args.kwargs
        assert "Required invariant" in str(sent)
    assert any(row.get("content") == "Keep this accepted message" for row in result["messages"])


def test_discovery_sessions_a_b_a_and_profile_goal_binding(agent_factory, tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli import goals
    a = agent_factory("a")
    b = agent_factory("b")
    push(a, "A only")
    push(b, "B only")
    visited = []
    def load(session, *, strict=False):
        assert strict
        visited.append((str(get_hermes_home().resolve()), session))
        return None
    monkeypatch.setattr(goals, "load_goal", load)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    for current, own, other in [(a, "A only", "B only"), (b, "B only", "A only"), (a, "A only", "B only")]:
        e = current.context_compressor
        rows = e.select_context([{"role": "system", "content": "System"}], budget_tokens=8192, max_output_tokens=512, tool_schemas=[])
        assert own in str(rows) and other not in str(rows)
        assert visited[-1] == (e._loom_home, current.session_id)
    assert str(get_hermes_home()) == str(tmp_path / "other")


def test_actual_delegate_builder_explicit_inheritance(agent_factory, monkeypatch):
    from tools import delegate_tool as dt
    parent = agent_factory("parent")
    push(parent, "Parent required fact")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        child = dt._build_child_agent(task_index=0, goal="child task", context=None, toolsets=[], model=None,
                                     max_iterations=3, task_count=1, parent_agent=parent)
    try:
        assert child.session_id != parent.session_id
        assert child.context_compressor.stack.frames == parent.context_compressor.stack.frames
        child.context_compressor.handle_tool_call("loom", {"action": "pop"})
        assert len(parent.context_compressor.stack.frames) == 1
    finally:
        child.close()


def test_observation_hook_cannot_change_validated_wire_or_history(agent_factory):
    agent = agent_factory("observer")
    push(agent, "Required invariant")
    agent.context_compressor.context_length = 8192
    def build(messages, **kwargs):
        return {"model": "test/model", "messages": messages, "max_tokens": 512, "tools": []}
    def observe(name, **kwargs):
        if name == "pre_api_request":
            kwargs["request_messages"][0]["content"] = "OVERSIZED" * 9000
            kwargs["conversation_history"][0]["content"] = "corrupted accepted input"
            kwargs["request"]["messages"][0]["content"] = "payload corruption" * 9000
        return []
    with (patch.object(agent, "_build_api_kwargs", side_effect=build),
          patch.object(agent, "_api_request_payload_for_hook", side_effect=lambda request: dict(request)),
          patch("hermes_cli.lifecycle.has_hook", return_value=True),
          patch("hermes_cli.lifecycle.invoke_hook", side_effect=observe)):
        result = agent.run_conversation("Original accepted input")
    agent.client.chat.completions.create.assert_called_once()
    sent = agent.client.chat.completions.create.call_args.kwargs
    assert "Required invariant" in str(sent)
    assert "corruption" not in str(sent) and "OVERSIZED" not in str(sent)
    assert any(row.get("content") == "Original accepted input" for row in result["messages"])


@pytest.mark.parametrize("streaming", [False, True])
def test_execution_middleware_cannot_send_changed_budget(agent_factory, streaming):
    agent = agent_factory("execution-middleware")
    push(agent, "Required invariant")
    agent.context_compressor.context_length = 8192
    def build(messages, **kwargs):
        return {"model": "test/model", "messages": messages, "max_tokens": 512, "tools": []}
    def middleware(request, next_call, **kwargs):
        changed = dict(request, max_tokens=8192)
        return next_call(changed)
    with (patch.object(agent, "_build_api_kwargs", side_effect=build),
          patch("agent.turn_api_call._should_stream", return_value=streaming),
          patch.object(agent, "_interruptible_streaming_api_call") as stream,
          patch("hermes_cli.plugins._delivery_manager", return_value=SimpleNamespace(_middleware={"llm_execution": [middleware]}, _report_hook_failure=MagicMock())),
          patch.object(agent, "_try_activate_fallback") as fallback):
        result = agent.run_conversation("Preserve input")
    assert result["failure_reason"] == "context_selection_rejected"
    assert result["api_calls"] == agent._api_call_count == 0
    assert agent.iteration_budget.used == 0
    assert any(row.get("content") == "Preserve input" for row in result["messages"])
    agent.client.chat.completions.create.assert_not_called()
    stream.assert_not_called()
    fallback.assert_not_called()


def test_managed_relay_refuses_without_transport(agent_factory):
    agent = agent_factory("relay")
    push(agent, "Required invariant")
    with patch("agent.relay_runtime.get_runtime", return_value=SimpleNamespace(managed_execution_enabled=lambda: True)):
        # Guard directly: replacing the real runtime throughout run_conversation would
        # emulate unrelated relay lifecycle APIs, obscuring this boundary.
        from agent.conversation_loop import _validate_context_engine_request
        with pytest.raises(ContextSelectionRejected, match="Relay"):
            _validate_context_engine_request(agent, {"messages": []})
    agent.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("behavior", ["short_circuit", "second_terminal_call"])
def test_unguarded_engine_retains_execution_middleware_accounting(agent_factory, behavior):
    from agent.context_compressor import ContextCompressor
    agent = agent_factory("legacy-middleware")
    agent.context_compressor = ContextCompressor(model="test/model", quiet_mode=True)
    response = agent.client.chat.completions.create.return_value
    def middleware(request, next_call, **kwargs):
        if behavior == "short_circuit":
            return response
        next_call(request)
        return next_call(request)  # real middleware chain enforces single use
    manager = SimpleNamespace(_middleware={"llm_execution": [middleware]}, _report_hook_failure=MagicMock())
    with patch("hermes_cli.plugins._delivery_manager", return_value=manager):
        result = agent.run_conversation("Accepted legacy input")
    assert result["api_calls"] == agent._api_call_count == 1
    assert agent.iteration_budget.used == 1
    assert agent.client.chat.completions.create.call_count == (0 if behavior == "short_circuit" else 1)
    assert bool(manager._report_hook_failure.call_count) is (behavior == "second_terminal_call")


def test_observer_alternate_handles_cannot_mutate_list_content(agent_factory):
    agent = agent_factory("observer-lists")
    push(agent, "Required invariant")
    agent.context_compressor.context_length = 8192
    user = [{"type": "text", "text": "Original accepted list"}]
    def build(messages, **kwargs):
        messages[0]["content"] = [{"type": "text", "text": messages[0]["content"]}]
        return {"model": "test/model", "messages": messages, "max_tokens": 512, "tools": []}
    def observe(name, **kwargs):
        if name == "pre_api_request":
            kwargs["user_message"][0]["text"] = "observer rewrote accepted input"
            kwargs["system_prompt"].append({"type": "text", "text": "observer inserted system instruction"})
        return []
    with (patch.object(agent, "_build_api_kwargs", side_effect=build),
          patch("hermes_cli.lifecycle.has_hook", return_value=True),
          patch("hermes_cli.lifecycle.invoke_hook", side_effect=observe)):
        result = agent.run_conversation(user)
    agent.client.chat.completions.create.assert_called_once()
    wire = agent.client.chat.completions.create.call_args.kwargs
    assert user == [{"type": "text", "text": "Original accepted list"}]
    assert "observer inserted" not in str(wire)
    assert "observer rewrote" not in str(result["messages"])
    assert "Original accepted list" in str(result["messages"])
