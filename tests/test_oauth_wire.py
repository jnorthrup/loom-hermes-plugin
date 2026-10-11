"""Anthropic OAuth wire names: Hermes sends every tool as ``mcp__<name>`` and maps replies back
through its tool registry, which does not hold context-engine tools."""
import agent.transports.anthropic as transport


def test_oauth_wire_name_maps_back_to_the_engine_tool(loom, monkeypatch):
    original = transport._unprefix_oauth_tool_name
    monkeypatch.setattr(transport, "_unprefix_oauth_tool_name", original)
    loom._install_oauth_name_fallback()
    unprefix = transport._unprefix_oauth_tool_name
    assert unprefix("mcp__loom") == "loom"
    assert unprefix("mcp__not_a_tool") == original("mcp__not_a_tool")  # other names keep the host's mapping
    loom._install_oauth_name_fallback()
    assert transport._unprefix_oauth_tool_name is unprefix  # installed once
