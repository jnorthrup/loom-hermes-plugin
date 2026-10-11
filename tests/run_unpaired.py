"""Run these tests on a Hermes host that lacks the paired seams (README.txt: State and host compatibility).

Test-only stand-ins, installed before pytest imports anything:
  agent.context_engine.ContextSelectionRejected  a ValueError subclass (the tool handler reports
                                                 refusals through except (ValueError, TypeError))
  hermes_cli.goals.load_goal(strict=...)         accepted and ignored
The 13 host-pair tests and the 6 strict-goal tests need the real paired host and fail here; every
other test must pass. Nothing here makes the plugin usable on such a host.

  cd loom-hermes-plugin
  HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME="$(mktemp -d)" PYTHONDONTWRITEBYTECODE=1 \\
    /path/to/hermes-agent/.venv/bin/python tests/run_unpaired.py /path/to/hermes-agent \\
    tests --import-mode=importlib -p no:cacheprovider -q

HERMES_DISABLE_LAZY_INSTALLS=1 keeps the host from syncing a dependency environment for the
throwaway HERMES_HOME and relaunching without pytest (hermes_cli/venv_sync.py).
"""
import inspect
import sys

sys.path.insert(0, sys.argv.pop(1))

import agent.context_engine as context_engine  # noqa: E402

if not hasattr(context_engine, "ContextSelectionRejected"):
    class ContextSelectionRejected(ValueError):
        """Stand-in for the paired host's refusal type."""

    context_engine.ContextSelectionRejected = ContextSelectionRejected

import hermes_cli.goals as goals  # noqa: E402

if "strict" not in inspect.signature(goals.load_goal).parameters:
    _load_goal = goals.load_goal

    def load_goal(session_id, strict=False):
        return _load_goal(session_id)

    goals.load_goal = load_goal

import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
