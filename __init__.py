"""loom: a Hermes context engine that keeps a /goal nesting stack at the front of every request.

The model works like nested for loops. Entering a loop pushes a frame (its stable instructions);
leaving it pops the frame. The frames on the stack, outermost first, are rendered onto the end
of the system prompt after the /goal text. Each frame is an independent obstack frame: it holds
only its own text, nothing refers across frames, and a pop simply discards the top.

That ordering is the nested-loop shape: outer frames change least often and sit furthest
forward, the innermost frame changes most often and sits last, and the conversation (the loop
body) follows. The next sibling iteration is a pop plus a push, so only the tail after the
common frames moves. A prefix cache on the server (koboldcpp --loomcache, or any provider
prompt cache) keeps everything before the change; nothing about frames or boundaries is sent.

Compaction is inherited unchanged from the built-in compressor; it only rewrites conversation
turns, never the frames.

Enable with ``context.engine: loom`` in config.yaml.

Provenance: see PROVENANCE.txt (origin jnorthrup/koboldcpp-loom@9c66f41e6, path hermes-plugin/loom).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextSelectionRejected, ContextEngine  # noqa: F401  (discovery scans the file head for this name)

from .loomframes import MAX_DEPTH, render_frames, validate_frame, validate_frames
from . import loomstate

logger = logging.getLogger(__name__)

TOOL_NAME = "loom"

def _clean_text(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


class LoomStack:
    """Obstack of prompt frames: push appends, pop discards the top. Nothing else."""

    def __init__(self) -> None:
        self.frames: List[Dict[str, str]] = []  # [{"label", "text"}], outermost first

    def to_dict(self) -> Dict[str, Any]:
        return {"frames": list(self.frames)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoomStack":
        stack = cls()
        if not isinstance(data, dict) or set(data) != {"frames"}:
            raise ValueError("stack must contain only frames")
        stack.frames = validate_frames(data["frames"])
        return stack

    def push(self, text: Any, label: Any = "") -> None:
        frame = validate_frame({"text": text, "label": "" if label is None else label})
        if len(self.frames) >= MAX_DEPTH:
            raise ValueError(f"nesting is {MAX_DEPTH} deep; pop before pushing")
        self.frames.append(frame)

    def pop(self, count: int = 1) -> int:
        if type(count) is not int or count < 1:
            raise ValueError("pop count must be at least 1")
        if count > len(self.frames):
            raise ValueError(f"only {len(self.frames)} frame(s) to pop")
        del self.frames[len(self.frames) - count:]
        return count

    def _name(self, depth: int) -> str:
        label = self.frames[depth]["label"]
        return f"{depth + 1}" + (f" {label}" if label else "")

    def render(self, goal_block: str) -> str:
        return render_frames(self.frames, goal_block)

    def outline(self) -> str:
        if not self.frames:
            return "(no frames)"
        lines = []
        for depth, frame in enumerate(self.frames):
            first = frame["text"].split("\n", 1)[0]
            if len(first) > 70:
                first = first[:67] + "..."
            lines.append(f"{'  ' * depth}{self._name(depth)}: {first}")
        return "\n".join(lines)


def _goal_block(session_id: str, home: str) -> str:
    """Goal text plus subgoals. Status is left out on purpose: it changes every turn and would
    move every frame behind it."""
    if not session_id:
        return ""
    try:
        from hermes_cli.goals import load_goal
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            state = load_goal(session_id, strict=True)
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        raise ContextSelectionRejected("loom: required goal state could not be read safely; request was not sent.") from exc
    if state is None or state.status == "cleared" or not (state.goal or "").strip():
        return ""
    block = _clean_text(state.goal)
    subgoals = state.render_subgoals_block()
    return block + ("\n" + subgoals if subgoals else "")


def _append_to_content(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{content}\n\n{text}" if content else text
    if isinstance(content, list):
        return list(content) + [{"type": "text", "text": text}]
    if content is None:
        return text
    raise ContextSelectionRejected("loom: unsupported system content; required frames were not sent.")


def _compression_kwargs() -> Dict[str, Any]:
    """Mirror the user's ``compression`` config for the inherited compressor (plugin engines are
    constructed without it)."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = (load_config_readonly() or {}).get("compression") or {}
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for key, name, cast in (("threshold", "threshold_percent", float), ("target_ratio", "summary_target_ratio", float),
                            ("protect_first_n", "protect_first_n", int), ("protect_last_n", "protect_last_n", int)):
        if cfg.get(key) is not None:
            try:
                out[name] = cast(cfg[key])
            except (TypeError, ValueError):
                pass
    if cfg.get("tail_mode"):
        out["tail_mode"] = str(cfg["tail_mode"]).strip().lower()
    return out


_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Nest work for the active /goal like nested loops. 'push' a frame when you enter a level of work "
        "(its stable instructions, not findings); 'pop' when you leave it. Frames are shown to you at the "
        "front of the context, outermost first, and the last frame is your current loop. Move to a sibling "
        "with pop then push. Keep results and status in the conversation. "
        "Actions: push {text, label?}; pop {count?}; show."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["push", "pop", "show"]},
            "text": {"type": "string", "description": "push: the frame's stable instructions."},
            "label": {"type": "string", "description": "push: optional short name for the frame."},
            "count": {"type": "integer", "description": "pop: frames to pop (default 1)."},
        },
        "required": ["action"],
    },
}


class LoomEngine(ContextCompressor):
    """ContextCompressor plus a front-of-context /goal frame stack."""

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        self._loom_budget = kwargs.pop("loom_budget", None)
        if self._loom_budget is None:
            try:
                from hermes_cli.config import load_config_readonly
                self._loom_budget = (load_config_readonly().get("context") or {}).get("loom")
            except Exception:
                self._loom_budget = None
        settings = _compression_kwargs()
        settings.update(kwargs)
        settings.setdefault("quiet_mode", True)
        super().__init__(model=model, **settings)
        self.stack = LoomStack()
        self._loom_session = ""
        self._loom_home = ""
        self._loom_error = ""
        self._loom_lock = RLock()
        self._loom_init_kwargs = dict(kwargs, loom_budget=self._loom_budget)

    @property
    def name(self) -> str:
        return "loom"

    def clone_for_agent(self):
        # Clone configuration only. Session DBs, locks and frames never enter a new agent implicitly.
        return type(self)(model=self.model, **self._loom_init_kwargs)

    # -- state -------------------------------------------------------------------------------
    def _state_path(self) -> str:
        if not (self._loom_home and self._loom_session):
            return ""
        return str(loomstate.state_path(self._loom_home, self._loom_session))

    def _require_ready(self):
        if self._loom_error or not self._state_path():
            raise ContextSelectionRejected(self._loom_error or "loom: session identity is unavailable.")

    def _save(self, stack=None) -> None:
        self._require_ready()
        loomstate.save(Path(self._state_path()), self._loom_home, self._loom_session,
                       (self.stack if stack is None else stack).frames)

    def _commit(self, stack):
        try:
            self._save(stack)
        except Exception as exc:
            self._loom_error = "loom: frame persistence failed; reload the session before continuing."
            raise ContextSelectionRejected(self._loom_error) from exc
        self.stack = stack

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        with self._loom_lock:
            old_home, old_session = self._loom_home, self._loom_session
            old_frames = self.stack.to_dict()
            old_error = self._loom_error
            # Clear first: even a failed load cannot expose the previous session.
            self.stack = LoomStack()
            self._loom_session = session_id if isinstance(session_id, str) else ""
            self._loom_home = ""
            self._loom_error = ""
            try:
                home = kwargs.get("hermes_home") or (old_home if kwargs.get("boundary_reason") == "compression" else None)
                if not home:
                    from hermes_constants import get_hermes_home
                    home = str(get_hermes_home())
                self._loom_home = str(Path(home).expanduser().resolve())
                if not self._loom_session or len(self._loom_session) > 4096:
                    raise ValueError("session identity is unavailable")
                super().on_session_start(session_id, **kwargs)
                path = Path(self._state_path())
                if kwargs.get("boundary_reason") == "compression" and (
                    old_error or old_home != self._loom_home or kwargs.get("old_session_id") != old_session
                ):
                    raise ValueError("unbound compression lineage")
                frames = loomstate.load(path, self._loom_home, self._loom_session)
                if frames is not None:
                    self.stack = LoomStack.from_dict({"frames": frames})
                elif kwargs.get("boundary_reason") == "compression":
                    self._commit(LoomStack.from_dict(old_frames))
                else:
                    # Old sanitized filenames have no identity and can collide. Never guess ownership.
                    legacy = path.parent / (re.sub(r"[^A-Za-z0-9_.-]", "_", self._loom_session) + ".json")
                    if os.path.lexists(legacy):
                        raise ValueError("legacy state requires explicit migration")
            except Exception:
                self.stack = LoomStack()
                self._loom_error = "loom: saved frames could not be safely loaded; repair or explicitly reset this session."

    def on_session_reset(self) -> None:
        with self._loom_lock:
            # Explicit reset may recover malformed state, but only an established identity.
            self._loom_error = ""
            self._commit(LoomStack())
            super().on_session_reset()

    def inherit_from_parent(self, parent: ContextEngine) -> None:
        if not isinstance(parent, type(self)):
            raise ContextSelectionRejected("loom: parent engine does not support explicit frame inheritance.")
        # Snapshot the parent before locking the child; no lock ordering cycle.
        with parent._loom_lock:
            parent._require_ready()
            home, session, data = parent._loom_home, parent._loom_session, parent.stack.to_dict()
        with self._loom_lock:
            self._require_ready()
            if home != self._loom_home or session == self._loom_session:
                raise ContextSelectionRejected("loom: frame inheritance requires distinct sessions in the same profile.")
            if self.stack.frames:
                raise ContextSelectionRejected("loom: refusing to replace an initialized child frame stack.")
            self._commit(LoomStack.from_dict(data))

    # -- per-request selection ---------------------------------------------------------------
    def select_context(self, request_messages: List[Dict[str, Any]], **kwargs: Any) -> Optional[List[Dict[str, Any]]]:
        with self._loom_lock:
            self._require_ready()
            goal = _goal_block(self._loom_session, self._loom_home)
            if not goal and not self.stack.frames:
                return None
            if not request_messages or request_messages[0].get("role") != "system":
                raise ContextSelectionRejected("loom: required frames need a system message; request was not sent.")
            first = dict(request_messages[0])
            first["content"] = _append_to_content(first.get("content"), self.stack.render(goal))
            selected = [first] + list(request_messages[1:])
            # New hosts validate the final wire request, after caps/overrides/middleware.
            if not kwargs.get("request_validation"):
                self._check_budget(selected, kwargs.get("tool_schemas"), kwargs.get("max_output_tokens"), kwargs.get("budget_tokens"))
            return selected

    def _check_budget(self, messages, tools, output, budget, model=None):
        contract = self._loom_budget
        keys = {"model", "tokenizer", "prompt_overhead", "message_overhead", "tool_overhead"}
        if (not isinstance(contract, dict) or set(contract) != keys
                or contract.get("tokenizer") != "utf8-byte-fallback"
                or contract.get("model") != (self.model if model is None else model)
                or any(type(contract.get(key)) is not int or contract[key] < 0
                       for key in ("prompt_overhead", "message_overhead", "tool_overhead"))):
            raise ContextSelectionRejected("loom: a reviewed model-specific byte-fallback/template budget contract is required; request was not sent.")
        if type(budget) is not int or budget <= 0 or type(output) is not int or output <= 0 or not isinstance(tools, list):
            raise ContextSelectionRejected("loom: context window, explicit output cap and tool schemas are required; request was not sent.")
        for message in messages:
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                if not isinstance(content, list) or any(not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str) for part in content):
                    raise ContextSelectionRejected("loom: this content type has no supported budget; request was not sent.")
        # Intentionally conservative text budget, not a provider-exact tokenizer.
        # UTF-8 bytes upper-bound ordinary byte-fallback tokens; reserve framing separately.
        try:
            charge = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode("utf-8"))
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ContextSelectionRejected("loom: request cannot be safely budgeted.") from exc
        charge += contract["prompt_overhead"] + contract["message_overhead"] * len(messages) + contract["tool_overhead"] * len(tools)
        if charge + output > budget:
            raise ContextSelectionRejected("loom: required context plus output exceeds the conservative context budget; shorten frames/history explicitly or select a larger window.")

    def validate_request(self, request, *, budget_tokens=0, **kwargs):
        with self._loom_lock:
            self._require_ready()
            goal = _goal_block(self._loom_session, self._loom_home)
            if not goal and not self.stack.frames:
                return
            if kwargs.get("relay_enabled"):
                raise ContextSelectionRejected("loom: Relay may rewrite requests after validation and is unsupported by this budget contract; request was not sent.")
            if kwargs.get("api_mode") != "chat_completions" or not isinstance(request.get("messages"), list):
                raise ContextSelectionRejected("loom: final budget validation supports text chat completions only; request was not sent.")
            if request.get("extra_body"):
                raise ContextSelectionRejected("loom: opaque request overrides cannot be safely budgeted; request was not sent.")
            caps = [request[k] for k in ("max_tokens", "max_completion_tokens") if k in request]
            if len(caps) != 1:
                raise ContextSelectionRejected("loom: exactly one explicit output cap is required; request was not sent.")
            # A middleware must not silently remove or rewrite the required block.
            block = self.stack.render(goal)
            content = request["messages"][0].get("content") if request["messages"] else None
            texts = [content] if isinstance(content, str) else [p.get("text", "") for p in (content or []) if isinstance(p, dict)] if isinstance(content, list) else []
            if not request["messages"] or request["messages"][0].get("role") != "system" or not any(block in text for text in texts):
                raise ContextSelectionRejected("loom: required frames changed during request assembly; request was not sent.")
            self._check_budget(request["messages"], request.get("tools", []), caps[0], budget_tokens, request.get("model"))

    # -- tool --------------------------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_TOOL_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if name != TOOL_NAME:
            return super().handle_tool_call(name, args, **kwargs)
        args = args or {}
        action = args.get("action")
        with self._loom_lock:
            try:
                self._require_ready()
                candidate = LoomStack.from_dict(self.stack.to_dict())
                if action == "push":
                    candidate.push(args.get("text"), args.get("label"))
                elif action == "pop":
                    candidate.pop(args.get("count", 1))
                elif action != "show":
                    raise ValueError("use push, pop or show")
                if action != "show":
                    self._commit(candidate)
            except (ValueError, TypeError) as exc:
                return json.dumps({"error": str(exc)})
            return json.dumps({"ok": True, "depth": len(self.stack.frames), "frames": self.stack.outline()})


def _loom_command(raw_args: str = "") -> str:
    # Hermes slash callbacks carry only raw_args, never a trustworthy agent/session.
    return "loom: slash commands cannot identify the active session safely. Ask the agent to use the loom show tool."


def register(ctx: Any) -> None:
    ctx.register_context_engine(LoomEngine())
    register_command = getattr(ctx, "register_command", None)
    if not callable(register_command):
        return
    try:  # the engine is loaded once per agent; register /loom only the first time
        from hermes_cli.plugins import get_plugin_manager
        if "loom" in get_plugin_manager()._plugin_commands:
            return
    except Exception:
        pass
    register_command("loom", _loom_command, description="Explain how to show the active agent’s frame stack safely")
