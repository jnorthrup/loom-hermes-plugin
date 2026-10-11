"""loom: a Hermes context engine that keeps a /goal nesting stack at the front of every request.

The model works like nested for loops. Entering a loop pushes a frame (its stable instructions);
leaving it pops the frame. The frames on the stack, outermost first, are rendered onto the end
of the system prompt after the /goal text. Each frame is an independent obstack frame: it holds
only its own text, nothing refers across frames, and a pop simply discards the top.

That ordering is the nested-loop shape: outer frames change least often and sit furthest
forward, and the conversation (the loop body) follows. The front moves only when the
conversation behind it has already been rewritten (first request, reload, compaction, undo,
pruning) or the goal changed. A push or pop does not move it: the loom tool call and result
carry the change in the conversation, which only grows, and the next rewrite folds it into the
front. So no prefix cache (koboldcpp --loomcache, Anthropic cache_control, OpenAI and Codex
automatic prefix caching) loses the conversation to a frame change. Every vendor gets the same
bytes; nothing about frames, caches or boundaries is sent, and cache signals stay the host's.

Compaction is inherited unchanged from the built-in compressor; it only rewrites conversation
turns, never the frames.

Enable with ``context.engine: loom`` in config.yaml.

Provenance: see PROVENANCE.txt (origin jnorthrup/koboldcpp-loom@9c66f41e6, path hermes-plugin/loom).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine  # noqa: F401  (discovery scans the file head for this name)

try:  # Paired hosts type refusals. Older hosts lack the type and fail open: a refused request goes out unframed.
    from agent.context_engine import ContextSelectionRejected
except ImportError:  # depends on the host
    class ContextSelectionRejected(ValueError):
        """Refusal on a host without the paired seam. A ValueError, so the loom tool reports it."""

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
        try:
            from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        except ImportError:  # unpaired host: the agent's own home is the session's home
            set_hermes_home_override = reset_hermes_home_override = None
        token = set_hermes_home_override(home) if set_hermes_home_override else None
        try:
            try:
                state = load_goal(session_id, strict=True)
            except TypeError:  # unpaired host: no strict read; an unreadable goal reads as none
                state = load_goal(session_id)
        finally:
            if reset_hermes_home_override:
                reset_hermes_home_override(token)
    except Exception as exc:
        raise ContextSelectionRejected("loom: required goal state could not be read safely; request was not sent.") from exc
    if state is None or state.status == "cleared" or not (state.goal or "").strip():
        return ""
    block = _clean_text(state.goal)
    render = getattr(state, "render_subgoals_block", None)
    subgoals = render() if callable(render) else ""
    return block + ("\n" + subgoals if subgoals else "")


def _append_to_content(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{content}\n\n{text}" if content else text
    if isinstance(content, list):
        return list(content) + [{"type": "text", "text": text}]
    if content is None:
        return text
    raise ContextSelectionRejected("loom: unsupported system content; required frames were not sent.")


# Provider-visible message fields. One digest per message tells a conversation that only grew
# (the front may stay) from one that was rewritten (compaction, undo, pruning: the cache is
# already lost, so the front may catch up for free).
_HISTORY_FIELDS = ("role", "content", "tool_calls", "tool_call_id", "name")


def _history_digests(messages: List[Any]) -> Optional[List[str]]:
    try:
        return [hashlib.sha256(json.dumps(
                    {key: m.get(key) for key in _HISTORY_FIELDS} if isinstance(m, dict) else m,
                    sort_keys=True, ensure_ascii=True, default=str).encode("ascii")).hexdigest()
                for m in messages]
    except (TypeError, ValueError, RecursionError):
        return None  # cannot prove the conversation only grew: the front is re-rendered


def _loom_calls(messages: List[Any]) -> int:
    """Loom tool calls recorded in these messages: the conversation's own record of push/pop."""
    count = 0
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        for call in calls if isinstance(calls, list) else ():
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if (function.get("name") if isinstance(function, dict) else call.get("name")) == TOOL_NAME:
                count += 1
    return count


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


def _check_front_share(block: str, budget: Any) -> None:
    """Hosts without the paired seam pass only the context window and size the request themselves,
    compaction included. Keep the front within a tenth of that window, counted in UTF-8 bytes (an
    upper bound on tokens)."""
    if type(budget) is not int or budget <= 0:
        raise ContextSelectionRejected("loom: the host reported no context window; frames were not sent.")
    if len(block.encode("utf-8")) > budget // 10:
        raise ContextSelectionRejected("loom: the frames exceed a tenth of the context window; pop or shorten frames.")


_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Nest work for the active /goal like nested loops. 'push' a frame when you enter a level of work "
        "(its stable instructions, not findings); 'pop' when you leave it. Move to a sibling with pop then "
        "push. A push or pop takes effect at once and its result lists the current stack, outermost first; "
        "the last frame is your current loop. The front of the context shows the frames as of the last "
        "context rebuild (start, reload, compaction), so prompt caches keep the conversation; where the "
        "two differ, the latest loom result is current. Keep results and status in the conversation. "
        "Actions: push {text, label?}; pop {count?}; show. Call this tool by its exact name, loom, with no prefix."
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
        self._front: Optional[Dict[str, Any]] = None  # the block at the front of requests; see _front_block
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
            self._front = None
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
            self._front = None
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
            self._front = None

    # -- per-request selection ---------------------------------------------------------------
    def _front_block(self, goal: str, history: List[Any]) -> str:
        """The frame block for the front of this request.

        Reused byte for byte while the conversation only grows, the goal is unchanged and every
        push or pop since it was rendered has its loom tool call recorded in the conversation.
        Re-rendered otherwise: first request, reload, compaction, undo, pruning, a goal change or
        an unrecorded stack change. A recorded push or pop therefore never moves the front, and
        an unprovable case costs one cache miss, never a frame."""
        digests = _history_digests(history)
        front = self._front
        if (front is not None and digests is not None and front["seen"] is not None
                and front["goal"] == goal
                and digests[:len(front["seen"])] == front["seen"]
                and _loom_calls(history[front["start"]:]) >= front["pending"]):
            front["seen"] = digests
            return front["block"]
        block = self.stack.render(goal) if (goal or self.stack.frames) else ""
        self._front = {"goal": goal, "block": block, "start": len(history), "seen": digests, "pending": 0}
        return block

    def select_context(self, request_messages: List[Dict[str, Any]], **kwargs: Any) -> Optional[List[Dict[str, Any]]]:
        with self._loom_lock:
            self._require_ready()
            goal = _goal_block(self._loom_session, self._loom_home)
            history = kwargs.get("conversation_messages")
            if not isinstance(history, list):
                has_system = bool(request_messages) and request_messages[0].get("role") == "system"
                history = request_messages[1:] if has_system else request_messages
            block = self._front_block(goal, list(history))
            if not block and not goal and not self.stack.frames:
                return None
            selected = list(request_messages)
            if block:
                if not request_messages or request_messages[0].get("role") != "system":
                    raise ContextSelectionRejected("loom: required frames need a system message; request was not sent.")
                first = dict(request_messages[0])
                first["content"] = _append_to_content(first.get("content"), block)
                selected = [first] + list(request_messages[1:])
            # New hosts validate the final wire request, after caps/overrides/middleware.
            if kwargs.get("request_validation"):
                pass
            elif kwargs.get("max_output_tokens") is None and kwargs.get("tool_schemas") is None:
                _check_front_share(block, kwargs.get("budget_tokens"))  # unpaired host: it sizes the request
            else:
                self._check_budget(selected, kwargs.get("tool_schemas"), kwargs.get("max_output_tokens"), kwargs.get("budget_tokens"))
            return selected if block else None

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
            # The block select_context placed at the front; it may predate recorded pushes/pops.
            block = (self._front["block"] if self._front is not None
                     else self.stack.render(goal) if (goal or self.stack.frames) else "")
            if not block and not goal and not self.stack.frames:
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
            content = request["messages"][0].get("content") if request["messages"] else None
            texts = [content] if isinstance(content, str) else [p.get("text", "") for p in (content or []) if isinstance(p, dict)] if isinstance(content, list) else []
            if not request["messages"] or request["messages"][0].get("role") != "system" or (block and not any(block in text for text in texts)):
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
                    if self._front is not None:
                        self._front["pending"] += 1  # its tool call must reach the conversation
            except (ValueError, TypeError) as exc:
                return json.dumps({"error": str(exc)})
            return json.dumps({"ok": True, "depth": len(self.stack.frames), "frames": self.stack.outline()})


def _loom_command(raw_args: str = "") -> str:
    # Hermes slash callbacks carry only raw_args, never a trustworthy agent/session.
    return "loom: slash commands cannot identify the active session safely. Ask the agent to use the loom show tool."


def _install_oauth_name_fallback() -> None:
    """Hermes' Anthropic OAuth wire renames every tool ``mcp__<name>`` and maps replies back only
    through the tool registry, which does not hold context-engine tools, so ``mcp__loom`` never
    reached this engine. Map exactly that name back to ``loom``; every other name keeps the host's
    mapping, and a host that resolves engine tools itself passes through unchanged."""
    try:
        import agent.transports.anthropic as transport
    except Exception:
        return
    original = getattr(transport, "_unprefix_oauth_tool_name", None)
    if not callable(original) or getattr(original, "_loom_fallback", False):
        return

    def unprefix(name: str) -> str:
        resolved = original(name)
        return TOOL_NAME if resolved == "mcp__" + TOOL_NAME else resolved

    unprefix._loom_fallback = True  # type: ignore[attr-defined]
    transport._unprefix_oauth_tool_name = unprefix


def register(ctx: Any) -> None:
    _install_oauth_name_fallback()
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
