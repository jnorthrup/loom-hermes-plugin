#!/usr/bin/env python3
"""loom-loop-precook: a book outline of a context evolution, curated as nested Python loops.

A loom script (JSON, format "loom-script/1") is an outline tree. `emit` turns it into a plain
Python program whose nested `for` loops walk the outline depth-first; each loop variable is a
frame that is pushed on entry and popped on exit, and every leaf is one chat request carrying
system prompt + goal + the frames of every enclosing loop. That program is the curated artifact:
read it, edit it, run it. `run` executes exactly that emitted program.

Walking depth-first means siblings share every enclosing frame byte for byte, so a server with
prefix reuse (koboldcpp --loomcache, provider prompt caches) keeps the shared front and computes
only what differs. `--prewarm` asks for one token per leaf: the walk then only cooks the cache.

Nothing but the ordinary OpenAI /v1/chat/completions request is sent; no cache hints.

  loom_loop.py emit    SCRIPT.json [-o book.py]
  loom_loop.py run     SCRIPT.json --base-url URL [--prewarm] [--dry-run] [--out RUN.jsonl]
  loom_loop.py replay  RUN.jsonl  --base-url URL [--out RUN2.jsonl]
  loom_loop.py prompts SCRIPT.json                    (one flattened prompt per leaf, NUL separated)
  loom_loop.py next    SCRIPT.json STATE.json         (ralph iterator: print next prompt, exit 3 when done)

Loom script:
  {"format": "loom-script/1",
   "system": "...", "goal": "...",
   "loops": ["epoch", "chapter", "bullet"],   # optional loop-variable names per depth
   "request": {"model": "...", "max_tokens": 512},   # merged into every request
   "outline": [{"label": "e1", "text": "...", "task": "...", "children": [...]}]}
A node with no children is a leaf; its `task` (default: DEFAULT_TASK) is the user message.
Depth may vary between branches; nothing here assumes a fixed number of levels.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import keyword
import os
import re
import ssl
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loomframes import FORMAT as FRAMES_FORMAT, with_frames  # noqa: E402

SCRIPT_FORMAT = "loom-script/1"
DEFAULT_TASK = "Carry out the work of the innermost frame."
_IDENT = re.compile(r"[^A-Za-z0-9_]")


class Node:
    __slots__ = ("label", "text", "task", "children")

    def __init__(self, d):
        if not isinstance(d, dict) or not str(d.get("text") or "").strip():
            raise ValueError(f"outline node needs text: {d!r:.80}")
        self.label = str(d.get("label") or "")
        self.text = str(d["text"]).replace("\r\n", "\n").strip()
        self.task = str(d.get("task") or "").strip()
        self.children = [Node(c) for c in (d.get("children") or [])]

    @property
    def leaf(self):
        return not self.children

    def depth(self):
        return 1 + max((c.depth() for c in self.children), default=0)


def load_script(path_or_obj):
    data = path_or_obj
    if not isinstance(data, dict):
        with open(path_or_obj, encoding="utf-8") as f:
            data = json.load(f)
    if data.get("format", SCRIPT_FORMAT) != SCRIPT_FORMAT:
        raise ValueError(f"unsupported format {data.get('format')!r}; expected {SCRIPT_FORMAT}")
    outline = [Node(n) for n in data.get("outline") or []]
    if not outline:
        raise ValueError("script has an empty outline")
    return data, outline


def loop_names(script, depth):
    names, seen = [], set()
    given = list(script.get("loops") or [])
    for i in range(depth):
        raw = given[i] if i < len(given) else f"level{i + 1}"
        name = _IDENT.sub("_", str(raw)) or f"level{i + 1}"
        if name[0].isdigit() or keyword.iskeyword(name) or name in seen or name in ("L", "loom"):
            name = f"{name}_{i + 1}"
        seen.add(name)
        names.append(name)
    return names


# -- the runtime the emitted program uses ----------------------------------------------------

class Loom:
    """Frame stack + request sender. frame() pushes on entry and pops on exit, nothing else."""

    def __init__(self, script, base_url="", api_key="", prewarm=False, dry_run=False, out=None, log=sys.stderr):
        self.script, self.outline = load_script(script)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.prewarm, self.dry_run, self.out, self.log = prewarm, dry_run, out, log
        self.frames = []
        self.path = []
        self.leaves = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @contextlib.contextmanager
    def frame(self, node):
        self.frames.append({"label": node.label, "text": node.text})
        self.path.append(node.label)
        try:
            yield node
        finally:
            self.frames.pop()
            self.path.pop()

    def messages(self, node):
        system = with_frames(str(self.script.get("system") or ""), self.frames, str(self.script.get("goal") or ""))
        return [{"role": "system", "content": system}, {"role": "user", "content": node.task or DEFAULT_TASK}]

    def request_body(self, node):
        body = dict(self.script.get("request") or {})
        body["messages"] = self.messages(node)
        if self.prewarm:
            body["max_tokens"] = 1
        return body

    def ask(self, node):
        self.leaves += 1
        body = self.request_body(node)
        where = "/".join(self.path)
        if self.dry_run:
            print(f"[{self.leaves}] {where}  system={len(body['messages'][0]['content'])} chars", file=self.log)
            record = {"path": list(self.path), "request": body}
        else:
            started = time.time()
            reply = post_chat(self.base_url, self.api_key, body)
            usage = reply.get("usage") or {}
            for k in self.usage:
                self.usage[k] += int(usage.get(k) or 0)
            print(f"[{self.leaves}] {where}  prompt={usage.get('prompt_tokens')} "
                  f"completion={usage.get('completion_tokens')} {time.time() - started:.2f}s", file=self.log)
            record = {"path": list(self.path), "request": body, "response": reply}
        if self.out:
            record["format"] = "loom-run/1"
            record["frames_format"] = FRAMES_FORMAT
            self.out.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.out.flush()
        return record


def post_chat(base_url, api_key, body, timeout=600):
    req = urllib.request.Request(f"{base_url}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    ctx = ssl.create_default_context() if base_url.startswith("https") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.load(r)


# -- emit: outline -> nested loops -------------------------------------------------------------

def emit(script_path, script_obj=None):
    script, outline = load_script(script_obj if script_obj is not None else script_path)
    depth = max(n.depth() for n in outline)
    names = loop_names(script, depth)
    lines = [
        "#!/usr/bin/env python3",
        f"# loom-loop-precook: nested loops over {os.path.basename(script_path) if script_path else 'script'}"
        f" ({depth} level{'s' if depth != 1 else ''}, {sum(1 for _ in _leaves(outline))} leaves).",
        "# Each loop variable is a frame: pushed on entry, popped on exit. Edit freely.",
        "import sys",
        "from loom_loop import Loom, main_runtime",
        "",
        "def walk(L):",
    ]
    indent = "    "

    def level(i, source):
        nonlocal indent
        var = names[i]
        lines.append(f"{indent}for {var} in {source}:")
        indent += "    "
        lines.append(f"{indent}with L.frame({var}):")
        indent += "    "
        if i + 1 < depth:
            lines.append(f"{indent}if {var}.leaf:")
            lines.append(f"{indent}    L.ask({var})")
            lines.append(f"{indent}    continue")
            level(i + 1, f"{var}.children")
        else:
            lines.append(f"{indent}L.ask({var})")

    level(0, "L.outline")
    lines += ["", "if __name__ == '__main__':", "    main_runtime(walk)", ""]
    return "\n".join(lines)


def _leaves(nodes, path=()):
    for n in nodes:
        p = path + (n,)
        if n.leaf:
            yield p
        else:
            yield from _leaves(n.children, p)


def run_walk(script, walk_src, **kw):
    """Execute emitted loop source against a Loom runtime; `run` always goes through here."""
    namespace = {"__name__": "loom_book"}
    sys.modules.setdefault("loom_loop", sys.modules[__name__])
    exec(compile(walk_src, "<loom-book>", "exec"), namespace)
    loom = Loom(script, **kw)
    namespace["walk"](loom)
    return loom


def main_runtime(walk):
    """Entry point used by an emitted, possibly hand-edited, book.py."""
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    _run_args(ap)
    a = ap.parse_args()
    with _open_out(a.out) as out:
        loom = Loom(a.script, base_url=a.base_url, api_key=os.environ.get(a.api_key_env, ""),
                    prewarm=a.prewarm, dry_run=a.dry_run, out=out)
        walk(loom)
    _summary(loom)


# -- flattened prompts for single-string harnesses (ralph) ------------------------------------

def flat_prompts(script_path):
    script, outline = load_script(script_path)
    loom = Loom(script)
    out = []
    for path in _leaves(outline):
        with contextlib.ExitStack() as stack:
            for node in path:
                stack.enter_context(loom.frame(node))
            system, user = (m["content"] for m in loom.messages(path[-1]))
            out.append({"path": [n.label for n in path], "prompt": f"{system}\n\n{user}" if system else user})
    return out


def cmd_next(script_path, state_path):
    """Print the next prompt and advance STATE; exit 3 when every leaf has been handed out."""
    prompts = flat_prompts(script_path)
    state = {"next": 0}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    i = int(state.get("next", 0))
    if i >= len(prompts):
        return 3
    sys.stdout.write(prompts[i]["prompt"])
    state.update({"next": i + 1, "total": len(prompts), "last_path": prompts[i]["path"]})
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path)
    print(f"loom: leaf {i + 1}/{len(prompts)} {'/'.join(prompts[i]['path'])}", file=sys.stderr)
    return 0


# -- CLI ----------------------------------------------------------------------------------------

def _run_args(ap):
    ap.add_argument("--base-url", default=os.environ.get("LOOM_BASE_URL", "http://localhost:5001/v1"))
    ap.add_argument("--api-key-env", default="LOOM_API_KEY", help="env var holding the bearer key (never a literal)")
    ap.add_argument("--prewarm", action="store_true", help="max_tokens=1 per leaf: only cook the cache")
    ap.add_argument("--dry-run", action="store_true", help="print the walk, send nothing")
    ap.add_argument("--out", help="write a loom-run/1 JSONL record per leaf (for replay/artifacts)")


@contextlib.contextmanager
def _open_out(path):
    if not path:
        yield None
        return
    with open(path, "w", encoding="utf-8") as f:
        yield f


def _summary(loom):
    print(f"loom: {loom.leaves} leaves, prompt_tokens={loom.usage['prompt_tokens']} "
          f"completion_tokens={loom.usage['completion_tokens']}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("emit"); p.add_argument("script"); p.add_argument("-o", "--output")
    p = sub.add_parser("run"); p.add_argument("script"); _run_args(p)
    p = sub.add_parser("replay"); p.add_argument("run"); _run_args(p)
    p = sub.add_parser("prompts"); p.add_argument("script")
    p = sub.add_parser("next"); p.add_argument("script"); p.add_argument("state")
    a = ap.parse_args(argv)

    if a.cmd == "emit":
        src = emit(a.script)
        if a.output:
            with open(a.output, "w", encoding="utf-8") as f:
                f.write(src)
        else:
            sys.stdout.write(src)
        return 0
    if a.cmd == "run":
        with _open_out(a.out) as out:
            loom = run_walk(a.script, emit(a.script), base_url=a.base_url,
                            api_key=os.environ.get(a.api_key_env, ""), prewarm=a.prewarm, dry_run=a.dry_run, out=out)
        _summary(loom)
        return 0
    if a.cmd == "replay":
        key = os.environ.get(a.api_key_env, "")
        n = 0
        with open(a.run, encoding="utf-8") as src, _open_out(a.out) as out:
            for line in src:
                rec = json.loads(line)
                body = dict(rec["request"])
                if a.prewarm:
                    body["max_tokens"] = 1
                n += 1
                if a.dry_run:
                    print(f"[{n}] {'/'.join(rec.get('path') or [])}", file=sys.stderr)
                    continue
                reply = post_chat(a.base_url.rstrip("/"), key, body)
                u = reply.get("usage") or {}
                print(f"[{n}] {'/'.join(rec.get('path') or [])} prompt={u.get('prompt_tokens')}", file=sys.stderr)
                if out:
                    out.write(json.dumps({**rec, "response": reply, "replayed_from": a.run}, ensure_ascii=False) + "\n")
        return 0
    if a.cmd == "prompts":
        for item in flat_prompts(a.script):
            sys.stdout.write(item["prompt"] + "\0")
        return 0
    if a.cmd == "next":
        return cmd_next(a.script, a.state)
    return 2


if __name__ == "__main__":
    sys.exit(main())
