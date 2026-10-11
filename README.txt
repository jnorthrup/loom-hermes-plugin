loom - Hermes context engine: /goal work as nested loops
========================================================

What it does
  Work nests like for loops. Entering a level of work pushes a frame (its stable
  instructions); leaving it pops the frame. The frames on the stack, outermost first,
  sit at the front of every request after the /goal text:

    system prompt | goal | frame 1 | ... | innermost frame | conversation | push/pop calls
                   '------- front, as of the last rebuild -------'           '-- since --'

  Each frame is an independent obstack frame: it holds only its own text and nothing
  refers across frames. The front moves only when the conversation behind it has
  already been rewritten (first request, reload, compaction, undo, pruning), the goal
  changed, or a stack change is missing from the conversation. A push or pop is
  carried by its own tool call and result, which only append; the next rewrite folds
  it into the front. So a prefix cache (koboldcpp --loomcache, Anthropic
  cache_control, OpenAI and Codex automatic prefix caching) keeps the frames and the
  whole conversation across frame changes, and an unprovable case costs one cache
  miss, never a frame. Every vendor gets the same bytes. Nothing about frames, depth
  or boundaries is sent; provider cache signals stay the host's (Hermes
  prompt_caching; `prompt_caching.cache_ttl: auto` keeps human-paced sessions on
  Anthropic's 1h tier).

  Compaction is the built-in compressor, unchanged; it rewrites conversation turns only.

Install
  git clone https://github.com/jnorthrup/loom-hermes-plugin.git
  ln -sfn "$PWD/loom-hermes-plugin" "${HERMES_HOME:-$HOME/.hermes}/plugins/loom"
  # config.yaml
  context:
    engine: loom

  The link must be named `loom`: Hermes selects the engine by directory name.

Use
  The model gets one tool, `loom`:
    push {text, label?}   enter a level: its stable instructions, not findings
    pop  {count?}         leave one or more levels
    show                  print the frames
  /loom prints the frames in the interactive CLI, which holds one conversation per process:
  it reads only that CLI agent's own engine, and only while the engine is bound to the
  agent's session. Hermes passes slash handlers nothing but raw arguments, so in the TUI and
  gateways, which serve many sessions per process, /loom refuses and points to the show tool.

State and host compatibility
  Requires the paired Hermes ContextSelectionRejected, validate_request and
  inherit_from_parent hooks for its strict guarantees. Older hosts still run it:
  a refusal fails open (the request goes out unframed), goal reads are not
  strict, and the front stays within a tenth of the context window.
  No host or gateway is installed/restarted by these source changes.

  $HERMES_HOME/loom/session-<sha256 of exact session ID>.json carries the exact
  session and resolved profile home. Files are bounded, validated, written privately
  through unique temporary files, fsynced and atomically replaced. A successful
  reset writes an empty stack, so restart cannot restore cleared frames. A failed
  load or write refuses requests until repaired or explicitly reset; required frames
  are never silently truncated. Separate sessions/profiles have separate engine state.
  One active writer per session remains required; atomic replacement is not a
  cross-process compare-and-swap ledger.

  Legacy sanitized filenames do not identify their owner and can collide. They are
  preserved, never auto-imported. An operator can explicitly reset the affected
  session, or review its original ownership and re-enter frames through the tool.

  Delegation uses a host call after child session initialization to copy only frames
  from an explicitly supplied parent in the same resolved profile. Child edits do
  not change the parent. Unknown or cross-profile inheritance refuses. Compression
  rotation carries frames only with an explicit matching parent/session boundary.
  Goals are read strictly in the engine's bound profile, even on a differently scoped
  thread. A missing goal is empty; unavailable storage or malformed persisted goal
  content refuses. This requires the paired Hermes load_goal(strict=True) seam.

Context budget
  Required frames are appended without changing persisted history. The paired host
  validates the FINAL built request after provider cap/schema overrides and middleware,
  and again at the execution-middleware terminal callback before its main provider
  dispatch. Observation callbacks receive isolated copies. Rejection preserves accepted input, does not
  retry/fallback, and never trims instructions. Earlier auxiliary compression calls
  may already have happened. Incidental final-validator errors also refuse.

  Budget enforcement currently supports text chat-completions with an explicit output
  cap and a reviewed model/template contract in context.loom. No contract is enabled
  by default. Native Responses/Anthropic wire formats, image/audio content and opaque
  extra_body overrides and managed Relay execution refuse when frames or a goal are required.

  The contract is an OPERATOR ASSERTION, not tokenizer detection: verify the deployed
  model uses a byte-fallback tokenizer (at most one ordinary token per UTF-8 byte),
  and bound its template/framing costs beyond serialized request bytes. Unknown models
  or templates must remain disabled. Pin the endpoint/model operationally; matching
  a model name alone does not prove what a remote server actually loaded.

    context:
      engine: loom
      loom:
        model: <exact reviewed model name>
        tokenizer: utf8-byte-fallback
        prompt_overhead: <reviewed framing token upper bound>
        message_overhead: <reviewed extra tokens per message, including tool calls>
        tool_overhead: <reviewed extra tokens per tool schema>

  Charge = serialized messages/tools UTF-8 bytes + the declared framing bounds +
  actual wire output cap. This conservative bound may reject requests that an exact
  tokenizer would fit. It is not a universal or provider-exact token estimate.
  Review a contract before activation; do not copy the synthetic test bounds into
  production. If the selected model differs or no defensible bound exists, it refuses.

Precook (precook/, stdlib only, runs without Hermes)
  A loom script (precook/examples/ashford.json) is the book outline of a context
  evolution: goal, then up to 64 labelled frames, leaves carry the task.
  Both precook variants use the shared strict frame bounds: 4000 characters per
  frame and the same label syntax. They are standalone generators, not Hermes
  engines; they do not claim the engine persistence or final request budget guard.

  loom-loop-precook   precook/loom_loop.py
    emit SCRIPT -o book.py     the outline as literal nested `for` loops, one loop
                               variable per level; each variable is a frame pushed on
                               entry and popped on exit; each leaf is one chat request.
                               book.py is the curated artifact: read it, edit it, run it.
    run SCRIPT --base-url URL  run exactly the emitted program; --prewarm sends
                               max_tokens=1 so the walk only cooks the server's cache;
                               --out RUN.jsonl keeps a loom-run/1 artifact; --dry-run.
    replay RUN.jsonl           re-send a stored run's exact requests.
    prompts SCRIPT             one flattened prompt per leaf (NUL separated).
    Bearer key comes from $LOOM_API_KEY (name via --api-key-env), never the command line.

  ralph-precook       precook/ralph.sh SCRIPT [STATE]
    The Ralph loop, with loom_loop.py as the generator: each lap pipes the next
    leaf's prompt (goal + enclosing frames + task) into LOOM_HARNESS, depth-first,
    so consecutive prompts share their front. LOOM_HARNESS is any command reading
    the prompt on stdin: 'claude -p', 'codex exec -', 'hermes chat -Q -q "$(cat)"'.
    LOOM_LAPS (0 = forever), LOOM_DONE stop file, resumable STATE.

  Server mate: koboldcpp-loom --loomdir DIR serves /v1/looms, which stores the same
  scripts, runs the same walk through its own /v1/chat/completions, and keeps run
  artifacts for replay. loomframes.py here and the copy in koboldcpp-loom's
  loomstore.py must render the same bytes for valid frames (format loom-frames/1).
  Validation helpers may differ; the frame rendering format has not changed.

TODO
  - Hermes-driven precook: let the context engine push/pop frames from a loom script
    so a live /goal session follows the outline instead of the model's own pushes.
  - Stored state: text only (recipes, run artifacts); KV/state snapshots ruled out as
    GB-scale. See koboldcpp-loom README-loom.txt.

Local verification
  Use the paired Hermes test interpreter and runner, with no runtime activation:
    HERMES_PYTHON=/path/to/hermes-agent/.venv/bin/python /path/to/hermes-agent/scripts/run_tests.sh /path/to/loom-hermes-plugin/tests --import-mode=importlib
  Tests use disposable profile homes and deny socket connections.
  On a host without the paired seams, tests/run_unpaired.py supplies test-only stand-ins
  (see its docstring): 12 host-pair and 6 strict-goal tests fail there; the rest must pass.
  tests/pytest.ini makes tests/ the pytest rootdir, so the suite collects under any checkout
  name (a root named loom-hermes-plugin is not an importable package).

License and home
  MIT (LICENSE), copyright Jim Northrup. This repository is the plugin's home:
  changes land here first, and integrations pin a commit from it.
