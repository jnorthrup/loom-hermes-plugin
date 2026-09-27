loom - Hermes context engine: /goal work as nested loops
========================================================

What it does
  Work nests like for loops. Entering a level of work pushes a frame (its stable
  instructions); leaving it pops the frame. The frames on the stack, outermost first,
  sit at the front of every request after the /goal text:

    system prompt | goal | frame 1 | frame 2 | ... | innermost frame | conversation

  Each frame is an independent obstack frame: it holds only its own text and nothing
  refers across frames. The next sibling iteration is pop + push, so only what follows
  the shared enclosing frames changes. A prefix cache on the server (koboldcpp
  --loomcache, or a provider prompt cache) keeps everything before the change. Nothing
  about frames, depth or boundaries is sent to the server.

  Compaction is the built-in compressor, unchanged; it rewrites conversation turns only.

Install
  git clone git@github.com:jnorthrup/loom-hermes-plugin.git
  ln -sfn "$PWD/loom-hermes-plugin" ~/.hermes/plugins/loom
  # config.yaml
  context:
    engine: loom

  The link must be named `loom`: Hermes selects the engine by directory name.

Use
  The model gets one tool, `loom`:
    push {text, label?}   enter a level: its stable instructions, not findings
    pop  {count?}         leave one or more levels
    show                  print the frames
  /loom in the session prints the stack.

State
  $HERMES_HOME/loom/<session_id>.json. A delegated subagent starts inside its parent's
  frames, like an inner loop.

Precook (precook/, stdlib only, runs without Hermes)
  A loom script (precook/examples/ashford.json) is the book outline of a context
  evolution: goal, then any depth of labelled frames, leaves carry the task.

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
  loomstore.py must stay byte-identical (format loom-frames/1).

TODO
  - Hermes-driven precook: let the context engine push/pop frames from a loom script
    so a live /goal session follows the outline instead of the model's own pushes.
  - Stored state (the persistence continuum): see koboldcpp-loom README-loom.txt and
    runpod experiments/loom/docs/looms-hosting-proposal.md.
