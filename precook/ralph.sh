#!/usr/bin/env bash
# ralph-precook: the Ralph Wiggum loop ("while :; do cat PROMPT | agent; done"), except each
# lap takes the next prompt from a loom script. The loom plugin's loop precook is the generator:
# it renders goal + the enclosing frames + the leaf task, depth-first, so consecutive prompts
# share their front byte for byte. Any agent harness that reads a prompt works.
#
#   ralph.sh SCRIPT.json [STATE.json]
#
# Environment:
#   LOOM_HARNESS   command that receives the prompt on stdin (default below). Examples:
#                    'claude -p'                      Claude Code, print mode
#                    'codex exec -'                   Codex
#                    'hermes chat -Q -q "$(cat)"'     Hermes, one-shot quiet
#   LOOM_LAPS      how many times to walk the whole outline (default 1; 0 = forever, like Ralph)
#   LOOM_DONE      stop early when this file exists (default: .loom-done next to STATE)
#   LOOM_LOG       append each lap's harness output here (default: STATE with .log)
set -u -o pipefail

script=${1:?usage: ralph.sh SCRIPT.json [STATE.json]}
state=${2:-${script%.json}.state.json}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py=${PYTHON:-python3}
harness=${LOOM_HARNESS:-'claude -p'}
laps=${LOOM_LAPS:-1}
done_file=${LOOM_DONE:-$(dirname "$state")/.loom-done}
log=${LOOM_LOG:-${state%.json}.log}

lap=1
while :; do
  [ -e "$done_file" ] && { echo "ralph: $done_file present, stopping" >&2; break; }
  prompt_file=$(mktemp "${TMPDIR:-/tmp}/loom-prompt.XXXXXX")
  "$py" "$here/loom_loop.py" next "$script" "$state" > "$prompt_file"
  rc=$?
  if [ $rc -eq 3 ]; then                       # outline exhausted: one lap done
    rm -f "$prompt_file" "$state"
    if [ "$laps" -ne 0 ] && [ "$lap" -ge "$laps" ]; then break; fi
    lap=$((lap + 1)); echo "ralph: lap $lap" >&2; continue
  elif [ $rc -ne 0 ]; then
    rm -f "$prompt_file"; echo "ralph: generator failed ($rc)" >&2; exit $rc
  fi
  { printf '\n=== lap %s  %s\n' "$lap" "$(date -u +%FT%TZ)"; } >> "$log"
  bash -c "$harness" < "$prompt_file" 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  rm -f "$prompt_file"
  if [ "$status" -ne 0 ]; then echo "ralph: harness exited $status; state kept, rerun to resume" >&2; exit "$status"; fi
done
