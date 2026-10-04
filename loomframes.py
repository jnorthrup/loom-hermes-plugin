"""Frame rendering shared by the Hermes plugin, the loop precook and koboldcpp-loom's /v1/looms.

Format "loom-frames/1". koboldcpp-loom carries a byte-identical copy (koboldcpp.py,
loom_render_frames); change both together and bump FORMAT if the text changes, because every
cached prefix and every stored loom script depends on these exact bytes. Stdlib only.
"""

import re

MAX_TEXT = 4000
MAX_DEPTH = 64
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")

def validate_frame(frame):
    """Validate without silently coercing, dropping or shortening required instructions."""
    if not isinstance(frame, dict) or set(frame) - {"label", "text"}:
        raise ValueError("frame must contain only label and text")
    text, label = frame.get("text"), frame.get("label", "")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f"frame text must contain 1-{MAX_TEXT} characters")
    if not isinstance(label, str) or (label and not _LABEL.fullmatch(label)):
        raise ValueError("frame label must use 1-64 letters, digits, underscore, dot or hyphen")
    return {"label": label, "text": text}

def validate_frames(frames):
    if not isinstance(frames, list) or len(frames) > MAX_DEPTH:
        raise ValueError(f"frames must be a list of at most {MAX_DEPTH} entries")
    return [validate_frame(frame) for frame in frames]

FORMAT = "loom-frames/1"
OPEN = "[loom: nested work frames, outermost first; the last frame is the current loop]"
CLOSE = "[/loom]"


def frame_name(depth, label):
    return f"{depth + 1}" + (f" {label}" if label else "")


def render_frames(frames, goal=""):
    """frames: [{"label", "text"}], outermost first."""
    frames = validate_frames(frames)
    parts = [OPEN]
    if goal:
        parts.append("== goal ==\n" + goal)
    for depth, frame in enumerate(frames):
        parts.append(f"== {frame_name(depth, frame.get('label') or '')} ==\n{frame['text']}")
    parts.append(CLOSE)
    return "\n".join(parts)


def with_frames(system, frames, goal=""):
    """System prompt text with the frame block appended; unchanged when there is nothing to add."""
    if not frames and not goal:
        return system
    block = render_frames(frames, goal)
    return f"{system}\n\n{block}" if system else block
