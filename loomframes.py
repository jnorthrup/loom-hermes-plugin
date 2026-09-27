"""Frame rendering shared by the Hermes plugin, the loop precook and koboldcpp-loom's /v1/looms.

Format "loom-frames/1". koboldcpp-loom carries a byte-identical copy (koboldcpp.py,
loom_render_frames); change both together and bump FORMAT if the text changes, because every
cached prefix and every stored loom script depends on these exact bytes. Stdlib only.
"""

FORMAT = "loom-frames/1"
OPEN = "[loom: nested work frames, outermost first; the last frame is the current loop]"
CLOSE = "[/loom]"


def frame_name(depth, label):
    return f"{depth + 1}" + (f" {label}" if label else "")


def render_frames(frames, goal=""):
    """frames: [{"label", "text"}], outermost first."""
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
