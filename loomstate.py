"""Bounded, identity-bound frame persistence. No process-global session state."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from .loomframes import validate_frames

MAX_STATE_BYTES = 2 * 1024 * 1024


def state_path(home, session):
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()
    return Path(home) / "loom" / f"session-{digest}.json"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate state field")
        result[key] = value
    return result


def load(path, home, session):
    # Never follow a state-file symlink, including a dangling link.
    if path.parent.is_symlink():
        raise ValueError("state directory must not be a symlink")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STATE_BYTES:
            raise ValueError("state file is not a bounded regular file")
        raw = f.read(MAX_STATE_BYTES + 1)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("state file exceeds limit")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {"version", "home", "session", "frames"}:
        raise ValueError("state format requires explicit identity")
    if type(value["version"]) is not int or value["version"] != 1 or value["home"] != home or value["session"] != session:
        raise ValueError("state identity does not match session")
    return validate_frames(value["frames"])


def save(path, home, session, frames):
    frames = validate_frames(frames)
    data = json.dumps({"version": 1, "home": home, "session": session, "frames": frames}, ensure_ascii=True).encode()
    if len(data) > MAX_STATE_BYTES:
        raise ValueError("state file exceeds limit")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("state paths must not be symlinks")
    fd, temporary = tempfile.mkstemp(prefix=".loom-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
