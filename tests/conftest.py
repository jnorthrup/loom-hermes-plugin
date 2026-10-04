"""Local-only tests; import the real paired Hermes host under a disposable home."""
import atexit
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile

import pytest

_home = tempfile.mkdtemp(prefix="loom-plugin-tests-")
atexit.register(shutil.rmtree, _home, ignore_errors=True)
os.environ["HERMES_HOME"] = _home
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hermes-agent"))

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    def deny(*args, **kwargs):
        raise AssertionError("network access is forbidden in plugin tests")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)

@pytest.fixture
def loom():
    path = Path(__file__).resolve().parents[1]
    name = "test_loom_plugin"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def precook():
    path = Path(__file__).resolve().parents[1] / "precook" / "loom_loop.py"
    spec = importlib.util.spec_from_file_location("test_precook", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
