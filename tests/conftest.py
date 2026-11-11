import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pipeline():
    """BENCHMARK_SOURCE_ROOT permits the exact same regressions on old bytes."""
    root = Path(os.environ.get("BENCHMARK_SOURCE_ROOT", Path(__file__).parents[1]))
    return SimpleNamespace(**{name: _load(root / "src" / f"{name}.py", name)
                              for name in ("train", "preprocess", "evaluate")})


@pytest.fixture(autouse=True)
def deterministic_test_runtime():
    torch.manual_seed(7)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
