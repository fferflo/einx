import pytest
import sys
import importlib
import einx


@pytest.mark.mlx
@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="mlx is not installed")
def test_import_mlx():
    try:
        import mlx.core as mx
    except ImportError:
        return

    x = mx.zeros((10,))

    version = tuple(int(i) for i in mx.__version__.split(".")[:3])
    if version < (0, 16, 1):
        with pytest.raises(einx.backend.InvalidBackendException):
            einx.add("a, a", x, x)
        with pytest.raises(einx.backend.InvalidBackendException):
            einx.add("a, a", x, x, backend="mlx")
    else:
        einx.add("a, a", x, x)
        einx.add("a, a", x, x, backend="mlx")


@pytest.mark.torch
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_import_torch():
    import torch

    x = torch.zeros((10,))

    major = int(torch.__version__.split(".")[0])
    if major < 2:
        with pytest.raises(einx.backend.InvalidBackendException):
            einx.add("a, a", x, x)
        with pytest.raises(einx.backend.InvalidBackendException):
            einx.add("a, a", x, x, backend="torch")
    else:
        einx.add("a, a", x, x)
        einx.add("a, a", x, x, backend="torch")
