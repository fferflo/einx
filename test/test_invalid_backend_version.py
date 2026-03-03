import pytest
import sys
import importlib
import einx


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_import_torch():
    try:
        import torch
    except ImportError:
        return

    x = torch.zeros((10, 11))

    version = tuple(int(i) for i in torch.__version__.split(".")[:2])
    if version < (2, 2):
        with pytest.raises(einx.errors.BackendResolutionError):
            einx.id("a b -> b a", x)
        with pytest.raises(einx.errors.ImportBackendError):
            einx.id("a b -> b a", x, backend="torch")
    else:
        einx.id("a b -> b a", x)
        einx.id("a b -> b a", x, backend="torch")
