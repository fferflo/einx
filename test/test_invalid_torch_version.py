import pytest
import sys
import importlib
import einx


@pytest.mark.skipif(importlib.find_loader("torch") is None, reason="torch is not installed")
def test_import():
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
