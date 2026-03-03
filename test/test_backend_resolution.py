import pytest
import sys
import importlib
import einx


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("jax") is None, reason="torch and jax must be installed")
def test_backend_resolution():
    import numpy as np
    import torch
    import jax.numpy as jnp

    x_numpy = np.zeros((2, 3))
    x_torch = torch.zeros(2, 3)
    x_jax = jnp.zeros((2, 3))

    einx.add("..., ...", x_numpy, x_numpy)
    einx.add("..., ...", x_numpy, x_torch)
    einx.add("..., ...", x_numpy, x_jax)
    einx.add("..., ...", x_numpy, x_torch, backend="torch")
    einx.add("..., ...", x_numpy, x_jax, backend="jax")
    with pytest.raises(einx.errors.BackendResolutionError):
        einx.add("..., ...", x_torch, x_jax)
    with pytest.raises(einx.errors.BackendResolutionError):
        einx.add("..., ..., ...", x_numpy, x_torch, x_jax)
