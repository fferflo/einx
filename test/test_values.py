import importlib
if importlib.util.find_spec("jax"):
    import jax
if importlib.util.find_spec("torch"):
    import torch
if importlib.util.find_spec("tensorflow"):
    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow

import einx, pytest
import numpy as np

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_values(backend):
    rng = np.random.default_rng(42)

    x = backend.to_tensor(rng.uniform(size=(13,)).astype("float32"))
    assert backend.allclose(
        einx.vmap("b -> b [3]", x, op=lambda x: x + backend.ones((3,))),
        einx.add("b, -> b 3", x, 1),
    )

    x = backend.to_tensor(rng.uniform(size=(10, 20, 3)).astype("float32"))
    y = backend.to_tensor(rng.uniform(size=(10, 24)).astype("float32"))
    assert backend.allclose(
        einx.dot("a b c, a d -> a b c d", x, y),
        einx.vmap("a [b c], a [d] -> a [b c d]", x, y, op=lambda x, y: einx.dot("b c, d -> b c d", x, y)),
    )

    assert backend.allclose(
        einx.mean("a b [c]", x),
        einx.vmap("a b [c] -> a b", x, op=backend.mean),
    )
