import importlib
if importlib.util.find_spec("jax"):
    import jax
if importlib.util.find_spec("torch"):
    import torch
if importlib.util.find_spec("tensorflow"):
    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow
    import tensorflow.experimental.numpy as tnp
    tnp.experimental_enable_numpy_behavior()

import einx, pytest
import numpy as np

backends = [b for b in einx.backend.backends if b != einx.backend.tracer]

@pytest.mark.parametrize("backend", backends)
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

    x = backend.ones((10,), "float32")
    assert einx.dot("(a + a) -> ", x) == 5

    assert einx.dot("[|]", 1, 1) == 1

    x = backend.ones((10, 10), "float32")
    y = backend.ones((10,), "float32")
    assert backend.allclose(
        einx.dot("a [|]", y, 1),
        y,
    )
    assert backend.allclose(
        einx.dot("a [b|]", x, y),
        y * 10,
    )
    assert backend.allclose(
        einx.dot("a [|b]", y, y),
        x,
    )
