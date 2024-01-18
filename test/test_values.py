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
        einx.multiply("a b c, a b c, a b c", x, x, x),
        x * x * x,
    )

    assert backend.allclose(
        einx.mean("a b [c]", x),
        einx.vmap("a b [c] -> a b", x, op=backend.mean),
    )

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
    assert backend.allclose(
        einx.dot("a [b|b]", x, y),
        einx.multiply("a b, b -> a b", x, y),
    )

    x = backend.to_tensor(np.arange(6)[np.newaxis])
    q, k, v = einx.rearrange("b (q+k+v) -> b q, b k, b v", x, q=2, k=2, v=2)
    assert backend.allclose(q, backend.to_tensor([[0, 1]]))
    assert backend.allclose(k, backend.to_tensor([[2, 3]]))
    assert backend.allclose(v, backend.to_tensor([[4, 5]]))

    x = backend.to_tensor(np.arange(4)[np.newaxis])
    q, k = einx.rearrange("b (q+k) -> b q, b k", x, q=2)
    assert backend.allclose(q, backend.to_tensor([[0, 1]]))
    assert backend.allclose(k, backend.to_tensor([[2, 3]]))

    x = backend.to_tensor(np.arange(4).reshape((2, 2)))
    a, b, c, d = einx.rearrange("(a + b) (c + d) -> (a c), (a d), (b c), (b d)", x, a=1, b=1, c=1, d=1)
    assert backend.allclose(a, backend.to_tensor([0]))
    assert backend.allclose(b, backend.to_tensor([1]))
    assert backend.allclose(c, backend.to_tensor([2]))
    assert backend.allclose(d, backend.to_tensor([3]))

    x = backend.to_tensor(np.arange(4)[np.newaxis])
    assert backend.allclose(
        einx.flip("a [b]", x),
        backend.to_tensor([[3, 2, 1, 0]]),
    )
    assert backend.allclose(
        einx.roll("a [b]", x, shift=2),
        backend.to_tensor([[2, 3, 0, 1]]),
    )

    x = backend.to_tensor(np.arange(10))
    y = backend.to_tensor(np.arange(10)[::-1].copy())
    z = backend.to_tensor(np.arange(10))
    assert backend.allclose(
        einx.get_at("[h], h2 -> h2", x, y),
        y,
    )
    assert backend.allclose(
        einx.set_at("[h], h2, h2 -> [h]", x, y, z),
        y,
    )

    assert backend.allclose(
        backend.cast(einx.arange("a b [2]", a=5, b=6, backend=backend), "int32"),
        backend.to_tensor(np.stack(np.meshgrid(np.arange(5), np.arange(6), indexing="ij"), axis=-1).astype("int32")),
    )
    assert backend.allclose(
        backend.cast(einx.arange("b a -> a b [2]", a=5, b=6, backend=backend), "int32"),
        backend.to_tensor(np.stack(np.meshgrid(np.arange(6), np.arange(5), indexing="xy"), axis=-1).astype("int32")),
    )

    coord_dtype = "int32" if backend.name != "torch" else "long"
    x = backend.to_tensor(rng.uniform(size=(4, 5, 6)).astype("float32"))
    y = backend.cast(backend.ones((4, 5)) * 3, coord_dtype)
    assert backend.allclose(
        einx.get_at("... [d], ... -> ...", x, y),
        x[:, :, 3],
    )

@pytest.mark.parametrize("backend", backends)
def test_compare_backends(backend):
    x = np.random.uniform(size=(10, 3, 10)).astype("float32")

    assert backend.allclose(
        backend.to_tensor(einx.sum("a [b] c", x)),
        einx.sum("a [b] c", backend.to_tensor(x)),
    )
    assert backend.allclose(
        backend.to_tensor(einx.softmax("a [b] c", x)),
        einx.softmax("a [b] c", backend.to_tensor(x)),
    )
    assert backend.allclose(
        backend.to_tensor(einx.log_softmax("a [b] c", x)),
        einx.log_softmax("a [b] c", backend.to_tensor(x)),
    )
    assert backend.allclose(
        backend.to_tensor(einx.logsumexp("a [b] c", x)),
        einx.logsumexp("a [b] c", backend.to_tensor(x)),
    )