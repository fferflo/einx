import pytest
import functools
import importlib
import types
from functools import partial
import numpy as np
import einx
from contextlib import suppress
from conftest import use_backend
from conftest import assert_allclose

OperationNotSupportedError = einx.errors.OperationNotSupportedError


@pytest.mark.computes_values
@use_backend
def test_values(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    rng = np.random.default_rng(42)

    x = setup.to_tensor(rng.uniform(size=(10, 20, 3)).astype("float32"))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.multiply("a b c, a b c, a b c", x, x, x),
            x * x * x,
            setup=setup,
        )

    x = setup.full((10, 10), dtype="float32", value=1)
    y = setup.full((10,), dtype="float32", value=1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.dot("a, -> a", y, 1.0),
            y,
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.dot("a b, b -> a", x, y),
            y * 10,
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.dot("a, b -> a b", y, y),
            x,
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.dot("a b, b -> a b", x, y),
            einx.multiply("a b, b -> a b", x, y),
            setup=setup,
        )

    if "mlx.vmap" not in setup.name and "torch.vmap" not in setup.name:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            x = setup.to_tensor(np.arange(6)[np.newaxis])
            q, k, v = einx.id("b (q+k+v) -> b q, b k, b v", x, q=2, k=2, v=2)
            assert_allclose(q, [[0, 1]], setup=setup)
            assert_allclose(k, [[2, 3]], setup=setup)
            assert_allclose(v, [[4, 5]], setup=setup)

        with suppress((OperationNotSupportedError, *setup.exceptions)):
            x = setup.to_tensor(np.arange(4)[np.newaxis])
            q, k = einx.id("b (q+k) -> b q, b k", x, q=2)
            assert_allclose(q, [[0, 1]], setup=setup)
            assert_allclose(k, [[2, 3]], setup=setup)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        x = setup.to_tensor(np.arange(4).reshape((2, 2)))
        a, b, c, d = einx.id("(a + b) (c + d) -> (a c), (a d), (b c), (b d)", x, a=1, b=1, c=1, d=1)
        assert_allclose(a, [0], setup=setup)
        assert_allclose(b, [1], setup=setup)
        assert_allclose(c, [2], setup=setup)
        assert_allclose(d, [3], setup=setup)

    x = setup.to_tensor(np.arange(4)[np.newaxis])
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.flip("a [b]", x),
            [[3, 2, 1, 0]],
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.roll("a [b]", x, shift=2),
            [[2, 3, 0, 1]],
            setup=setup,
        )

    x = setup.to_tensor(np.arange(10))
    y = setup.to_tensor(np.arange(10)[::-1].copy())
    z = setup.to_tensor(np.arange(10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.get_at("[h], h2 -> h2", x, y),
            y,
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.set_at("[h], h2, h2 -> [h]", x, y, z),
            y,
            setup=setup,
        )

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        x = setup.to_tensor(rng.uniform(size=(4, 5, 6)).astype("float32"))
        y = setup.full((4, 5), value=3, dtype="int64")
        assert_allclose(
            einx.get_at("... [d], ... -> ...", x, y),
            x[:, :, 3],
            setup=setup,
        )

    if not ("torch" in setup.name and "compile" in setup.name):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            x = setup.to_tensor(np.zeros((3,)).astype("float32"))
            indices = 0
            updates = setup.to_tensor(np.arange(10).astype("float32"))
            x = einx.add_at("[h], , p", x, indices, updates)
            assert_allclose(
                x,
                np.asarray([np.sum(np.arange(10)), 0, 0]),
                setup=setup,
            )

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        x = setup.to_tensor(np.zeros((10,)).astype("int32"))
        indices = setup.to_tensor(np.arange(10).astype("int32"))
        updates = setup.to_tensor(np.arange(10).astype("int32"))
        assert_allclose(
            einx.set_at("[h], p, p", x, indices, updates),
            np.arange(10),
            setup=setup,
        )

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        x = setup.full((10, 10, 2), dtype="float32", value=1)
        assert_allclose(
            einx.sum("[a a] b -> b", x),
            np.full((2,), 100),
            setup=setup,
        )


@pytest.mark.computes_values
def test_compare_backends(setup_backend):
    import einx

    einx2, backend2, setup = setup_backend.einx, setup_backend.backend, setup_backend

    x = np.random.uniform(size=(10, 3, 10)).astype("float32")
    xb = setup.to_tensor(x)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.sum("a [b] c", x),
            einx2.sum("a [b] c", xb, backend=backend2),
            setup=setup,
        )
    if not ("torch" in setup.name and "compile" in setup.name) and "mlx" not in setup.name:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert_allclose(
                einx.softmax("a [b] c", x),
                einx2.softmax("a [b] c", xb, backend=backend2),
                setup=setup,
            )
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert_allclose(
                einx.log_softmax("a [b] c", x),
                einx2.log_softmax("a [b] c", xb, backend=backend2),
                setup=setup,
            )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.logsumexp("a [b] c", x),
            einx2.logsumexp("a [b] c", xb, backend=backend2),
            setup=setup,
        )

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.flip("a [b c]", x),
            einx2.flip("a [b c]", xb, backend=backend2),
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.flip("a [b c]", x),
            einx2.flip("a [b c]", xb, backend=backend2),
            setup=setup,
        )

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.roll("a [b] c", x, shift=2),
            einx2.roll("a [b] c", xb, shift=2, backend=backend2),
            setup=setup,
        )
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert_allclose(
            einx.roll("a [b c]", x, shift=(-2, -3)),
            einx2.roll("a [b c]", xb, shift=(-2, -3), backend=backend2),
            setup=setup,
        )

    x = np.random.uniform(size=(10, 3, 10)).astype("float32")
    xb = setup.to_tensor(x)

    y = np.random.uniform(size=(3, 10, 2, 10)).astype("float32")
    yb = setup.to_tensor(y)

    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 3, 0)):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert_allclose(
                einx.dot("[a] b [c], b [c] x [a] -> b x", x, y),
                einx2.dot("[a] b [c], b [c] x [a] -> b x", xb, yb, backend=backend2),
                setup=setup,
            )
