import conftest
import einx
import pytest
import numpy as np


def allclose(x, y, setup):
    if isinstance(x, (list, int, float, tuple, np.ndarray)):
        x = np.asarray(x)
    else:
        x = setup.to_numpy(x)
    if isinstance(y, (list, int, float, tuple, np.ndarray)):
        y = np.asarray(y)
    else:
        y = setup.to_numpy(y)
    return np.allclose(x, y)


@pytest.mark.parametrize("test", conftest.tests)
def test_values(test):
    einx, backend, setup = test

    rng = np.random.default_rng(42)

    if backend.name not in {"mlx", "dask", "tinygrad"}:
        x = setup.to_tensor(rng.uniform(size=(13,)).astype("float32"))
        assert allclose(
            einx.vmap("b -> b [3]", x, op=lambda x: x + setup.full((3,), value=1)),
            einx.add("b, -> b 3", x, 1),
            setup=setup,
        )

    x = setup.to_tensor(rng.uniform(size=(10, 20, 3)).astype("float32"))
    y = setup.to_tensor(rng.uniform(size=(10, 24)).astype("float32"))
    if backend.name not in {"mlx", "dask", "tinygrad"}:
        assert allclose(
            einx.dot("a b c, a d -> a b c d", x, y),
            einx.vmap(
                "a [b c], a [d] -> a [b c d]",
                x,
                y,
                op=lambda x, y: einx.dot("b c, d -> b c d", x, y),
            ),
            setup=setup,
        )

    assert allclose(
        einx.multiply("a b c, a b c, a b c", x, x, x),
        x * x * x,
        setup=setup,
    )

    if backend.name not in {"mlx", "dask", "tinygrad"}:
        assert allclose(
            einx.mean("a b [c]", x),
            einx.vmap("a b [c] -> a b", x, op=backend.mean),
            setup=setup,
        )

        assert einx.dot("[->]", 1, 1) == 1

        x = setup.full((10, 10), dtype="float32", value=1)
        y = setup.full((10,), dtype="float32", value=1)
        if backend.name != "torch":
            assert allclose(
                einx.dot("a [->]", y, 1),
                y,
                setup=setup,
            )
        assert allclose(
            einx.dot("a [b->]", x, y),
            y * 10,
            setup=setup,
        )
        assert allclose(
            einx.dot("a [->b]", y, y),
            x,
            setup=setup,
        )
        assert allclose(
            einx.dot("a [b->b]", x, y),
            einx.multiply("a b, b -> a b", x, y),
            setup=setup,
        )

    x = setup.to_tensor(np.arange(6)[np.newaxis])
    q, k, v = einx.rearrange("b (q+k+v) -> b q, b k, b v", x, q=2, k=2, v=2)
    assert allclose(q, [[0, 1]], setup=setup)
    assert allclose(k, [[2, 3]], setup=setup)
    assert allclose(v, [[4, 5]], setup=setup)

    x = setup.to_tensor(np.arange(4)[np.newaxis])
    q, k = einx.rearrange("b (q+k) -> b q, b k", x, q=2)
    assert allclose(q, [[0, 1]], setup=setup)
    assert allclose(k, [[2, 3]], setup=setup)

    x = setup.to_tensor(np.arange(4).reshape((2, 2)))
    a, b, c, d = einx.rearrange(
        "(a + b) (c + d) -> (a c), (a d), (b c), (b d)", x, a=1, b=1, c=1, d=1
    )
    assert allclose(a, [0], setup=setup)
    assert allclose(b, [1], setup=setup)
    assert allclose(c, [2], setup=setup)
    assert allclose(d, [3], setup=setup)

    x = setup.to_tensor(np.arange(4)[np.newaxis])
    assert allclose(
        einx.flip("a [b]", x),
        [[3, 2, 1, 0]],
        setup=setup,
    )
    assert allclose(
        einx.roll("a [b]", x, shift=2),
        [[2, 3, 0, 1]],
        setup=setup,
    )

    x = setup.to_tensor(np.arange(10))
    y = setup.to_tensor(np.arange(10)[::-1].copy())
    z = setup.to_tensor(np.arange(10))
    if backend.name not in {"mlx", "dask", "tinygrad"}:
        assert allclose(
            einx.get_at("[h], h2 -> h2", x, y),
            y,
            setup=setup,
        )
        assert allclose(
            einx.set_at("[h], h2, h2 -> [h]", x, y, z),
            y,
            setup=setup,
        )

    assert allclose(
        einx.arange("a b [2]", a=5, b=6, backend=backend),
        np.stack(np.meshgrid(np.arange(5), np.arange(6), indexing="ij"), axis=-1).astype("int32"),
        setup=setup,
    )
    assert allclose(
        einx.arange("b a -> a b [2]", a=5, b=6, backend=backend),
        np.stack(np.meshgrid(np.arange(6), np.arange(5), indexing="xy"), axis=-1).astype("int32"),
        setup=setup,
    )

    if backend.name not in {"mlx", "dask", "tinygrad"}:
        coord_dtype = "int32" if backend.name != "torch" else "long"
        x = setup.to_tensor(rng.uniform(size=(4, 5, 6)).astype("float32"))
        y = setup.full((4, 5), value=3, dtype=coord_dtype)
        assert allclose(
            einx.get_at("... [d], ... -> ...", x, y),
            x[:, :, 3],
            setup=setup,
        )


@pytest.mark.parametrize("test", conftest.tests)
def test_compare_backends(test):
    einx, backend, setup = test

    x = np.random.uniform(size=(10, 3, 10)).astype("float32")
    y = setup.to_tensor(x)

    assert allclose(
        einx.sum("a [b] c", x),
        einx.sum("a [b] c", y),
        setup=setup,
    )
    assert allclose(
        einx.softmax("a [b] c", x),
        einx.softmax("a [b] c", y),
        setup=setup,
    )
    assert allclose(
        einx.log_softmax("a [b] c", x),
        einx.log_softmax("a [b] c", y),
        setup=setup,
    )
    assert allclose(
        einx.logsumexp("a [b] c", x),
        einx.logsumexp("a [b] c", y),
        setup=setup,
    )

    assert allclose(
        einx.flip("a [b c]", x),
        einx.flip("a [b c]", y),
        setup=setup,
    )
    assert allclose(
        einx.flip("a [b c]", x),
        einx.flip("a [b c]", y),
        setup=setup,
    )

    assert allclose(
        einx.roll("a [b] c", x, shift=2),
        einx.roll("a [b] c", y, shift=2),
        setup=setup,
    )
    assert allclose(
        einx.roll("a [b c]", x, shift=(-2, -3)),
        einx.roll("a [b c]", y, shift=(-2, -3)),
        setup=setup,
    )
