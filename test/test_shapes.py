import pytest
import conftest
import functools
import einx
import numpy as np
from contextlib import suppress
from conftest import use_backend
import warnings

OperationNotSupportedError = einx.errors.OperationNotSupportedError
EinxError = einx.errors.EinxError
BackendResolutionError = einx.errors.BackendResolutionError


@use_backend
def test_shape_id(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    x = setup.full((10, 20, 1))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c -> (a b) c 1 1 1", x).shape == (200, 1, 1, 1, 1)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a (a + b) c -> (a b) c 1", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a b c -> a c b -> b c a", x)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c -> a b", x).shape == (10, 20)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a b c -> a b c c", x)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c", x).shape == (10, 20, 1)

    x = setup.full((10,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.id("x", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("([x)] -> x", x)

    x = setup.full((10, 10, 10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a a a a -> a", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a a a b -> a b", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a a b a -> a b", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b a a -> a b", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("b a a a -> a b", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("b a c a -> a b c", x).shape == (10, 10, 10)
    with suppress((OperationNotSupportedError, BackendResolutionError, *setup.exceptions)):
        assert einx.id("b a c a -> a b c", setup.full, a=10, b=10, c=10).shape == (10, 10, 10)

    x = setup.full((100,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("(a a) -> a", x).shape == (10,)

    x = setup.full((10, 10))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a a -> a a", x)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 1 ->", np.asarray([[0]])).shape == ()
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 ->", np.asarray([0])).shape == ()
    if "arrayapi" not in setup.name:
        # array_api_compat.array_namespace fails to determine correct array_namespace here
        with suppress((OperationNotSupportedError, BackendResolutionError, *setup.exceptions)):
            assert einx.id("->", 1) == 1
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("1 1 ->", np.asarray([0]))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("1 ->", np.asarray([[0]]))

    x = setup.full((10, 10, 20, 1))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a a b c -> (a b) c 1", x).shape == (200, 1, 1)

    x = setup.full((200,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id(" ( a   b  ) ->   b   a   ", x, a=10).shape == (20, 10)

    x = setup.full((20, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("(a) (b) -> b a", x).shape == (10, 20)

    x = setup.full((10, 20, 20, 2))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("b s... c -> b (s...) c", x).shape == (10, 400, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("b ... c -> b (...) c", x).shape == (10, 400, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("b (s...) (r...) c -> b (s...) r... c", x, r=(10, 2)).shape == (10, 20, 10, 2, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("b s... c x... -> x... b (s...) c", x, x=()).shape == (10, 400, 2)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 -> (x)", np.asarray([1]), x=10).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 -> (x y)", np.asarray([1]), x=10, y=20).shape == (200,)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 -> (x)", setup.to_tensor([1]), x=10).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 -> (x y)", setup.to_tensor([1]), x=10, y=20).shape == (200,)

    x = setup.full((1,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 -> (x)", x, x=10).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("1 -> (x y)", x, x=10, y=20).shape == (200,)

    x = setup.full((10, 20, 1))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c d... -> a b c (d...)", x).shape == (10, 20, 1, 1)

    x = setup.full((10, 20, 1, 2))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a (b...) c d -> a (b... c) d", x).shape == (10, 20, 2)

    x = setup.full((10, 20, 1, 2, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a (b... c) d... e -> a (b...) (c d...) e", x, b=[2, 5]).shape == (10, 10, 4, 3)

    x = setup.full((10, 20, 6, 24))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b (c...) (d...) -> a c... b d...", x, c=[2, 3], d=[4, 6]).shape == (10, 2, 3, 20, 4, 6)

    x = setup.full((10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a... -> 1 (a...)", x).shape == (1, 100)

    x = setup.full((10, 20, 5))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x).shape == (1, 5, 10, 20)

    x = setup.full((10, 20))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x)

    x = setup.full((10, 20, 1))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a b c -> (a b) c d", x)

    x = setup.full((10, 20, 1))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a b... c... -> a (b...) c...", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a b... -> a b", x)

    x = setup.full((1, 10, 20, 6))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a (b...) (e f...) (d c) -> a d (b...) (e f...) c", x, d=2).shape == (1, 2, 10, 20, 3)

    x = setup.full((1, 10, 20, 6, 7, 12))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("a b c d... (e f...) -> a b c d... ((e 2 2) f...)", x, f=[2, 2]).shape == (1, 10, 20, 6, 7, 12 * 2 * 2)

    x = setup.full((10, 20, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("(s s2)... c -> s... s2... c", x, s2=(2, 2)).shape == (5, 10, 2, 2, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("(s s2)... c -> s... s2... c", x, s2=2).shape == (5, 10, 2, 2, 3)

    x = setup.full((10, 10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.id("(a b) (c d) (e f) -> a (b c d e) f", x, a=2, f=2).shape == (2, 250, 2)

    x = setup.full((10,))
    y = setup.full((20,))
    if "torch.vmap" not in setup.name and "mlx.vmap" not in setup.name:
        # torch with vmap rauses error: "can not return a BatchedTensor when out_dim is None"
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("a, b -> (a + b)", x, y).shape == (30,)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("a, b -> (a+ b)", x, y).shape == (30,)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("a, b -> (a +b)", x, y).shape == (30,)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("a, b ->(a+b)", x, y).shape == (30,)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        assert einx.id("a, b -> (b + a)", x, y).shape == (30,)

    if "torch.vmap" not in setup.name and "mlx.vmap" not in setup.name:
        # torch with vmap rauses error: "can not return a BatchedTensor when out_dim is None"
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("a, b -> a b (1 + 1)", x, y).shape == (10, 20, 2)
    if "torch.vmap" not in setup.name and "mlx.vmap" not in setup.name:
        # torch with vmap rauses error: "can not return a BatchedTensor when out_dim is None"
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert [x.shape for x in einx.id("(a + b) -> a, b 1", x, a=4)] == [(4,), (6, 1)]
    with pytest.raises((OperationNotSupportedError, ValueError, *setup.exceptions)):
        einx.id("a, b -> a b (1 + 1)", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("(a + b) -> a b (1 + 1)", x)
    if "torch.vmap" not in setup.name and "mlx.vmap" not in setup.name:
        # torch with vmap rauses error: "can not return a BatchedTensor when out_dim is None"
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("a, (b c) -> c (a + b)", x, y, c=2).shape == (2, 20)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.id("a, -> (a +)", x, 1)

    x = setup.full((10, 10))
    if "arrayapi" not in setup.name and "mlx.vmap" not in setup.name and "dask" not in setup.name:
        # array_api_compat.array_namespace fails to determine correct array_namespace here
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("b c, 1 -> b (c + 1)", x, np.asarray([42.0]).astype("float32")).shape == (10, 11)
    if "mlx.vmap" not in setup.name:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("b c, -> b (c + 1)", x, 42.0).shape == (10, 11)
        s = setup.full(())
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("b c, -> b (c + 1)", x, s).shape == (10, 11)

    x = setup.full((10, 20), dtype="bool")
    y = setup.full((4, 10, 20, 3))
    if "torch.vmap" not in setup.name and "mlx.vmap" not in setup.name:
        # torch with vmap rauses error: "can not return a BatchedTensor when out_dim is None"
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            x, y = einx.id("h w, b h w c -> 1 h w 1, b h w c", x, y)

        x = setup.full((5, 3))
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            x = einx.id("(a + b) a -> a, b a", x)
            assert x[0].shape == (3,)
            assert x[1].shape == (2, 3)

        x = np.zeros((10, 10))
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            y = einx.id("(a + b) (c + d) -> a c, a d, b c, b d", x, b=3, d=4)
            assert y[0].shape == (7, 6)
            assert y[1].shape == (7, 4)
            assert y[2].shape == (3, 6)
            assert y[3].shape == (3, 4)

        with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
            einx.id("(a + b) (c + d) -> a d, a c, b c, b d", x, b=3, d=4)

        x = setup.full((3, 10, 11, 2))
        y = setup.full((3, 2))
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.id("b c, b s... c -> b (1 + (s...)) c", y, x).shape == (3, 1 + 10 * 11, 2)

        z = setup.full((3, 1 + 10 * 11, 2))
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            y, x = einx.id("b (1 + (s...)) c -> b c, b s... c", z, s=(10, 11))
            assert y.shape == (3, 2)
            assert x.shape == (3, 10, 11, 2)


@use_backend
def test_shape_dot(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    x = setup.full((10, 10))
    y = setup.full((10 * 10,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a..., a... -> 1", x, x).shape == (1,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("[a...], [a...] -> 1", x, x).shape == (1,)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("a..., [a]... -> 1", x, x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("a b1, c b2 -> a c", x, x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("a b -> a", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("a b, a c -> b", x, x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("a [b], a [b] -> 1", x, x)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("([a () b] ()), [a b] -> 1", y, x).shape == (1,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("[([a () b] ())], [[a] b] -> 1", y, x).shape == (1,)

    x = setup.full((10, 20, 1))
    y = setup.full((10, 24))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a b c, a d -> 1 b c d", x, y).shape == (1, 20, 1, 24)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("[a] b c, [a] d -> 1 b c d", x, y).shape == (1, 20, 1, 24)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a b c, a d -> 1 b c d", x, setup.full, d=24).shape == (1, 20, 1, 24)

    x = setup.full((10, 10, 20))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a a [b], c c [b] -> a c", x, x).shape == (10, 10)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("[a a] b, [a a] c -> b c", x, x)

    x = setup.full((10, 20, 1))
    with pytest.raises((OperationNotSupportedError, ValueError, *setup.exceptions)):
        einx.dot("a b c -> a b c", x, x)
    with pytest.raises((OperationNotSupportedError, ValueError, *setup.exceptions)):
        einx.dot("a b c, a -> a b c", x)

    x = setup.full((10, 20))
    y = setup.full((20, 30))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a b, b c -> a c", x, y).shape == (10, 30)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a [b], [b] c -> a c", x, y).shape == (10, 30)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a [b...], [b...] c -> a c", x, y).shape == (10, 30)

    x = setup.full((10, 20))
    y = setup.full((10, 20, 30))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a b, a b c -> a c", x, y).shape == (10, 30)

    x = setup.full((10,))
    y = setup.full((30,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a, a ->", x, x).shape == ()
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a, c -> a c", x, y).shape == (10, 30)

    x = setup.full((4, 128, 128, 16))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("b s... [c1], [c1] c2 -> b s... c2", x, setup.full, c2=32).shape == (4, 128, 128, 32)
    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 3, 0)):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.dot("b [s...] c, [s...] s2 -> b s2 c", x, setup.full, s2=32).shape == (4, 32, 16)

    w = setup.full((2, 2, 16, 32))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("b (s [s2])... [c1], [s2... c1] c2 -> b s... c2", x, w, s2=2, c2=32).shape == (4, 64, 64, 32)

    x = setup.full((4, 16, 16, 16))

    x = setup.full((11, 10))
    y = setup.full((11,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("... b, ... -> b", x, y).shape == (10,)

    x = setup.full((10,))
    y = setup.full(())
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("... b, ... -> b", x, y).shape == (10,)

    k = setup.full((2, 4, 100))
    v = setup.full((2, 4, 100))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("b t (h ck), b t (h cv) -> b h ck cv", k, v, h=32, graph=True)

    x = setup.full((10, 20))
    y = setup.full((10, 24))
    z = setup.full((3, 24))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("[a] b, [a c], d [c] -> b d", x, y, z).shape == (20, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a b, a c, d c -> b d", x, y, z).shape == (20, 3)

    x = setup.full((10, 20, 24))
    y = setup.full((10,))
    z = setup.full((3, 24))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("[a] b [c], [a], d [c] -> b d", x, y, z).shape == (20, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a b c, a, d c -> b d", x, y, z).shape == (20, 3)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("[a] b c, a, d [c] -> b d", x, y, z)

    x = setup.full((10, 5, 1))
    y = setup.full((5,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.dot("a [b] 1, [b] -> a", x, y).shape == (10,)

    x = setup.full((10, 5))
    y = setup.full((5,))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("[a b], [b] ->", x, y)

    x = setup.full((10, 10))
    y = setup.full((10,))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("[a a], [a] ->", x, y)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("[a b] ->", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.dot("a b -> b a", x)
    with suppress((OperationNotSupportedError, ValueError, *setup.exceptions)):
        assert einx.dot("a b, c -> b a c", x).shape == (10, 10, 10)


@use_backend
def test_shape_reduce(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    x = setup.full((10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("a b -> 1 a", x).shape == (1, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("[a] b -> 1 b", x).shape == (1, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("[([a] ())] b -> 1 b", x).shape == (1, 10)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.mean("a [b] -> 1", x)

    x = setup.full((10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.var("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.std("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.prod("[a] b", x).shape == (10,)
    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 3, 0)):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.count_nonzero("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.min("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.max("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logsumexp("[a] b", x).shape == (10,)

    x = setup.full((10, 10), dtype="bool")
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.all("[a] b", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.any("[a] b", x).shape == (10,)

    x = setup.full((10, 10, 10))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.sum("a [b] c -> a b", x)

    x = setup.full((10, 10, 30))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("a a [b] -> a", x).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[a a] b -> b", x).shape == (30,)

    x = setup.full((10, 3, 1))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("(a [b]) c 1", x, b=2).shape == (5, 3, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("([a b]) c 1", x).shape == (1, 3, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("[(a b)] c 1", x).shape == (3, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("[(a...)] c 1", x).shape == (3, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("(b... [a...]) c 1", x, b=(1, 1)).shape == (1, 3, 1)

    x = setup.full((1, 10, 3, 2))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("1 [a...] b", x).shape == (1, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("1 [a]... b", x).shape == (1, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("1 ([a])... b", x).shape == (1, 1, 1, 2)

    x = setup.full((16, 1, 20, 30, 64))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.mean("(b rg) pv [s...] c", x).shape == (16, 1, 64)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logsumexp("a [...]", x).shape == (16,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logsumexp("[a]", np.asarray([0.0, 1.0])).shape == ()
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logsumexp("[a] 1", np.asarray([[0.0], [1.0]])).shape == (1,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logsumexp("[a]", np.asarray([0.0] * 10)).shape == ()

    x = setup.full((16, 15))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[b] a []", x).shape == (15,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[b] a [...]", x).shape == (15,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("b [p] -> b p2", x, p2=7).shape == (16, 7)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("16 [a]", x).shape == (16,)

    x = setup.full((10, 5, 1))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("a [b] 1 -> a", x).shape == (10,)

    x = setup.full((10, 10, 20))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.sum("a a b -> b", x)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[a a] b -> b", x).shape == (20,)

    x = setup.full((10, 10, 20))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[...] b", x).shape == (20,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[s...] b", x).shape == (20,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[s]... b", x).shape == (20,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sum("[(...)]... b", x).shape == (20,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        x = setup.full((2, 3, 4, 5))
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.sum("[a] b c [d]", x, keepdims=True).shape == (1, 3, 4, 1)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.sum("[a] b c [d]", x, keepdims=True).shape == (1, 3, 4, 1)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.sum("[...]", x, keepdims=True).shape == (1,)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.sum("[a...]", x, keepdims=True).shape == (1,)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.sum("[a]...", x, keepdims=True).shape == (1, 1, 1, 1)


@use_backend
def test_shape_elementwise(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    x = setup.full((10, 5, 1))
    y = setup.full((13,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b 1, l -> a b l", x, y).shape == (10, 5, 13)
    for _ in range(10):  # Test cache
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.add("a b 1, l -> a b l", x, y).shape == (10, 5, 13)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.add("a b 1, l -> b l a 1, b l a 1", x, y)

    mask = setup.full((10, 5), dtype="bool")
    x = setup.full((10, 5, 1))
    y = setup.full((13,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.subtract("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.multiply("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.true_divide("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.floor_divide("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.divide("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.maximum("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.minimum("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.less("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.less_equal("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.greater("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.greater_equal("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.equal("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    if not ("vmap" in setup.name and "mlx" in setup.name):  # Fails with mlx 0.30.1: Seems to be a problem with vmap in mlx
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.not_equal("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logaddexp("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.where("a b, a b 1, l -> b l a 1", mask, x, y).shape == (5, 13, 10, 1)

    x = setup.full((10, 5, 1), dtype="bool")
    y = setup.full((13,), dtype="bool")
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logical_and("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logical_or("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)

    z = setup.full((10, 5))
    x = setup.full((10, 5, 1))
    y = setup.full((13,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, a b 1, l -> b l a 1", z, x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.multiply("a b, a b 1, l -> b l a 1", z, x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.maximum("a b, a b 1, l -> b l a 1", z, x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.minimum("a b, a b 1, l -> b l a 1", z, x, y).shape == (5, 13, 10, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.logaddexp("a b, a b 1, l -> b l a 1", z, x, y).shape == (5, 13, 10, 1)

    x = setup.full((10, 10))
    y = setup.full((10,))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a, a b", y, x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, a", x, y).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, b", x, y).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, a b", x, x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, ", x, 1.0).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add(", a b", 1.0, x).shape == (10, 10)
    if "arrayapi" not in setup.name and "dask" not in setup.name:  # array_api_compat.array_namespace fails to determine correct array_namespace here
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.add("a b, 1", x, np.asarray([1.0]).astype("float32")).shape == (10, 10)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.add("1, a b", np.asarray([1.0]).astype("float32"), x).shape == (10, 10)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.add("a a, a -> a a", x, y)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, a b", x, setup.full).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a, a", y, y).shape == (10,)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("b, -> b 3", y, 1.0).shape == (10, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a a, a -> a", x, y).shape == (10,)

    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 3, 0)):

        def param(shape, x, y):
            return setup.full(shape)

        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.multiply("... c, c", x, functools.partial(param, x="a", y={"b": True, "c": False})).shape == (10, 10)

    x = setup.full((2, 3))
    y = setup.full((10,))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.add("a b, c", x, y)

    x = setup.full((3, 128, 196, 64))
    y = setup.full((3, 4, 16))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("b h w (g c), b (g) c -> b h w (g c)", x, y).shape == (3, 128, 196, 64)

    x = setup.full((10, 20))
    y = setup.full((10, 20, 30))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.add("a b, a b c -> a b c", x, y).shape == (10, 20, 30)

    x = setup.full((10, 20))
    y = setup.full((30, 20))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.subtract("ba c, i c -> i ba", x, y)

    ops = [
        ("add", ("float32", "float32")),
        ("subtract", ("float32", "float32")),
        ("multiply", ("float32", "float32")),
        ("true_divide", ("float32", "float32")),
        ("floor_divide", ("float32", "float32")),
        ("divide", ("float32", "float32")),
        ("logical_and", ("bool", "bool")),
        ("logical_or", ("bool", "bool")),
        ("where", ("bool", "float32", "float32")),
        ("less", ("float32", "float32")),
        ("less_equal", ("float32", "float32")),
        ("greater", ("float32", "float32")),
        ("greater_equal", ("float32", "float32")),
        ("equal", ("float32", "float32")),
        ("not_equal", ("float32", "float32")),
        ("maximum", ("float32", "float32")),
        ("minimum", ("float32", "float32")),
        ("logaddexp", ("float32", "float32")),
    ]

    def create_scalar(dtype):
        if dtype == "float32":
            return 1.0
        elif dtype == "bool":
            return True
        else:
            assert False

    for op, dtypes in ops:
        tensor_args = [setup.full((3,), dtype=dtype, value=1) for dtype in dtypes]
        scalar_args = [create_scalar(dtype) for dtype in dtypes]

        for scalar_index in range(len(dtypes)):
            args = [scalar_args[scalar_index] if i == scalar_index else tensor_args[i] for i in range(len(dtypes))]
            expr = ", ".join(["" if i == scalar_index else "a" for i in range(len(dtypes))]) + " -> a"
            with suppress((OperationNotSupportedError, *setup.exceptions)):
                assert getattr(einx, op)(expr, *args).shape == (3,)


@use_backend
def test_shape_argfind(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    if "mlx.vmap" in setup.name:
        pytest.xfail(reason='Problems with mlx.vmap, causes "Aborted (core dumped)"')

    x = setup.full((4, 16, 16, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmax("b [h w] c", x).shape == (4, 2, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmax("b [h w] c -> b [2] c", x).shape == (4, 2, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmax("b [h w] c -> [2] b c", x).shape == (2, 4, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmax("b [h w] c -> [i] b c", x).shape == (2, 4, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmax("b [h w] c -> ([2] b) c", x).shape == (2 * 4, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmax("[b] h [w] c -> [2] h c", x).shape == (2, 16, 3)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argmax("b [h w] c -> b [3] c", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argmax("b [h w] c -> b [1] c", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argmax("b [h w] c -> b c", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argmax("b [h w] c -> b [i] c", x, i=3)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h] w c", x).shape == (4, 1, 16, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h] w c -> b [1] w c", x).shape == (4, 1, 16, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h] w c -> b w c", x).shape == (4, 16, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h->1] w c", x).shape == (4, 1, 16, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h->] w c", x).shape == (4, 16, 3)
    with suppress((OperationNotSupportedError, BackendResolutionError, *setup.exceptions)):
        assert einx.argmin("b [h->] w c", setup.full, b=4, h=16, w=16, c=3).shape == (4, 16, 3)

    x = setup.full((4 * 16, 16, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("(b [h]) w c", x, b=4).shape == (4, 16, 3)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argmin("(b [h]) [w] c", x, b=4)  # Cannot implicitly determine output expression
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("(b [h]) [w] c -> [2] b c", x, b=4).shape == (2, 4, 3)

    x = setup.full((4, 16 * 16, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b ([h w]) c", x, h=16).shape == (4, 2, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [(h w)] c", x, h=16).shape == (4, 2, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [(h h)] c", x, h=16).shape == (4, 2, 3)

    x = setup.full((4, 16, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b ([h]) c", x, h=16).shape == (4, 1, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [(h)] c", x, h=16).shape == (4, 1, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h] c", x, h=16).shape == (4, 1, 3)

    x = setup.full((4, 16, 16, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b [h h] c", x, h=16).shape == (4, 2, 3)

    x = setup.full((4, 4, 16, 16, 3))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argmin("b b [h h] c", x, h=16)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b b [h h] c -> b [2] c", x, h=16).shape == (4, 2, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argmin("b b [h h] c -> c b [2]", x, h=16).shape == (3, 4, 2)


@use_backend
def test_shape_index(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    x = setup.full((4, 16, 17, 3))
    y = setup.full((4, 128, 2), dtype="int64")
    y2 = setup.full((128, 4, 2), dtype="int64")
    z = setup.full((4, 128, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b p [1, 1] -> b p c", x, y[..., 0:1], y[..., 1:2]).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b p [,] -> b p c", x, y[..., 0], y[..., 1]).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, p b [2] -> b p c", x, y2).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b p, b p -> b p c", x, y[..., 0], y[..., 1]).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b (p [1]), b p -> b p c", x, y[..., 0], y[..., 1]).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b p, p b -> b p c", x, y[..., 0], y2[..., 1]).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, p, p b -> b p c", x, y[0, ..., 0], y2[..., 1]).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b (p [1]), p b -> b p c", x, y[..., 0], y2[..., 1]).shape == (4, 128, 3)
    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 3, 0)):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.get_at("b [h w] c, b p [2] -> b p c", x, lambda shape: setup.full(shape, dtype="int64", value=0), p=128).shape == (4, 128, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.get_at("b [h w] c, b p [l] -> b p c", x, lambda shape: setup.full(shape, dtype="int64", value=0), p=128).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [16 w] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [16 17] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, p [2] -> b p c", x, y[0]).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p [2], b p c -> b [h w] c", x, y, z).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p [2], b p c", x, y, z).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p, b p, b p c -> b [h w] c", x, y[..., 0], y[..., 1], z).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p, p b, b p c -> b [h w] c", x, y[..., 0], y2[..., 1], z).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p, p b, p c -> b [h w] c", x, y[..., 0], y2[..., 1], z[0]).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p, p b, c -> b [h w] c", x, y[..., 0], y2[..., 1], z[0, 0]).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, p [2], p c -> b [h w] c", x, y[0], z[0]).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p [2], b p c -> b [h w] c", x, y, z).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p [2], p c -> b [h w] c", x, y, z[0]).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, p [2], b p c -> b [h w] c", x, y[0], z).shape == (4, 16, 17, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, p [2], p c -> b [h w] c", x, y[0], z[0]).shape == (4, 16, 17, 3)
        with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
            op("b [h w] c, p [2], p c -> b [w h] c", x, y[0], z[0])

    x = setup.full((4, 16, 16, 3))
    y = setup.full((4, 128, 2), dtype="int64")
    z = setup.full((4, 128, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h h] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)

    x = setup.full((16, 4, 3, 16))
    y = setup.full((2, 4, 128), dtype="int64", value=0)
    z = setup.full((3, 4, 128))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("[w] b c [h], [2] b p -> b p c", x, y).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("[w] b c [h], [2] p -> b p c", x, y[:, 0]).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("[w] b c [h], [2] b p, c b p -> b [w h] c", x, y, z).shape == (4, 16, 16, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("[w] b c [h], [2] p, c p -> b [w h] c", x, y[:, 0], z[:, 0]).shape == (4, 16, 16, 3)

    x = setup.full((16, 4, 3 * 16))
    y = setup.full((2, 4, 128), dtype="int64")
    z = setup.full((3, 4, 128))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("[w] b (c [h]), [2] b p -> b p c", x, y, c=3).shape == (4, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("[w] b (c [h]), [2] p -> b p c", x, y[:, 0], c=3).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("[w] b (c [h]), [2] b p, c b p -> b ([w h]) c", x, y, z).shape == (4, 256, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("[w] b (c [h]), [2] p, c p -> b ([w h]) c", x, y[:, 0], z[:, 0]).shape == (4, 256, 3)

    x = setup.full((4, 16, 16, 3))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.get_at("b [h w] c -> b c", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.get_at("b [h w] c", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.set_at("b [h w] c -> b [h w] c", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.set_at("b [h w] c", x)

    x = setup.full((4, 16, 16, 3))
    y = setup.full((4, 3, 4, 5, 2), dtype="int64")
    z = setup.full((4, 3, 4, 5, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, b p q r [2] -> b p q r c", x, y).shape == (4, 3, 4, 5, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b [h w] c, p q r [2] -> b p q r c", x, y[0]).shape == (4, 3, 4, 5, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, b p q r [2], b p q r c -> b [h w] c", x, y, z).shape == (4, 16, 16, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h h] c, b p q r [2], b p q r c -> b [h h] c", x, y, z).shape == (4, 16, 16, 3)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert op("b [h w] c, p q r [2], p q r c -> b [h w] c", x, y[0], z[0]).shape == (4, 16, 16, 3)

    x = setup.full((4, 1, 1, 3))
    y = setup.full((4, 128, 2), dtype="int64")
    z = setup.full((4, 128, 3))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.get_at("b ([1 1]) c, b p [2] -> b p c", x, y)

    x = setup.full((4, 5, 6))
    y = setup.full((4, 5), dtype="int64")
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b t [d], b t -> b t", x, y).shape == (4, 5)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("... [d], ... -> ...", x, y).shape == (4, 5)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b t [d], b (t [1]) -> b (t 1)", x, y).shape == (4, 5)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.get_at("b t [d], b (t [1]) -> b (t [1])", x, y)

    x = setup.full((4, 128, 128, 3))
    y = setup.full((4, 0, 2), dtype="int64")
    y2 = setup.full((4, 2), dtype="int64")
    z = setup.full((4, 0, 3))
    z2 = setup.full((4, 3))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.set_at("b [h w] c, b p [2], b p c -> b [h w] c", x, y, z).shape == (4, 128, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.set_at("b [h w] c, b p [2], b c -> b [h w] c", x, y, z2).shape == (4, 128, 128, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.set_at("b [h w] c, b [2], b p c -> b [h w] c", x, y2, z).shape == (4, 128, 128, 3)

    x = setup.full((4, 128, 16))
    y = setup.full((4, 128), dtype="int64")
    z = setup.full((4, 128))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.get_at("b p [i,->]", x, y).shape == (4, 128)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.set_at("b p [i,,->i]", x, y, z).shape == (4, 128, 16)

    consts = {"b": 4, "h": 16, "w": 16, "c": 3, "p": 128}

    def make_coords(shape):
        return setup.full(shape, dtype="int64")

    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 4, 0)):
        xs = ["([h] b) [w] c", "[h] c [w]", "[h w]"]
        ys = ["b (p [2])", "[2] p", "[2]"]
        ys2 = ["p b", "p", "[1]"]
        zs = ["b p c", "c (p b)"]
        for x in xs:
            for z in zs:
                shape = None
                with suppress((OperationNotSupportedError, BackendResolutionError, *setup.exceptions)):
                    shape = einx.add(f"{z}, ", setup.full, 0.0, **consts).shape
                if shape is None:
                    continue
                for y in ys:
                    with suppress((OperationNotSupportedError, *setup.exceptions)):
                        assert einx.get_at(f"{x}, {y} -> {z}", setup.full, make_coords, **consts).shape == shape
                for y1 in ys2:
                    for y2 in ys2:
                        with suppress((OperationNotSupportedError, *setup.exceptions)):
                            assert einx.get_at(f"{x}, {y1}, {y2} -> {z}", setup.full, make_coords, make_coords, **consts).shape == shape

        for x in xs:
            shape = None
            with suppress((OperationNotSupportedError, BackendResolutionError, *setup.exceptions)):
                shape = einx.add(f"{x.replace('[', '').replace(']', '')}, ", setup.full, 0.0, **consts).shape
            if shape is None:
                continue
            for z in zs:
                z_axes = {a for a in z if a.isalpha()}
                for y in ys:
                    if all(a in (x + y) for a in z_axes):
                        with suppress((OperationNotSupportedError, *setup.exceptions)):
                            assert einx.set_at(f"{x}, {y}, {z} -> {x}", setup.full, make_coords, setup.full, **consts).shape == shape
                for y1 in ys2:
                    for y2 in ys2:
                        if all(a in (x + y1 + y2) for a in z_axes):
                            with suppress((OperationNotSupportedError, *setup.exceptions)):
                                assert einx.set_at(f"{x}, {y1}, {y2}, {z} -> {x}", setup.full, make_coords, make_coords, setup.full, **consts).shape == shape
                            with suppress((OperationNotSupportedError, *setup.exceptions)):
                                assert einx.set_at(f"{x}, {y1}, {y2}, {z}", setup.full, make_coords, make_coords, setup.full, **consts).shape == shape


@use_backend
def test_shape_preserve_shape(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    x = setup.full((10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.flip("a [b] -> a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.flip("a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sort("a [b] -> a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.sort("a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argsort("a [b] -> a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.argsort("a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a [b]", x, shift=5).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a [b]", x, shift=(5,)).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.softmax("a [b] -> a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.softmax("a [b]", x).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.softmax("a [b] -> (a [b]) c", x, c=3).shape == (100, 3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.softmax("a [b] -> a ([b] c)", x, c=3).shape == (10, 30)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.log_softmax("(a [b]) c", x, b=2).shape == (10, 10)

    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.roll("a [shift]", x, shift=5)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.flip("a ([b c])", x, b=2).shape == (10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a ([b c])", x, shift=(5, 5), b=2).shape == (10, 10)

    x = setup.full((3, 10, 10))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.flip("a [b c] -> a [b c]", x).shape == (3, 10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.flip("a [b b] -> a [b b]", x).shape == (3, 10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a [b c] -> a [b c]", x, shift=(5, 5)).shape == (3, 10, 10)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a [b b] -> a [b b]", x, shift=(5, 5)).shape == (3, 10, 10)
    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 8, 0)):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.softmax("a [b c] -> a [b c]", x).shape == (3, 10, 10)
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.softmax("a [b b] -> a [b b]", x).shape == (3, 10, 10)

    x = setup.full((3, 4, 5))
    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 8, 0)):
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            assert einx.softmax("a [b c]", x).shape == (3, 4, 5)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.log_softmax("a [b c]", x).shape == (3, 4, 5)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.sort("a [b c]", x)
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.argsort("a [b c]", x)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.flip("a [b c]", x).shape == (3, 4, 5)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a [b c]", x, shift=1).shape == (3, 4, 5)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        assert einx.roll("a [b c]", x, shift=(1,)).shape == (3, 4, 5)
    with pytest.raises((OperationNotSupportedError, ValueError, *setup.exceptions)):
        einx.roll("a [b c]", x, shift=(1, 1, 1))
    with pytest.raises((OperationNotSupportedError, ValueError, *setup.exceptions)):
        einx.roll("a [b c]", x, shift=())

    x = setup.full((10, 11))
    with pytest.raises((OperationNotSupportedError, EinxError, *setup.exceptions)):
        einx.softmax("[a b] -> [b a]", x)


def test_shape_solve():
    import numpy as np

    x = np.zeros((2, 3, 4))
    assert einx.matches("a b c", x)
    assert not einx.matches("a b", x)
    with pytest.raises(EinxError):
        einx.id("a b c d", x)
    einx.id("a b c", x)

    x = np.zeros((6, 4))
    assert einx.matches("(a b) c", x)

    x = np.zeros((2, 3, 4))
    assert einx.matches("a b...", x)

    x = np.zeros((5, 4))
    assert einx.matches("(a + b) c", x)
    assert einx.matches("(a + b) c", x, a=2)
    assert not einx.matches("(a + b) c", x, a=10)
    assert einx.solve_axes("(a + b) c", x, b=3) == {"a": 2, "b": 3, "c": 4}
    with pytest.raises(EinxError):
        einx.solve_axes("(a + b) c", x)

    axes = einx.solve_axes("a b, c b", x, x)
    assert axes["a"] == 5 and axes["b"] == 4 and axes["c"] == 5

    shape0, shape1 = einx.solve_shapes("a b, b a", x, None)
    assert shape0 == (5, 4) and shape1 == (4, 5)

    shape0, shape1 = einx.solve_shapes("a b, c b", x, x)
    assert shape0 == (5, 4) and shape1 == (5, 4)
