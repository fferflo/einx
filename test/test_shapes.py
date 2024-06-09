import einx
import pytest
import numpy as np
import conftest


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_rearrange(test):
    einx, backend, setup = test

    x = setup.full((10, 20, 1))

    assert einx.rearrange("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    assert einx.rearrange("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    assert einx.rearrange("a b c -> (a b) c 1 1 1", x).shape == (200, 1, 1, 1, 1)
    with pytest.raises(Exception):
        einx.rearrange("a a b c -> (a b) c 1", x)
        einx.rearrange("a (a + b) c -> (a b) c 1", x)

    x = setup.full((10, 20, 20, 2))
    assert einx.rearrange("b s... c -> b (s...) c", x).shape == (10, 400, 2)
    assert einx.rearrange("b ... c -> b (...) c", x).shape == (10, 400, 2)
    assert einx.rearrange("b (s...) (r...) c -> b (s...) r... c", x, r=(10, 2)).shape == (
        10,
        20,
        10,
        2,
        2,
    )
    assert einx.rearrange("b s... c x... -> x... b (s...) c", x, x=()).shape == (10, 400, 2)

    if backend.name != "torch":
        assert einx.rearrange("1 -> (x)", [1], x=10, backend=backend).shape == (10,)
        assert einx.rearrange("1 -> (x y)", [1], x=10, y=20, backend=backend).shape == (200,)

    assert einx.rearrange("1 -> (x)", setup.to_tensor([1]), x=10).shape == (10,)
    assert einx.rearrange("1 -> (x y)", setup.to_tensor([1]), x=10, y=20).shape == (200,)

    x = setup.full((1,))
    assert einx.rearrange("1 -> (x)", x, x=10).shape == (10,)
    assert einx.rearrange("1 -> (x y)", x, x=10, y=20).shape == (200,)

    x = setup.full((10, 20, 1))
    assert einx.rearrange("a b c d... -> a b c (d...)", x).shape == (10, 20, 1, 1)

    x = setup.full((10, 20, 1, 2))
    assert einx.rearrange("a (b...) c d -> a (b... c) d", x).shape == (10, 20, 2)

    x = setup.full((10, 20, 1, 2, 3))
    assert einx.rearrange("a (b... c) d... e -> a (b...) (c d...) e", x, b=[2, 5]).shape == (
        10,
        10,
        4,
        3,
    )

    x = setup.full((10, 20, 6, 24))
    assert einx.rearrange("a b (c...) (d...) -> a c... b d...", x, c=[2, 3], d=[4, 6]).shape == (
        10,
        2,
        3,
        20,
        4,
        6,
    )

    x = setup.full((10, 10))
    assert einx.rearrange("a... -> 1 (a...)", x).shape == (1, 100)

    x = setup.full((10, 20, 5))
    assert einx.rearrange("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x).shape == (1, 5, 10, 20)

    x = setup.full((10, 20))
    with pytest.raises(Exception):
        assert einx.rearrange("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x).shape == (1, 5, 10, 20)

    x = setup.full((10, 20, 1))
    with pytest.raises(Exception):
        einx.rearrange("a b c -> (a b) c d", x)

    x = setup.full((10, 20, 1))
    with pytest.raises(Exception):
        einx.rearrange("a b... c... -> a (b...) c...", x)
    with pytest.raises(Exception):
        einx.rearrange("a b... -> a b", x)

    x = setup.full((1, 10, 20, 6))
    assert einx.rearrange("a (b...) (e f...) (d c) -> a d (b...) (e f...) c", x, d=2).shape == (
        1,
        2,
        10,
        20,
        3,
    )

    x = setup.full((1, 10, 20, 6, 7, 12))
    assert einx.rearrange(
        "a b c d... (e f...) -> a b c d... ((e 2 2) f...)", x, f=[2, 2]
    ).shape == (
        1,
        10,
        20,
        6,
        7,
        12 * 2 * 2,
    )

    x = setup.full((10, 20, 3))
    assert einx.rearrange("(s s2)... c -> s... s2... c", x, s2=(2, 2)).shape == (5, 10, 2, 2, 3)
    assert einx.rearrange("(s s2)... c -> s... s2... c", x, s2=2).shape == (5, 10, 2, 2, 3)

    x = setup.full((10, 10, 10))
    assert einx.rearrange("(a b) (c d) (e f) -> a (b c d e) f", x, a=2, f=2).shape == (2, 250, 2)

    x = setup.full((10,))
    y = setup.full((20,))
    assert einx.rearrange("a, b -> a + b", x, y).shape == (30,)
    assert einx.rearrange("a, b -> b + a", x, y).shape == (30,)
    assert einx.rearrange("a, b -> a b (1 + 1)", x, y).shape == (10, 20, 2)
    assert [x.shape for x in einx.rearrange("(a + b) -> a, b 1", x, a=4)] == [(4,), (6, 1)]
    with pytest.raises(Exception):
        einx.rearrange("a, b -> a b (1 + 1)", x)
        einx.rearrange("(a + b) -> a b (1 + 1)", x)
    assert einx.rearrange("a, (b c) -> c (b + a)", x, y, c=2).shape == (2, 20)
    with pytest.raises(Exception):
        assert einx.rearrange("a, -> (a +)", x, 1).shape == (11,)

    x = setup.full((10, 10))
    assert einx.rearrange("b c, 1 -> b (c + 1)", x, [42]).shape == (10, 11)
    assert einx.rearrange("b c, -> b (c + 1)", x, 42).shape == (10, 11)
    s = setup.full(())
    assert einx.rearrange("b c, -> b (c + 1)", x, s).shape == (10, 11)

    assert einx.arange("c", c=2, backend=backend).shape == (2,)
    assert einx.arange("c... [2]", c=(4, 3), backend=backend).shape == (4, 3, 2)
    assert einx.arange("c... [l]", c=(4, 3), backend=backend).shape == (4, 3, 2)
    with pytest.raises(Exception):
        einx.arange("c... [3]", c=(4, 3), backend=backend)
    assert einx.arange("c1 c2 -> [l] c2 c1", c1=4, c2=3, backend=backend).shape == (2, 3, 4)
    assert einx.arange("(c...) [2]", c=(4, 3), backend=backend).shape == (4 * 3, 2)
    assert einx.arange("(c... [l])", c=(4, 3), backend=backend).shape == (4 * 3 * 2,)
    assert einx.arange("c1 c2 -> ([l] c2) c1", c1=4, c2=3, backend=backend).shape == (2 * 3, 4)

    x = setup.full((10, 20), dtype="bool")
    y = setup.full((4, 10, 20, 3))
    x, y = einx.rearrange("h w, b h w c -> 1 h w 1, b h w c", x, y)

    x = np.zeros((5, 4))
    x = einx.rearrange("(a + b + c) d -> b d, (a + c) d", x, a=1, b=2)
    assert x[0].shape == (2, 4)
    assert x[1].shape == (3, 4)
    x = np.zeros((5, 4))
    x = einx.rearrange("(a + b + c) d -> (a + c) d, b d", x, a=1, b=2)
    assert x[0].shape == (3, 4)
    assert x[1].shape == (2, 4)


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_dot(test):
    einx, backend, setup = test

    if backend.name == "mlx":
        pytest.xfail(reason="Backend does not support einsum")
    x = setup.full((10, 10))
    assert einx.dot("a..., a... -> 1", x, x).shape == (1,)
    assert einx.dot("[a...], [a...] -> 1", x, x).shape == (1,)
    with pytest.raises(Exception):
        einx.dot("a..., [a]... -> 1", x, x)

    x = setup.full((10, 20, 1))
    y = setup.full((10, 24))
    assert einx.dot("a b c, a d -> 1 b c d", x, y).shape == (1, 20, 1, 24)
    assert einx.dot("[a] b c, [a] d -> 1 b c d", x, y).shape == (1, 20, 1, 24)
    assert einx.dot("a b c, a d -> 1 b c d", x, setup.full, d=24).shape == (1, 20, 1, 24)

    x = setup.full((10, 20, 1))
    with pytest.raises(Exception):
        einx.dot("a b c -> a b c", x, x)
    with pytest.raises(Exception):
        einx.dot("a b c, a -> a b c", x)

    x = setup.full((10, 20))
    y = setup.full((20, 30))
    assert einx.dot("a [b] -> a [c]", x, y).shape == (10, 30)
    assert einx.dot("a b, b c -> a c", x, y).shape == (10, 30)
    assert einx.dot("a [b], [b] c -> a c", x, y).shape == (10, 30)
    assert einx.dot("a [b->c]", x, y).shape == (10, 30)
    assert einx.dot("a [b...->c]", x, y).shape == (10, 30)

    x = setup.full((10, 20))
    y = setup.full((10, 20, 30))
    assert einx.dot("a b, a b c -> a c", x, y).shape == (10, 30)
    assert einx.dot("[a b] -> [a c]", x, y).shape == (10, 30)
    assert einx.dot("[a b->a c]", x, y).shape == (10, 30)

    x = setup.full((10,))
    y = setup.full((30,))
    assert einx.dot("a, a ->", x, x).shape == ()
    assert einx.dot("[a->]", x, x).shape == ()
    assert einx.dot("a, c -> a c", x, y).shape == (10, 30)
    assert einx.dot("a [->c]", x, y).shape == (10, 30)
    assert einx.dot("a [b...->c]", x, y).shape == (10, 30)

    x = setup.full((4, 128, 128, 16))
    assert einx.dot("b s... [c1->c2]", x, setup.full, c2=32).shape == (4, 128, 128, 32)
    assert einx.dot("b [s...->s2] c", x, setup.full, s2=32).shape == (4, 32, 16)

    w = setup.full((2, 2, 16, 32))
    assert einx.dot("b (s [s2->])... [c1->c2]", x, w, s2=2, c2=32).shape == (4, 64, 64, 32)

    x = setup.full((4, 16, 16, 16))

    def w(shape):
        return setup.full(shape)

    assert einx.dot("b [(s s2)->s]... [c1->c2]", x, w, s2=4, c2=4).shape == (4, 4, 4, 4)
    assert einx.dot("b (s [s2->])... [c1->c2]", x, w, s2=4, c2=4).shape == (4, 4, 4, 4)

    s = setup.full(())
    x = setup.full((10, 10))
    y = setup.full((10,))
    assert einx.dot("[->]", s, s, backend=backend).shape == ()
    assert einx.dot("a [->]", y, s).shape == (10,)
    if backend.name not in {"torch"}:
        assert einx.dot("[->]", 1, 1, backend=backend).shape == ()
    assert einx.dot("a [->]", y, 1).shape == (10,)
    assert einx.dot("a [b->]", x, y).shape == (10,)
    assert einx.dot("a [->b]", y, y).shape == (10, 10)

    x = setup.full((11, 10))
    y = setup.full((11,))
    assert einx.dot("... b, ... -> b", x, y).shape == (10,)
    assert einx.dot("[...] b -> b", x, y).shape == (10,)

    x = setup.full((10,))
    y = setup.full(())
    assert einx.dot("... b, ... -> b", x, y).shape == (10,)
    assert einx.dot("[...] b -> b", x, y).shape == (10,)

    k = setup.full((2, 4, 100))
    v = setup.full((2, 4, 100))
    with pytest.raises(Exception):
        einx.dot("b t (h ck), b t (h cv) -> b h ck cv", k, v, h=32, graph=True)


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_reduce(test):
    einx, backend, setup = test

    x = setup.full((10, 10))
    assert einx.reduce("a b -> 1 a", x, op=backend.mean).shape == (1, 10)
    op = lambda tensor, axis: einx.jit(lambda tensor, backend: backend.mean(tensor, axis))(tensor)
    assert einx.reduce("a b -> 1 a", x, op=op).shape == (
        1,
        10,
    )
    assert einx.mean("a b -> 1 a", x).shape == (1, 10)
    assert einx.mean("[a] b", x).shape == (10,)
    assert einx.mean("[a] b -> 1 b", x).shape == (1, 10)

    x = setup.full((10, 10, 10))
    with pytest.raises(Exception):
        einx.sum("a [b] c -> a b", x)

    x = setup.full((10, 3, 1))
    assert einx.mean("(a [b]) c 1", x, b=2).shape == (5, 3, 1)
    assert einx.mean("([a b]) c 1", x).shape == (1, 3, 1)
    assert einx.mean("[(a b)] c 1", x).shape == (3, 1)
    assert einx.mean("[(a...)] c 1", x).shape == (3, 1)
    assert einx.mean("(b... [a...]) c 1", x, b=(1, 1)).shape == (1, 3, 1)

    x = setup.full((1, 10, 3, 2))
    assert einx.mean("1 [a...] b", x).shape == (1, 2)
    assert einx.mean("1 [a]... b", x).shape == (1, 2)
    assert einx.mean("1 ([a])... b", x).shape == (1, 1, 1, 2)
    assert einx.mean("1 [a]... b", x, keepdims=True).shape == (1, 1, 1, 2)
    assert einx.mean("1 [a...] b", x, keepdims=True).shape == (1, 1, 2)

    x = setup.full((16, 1, 20, 30, 64))
    assert einx.mean("(b rg) pv [s...] c", x).shape == (16, 1, 64)

    x = setup.full((16, 16, 32))
    bias = setup.full((4,))
    assert einx.add("b... (g [c])", x, bias).shape == (16, 16, 32)

    assert einx.logsumexp("a [...]", x).shape == (16,)

    if backend.name != "torch":
        assert einx.logsumexp("[a]", [0.0, 1.0], backend=backend).shape == ()
        assert einx.logsumexp("[a] 1", [[0.0], [1.0]], backend=backend).shape == (1,)
        assert einx.logsumexp("[a]", [0.0] * 10, backend=backend).shape == ()
        with pytest.raises(Exception):
            einx.logsumexp("a", [0.0, [1.0]], backend=backend)

    x = setup.full((16, 15))
    assert einx.sum("[b] a []", x).shape == (15,)
    assert einx.sum("[b] a [...]", x).shape == (15,)

    assert einx.sum("b [p] -> b p2", x, p2=7).shape == (16, 7)


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_elementwise(test):
    einx, backend, setup = test

    x = setup.full((10, 5, 1))
    y = setup.full((13,))
    assert einx.elementwise("a b 1, l -> b l a 1", x, y, op=backend.add).shape == (5, 13, 10, 1)
    assert einx.elementwise("a b 1, l -> b l a 1", x, y, op=lambda x, y: x + y).shape == (
        5,
        13,
        10,
        1,
    )
    assert einx.add("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    assert einx.add("a b 1, l -> a b l", x, y).shape == (10, 5, 13)

    x = setup.full((10, 10))
    y = setup.full((10,))
    assert einx.add("a, a b", y, x).shape == (10, 10)
    assert einx.add("a b, a", x, y).shape == (10, 10)
    assert einx.add("a b, b", x, y).shape == (10, 10)
    assert einx.add("a [b]", x, y).shape == (10, 10)
    assert einx.add("a b, a b", x, x).shape == (10, 10)
    assert einx.add("a b, ", x, 1).shape == (10, 10)
    assert einx.add(", a b", 1, x).shape == (10, 10)
    assert einx.add("a b, 1", x, [1]).shape == (10, 10)
    assert einx.add("1, a b", [1], x).shape == (10, 10)
    with pytest.raises(Exception):
        einx.add("a a, a -> a a", x, y)
    assert einx.add("a b, a b", x, setup.full).shape == (10, 10)
    assert einx.add("a, a", y, y).shape == (10,)
    assert einx.add("[a]", y, y).shape == (10,)
    assert einx.add("b, -> b 3", y, 1).shape == (10, 3)

    x = setup.full((2, 3))
    y = setup.full((10,))
    with pytest.raises(Exception):
        einx.add("a b, c", x, y)

    x = setup.full((3, 128, 196, 64))
    y = setup.full((3, 4, 16))
    assert einx.add("b h w (g c), b (g) c -> b h w (g c)", x, y).shape == (3, 128, 196, 64)

    x = setup.full((10, 20))
    y = setup.full((10, 20, 30))
    assert einx.add("a b, a b c -> a b c", x, y).shape == (10, 20, 30)
    assert einx.add("(a [1])...", x, setup.full).shape == (10, 20)

    x = setup.full((10, 20))
    y = setup.full((30, 20))
    with pytest.raises(Exception):
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
    ]

    def create_scalar(dtype):
        if dtype == "float32":
            return 1.0
        elif dtype == "bool":
            return True
        else:
            assert False

    for op, dtypes in ops:
        tensor_args = [setup.full((10,), dtype=dtype, value=1) for dtype in dtypes]
        scalar_args = [create_scalar(dtype) for dtype in dtypes]

        for scalar_index in range(len(dtypes)):
            args = [
                scalar_args[scalar_index] if i == scalar_index else tensor_args[i]
                for i in range(len(dtypes))
            ]
            expr = (
                ", ".join(["" if i == scalar_index else "a" for i in range(len(dtypes))]) + " -> a"
            )
            assert getattr(einx, op)(expr, *args).shape == (10,)


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_vmap(test):
    einx, backend, setup = test

    if backend.name in {"mlx", "dask", "tinygrad"}:
        pytest.xfail(reason="Backend does not fully support vmap")

    x = setup.full((13,))
    assert einx.vmap("b -> b [3]", x, op=lambda x: x + setup.full((3,))).shape == (13, 3)

    with pytest.raises(Exception):
        einx.vmap("b -> [b] 3", x, op=lambda x: x + setup.full((3,)))
    with pytest.raises(Exception):
        einx.vmap("b -> b 3", x, op=einx.trace(lambda x: x + setup.full((3,))))
    with pytest.raises(Exception):
        einx.vmap("b -> b 3", x, op=lambda x: x + setup.full((3,)))

    x = setup.full((4, 13, 2))
    y = setup.full((13, 4, 5, 5))

    def f(x, y):
        assert x.shape == (4, 2)
        assert y.shape == (4, 5)
        x = x[:, 0] + y[:, 0]
        return einx.rearrange("a -> a 15", x)

    assert einx.vmap("[a] b [e], b [a] c [d] -> [a] b [g] c", x, y, op=f, g=15).shape == (
        4,
        13,
        15,
        5,
    )
    assert einx.vmap("[a] b [e], b [a] c [d] -> [a] b ([g] c)", x, y, op=f, g=15).shape == (
        4,
        13,
        15 * 5,
    )
    with pytest.raises(Exception):
        einx.vmap("[a] b [e], b [a] c [d] -> [g] b [a] c", x, y, op=f, g=15)

    with pytest.raises(Exception):

        def f(x, y):
            assert x.shape == (4, 2)
            assert y.shape == (4, 5)
            x = x[:, 0] + y[:, 0]
            return einx.rearrange("a -> a 16", x)

        einx.vmap("[a] b [e], b [a] c [d] -> [a] b [g] c", x, y, op=f, g=15)

    x = setup.full((4, 16))
    y = setup.full((16, 32))
    op = lambda x, y: einx.jit(lambda x, y, backend: backend.sum(x * y))(x, y)
    assert einx.vmap("b [c1], [c1] c2 -> b c2", x, y, op=op).shape == (
        4,
        32,
    )

    x = setup.full((4,))
    y = setup.full((16, 32))
    assert einx.vmap("a, b c -> a b c", x, y, op=backend.add).shape == (4, 16, 32)

    func = lambda x: einx.jit(lambda x, backend: backend.stack([backend.mean(x), backend.max(x)]))(
        x
    )  # c -> 2

    x = setup.full(
        (
            16,
            64,
            3,
        ),
    )
    assert einx.vmap("b [c] a -> a b [2]", x, op=func).shape == (3, 16, 2)

    func = lambda x, y: einx.jit(
        lambda x, y, backend: backend.stack([backend.mean(x), backend.max(y)])
    )(x, y)  # c, d -> 2

    x = setup.full((16, 64))  # b c
    y = setup.full((16, 72))  # b d
    assert einx.vmap("b [c], b [d] -> b [2]", x, y, op=func).shape == (16, 2)

    x = setup.full((16, 64, 3))  # b1 c b2
    y = setup.full((3, 72))  # b2 d
    assert einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=func).shape == (3, 2, 16)

    @einx.trace
    def func(x):  # (c d) -> 2
        x = einx.vmap("([c] d) -> d", x, op=backend.mean, c=16)
        x = backend.max(x)
        return backend.stack([x, x])

    x = setup.full((16, 64))  # b c
    assert einx.vmap("b ([c d]) -> b [2]", x, op=func, c=16).shape == (16, 2)
    assert einx.vmap("b ([c d]) -> b [2] 1", x, op=func, c=16).shape == (16, 2, 1)
    assert einx.vmap("b [(c d)->2]", x, op=func, c=16).shape == (16, 2)
    assert einx.vmap("b ([c d->2])", x, op=func, c=16).shape == (16, 2)
    with pytest.raises(Exception):
        einx.vmap("b ([c d]) -> [2]", x, op=func, c=16)

    @einx.trace
    def func(x):  # c d -> 2
        x = einx.vmap("[c] d -> d", x, op=backend.mean, c=16)
        x = backend.max(x)
        return backend.stack([x, x])

    x = setup.full((16, 64))  # b c
    assert einx.vmap("b ([c d]) -> b [2]", x, op=func, c=16, flat=True).shape == (16, 2)
    assert einx.vmap("b ([c d]) -> b [2] 1", x, op=func, c=16, flat=True).shape == (16, 2, 1)
    assert einx.vmap("b [(c d)->2]", x, op=func, c=16, flat=True).shape == (16, 2)
    assert einx.vmap("b ([c d->2])", x, op=func, c=16, flat=True).shape == (16, 2)
    with pytest.raises(Exception):
        einx.vmap("b ([c d]) -> [2]", x, op=func, c=16, flat=True)

    op = lambda tensor, axis: einx.jit(
        lambda tensor, backend: backend.roll(tensor, axis=axis, shift=(2, 2))
    )(tensor)
    with pytest.raises(Exception):
        einx.vmap_with_axis("a ([b c]) -> a ([b c])", x, op=op)
    assert einx.vmap_with_axis(
        "a ([b c]) -> a ([b c])",
        x,
        op=op,
        b=2,
    ).shape == (
        16,
        64,
    )


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_index(test):
    einx, backend, setup = test

    if backend.name in {"mlx", "dask", "tinygrad"}:
        pytest.xfail(reason="Backend does not fully support vmap")

    coord_dtype = "int32" if backend.name != "torch" else "long"
    x = setup.full((4, 16, 16, 3))
    y = setup.full((4, 128, 2), dtype=coord_dtype)
    y2 = setup.full((128, 4, 2), dtype=coord_dtype)
    z = setup.full((4, 128, 3))
    assert einx.get_at("b [h w] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, p b [2] -> b p c", x, y2).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b p, b p -> b p c", x, y[..., 0], y[..., 1]).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b (p [1]), b p -> b p c", x, y[..., 0], y[..., 1]).shape == (
        4,
        128,
        3,
    )
    assert einx.get_at("b [h w] c, b p, p b -> b p c", x, y[..., 0], y2[..., 1]).shape == (
        4,
        128,
        3,
    )
    assert einx.get_at("b [h w] c, p, p b -> b p c", x, y[0, ..., 0], y2[..., 1]).shape == (
        4,
        128,
        3,
    )
    assert einx.get_at("b [h w] c, b (p [1]), p b -> b p c", x, y[..., 0], y2[..., 1]).shape == (
        4,
        128,
        3,
    )
    assert einx.get_at(
        "b [h w] c, b p [2] -> b p c",
        x,
        lambda shape: setup.full(shape, dtype=coord_dtype, value=0),
        p=128,
    ).shape == (4, 128, 3)
    assert einx.get_at(
        "b [h w] c, b p [l] -> b p c",
        x,
        lambda shape: setup.full(shape, dtype=coord_dtype, value=0),
        p=128,
    ).shape == (4, 128, 3)
    assert einx.get_at("b [16 w] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("b [16 16] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, p [2] -> b p c", x, y[0]).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("b [h w] c, b p [2], b p c -> b [h w] c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p [2], b p c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p, b p, b p c -> b [h w] c", x, y[..., 0], y[..., 1], z).shape == (
            4,
            16,
            16,
            3,
        )
        assert op("b [h w] c, b p, p b, b p c -> b [h w] c", x, y[..., 0], y2[..., 1], z).shape == (
            4,
            16,
            16,
            3,
        )
        assert op(
            "b [h w] c, b p, p b, p c -> b [h w] c", x, y[..., 0], y2[..., 1], z[0]
        ).shape == (4, 16, 16, 3)
        assert op(
            "b [h w] c, b p, p b, c -> b [h w] c", x, y[..., 0], y2[..., 1], z[0, 0]
        ).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p [2], p c -> b [h w] c", x, y[0], z[0]).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p [2], b p c -> b h w c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p [2], p c -> b h w c", x, y, z[0]).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p [2], b p c -> b h w c", x, y[0], z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p [2], p c -> b h w c", x, y[0], z[0]).shape == (4, 16, 16, 3)

    x = setup.full((16, 4, 3, 16))
    y = setup.full((2, 4, 128), dtype=coord_dtype, value=0)
    z = setup.full((3, 4, 128))
    assert einx.get_at("[w] b c [h], [2] b p -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("[w] b c [h], [2] p -> b p c", x, y[:, 0]).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("[w] b c [h], [2] b p, c b p -> b [w h] c", x, y, z).shape == (4, 16, 16, 3)
        assert op("[w] b c [h], [2] p, c p -> b [w h] c", x, y[:, 0], z[:, 0]).shape == (
            4,
            16,
            16,
            3,
        )

    x = setup.full((16, 4, 3 * 16))
    y = setup.full((2, 4, 128), dtype=coord_dtype)
    z = setup.full((3, 4, 128))
    assert einx.get_at("[w] b (c [h]), [2] b p -> b p c", x, y, c=3).shape == (4, 128, 3)
    assert einx.get_at("[w] b (c [h]), [2] p -> b p c", x, y[:, 0], c=3).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("[w] b (c [h]), [2] b p, c b p -> b ([w h]) c", x, y, z).shape == (4, 256, 3)
        assert op("[w] b (c [h]), [2] p, c p -> b ([w h]) c", x, y[:, 0], z[:, 0]).shape == (
            4,
            256,
            3,
        )

    x = setup.full((4, 16, 16, 3))
    y = setup.full((4, 3, 4, 5, 2), dtype=coord_dtype)
    z = setup.full((4, 3, 4, 5, 3))
    assert einx.get_at("b [h w] c, b p q r [2] -> b p q r c", x, y).shape == (4, 3, 4, 5, 3)
    assert einx.get_at("b [h w] c, p q r [2] -> b p q r c", x, y[0]).shape == (4, 3, 4, 5, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("b [h w] c, b p q r [2], b p q r c -> b [h w] c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p q r [2], p q r c -> b [h w] c", x, y[0], z[0]).shape == (
            4,
            16,
            16,
            3,
        )

    x = setup.full((4, 1, 1, 3))
    y = setup.full((4, 128, 2), dtype=coord_dtype)
    z = setup.full((4, 128, 3))
    with pytest.raises(Exception):
        einx.get_at("b ([1 1]) c, b p [2] -> b p c", x, y)

    x = setup.full((4, 5, 6))
    y = setup.full((4, 5), dtype=coord_dtype)
    assert einx.get_at("b t [d], b t -> b t", x, y).shape == (4, 5)
    assert einx.get_at("... [d], ... -> ...", x, y).shape == (4, 5)
    assert einx.get_at("b t [d], b (t [1]) -> b (t 1)", x, y).shape == (4, 5)
    with pytest.raises(Exception):
        einx.get_at("b t [d], b (t [1]) -> b (t [1])", x, y)

    x = setup.full((4, 128, 128, 3))
    y = setup.full((4, 0, 2), dtype=coord_dtype)
    y2 = setup.full((4, 2), dtype=coord_dtype)
    z = setup.full((4, 0, 3))
    z2 = setup.full((4, 3))
    assert einx.set_at("b [h w] c, b p [2], b p c -> b [h w] c", x, y, z).shape == (4, 128, 128, 3)
    assert einx.set_at("b [h w] c, b p [2], b c -> b [h w] c", x, y, z2).shape == (4, 128, 128, 3)
    assert einx.set_at("b [h w] c, b [2], b p c -> b [h w] c", x, y, z2).shape == (4, 128, 128, 3)

    x = setup.full((4, 128, 16))
    y = setup.full((4, 128), dtype=coord_dtype)
    z = setup.full((4, 128))
    assert einx.get_at("b p [i,->]", x, y).shape == (4, 128)
    assert einx.set_at("b p [i,,->i]", x, y, z).shape == (4, 128, 16)

    consts = {"b": 4, "h": 16, "w": 16, "c": 3, "p": 128}

    def make_coords(shape):
        return setup.full(shape, dtype=coord_dtype)

    xs = ["([h] b) [w] c", "[h] c [w]", "[h w]"]
    ys = ["b (p [2])", "[2] p", "[2]"]
    ys2 = ["p b", "p", "[1]"]
    zs = ["b p c", "c (p b)"]
    for x in xs:
        for z in zs:
            if not (z == "c (p b)" and getattr(einx, "name", "") == "torch.compile"):
                shape = einx.add(f"{z}, ", setup.full, 0, **consts, backend=backend).shape
                for y in ys:
                    assert (
                        einx.get_at(
                            f"{x}, {y} -> {z}",
                            setup.full,
                            make_coords,
                            **consts,
                            backend=backend,
                        ).shape
                        == shape
                    )
                for y1 in ys2:
                    for y2 in ys2:
                        assert (
                            einx.get_at(
                                f"{x}, {y1}, {y2} -> {z}",
                                setup.full,
                                make_coords,
                                make_coords,
                                **consts,
                                backend=backend,
                            ).shape
                            == shape
                        )

    for x in xs:
        shape = einx.add(
            f"{x.replace('[', '').replace(']', '')}, ",
            setup.full,
            0,
            **consts,
            backend=backend,
        ).shape
        for z in zs:
            z_axes = {a for a in z if a.isalpha()}
            for y in ys:
                if all(a in (x + y) for a in z_axes):
                    assert (
                        einx.set_at(
                            f"{x}, {y}, {z} -> {x}",
                            setup.full,
                            make_coords,
                            setup.full,
                            **consts,
                            backend=backend,
                        ).shape
                        == shape
                    )
            for y1 in ys2:
                for y2 in ys2:
                    if all(a in (x + y1 + y2) for a in z_axes):
                        assert (
                            einx.set_at(
                                f"{x}, {y1}, {y2}, {z} -> {x}",
                                setup.full,
                                make_coords,
                                make_coords,
                                setup.full,
                                **consts,
                                backend=backend,
                            ).shape
                            == shape
                        )
                        assert (
                            einx.set_at(
                                f"{x}, {y1}, {y2}, {z}",
                                setup.full,
                                make_coords,
                                make_coords,
                                setup.full,
                                **consts,
                                backend=backend,
                            ).shape
                            == shape
                        )


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_vmap_with_axis(test):
    einx, backend, setup = test

    x = setup.full((10, 10))
    assert einx.flip("a [b] -> a [b]", x).shape == (10, 10)
    assert einx.flip("a [b]", x).shape == (10, 10)
    assert einx.roll("a [b]", x, shift=5).shape == (10, 10)
    assert einx.roll("a [b]", x, shift=(5,)).shape == (10, 10)
    assert einx.softmax("a [b] -> a [b]", x).shape == (10, 10)
    assert einx.softmax("a [b]", x).shape == (10, 10)
    assert einx.softmax("a [b] -> (a [b]) c", x, c=3).shape == (100, 3)
    assert einx.softmax("a [b] -> a ([b] c)", x, c=3).shape == (10, 30)
    assert einx.log_softmax("(a [b]) c", x, b=2).shape == (10, 10)

    assert einx.flip("a ([b c])", x, b=2).shape == (10, 10)
    assert einx.roll(
        "a ([b c])",
        x,
        shift=(
            5,
            5,
        ),
        b=2,
    ).shape == (10, 10)


@pytest.mark.parametrize("test", conftest.tests)
def test_shape_solve(test):
    einx, backend, setup = test

    x = setup.full((2, 3, 4))
    assert einx.matches("a b c", x)
    assert not einx.matches("a b", x)
    with pytest.raises(Exception):
        einx.check("a b c d", x)
    einx.check("a b c", x)

    x = setup.full((6, 4))
    assert einx.matches("(a b) c", x)

    x = setup.full((2, 3, 4))
    assert einx.matches("a b...", x)

    x = setup.full((5, 4))
    assert einx.matches("(a + b) c", x)
    assert einx.matches("(a + b) c", x, a=2)
    assert not einx.matches("(a + b) c", x, a=10)

    params = einx.solve("a b, c b", x, x)
    assert params["a"] == 5 and params["b"] == 4 and params["c"] == 5
