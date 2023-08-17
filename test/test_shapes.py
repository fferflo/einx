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

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_shape_rearrange(backend):
    x = backend.zeros((10, 20, 1), "float32")
    assert einx.rearrange("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    assert einx.rearrange("a b c -> (a b) c 1...", x, output_ndims=5).shape == (200, 1, 1, 1, 1)

    x = backend.zeros((10, 20, 20, 2), "float32")
    assert einx.rearrange("b s... c -> b (s...) c", x).shape == (10, 400, 2)
    assert einx.rearrange("b (s...) (r...) c -> b (s...) r... c", x, r=(10, 2)).shape == (10, 20, 10, 2, 2)

    assert einx.rearrange("1 -> (x)", backend.to_tensor([1]), x=10).shape == (10,)
    assert einx.rearrange("1 -> (x y)", backend.to_tensor([1]), x=10, y=20).shape == (200,)

    x = backend.zeros((10, 20, 1), "float32")
    assert einx.rearrange("a b c d... -> a b c (d...)", x).shape == (10, 20, 1, 1)

    x = backend.zeros((10, 20, 1, 2), "float32")
    assert einx.rearrange("a (b...) c d -> a (b... c) d", x).shape == (10, 20, 2)

    x = backend.zeros((10, 20, 1, 2, 3), "float32")
    assert einx.rearrange("a (b... c) d... e -> a (b...) (c d...) e", x, b=[2, 5]).shape == (10, 10, 4, 3)

    x = backend.zeros((10, 20, 6, 24), "float32")
    assert einx.rearrange("a b (c...) (d...) -> a c... b d...", x, c=[2, 3], d=[4, 6]).shape == (10, 2, 3, 20, 4, 6)

    x = backend.zeros((10, 10), "float32")
    assert einx.rearrange("a... -> 1 (a...)", x).shape == (1, 100)

    x = backend.zeros((10, 20, 5), "float32")
    assert einx.rearrange("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x).shape == (1, 5, 10, 20)

    x = backend.zeros((10, 20), "float32")
    with pytest.raises(ValueError):
        assert einx.rearrange("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x).shape == (1, 5, 10, 20)

    x = backend.zeros((10, 20, 1), "float32")
    with pytest.raises(ValueError):
        einx.rearrange("a b c -> (a b) c d", x)
    with pytest.raises(ValueError):
        einx.rearrange("a... a__0 c -> a... c a__0", x)

    x = backend.zeros((10, 20, 1), "float32")
    with pytest.raises(ValueError):
        einx.rearrange("a b... c... -> a (b...) c...", x)

    x = backend.zeros((1, 10, 20, 6), "float32")
    assert einx.rearrange("a (b...) (e f...) (d c) -> a d (b...) (e f...) c", x, d=2).shape == (1, 2, 10, 20, 3)

    x = backend.zeros((1, 10, 20, 6, 7, 12), "float32")
    assert einx.rearrange("a b c d... (e f...) -> a b c d... ((e 2 2) f...)", x, f=[2, 2]).shape == (1, 10, 20, 6, 7, 12 * 2 * 2)

    x = backend.zeros((10, 20, 3), "float32")
    assert einx.rearrange("(s s2)... c -> s... s2... c", x, s2=(2, 2)).shape == (5, 10, 2, 2, 3)
    assert einx.rearrange("(s s2)... c -> s... s2... c", x, s2=2).shape == (5, 10, 2, 2, 3)

    x = backend.zeros((10, 10, 10), "float32")
    assert einx.rearrange("(a b) (c d) (e f) -> a (b c d e) f", x, a=2, f=2).shape == (2, 250, 2)

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_shape_dot(backend):
    x = backend.zeros((10, 10), "float32")
    assert einx.dot("a..., a... -> 1", x, x).shape == (1,)
    with pytest.raises(ValueError):
        einx.dot("a..., [a]... -> 1", x, x)

    x = backend.zeros((10, 20, 1), "float32")
    y = backend.zeros((10, 24), "float32")
    assert einx.dot("a b c, a d -> 1 b c d", x, y).shape == (1, 20, 1, 24)
    assert einx.dot("a b c, a d -> 1 b c d", x, backend.zeros, d=24).shape == (1, 20, 1, 24)

    x = backend.zeros((10, 20, 1), "float32")
    with pytest.raises(ValueError):
        einx.dot("a b c -> a b c", x, x)
    with pytest.raises(ValueError):
        einx.dot("a b c, a -> a b c", x)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((20, 30), "float32")
    assert einx.dot("a b -> a c", x, y).shape == (10, 30)
    assert einx.dot("a b, b c -> a c", x, y).shape == (10, 30)
    assert einx.dot("a [b|c]", x, y).shape == (10, 30)
    assert einx.dot("a [b...|c]", x, y).shape == (10, 30)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((10, 20, 30), "float32")
    assert einx.dot("a b, a b c -> a c", x, y).shape == (10, 30)
    assert einx.dot("[a] b -> a c", x, y).shape == (10, 30)

    x = backend.zeros((10,), "float32")
    y = backend.zeros((30,), "float32")
    assert einx.dot("a, a ->", x, x).shape == ()
    assert einx.dot("[a|]", x, x).shape == ()
    assert einx.dot("a, c -> a c", x, y).shape == (10, 30)
    assert einx.dot("a [|c]", x, y).shape == (10, 30)
    assert einx.dot("a [b...|c]", x, y).shape == (10, 30)

    x = backend.zeros((4, 128, 128, 16), "float32")
    assert einx.dot("b s... [c1|c2]", x, backend.zeros, c2=32).shape == (4, 128, 128, 32)
    assert einx.dot("b [s...|s2] c", x, backend.zeros, s2=32).shape == (4, 32, 16)

    w = backend.zeros((2, 2, 16, 1, 1, 32), "float32")
    assert einx.dot("b (s [s2|])... [c1|c2]", x, w, s2=2, c2=32).shape == (4, 64, 64, 32)

    w = lambda shape: backend.zeros(shape, "float32")
    assert einx.dot("b [(s s2)|s]... [c1|c2]", x, w, s2=4, c2=64).shape == (4, 32, 32, 64)
    assert einx.dot("b (s [s2|])... [c1|c2]", x, w, s2=4, c2=64).shape == (4, 32, 32, 64)

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_shape_reduce(backend):
    x = backend.zeros((10, 10), "float32")
    assert einx.reduce("a b -> 1 a", x, op=backend.mean).shape == (1, 10)
    assert einx.mean("a b -> 1 a", x).shape == (1, 10)
    assert einx.mean("[a] b", x).shape == (10,)

    x = backend.zeros((10, 3, 1), "float32")
    assert einx.mean("(a [b]) c 1", x, b=2).shape == (5, 3, 1)
    assert einx.mean("([a b]) c 1", x).shape == (1, 3, 1)
    assert einx.mean("[(a b)] c 1", x).shape == (3, 1)
    assert einx.mean("[(a...)] c 1", x).shape == (3, 1)
    assert einx.mean("(b... [a...]) c 1", x, b=(1, 1)).shape == (1, 3, 1)

    x = backend.zeros((1, 10, 3, 2), "float32")
    assert einx.mean("1 [a...] b", x).shape == (1, 2)
    assert einx.mean("1 [a]... b", x).shape == (1, 2)
    assert einx.mean("1 ([a])... b", x).shape == (1, 1, 1, 2)
    assert einx.mean("1 [a]... b", x, keepdims=True).shape == (1, 1, 1, 2)
    assert einx.mean("1 [a...] b", x, keepdims=True).shape == (1, 1, 1, 2)

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_shape_elementwise(backend):
    x = backend.zeros((10, 5, 1), "float32")
    y = backend.zeros((13,), "float32")
    assert einx.elementwise("a b 1, l -> b l a 1", x, y, op=backend.add).shape == (5, 13, 10, 1)
    assert einx.add("a b 1, l -> b l a 1", x, y).shape == (5, 13, 10, 1)
    assert einx.add("a b 1, l -> a b l", x, y).shape == (10, 5, 13)

    x = backend.zeros((10, 10), "float32")
    y = backend.zeros((10,), "float32")
    assert einx.add("a, a b", y, x).shape == (10, 10)
    assert einx.add("a b, a", x, y).shape == (10, 10)
    assert einx.add("a b, b", x, y).shape == (10, 10)
    assert einx.add("a [b]", x, y).shape == (10, 10)
    assert einx.add("a b, a b", x, x).shape == (10, 10)
    assert einx.add("a b, ", x, 1).shape == (10, 10)
    assert einx.add(", a b", 1, x).shape == (10, 10)
    assert einx.add("a b, 1", x, [1]).shape == (10, 10)
    assert einx.add("1, a b", [1], x).shape == (10, 10)
    with pytest.raises(ValueError):
        einx.add("a a, a -> a a", x, y)
    assert einx.add("a b, a b", x, backend.zeros).shape == (10, 10)
    assert einx.add("a, a", y, y).shape == (10,)
    assert einx.add("[a]", y, y).shape == (10,)

    # TODO: make this work
    # x = backend.zeros((2, 3), "float32")
    # y = backend.zeros((10,), "float32")
    # assert einx.add("a b, c", x, y).shape == (2, 3, 10)
    # x = backend.zeros((2, 3, 10), "float32")
    # y = backend.zeros((10, 4), "float32")
    # assert einx.add("a b c, c d", x, y).shape == (2, 3, 10, 4)

    x = backend.zeros((16, 128, 196, 64), "float32")
    y = backend.zeros((16, 4, 16), "float32")
    assert einx.add("b h w (g c), b (g) c", x, y).shape == (16, 128, 196, 64)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((10, 20, 30), "float32")
    assert einx.add("a b, a b c -> a b c", x, y).shape == (10, 20, 30)

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_anonymous_ellipsis_success(backend):
    x = backend.zeros((10, 5, 1), "float32")

    einx.rearrange("b ... -> ... b", x) # Succeeds

    einx.anonymous_ellipsis_name = None

    from einx.api.rearrange import _parse
    if "cache_clear" in dir(_parse):
        _parse.cache_clear()

    with pytest.raises(ValueError):
        einx.rearrange("b ... -> ... b", x) # Fails

@pytest.mark.parametrize("backend", einx.backend.backends)
def test_shape_vmap(backend):
    x = backend.zeros((13,), "float32")
    assert einx.vmap("b -> b [3]", x, op=lambda x: x + backend.zeros((3,))).shape == (13, 3)

    with pytest.raises(ValueError):
        einx.vmap("b -> [b] 3", x, op=lambda x: x + backend.zeros((3,)))
    with pytest.raises(ValueError):
        einx.vmap("b -> b 3", x, op=lambda x: x + backend.ones((3,)))

    x = backend.zeros((4, 13, 2), "float32")
    y = backend.zeros((13, 4, 5, 5), "float32")
    def f(x, y):
        assert x.shape == (4, 2)
        assert y.shape == (4, 5)
        x = x[:, 0] + y[:, 0]
        return einx.rearrange("a -> a 15", x)
    assert einx.vmap("[a] b [e], b [a] c [d] -> [a] b [g] c", x, y, op=f, g=15).shape == (4, 13, 15, 5)
    assert einx.vmap("[a] b [e], b [a] c [d] -> [a] b ([g] c)", x, y, op=f, g=15).shape == (4, 13, 15 * 5)
    with pytest.raises(ValueError):
        einx.vmap("[a] b [e], b [a] c [d] -> [g] b [a] c", x, y, op=f, g=15)
    
    with pytest.raises(ValueError):
        def f(x, y):
            assert x.shape == (4, 2)
            assert y.shape == (4, 5)
            x = x[:, 0] + y[:, 0]
            return einx.rearrange("a -> a 16", x)
        einx.vmap("[a] b [e], b [a] c [d] -> [a] b [g] c", x, y, op=f, g=15)

    x = backend.zeros((4, 16), "float32")
    y = backend.zeros((16, 32), "float32")
    assert einx.vmap("b [c1], [c1] c2 -> b c2", x, y, op=backend.dot).shape == (4, 32)

    x = backend.zeros((4,), "float32")
    y = backend.zeros((16, 32), "float32")
    assert einx.vmap("a, b c -> a b c", x, y, op=backend.add).shape == (4, 16, 32)

    def func(x): # c -> 2
        return backend.stack([backend.mean(x), backend.max(x)])
    x = backend.zeros((16, 64, 3,), "float32")
    assert einx.vmap("b [c] a -> a b [2]", x, op=func).shape == (3, 16, 2)

    def func(x, y): # c, d -> 2
        return backend.stack([backend.mean(x), backend.max(y)])
    x = backend.zeros((16, 64), "float32") # b c
    y = backend.zeros((16, 72), "float32") # b d
    assert einx.vmap("b [c], b [d] -> b [2]", x, y, op=func).shape == (16, 2)

    x = backend.zeros((16, 64, 3), "float32") # b1 c b2
    y = backend.zeros((3, 72), "float32") # b2 d
    assert einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=func).shape == (3, 2, 16)