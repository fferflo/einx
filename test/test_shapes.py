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
from functools import partial
import numpy as np

backends = [b for b in einx.backend.backends if b != einx.backend.tracer]

@pytest.mark.parametrize("backend", backends)
def test_shape_rearrange(backend):
    x = backend.zeros((10, 20, 1), "float32")
    assert einx.rearrange("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    assert einx.rearrange("a b c -> (a b) c 1", x).shape == (200, 1, 1)
    assert einx.rearrange("a b c -> (a b) c 1 1 1", x).shape == (200, 1, 1, 1, 1)
    with pytest.raises(Exception):
        einx.rearrange("a a b c -> (a b) c 1", x)
        einx.rearrange("a (a + b) c -> (a b) c 1", x)

    x = backend.zeros((10, 20, 20, 2), "float32")
    assert einx.rearrange("b s... c -> b (s...) c", x).shape == (10, 400, 2)
    assert einx.rearrange("b ... c -> b (...) c", x).shape == (10, 400, 2)
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
    with pytest.raises(Exception):
        assert einx.rearrange("(s1...) (s2...) h -> 1 h (s1...) (s2...)", x).shape == (1, 5, 10, 20)

    x = backend.zeros((10, 20, 1), "float32")
    with pytest.raises(Exception):
        einx.rearrange("a b c -> (a b) c d", x)

    x = backend.zeros((10, 20, 1), "float32")
    with pytest.raises(Exception):
        einx.rearrange("a b... c... -> a (b...) c...", x)
    with pytest.raises(Exception):
        einx.rearrange("a b... -> a b", x)


    x = backend.zeros((1, 10, 20, 6), "float32")
    assert einx.rearrange("a (b...) (e f...) (d c) -> a d (b...) (e f...) c", x, d=2).shape == (1, 2, 10, 20, 3)

    x = backend.zeros((1, 10, 20, 6, 7, 12), "float32")
    assert einx.rearrange("a b c d... (e f...) -> a b c d... ((e 2 2) f...)", x, f=[2, 2]).shape == (1, 10, 20, 6, 7, 12 * 2 * 2)

    x = backend.zeros((10, 20, 3), "float32")
    assert einx.rearrange("(s s2)... c -> s... s2... c", x, s2=(2, 2)).shape == (5, 10, 2, 2, 3)
    assert einx.rearrange("(s s2)... c -> s... s2... c", x, s2=2).shape == (5, 10, 2, 2, 3)

    x = backend.zeros((10, 10, 10), "float32")
    assert einx.rearrange("(a b) (c d) (e f) -> a (b c d e) f", x, a=2, f=2).shape == (2, 250, 2)

    x = backend.zeros((10,), "float32")
    y = backend.zeros((20,), "float32")
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

    x = backend.zeros((10, 10), "float32")
    assert einx.rearrange("b c, 1 -> b (c + 1)", x, [42]).shape == (10, 11)

    assert einx.arange("c", c=2, backend=backend).shape == (2,)
    assert einx.arange("c... [2]", c=(4, 3), backend=backend).shape == (4, 3, 2)
    assert einx.arange("c... [l]", c=(4, 3), backend=backend).shape == (4, 3, 2)
    with pytest.raises(Exception):
        einx.arange("c... [3]", c=(4, 3), backend=backend)
    assert einx.arange("c1 c2 -> [l] c2 c1", c1=4, c2=3, backend=backend).shape == (2, 3, 4)
    assert einx.arange("(c...) [2]", c=(4, 3), backend=backend).shape == (4 * 3, 2)
    assert einx.arange("(c... [l])", c=(4, 3), backend=backend).shape == (4 * 3 * 2,)
    assert einx.arange("c1 c2 -> ([l] c2) c1", c1=4, c2=3, backend=backend).shape == (2 * 3, 4)

    x = backend.zeros((10, 20), "bool")
    y = backend.zeros((4, 10, 20, 3), "float32")
    x, y = einx.rearrange("h w, b h w c -> 1 h w 1, b h w c", x, y)
    assert backend.where(x, y, 0.0).shape == (4, 10, 20, 3)

@pytest.mark.parametrize("backend", backends)
def test_shape_dot(backend):
    x = backend.zeros((10, 10), "float32")
    assert einx.dot("a..., a... -> 1", x, x).shape == (1,)
    with pytest.raises(Exception):
        einx.dot("a..., [a]... -> 1", x, x)

    x = backend.zeros((10, 20, 1), "float32")
    y = backend.zeros((10, 24), "float32")
    assert einx.dot("a b c, a d -> 1 b c d", x, y).shape == (1, 20, 1, 24)
    assert einx.dot("a b c, a d -> 1 b c d", x, backend.zeros, d=24).shape == (1, 20, 1, 24)

    x = backend.zeros((10, 20, 1), "float32")
    with pytest.raises(Exception):
        einx.dot("a b c -> a b c", x, x)
    with pytest.raises(Exception):
        einx.dot("a b c, a -> a b c", x)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((20, 30), "float32")
    assert einx.dot("a [b] -> a [c]", x, y).shape == (10, 30)
    assert einx.dot("a b, b c -> a c", x, y).shape == (10, 30)
    assert einx.dot("a [b|c]", x, y).shape == (10, 30)
    assert einx.dot("a [b...|c]", x, y).shape == (10, 30)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((10, 20, 30), "float32")
    assert einx.dot("a b, a b c -> a c", x, y).shape == (10, 30)
    assert einx.dot("[a b] -> [a c]", x, y).shape == (10, 30)
    assert einx.dot("[a b|a c]", x, y).shape == (10, 30)

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

    w = backend.zeros((2, 2, 16, 32), "float32")
    assert einx.dot("b (s [s2|])... [c1|c2]", x, w, s2=2, c2=32).shape == (4, 64, 64, 32)

    x = backend.zeros((4, 16, 16, 16), "float32")
    w = lambda shape: backend.zeros(shape, "float32")
    assert einx.dot("b [(s s2)|s]... [c1|c2]", x, w, s2=4, c2=4).shape == (4, 4, 4, 4)
    assert einx.dot("b (s [s2|])... [c1|c2]", x, w, s2=4, c2=4).shape == (4, 4, 4, 4)

    x = backend.ones((10, 10), "float32")
    y = backend.ones((10,), "float32")
    assert einx.dot("[|]", 1, 1).shape == ()
    assert einx.dot("a [|]", y, 1).shape == (10,)
    assert einx.dot("a [b|]", x, y).shape == (10,)
    assert einx.dot("a [|b]", y, y).shape == (10, 10)

@pytest.mark.parametrize("backend", backends)
def test_shape_reduce(backend):
    x = backend.zeros((10, 10), "float32")
    assert einx.reduce("a b -> 1 a", x, op=backend.mean).shape == (1, 10)
    assert einx.mean("a b -> 1 a", x).shape == (1, 10)
    assert einx.mean("[a] b", x).shape == (10,)
    assert einx.mean("[a] b -> 1 b", x).shape == (1, 10)

    x = backend.zeros((10, 10, 10), "float32")
    with pytest.raises(Exception):
        einx.sum("a [b] c -> a b", x)

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
    assert einx.mean("1 [a...] b", x, keepdims=True).shape == (1, 1, 2)

    x = backend.zeros((16, 1, 20, 30, 64), "float32")
    assert einx.mean("(b rg) pv [s...] c", x).shape == (16, 1, 64)

    x = backend.ones((16, 16, 32))
    bias = backend.ones((4,))
    assert einx.add("b... (g [c])", x, bias).shape == (16, 16, 32)

    assert einx.logsumexp("a [...]", x).shape == (16,)

    assert einx.logsumexp("[a]", [0.0, 1.0]).shape == ()
    assert einx.logsumexp("[a]", [np.asarray(0.0), np.asarray(1.0)]).shape == ()
    assert einx.mean("[a]", [backend.to_tensor(0.0), np.asarray(1.0)]).shape == ()
    assert einx.sum("[a]", [backend.to_tensor(0.0), backend.to_tensor(1.0)]).shape == ()
    assert einx.logsumexp("[a] 1", [[0.0], [1.0]]).shape == (1,)
    assert einx.logsumexp("[a]", [0.0] * 10).shape == ()
    with pytest.raises(ValueError):
        einx.logsumexp("a", [0.0, [1.0]])

@pytest.mark.parametrize("backend", backends)
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
    with pytest.raises(Exception):
        einx.add("a a, a -> a a", x, y)
    assert einx.add("a b, a b", x, backend.zeros).shape == (10, 10)
    assert einx.add("a, a", y, y).shape == (10,)
    assert einx.add("[a]", y, y).shape == (10,)
    assert einx.add("b, -> b 3", y, 1).shape == (10, 3)

    x = backend.zeros((2, 3), "float32")
    y = backend.zeros((10,), "float32")
    with pytest.raises(Exception):
        einx.add("a b, c", x, y)

    x = backend.zeros((16, 128, 196, 64), "float32")
    y = backend.zeros((16, 4, 16), "float32")
    assert einx.add("b h w (g c), b (g) c -> b h w (g c)", x, y).shape == (16, 128, 196, 64)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((10, 20, 30), "float32")
    assert einx.add("a b, a b c -> a b c", x, y).shape == (10, 20, 30)
    assert einx.add("(a [1])...", x, backend.ones).shape == (10, 20)

    x = backend.zeros((10, 20), "float32")
    y = backend.zeros((30, 20), "float32")
    with pytest.raises(Exception):
        einx.subtract("ba c, i c -> i ba", x, y)

@pytest.mark.parametrize("backend", backends)
def test_shape_vmap(backend):
    x = backend.zeros((13,), "float32")
    assert einx.vmap("b -> b [3]", x, op=lambda x: x + backend.zeros((3,))).shape == (13, 3)

    with pytest.raises(ValueError):
        einx.vmap("b -> [b] 3", x, op=lambda x: x + backend.zeros((3,)))
    with pytest.raises(AssertionError):
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
    with pytest.raises(Exception):
        einx.vmap("[a] b [e], b [a] c [d] -> [g] b [a] c", x, y, op=f, g=15)
    
    with pytest.raises(Exception):
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

    def func(x): # (c d) -> 2
        x = einx.vmap("([c] d) -> d", x, op=backend.mean, c=16)
        x = backend.max(x)
        return backend.stack([x, x])
    x = backend.zeros((16, 64), "float32") # b c
    assert einx.vmap("b ([c d]) -> b [2]", x, op=func, c=16).shape == (16, 2)
    assert einx.vmap("b ([c d]) -> b [2] 1", x, op=func, c=16).shape == (16, 2, 1)
    assert einx.vmap("b [(c d)|2]", x, op=func, c=16).shape == (16, 2)
    assert einx.vmap("b ([c d|2])", x, op=func, c=16).shape == (16, 2)
    with pytest.raises(Exception):
        einx.vmap("b ([c d]) -> [2]", x, op=func, c=16)

    def func(x): # c d -> 2
        x = einx.vmap("[c] d -> d", x, op=backend.mean, c=16)
        x = backend.max(x)
        return backend.stack([x, x])
    x = backend.zeros((16, 64), "float32") # b c
    assert einx.vmap("b ([c d]) -> b [2]", x, op=func, c=16, flat=True).shape == (16, 2)
    assert einx.vmap("b ([c d]) -> b [2] 1", x, op=func, c=16, flat=True).shape == (16, 2, 1)
    assert einx.vmap("b [(c d)|2]", x, op=func, c=16, flat=True).shape == (16, 2)
    assert einx.vmap("b ([c d|2])", x, op=func, c=16, flat=True).shape == (16, 2)
    with pytest.raises(Exception):
        einx.vmap("b ([c d]) -> [2]", x, op=func, c=16, flat=True)

    with pytest.raises(Exception):
        einx.vmap_with_axis("a ([b c]) -> a ([b c])", x, op=partial(backend.roll, shift=(2, 2)))
    assert einx.vmap_with_axis("a ([b c]) -> a ([b c])", x, op=partial(backend.roll, shift=(2, 2)), b=2).shape == (16, 64)

@pytest.mark.parametrize("backend", backends)
def test_shape_index(backend):
    coord_dtype = "int32" if backend.name != "torch" else "long"
    x = backend.ones((4, 16, 16, 3))
    y = backend.cast(backend.ones((4, 128, 2)), coord_dtype)
    y2 = backend.cast(backend.ones((128, 4, 2)), coord_dtype)
    z = backend.ones((4, 128, 3))
    assert einx.get_at("b [h w] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, p b [2] -> b p c", x, y2).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b p, b p -> b p c", x, y[..., 0], y[..., 1]).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b (p [1]), b p -> b p c", x, y[..., 0], y[..., 1]).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b p, p b -> b p c", x, y[..., 0], y2[..., 1]).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, p, p b -> b p c", x, y[0, ..., 0], y2[..., 1]).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b (p [1]), p b -> b p c", x, y[..., 0], y2[..., 1]).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b p [2] -> b p c", x, lambda shape: backend.ones(shape, coord_dtype), p=128).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, b p [l] -> b p c", x, lambda shape: backend.ones(shape, coord_dtype), p=128).shape == (4, 128, 3)
    assert einx.get_at("b [16 w] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("b [16 16] c, b p [2] -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("b [h w] c, p [2] -> b p c", x, y[0]).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("b [h w] c, b p [2], b p c -> b [h w] c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p [2], b p c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p, b p, b p c -> b [h w] c", x, y[..., 0], y[..., 1], z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p, p b, b p c -> b [h w] c", x, y[..., 0], y2[..., 1], z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p, p b, p c -> b [h w] c", x, y[..., 0], y2[..., 1], z[0]).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p, p b, c -> b [h w] c", x, y[..., 0], y2[..., 1], z[0, 0]).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p [2], p c -> b [h w] c", x, y[0], z[0]).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p [2], b p c -> b h w c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, b p [2], p c -> b h w c", x, y, z[0]).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p [2], b p c -> b h w c", x, y[0], z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p [2], p c -> b h w c", x, y[0], z[0]).shape == (4, 16, 16, 3)

    x = backend.ones((16, 4, 3, 16))
    y = backend.cast(backend.ones((2, 4, 128)), coord_dtype)
    z = backend.ones((3, 4, 128))
    assert einx.get_at("[w] b c [h], [2] b p -> b p c", x, y).shape == (4, 128, 3)
    assert einx.get_at("[w] b c [h], [2] p -> b p c", x, y[:, 0]).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("[w] b c [h], [2] b p, c b p -> b [w h] c", x, y, z).shape == (4, 16, 16, 3)
        assert op("[w] b c [h], [2] p, c p -> b [w h] c", x, y[:, 0], z[:, 0]).shape == (4, 16, 16, 3)

    x = backend.ones((16, 4, 3 * 16))
    y = backend.cast(backend.ones((2, 4, 128)), coord_dtype)
    z = backend.ones((3, 4, 128))
    assert einx.get_at("[w] b (c [h]), [2] b p -> b p c", x, y, c=3).shape == (4, 128, 3)
    assert einx.get_at("[w] b (c [h]), [2] p -> b p c", x, y[:, 0], c=3).shape == (4, 128, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("[w] b (c [h]), [2] b p, c b p -> b ([w h]) c", x, y, z).shape == (4, 256, 3)
        assert op("[w] b (c [h]), [2] p, c p -> b ([w h]) c", x, y[:, 0], z[:, 0]).shape == (4, 256, 3)

    x = backend.ones((4, 16, 16, 3))
    y = backend.cast(backend.ones((4, 3, 4, 5, 2)), coord_dtype)
    z = backend.ones((4, 3, 4, 5, 3))
    assert einx.get_at("b [h w] c, b p q r [2] -> b p q r c", x, y).shape == (4, 3, 4, 5, 3)
    assert einx.get_at("b [h w] c, p q r [2] -> b p q r c", x, y[0]).shape == (4, 3, 4, 5, 3)
    for op in [einx.set_at, einx.add_at, einx.subtract_at]:
        assert op("b [h w] c, b p q r [2], b p q r c -> b [h w] c", x, y, z).shape == (4, 16, 16, 3)
        assert op("b [h w] c, p q r [2], p q r c -> b [h w] c", x, y[0], z[0]).shape == (4, 16, 16, 3)

    x = backend.ones((4, 1, 1, 3))
    y = backend.cast(backend.zeros((4, 128, 2)), coord_dtype)
    z = backend.ones((4, 128, 3))
    with pytest.raises(Exception):
        einx.get_at("b ([1 1]) c, b p [2] -> b p c", x, y)

    x = backend.zeros((4, 5, 6))
    y = backend.cast(backend.zeros((4, 5)), coord_dtype)
    assert einx.get_at("b t [d], b t -> b t", x, y).shape == (4, 5)
    assert einx.get_at("... [d], ... -> ...", x, y).shape == (4, 5)
    assert einx.get_at("b t [d], b (t [1]) -> b (t 1)", x, y).shape == (4, 5)
    with pytest.raises(ValueError):
        einx.get_at("b t [d], b (t [1]) -> b (t [1])", x, y)

    consts = {"b": 4, "h": 16, "w": 16, "c": 3, "p": 128}
    make_coords = lambda shape: backend.cast(backend.zeros(shape), coord_dtype)
    xs = ["([h] b) [w] c", "[h] c [w]", "[h w]"]
    ys = ["b (p [2])", "[2] p", "[2]"]
    ys2 = ["p b", "p", "[1]"]
    zs = ["b p c", "c (p b)"]
    for x in xs:
        for z in zs:
            shape = einx.add(f"{z}, ", backend.zeros, 0, **consts).shape
            for y in ys:
                assert einx.get_at(f"{x}, {y} -> {z}", backend.zeros, make_coords, **consts, backend=backend).shape == shape
            for y1 in ys2:
                for y2 in ys2:
                    assert einx.get_at(f"{x}, {y1}, {y2} -> {z}", backend.zeros, make_coords, make_coords, **consts, backend=backend).shape == shape

    for x in xs:
        shape = einx.add(f"{x.replace('[', '').replace(']', '')}, ", backend.zeros, 0, **consts).shape
        for z in zs:
            z_axes = set(a for a in z if a.isalpha())
            for y in ys:
                if all(a in (x + y) for a in z_axes):
                    assert einx.set_at(f"{x}, {y}, {z} -> {x}", backend.ones, make_coords, backend.zeros, **consts, backend=backend).shape == shape
            for y1 in ys2:
                for y2 in ys2:
                    if all(a in (x + y1 + y2) for a in z_axes):
                        assert einx.set_at(f"{x}, {y1}, {y2}, {z} -> {x}", backend.ones, make_coords, make_coords, backend.zeros, **consts, backend=backend).shape == shape
                        assert einx.set_at(f"{x}, {y1}, {y2}, {z}", backend.ones, make_coords, make_coords, backend.zeros, **consts, backend=backend).shape == shape

@pytest.mark.parametrize("backend", backends)
def test_shape_vmap_with_axis(backend):
    x = backend.ones((10, 10), "float32")
    assert einx.flip("a [b] -> a [b]", x).shape == (10, 10)
    assert einx.flip("a [b]", x).shape == (10, 10)
    assert einx.roll("a [b]", x, shift=5).shape == (10, 10)
    assert einx.roll("a [b]", x, shift=(5,)).shape == (10, 10)
    assert einx.softmax("a [b] -> a [b]", x).shape == (10, 10)
    assert einx.softmax("a [b]", x).shape == (10, 10)
    assert einx.log_softmax("(a [b]) c", x, b=2).shape == (10, 10)

    assert einx.flip("a ([b c])", x, b=2).shape == (10, 10)
    assert einx.roll("a ([b c])", x, shift=(5, 5,), b=2).shape == (10, 10)

@pytest.mark.parametrize("backend", backends)
def test_shape_solve(backend):
    x = backend.ones((2, 3, 4))
    assert einx.matches("a b c", x)
    assert not einx.matches("a b", x)
    with pytest.raises(Exception):
        einx.check("a b c d", x)
    einx.check("a b c", x)

    x = backend.ones((6, 4))
    assert einx.matches("(a b) c", x)

    x = backend.ones((2, 3, 4))
    assert einx.matches("a b...", x)

    x = backend.ones((5, 4))
    assert einx.matches("(a + b) c", x)
    assert einx.matches("(a + b) c", x, a=2)
    assert not einx.matches("(a + b) c", x, a=10)