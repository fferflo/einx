import importlib
if importlib.util.find_spec("einops"):
    import einops, einx
    import numpy as np

    def assert_equal(a, b):
        assert a.shape == b.shape
        if a.dtype.kind in "f":
            assert np.allclose(a, b)
        else:
            assert np.all(a == b)

    def test_compare():
        x = np.random.uniform(size=(4, 128, 128, 3))

        assert_equal(einx.mean("b [s...] c", x), einops.reduce(x, "b ... c -> b c", reduction="mean"))
        assert_equal(einx.mean("b ... c -> b c", x), einops.reduce(x, "b ... c -> b c", reduction="mean"))

        assert_equal(einx.mean("b [s...] c", x, keepdims=True), einops.reduce(x, "b h w c -> b 1 c", reduction="mean"))
        assert_equal(einx.mean("b [s]... c", x, keepdims=True), einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean"))
        assert_equal(einx.mean("b h w c -> b 1 1 c", x), einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean"))

        assert_equal(einx.sum("b (s [s2])... c", x, s2=2), einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="sum", h2=2, w2=2))
        assert_equal(einx.sum("b (h h2) (w w2) c -> b h w c", x, h2=2, w2=2), einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="sum", h2=2, w2=2))

        w = np.random.uniform(size=(3, 32))
        assert_equal(einx.dot("b... [c1|c2]", x, w), einops.einsum(x, w, "... c1, c1 c2 -> ... c2"))
        assert_equal(einx.dot("... c1, c1 c2 -> ... c2", x, w), einops.einsum(x, w, "... c1, c1 c2 -> ... c2"))

        w = np.random.uniform(size=(128, 128, 64))
        assert_equal(einx.dot("b [s...|s2] c", x, w), einops.einsum(x, w, "b h w c, h w s2 -> b s2 c"))
        assert_equal(einx.dot("b h w c, h w s2 -> b s2 c", x, w), einops.einsum(x, w, "b h w c, h w s2 -> b s2 c"))