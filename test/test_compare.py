import importlib
import einx

if importlib.util.find_spec("einops"):
    import einops
    import numpy as np

    def assert_equal_numpy(a, b):
        assert a.shape == b.shape
        if a.dtype.kind in "f":
            assert np.allclose(a, b)
        else:
            assert np.all(a == b)

    def test_compare_einops():
        x = np.random.uniform(size=(4, 128, 128, 3))

        assert_equal_numpy(
            einx.mean("b [s...] c", x), einops.reduce(x, "b ... c -> b c", reduction="mean")
        )
        assert_equal_numpy(
            einx.mean("b ... c -> b c", x), einops.reduce(x, "b ... c -> b c", reduction="mean")
        )

        assert_equal_numpy(
            einx.mean("b [s...] c", x, keepdims=True),
            einops.reduce(x, "b h w c -> b 1 c", reduction="mean"),
        )
        assert_equal_numpy(
            einx.mean("b [s]... c", x, keepdims=True),
            einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean"),
        )
        assert_equal_numpy(
            einx.mean("b h w c -> b 1 1 c", x),
            einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean"),
        )

        assert_equal_numpy(
            einx.sum("b (s [s2])... c", x, s2=2),
            einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="sum", h2=2, w2=2),
        )
        assert_equal_numpy(
            einx.sum("b (h h2) (w w2) c -> b h w c", x, h2=2, w2=2),
            einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="sum", h2=2, w2=2),
        )

        w = np.random.uniform(size=(3, 32))
        assert_equal_numpy(
            einx.dot("b... [c1->c2]", x, w), einops.einsum(x, w, "... c1, c1 c2 -> ... c2")
        )
        assert_equal_numpy(
            einx.dot("... c1, c1 c2 -> ... c2", x, w),
            einops.einsum(x, w, "... c1, c1 c2 -> ... c2"),
        )

        w = np.random.uniform(size=(128, 128, 64))
        assert_equal_numpy(
            einx.dot("b [s...->s2] c", x, w), einops.einsum(x, w, "b h w c, h w s2 -> b s2 c")
        )
        assert_equal_numpy(
            einx.dot("b h w c, h w s2 -> b s2 c", x, w),
            einops.einsum(x, w, "b h w c, h w s2 -> b s2 c"),
        )


if importlib.util.find_spec("torch"):
    import torch

    def assert_equal_torch(a, b):
        assert a.shape == b.shape
        if "float" in str(a.dtype):
            assert torch.allclose(a, b)
        else:
            assert torch.all(a == b)

    def test_compare_torch():
        # torch.gather torch.take_along_dim
        x = torch.rand(4, 128, 3)
        coords = torch.randint(4, (15, 128, 3))
        assert_equal_torch(
            torch.gather(x, 0, coords),
            einx.get_at("[_] ..., i ... -> i ...", x, coords),
        )
        assert_equal_torch(
            torch.take_along_dim(x, coords, dim=0),
            einx.get_at("[_] ..., i ... -> i ...", x, coords),
        )

        x = torch.rand(4, 128, 3)
        coords = torch.randint(128, (4, 15, 3))
        assert_equal_torch(
            torch.gather(x, 1, coords),
            einx.get_at("a [_] ..., a i ... -> a i ...", x, coords),
        )
        assert_equal_torch(
            torch.take_along_dim(x, coords, dim=1),
            einx.get_at("a [_] ..., a i ... -> a i ...", x, coords),
        )

        x = torch.rand(4, 128, 3)
        coords = torch.randint(3, (4, 128, 15))
        assert_equal_torch(
            torch.gather(x, 2, coords),
            einx.get_at("a b [_] ..., a b i ... -> a b i ...", x, coords),
        )
        assert_equal_torch(
            torch.take_along_dim(x, coords, dim=2),
            einx.get_at("a b [_] ..., a b i ... -> a b i ...", x, coords),
        )

        # torch.index_select
        x = torch.rand(4, 128, 3)
        indices = torch.randint(4, (15,))
        assert_equal_torch(
            torch.index_select(x, 0, indices),
            einx.get_at("[_] ..., i -> i ...", x, indices),
        )

        x = torch.rand(4, 128, 3)
        indices = torch.randint(128, (15,))
        assert_equal_torch(
            torch.index_select(x, 1, indices),
            einx.get_at("a [_] ..., i -> a i ...", x, indices),
        )

        x = torch.rand(4, 128, 3)
        indices = torch.randint(3, (15,))
        assert_equal_torch(
            torch.index_select(x, 2, indices),
            einx.get_at("a b [_] ..., i -> a b i ...", x, indices),
        )

        # torch.take
        x = torch.rand(128)
        indices = torch.randint(128, (15, 16, 3))
        assert_equal_torch(
            torch.take(x, indices),
            einx.get_at("[_], ... -> ...", x, indices),
        )

        # x[y]
        x = torch.rand((4, 128, 3))
        coords = (
            torch.rand((
                15,
                3,
            ))
            * torch.tensor(x.shape, dtype=torch.float32)
        ).to(torch.int32)
        assert_equal_torch(
            x[coords[..., 0], coords[..., 1], coords[..., 2]],
            einx.get_at("[a...], b... [3] -> b...", x, coords),
        )


if importlib.util.find_spec("tensorflow"):
    import os

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf

    def assert_equal_tf(a, b):
        assert a.shape == b.shape
        if "float" in str(a.dtype):
            assert tf.reduce_all(tf.abs(a - b) < 1e-6)
        else:
            assert tf.reduce_all(a == b)

    def test_compare_tensorflow():
        # tf.gather
        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((15,), maxval=4, dtype=tf.int32)
        assert_equal_tf(
            tf.gather(x, coords, axis=0),
            einx.get_at("[_] ..., i -> i ...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((15,), maxval=128, dtype=tf.int32)
        assert_equal_tf(
            tf.gather(x, coords, axis=1),
            einx.get_at("a [_] ..., i -> a i ...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((15,), maxval=3, dtype=tf.int32)
        assert_equal_tf(
            tf.gather(x, coords, axis=2),
            einx.get_at("a b [_] ..., i -> a b i ...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((4, 15), maxval=128, dtype=tf.int32)
        assert_equal_tf(
            tf.gather(x, coords, batch_dims=1, axis=1),
            einx.get_at("a [_] ..., a i -> a i ...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((4, 128, 15), maxval=3, dtype=tf.int32)
        assert_equal_tf(
            tf.gather(x, coords, batch_dims=2, axis=2),
            einx.get_at("a b [_] ..., a b i -> a b i ...", x, coords),
        )

        # tf.gather_nd
        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(
            tf.random.uniform((3,), dtype=tf.float32) * tf.cast(tf.shape(x), tf.float32), tf.int32
        )
        assert_equal_tf(
            tf.gather_nd(x, coords),
            einx.get_at("[a...], b... [3] -> b...", x, coords),
        )
        assert_equal_tf(
            x[coords[..., 0], coords[..., 1], coords[..., 2]],
            einx.get_at("[a...], b... [3] -> b...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(
            tf.random.uniform(
                (
                    15,
                    3,
                ),
                dtype=tf.float32,
            )
            * tf.cast(tf.shape(x), tf.float32),
            tf.int32,
        )
        assert_equal_tf(
            tf.gather_nd(x, coords),
            einx.get_at("[a...], b... [3] -> b...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(
            tf.random.uniform(
                (
                    15,
                    15,
                    3,
                ),
                dtype=tf.float32,
            )
            * tf.cast(tf.shape(x), tf.float32),
            tf.int32,
        )
        assert_equal_tf(
            tf.gather_nd(x, coords),
            einx.get_at("[a...], b... [3] -> b...", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(
            tf.random.uniform(
                (
                    4,
                    2,
                ),
                dtype=tf.float32,
            )
            * tf.cast(tf.shape(x)[1:], tf.float32),
            tf.int32,
        )
        assert_equal_tf(
            tf.gather_nd(x, coords, batch_dims=1),
            einx.get_at("a [...], a [2] -> a", x, coords),
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(
            tf.random.uniform(
                (
                    4,
                    15,
                    2,
                ),
                dtype=tf.float32,
            )
            * tf.cast(tf.shape(x)[1:], tf.float32),
            tf.int32,
        )
        assert_equal_tf(
            tf.gather_nd(x, coords, batch_dims=1),
            einx.get_at("a [...], a b [2] -> a b", x, coords),
        )


if importlib.util.find_spec("numpy"):
    import numpy as np

    def assert_equal_numpy(a, b):
        assert a.shape == b.shape
        if a.dtype.kind in "f":
            assert np.allclose(a, b)
        else:
            assert np.all(a == b)

    def test_compare_numpy():
        # np.matmul
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3, 4))
        assert_equal_numpy(
            np.matmul(x, y),
            einx.dot("a [b], [b] c -> a c", x, y),
        )

        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(2,))
        assert_equal_numpy(
            np.matmul(x, y),
            einx.dot("[a], [a] ->", x, y),
        )

        x = np.random.uniform(size=(16, 2, 3))
        y = np.random.uniform(size=(16, 3, 4))
        assert_equal_numpy(
            np.matmul(x, y),
            einx.dot("... a [b], ... [b] c -> ... a c", x, y),
        )

        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3,))
        assert_equal_numpy(
            np.matmul(x, y),
            einx.dot("... [b], [b] -> ...", x, y),
        )

        # np.dot
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3,))
        assert_equal_numpy(
            np.dot(x, y),
            einx.dot("... [b], [b] -> ...", x, y),
        )

        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(2,))
        assert_equal_numpy(
            np.dot(x, y),
            einx.dot("[a], [a] ->", x, y),
        )

        x = np.random.uniform(size=(5, 5, 2, 3))
        y = np.random.uniform(size=(5, 5, 3, 4))
        assert_equal_numpy(
            np.dot(x, y),
            einx.dot("x... [b], y... [b] c -> x... y... c", x, y),
        )

        # np.tensordot
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3, 4))
        assert_equal_numpy(
            np.tensordot(x, y, axes=1),
            einx.dot("a [b], [b] c -> a c", x, y),
        )

        x = np.random.uniform(size=(2, 3, 4))
        y = np.random.uniform(size=(5, 4, 6))
        assert_equal_numpy(
            np.tensordot(x, y, axes=([2], [1])),
            einx.dot("a b [c], d [c] e -> a b d e", x, y),
        )

        # np.inner
        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(2,))
        assert_equal_numpy(
            np.inner(x, y),
            einx.dot("x... [a], y... [a] -> x... y...", x, y),
        )

        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(4, 3))
        assert_equal_numpy(
            np.inner(x, y),
            einx.dot("x... [a], y... [a] -> x... y...", x, y),
        )

        # np.multiply
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(2, 3))
        assert_equal_numpy(
            np.multiply(x, y),
            einx.multiply("a b, a b -> a b", x, y),
        )

        # np.outer
        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(3,))
        assert_equal_numpy(
            np.outer(x, y),
            einx.multiply("a, b -> a b", x, y),
        )

        # np.kron
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(4, 5))
        assert_equal_numpy(
            np.kron(x, y),
            einx.multiply("a..., b... -> (a b)...", x, y),
        )

        # np.flip
        x = np.random.uniform(size=(2, 3))
        assert_equal_numpy(
            np.flip(x, axis=0),
            einx.flip("[a] b", x),
        )

        x = np.random.uniform(size=(2, 3))
        assert_equal_numpy(
            np.flip(x, axis=1),
            einx.flip("a [b]", x),
        )

        # np.fliplr
        x = np.random.uniform(size=(2, 3))
        assert_equal_numpy(
            np.fliplr(x),
            einx.flip("a [b]", x),
        )

        # np.flipud
        x = np.random.uniform(size=(2, 3))
        assert_equal_numpy(
            np.flipud(x),
            einx.flip("[a] b", x),
        )


if importlib.util.find_spec("scipy"):
    import numpy as np
    import scipy.linalg

    def assert_equal_numpy(a, b):
        assert a.shape == b.shape
        if a.dtype.kind in "f":
            assert np.allclose(a, b)
        else:
            assert np.all(a == b)

    def test_compare_scipy():
        # scipy.linalg.khatri_rao
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(5, 3))
        assert_equal_numpy(
            scipy.linalg.khatri_rao(x, y),
            einx.multiply("a c, b c -> (a b) c", x, y),
        )
