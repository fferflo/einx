import importlib
import types
from functools import partial
import numpy as np
import conftest
from conftest import assert_allclose
import pytest

if importlib.util.find_spec("einops"):
    import einops

    def test_compare_einops():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x)

        x = np.random.uniform(size=(4, 128, 128, 3)).astype("float32")

        assert_allclose(
            einx.mean("b [s...] c", x),
            einops.reduce(x, "b ... c -> b c", reduction="mean"),
            setup,
        )
        assert_allclose(
            einx.mean("b ... c -> b c", x),
            einops.reduce(x, "b ... c -> b c", reduction="mean"),
            setup,
        )

        assert_allclose(
            einx.mean("b ([s])... c", x),
            einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean"),
            setup,
        )
        assert_allclose(
            einx.mean("b h w c -> b 1 1 c", x),
            einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean"),
            setup,
        )

        assert_allclose(
            einx.sum("b (s [ds])... c", x, ds=2),
            einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="sum", h2=2, w2=2),
            setup,
        )
        assert_allclose(
            einx.sum("b (h h2) (w w2) c -> b h w c", x, h2=2, w2=2),
            einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="sum", h2=2, w2=2),
            setup,
        )

        w = np.random.uniform(size=(3, 32)).astype("float32")
        assert_allclose(
            einx.dot("... [c1], [c1] c2 -> ... c2", x, w),
            einops.einsum(x, w, "... c1, c1 c2 -> ... c2"),
            setup,
        )

        w = np.random.uniform(size=(128, 128, 64)).astype("float32")
        assert_allclose(
            einx.dot("b [s...] c, [s...] s2 -> b s2 c", x, w),
            einops.einsum(x, w, "b h w c, h w s2 -> b s2 c"),
            setup,
        )

        x = np.random.uniform(size=(2, 3, 4, 5)).astype("float32")
        y = np.random.uniform(size=(2, 3, 3)).astype("float32")
        assert_allclose(
            einops.pack([x, y], "a b *")[0],
            einx.id("a b c d, a b e -> a b ((c d) + e)", x, y),
            setup,
        )


if importlib.util.find_spec("torch"):
    import torch

    def test_compare_torch():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x.detach().cpu().numpy())

        # torch.gather torch.take_along_dim
        x = torch.rand(4, 128, 3)
        coords = torch.randint(4, (15, 128, 3))
        assert_allclose(
            torch.gather(x, 0, coords),
            einx.get_at("[_] ..., i ... -> i ...", x, coords),
            setup,
        )
        assert_allclose(
            torch.take_along_dim(x, coords, dim=0),
            einx.get_at("[_] ..., i ... -> i ...", x, coords),
            setup,
        )

        x = torch.rand(4, 128, 3)
        coords = torch.randint(128, (4, 15, 3))
        assert_allclose(
            torch.gather(x, 1, coords),
            einx.get_at("a [_] ..., a i ... -> a i ...", x, coords),
            setup,
        )
        assert_allclose(
            torch.take_along_dim(x, coords, dim=1),
            einx.get_at("a [_] ..., a i ... -> a i ...", x, coords),
            setup,
        )

        x = torch.rand(4, 128, 3)
        coords = torch.randint(3, (4, 128, 15))
        assert_allclose(
            torch.gather(x, 2, coords),
            einx.get_at("a b [_] ..., a b i ... -> a b i ...", x, coords),
            setup,
        )
        assert_allclose(
            torch.take_along_dim(x, coords, dim=2),
            einx.get_at("a b [_] ..., a b i ... -> a b i ...", x, coords),
            setup,
        )

        # torch.index_select
        x = torch.rand(4, 128, 3)
        indices = torch.randint(4, (15,))
        assert_allclose(
            torch.index_select(x, 0, indices),
            einx.get_at("[_] ..., i -> i ...", x, indices),
            setup,
        )

        x = torch.rand(4, 128, 3)
        indices = torch.randint(128, (15,))
        assert_allclose(
            torch.index_select(x, 1, indices),
            einx.get_at("a [_] ..., i -> a i ...", x, indices),
            setup,
        )

        x = torch.rand(4, 128, 3)
        indices = torch.randint(3, (15,))
        assert_allclose(
            torch.index_select(x, 2, indices),
            einx.get_at("a b [_] ..., i -> a b i ...", x, indices),
            setup,
        )

        # torch.take
        x = torch.rand(128)
        indices = torch.randint(128, (15, 16, 3))
        assert_allclose(
            torch.take(x, indices),
            einx.get_at("[_], ... -> ...", x, indices),
            setup,
        )

        # x[y]
        x = torch.rand((4, 128, 3))
        coords = (torch.rand((15, 3)) * torch.tensor(x.shape, dtype=torch.float32)).to(torch.int64)
        assert_allclose(
            x[coords[..., 0], coords[..., 1], coords[..., 2]],
            einx.get_at("[a...], b... [3] -> b...", x, coords),
            setup,
        )

        # from docs
        x = torch.rand((3, 15))
        x1, x2, x3 = torch.split(x, [4, 5, 6], dim=-1)
        x1e, x2e, x3e = einx.id("a (b1 + b2 + b3) -> a b1, a b2, a b3", x, b1=4, b2=5)
        assert_allclose(x1, x1e, setup)
        assert_allclose(x2, x2e, setup)
        assert_allclose(x3, x3e, setup)


if importlib.util.find_spec("tensorflow"):
    import os

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf

    def test_compare_tensorflow():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x.numpy())

        # tf.gather
        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((15,), maxval=4, dtype=tf.int32)
        assert_allclose(
            tf.gather(x, coords, axis=0),
            einx.get_at("[_] ..., i -> i ...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((15,), maxval=128, dtype=tf.int32)
        assert_allclose(
            tf.gather(x, coords, axis=1),
            einx.get_at("a [_] ..., i -> a i ...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((15,), maxval=3, dtype=tf.int32)
        assert_allclose(
            tf.gather(x, coords, axis=2),
            einx.get_at("a b [_] ..., i -> a b i ...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((4, 15), maxval=128, dtype=tf.int32)
        assert_allclose(
            tf.gather(x, coords, batch_dims=1, axis=1),
            einx.get_at("a [_] ..., a i -> a i ...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.random.uniform((4, 128, 15), maxval=3, dtype=tf.int32)
        assert_allclose(
            tf.gather(x, coords, batch_dims=2, axis=2),
            einx.get_at("a b [_] ..., a b i -> a b i ...", x, coords),
            setup,
        )

        # tf.gather_nd
        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(tf.random.uniform((3,), dtype=tf.float32) * tf.cast(tf.shape(x), tf.float32), tf.int32)
        assert_allclose(
            tf.gather_nd(x, coords),
            einx.get_at("[a...], b... [3] -> b...", x, coords),
            setup,
        )
        assert_allclose(
            x[coords[..., 0], coords[..., 1], coords[..., 2]],
            einx.get_at("[a...], b... [3] -> b...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(tf.random.uniform((15, 3), dtype=tf.float32) * tf.cast(tf.shape(x), tf.float32), tf.int32)
        assert_allclose(
            tf.gather_nd(x, coords),
            einx.get_at("[a...], b... [3] -> b...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(tf.random.uniform((15, 15, 3), dtype=tf.float32) * tf.cast(tf.shape(x), tf.float32), tf.int32)
        assert_allclose(
            tf.gather_nd(x, coords),
            einx.get_at("[a...], b... [3] -> b...", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(tf.random.uniform((4, 2), dtype=tf.float32) * tf.cast(tf.shape(x)[1:], tf.float32), tf.int32)
        assert_allclose(
            tf.gather_nd(x, coords, batch_dims=1),
            einx.get_at("a [...], a [2] -> a", x, coords),
            setup,
        )

        x = tf.random.uniform((4, 128, 3))
        coords = tf.cast(tf.random.uniform((4, 15, 2), dtype=tf.float32) * tf.cast(tf.shape(x)[1:], tf.float32), tf.int32)
        assert_allclose(
            tf.gather_nd(x, coords, batch_dims=1),
            einx.get_at("a [...], a b [2] -> a b", x, coords),
            setup,
        )


if importlib.util.find_spec("numpy"):

    def test_compare_numpy():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x)

        # np.matmul
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3, 4))
        assert_allclose(
            np.matmul(x, y),
            einx.dot("a [b], [b] c -> a c", x, y),
            setup,
        )

        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(2,))
        assert_allclose(
            np.matmul(x, y),
            einx.dot("[a], [a] ->", x, y),
            setup,
        )

        x = np.random.uniform(size=(16, 2, 3))
        y = np.random.uniform(size=(16, 3, 4))
        assert_allclose(
            np.matmul(x, y),
            einx.dot("... a [b], ... [b] c -> ... a c", x, y),
            setup,
        )

        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3,))
        assert_allclose(
            np.matmul(x, y),
            einx.dot("... [b], [b] -> ...", x, y),
            setup,
        )

        # np.dot
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3,))
        assert_allclose(
            np.dot(x, y),
            einx.dot("... [b], [b] -> ...", x, y),
            setup,
        )

        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(2,))
        assert_allclose(
            np.dot(x, y),
            einx.dot("[a], [a] ->", x, y),
            setup,
        )

        x = np.random.uniform(size=(5, 5, 2, 3))
        y = np.random.uniform(size=(5, 5, 3, 4))
        assert_allclose(
            np.dot(x, y),
            einx.dot("x... [b], y... [b] c -> x... y... c", x, y),
            setup,
        )

        # np.tensordot
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3, 4))
        assert_allclose(
            np.tensordot(x, y, axes=1),
            einx.dot("a [b], [b] c -> a c", x, y),
            setup,
        )

        x = np.random.uniform(size=(2, 3, 4))
        y = np.random.uniform(size=(5, 4, 6))
        assert_allclose(
            np.tensordot(x, y, axes=([2], [1])),
            einx.dot("a b [c], d [c] e -> a b d e", x, y),
            setup,
        )
        assert_allclose(
            np.tensordot(x, y, axes=(2, 1)),
            einx.dot("a b [c], d [c] e -> a b d e", x, y),
            setup,
        )

        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(4, 2))
        assert_allclose(
            np.tensordot(x, y, axes=(0, 1)),
            einx.dot("[a] b, c [a] -> b c", x, y),
            setup,
        )

        # np.inner
        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(2,))
        assert_allclose(
            np.inner(x, y),
            einx.dot("x... [a], y... [a] -> x... y...", x, y),
            setup,
        )

        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(4, 3))
        assert_allclose(
            np.inner(x, y),
            einx.dot("x... [a], y... [a] -> x... y...", x, y),
            setup,
        )

        # np.multiply
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.multiply(x, y),
            einx.multiply("a b, a b -> a b", x, y),
            setup,
        )

        # np.outer
        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(3,))
        assert_allclose(
            np.outer(x, y),
            einx.multiply("a, b -> a b", x, y),
            setup,
        )

        # np.kron
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(4, 5))
        assert_allclose(
            np.kron(x, y),
            einx.multiply("a..., b... -> (a b)...", x, y),
            setup,
        )

        # np.flip
        x = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.flip(x, axis=0),
            einx.flip("[a] b", x),
            setup,
        )

        x = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.flip(x, axis=1),
            einx.flip("a [b]", x),
            setup,
        )

        # np.fliplr
        x = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.fliplr(x),
            einx.flip("a [b]", x),
            setup,
        )

        # np.flipud
        x = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.flipud(x),
            einx.flip("[a] b", x),
            setup,
        )

        # np.stack
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.stack([x, y], axis=-1),
            einx.id("..., ... -> ... (1 + 1)", x, y),
            setup,
        )

        # np.concatenate
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(2, 3))
        assert_allclose(
            np.concatenate([x, y], axis=-1),
            einx.id("... a, ... b -> ... (a + b)", x, y),
            setup,
        )

        img = np.random.uniform(size=(2, 3, 4, 5))
        vec = np.random.uniform(size=(6,))
        assert_allclose(
            einx.id("b h w c1, c2 -> b h w (c1 + c2)", img, vec),
            np.concatenate([img, np.broadcast_to(vec[None, None, None, :], img.shape[:3] + vec.shape)], axis=-1),
            setup,
        )
        assert_allclose(
            einx.id("b c1 h w, c2 -> b (c1 + c2) h w", img, vec),
            np.concatenate([img, np.broadcast_to(vec[None, :, None, None], (img.shape[0], vec.shape[0], img.shape[2], img.shape[3]))], axis=1),
            setup,
        )

        # np.meshgrid
        x = np.random.uniform(size=(2,))
        y = np.random.uniform(size=(3,))
        assert_allclose(
            np.meshgrid(x, y, indexing="ij"),
            einx.id("x, y -> x y, x y", x, y),
            setup,
        )
        assert_allclose(
            np.meshgrid(x, y, indexing="xy"),
            einx.id("x, y -> y x, y x", x, y),
            setup,
        )

        # np.einsum
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(3, 4))
        assert_allclose(
            np.einsum("ab,bc->ac", x, y),
            einx.dot("a [b], [b] c -> a c", x, y),
            setup,
        )
        assert_allclose(
            np.einsum("ab->a", x),
            einx.sum("a [b]", x),
            setup,
        )
        assert_allclose(
            np.einsum("ab,bc->abc", x, y),
            einx.multiply("a b, b c -> a b c", x, y),
            setup,
        )

        x = np.random.uniform(size=(3, 3))
        assert_allclose(
            np.einsum("aa->a", x),
            einx.id("a a -> a", x),
            setup,
        )

        # indexing
        x = np.random.uniform(size=(4, 5))
        coords = (np.random.uniform(size=(6,))[:, None] * np.asarray(x.shape[:2])[None, :]).astype("int32")
        assert_allclose(
            x[coords[:, 0], coords[:, 1]],
            einx.get_at("[x y], a [2] -> a", x, coords),
            setup,
        )

        # np.zeros
        x = np.zeros((3, 4, 5))
        assert_allclose(
            x,
            einx.id("-> 3 4 5", 0.0),
            setup,
        )
        assert_allclose(
            x,
            einx.id("-> a...", 0.0, a=(3, 4, 5)),
            setup,
        )

        # from docs
        x = np.random.uniform(size=(2, 4))
        y = np.random.uniform(size=(2, 3, 5))
        assert_allclose(
            x[:, np.newaxis, :, np.newaxis] + y[:, :, np.newaxis, :],
            einx.add("a c, a b d -> a b c d", x, y),
            setup,
        )

        x = np.random.uniform(size=(2, 3, 4, 5))
        assert_allclose(
            np.transpose(x, (2, 1, 3, 0)),
            einx.id("a b c d -> c b d a", x),
            setup,
        )

        img = np.random.uniform(size=(4, 3, 64, 64))
        vec = np.random.uniform(size=(2,))
        vec_as_img = np.broadcast_to(vec[np.newaxis, :, np.newaxis, np.newaxis], (img.shape[0], vec.shape[0], img.shape[2], img.shape[3]))
        out = np.concatenate([img, vec_as_img], axis=1)
        assert_allclose(
            out,
            einx.id("b c1 h w, c2 -> b (c1 + c2) h w", img, vec),
            setup,
        )

        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(2, 3))
        assert_allclose(
            einx.dot("b [x], b [x] -> b", x, y),
            np.matmul(x[:, np.newaxis, :], y[:, :, np.newaxis]).squeeze(-1).squeeze(-1),
            setup,
        )

        x = np.random.uniform(size=(4,))
        assert_allclose(
            einx.id("c -> 3 c 5", x),
            np.broadcast_to(x[np.newaxis, :, np.newaxis], (3, 4, 5)),
            setup,
        )
        with pytest.raises(Exception):
            np.broadcast_to(x, (3, 4, 5))

        x = np.random.uniform(size=(3, 15))
        x1, x2, x3 = np.split(x, [4, 9], axis=-1)
        x1e, x2e, x3e = einx.id("a (b1 + b2 + b3) -> a b1, a b2, a b3", x, b1=4, b2=5)
        assert_allclose(x1, x1e, setup)
        assert_allclose(x2, x2e, setup)
        assert_allclose(x3, x3e, setup)

        x = np.random.uniform(size=(3, 4, 4))
        assert_allclose(
            np.diagonal(x, axis1=1, axis2=2),
            einx.id("a b b -> a b", x),
            setup,
        )


if importlib.util.find_spec("jax"):
    import jax.numpy as jnp
    import jax

    def test_compare_jax():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: np.asarray(x))

        einsolve = einx.jax.adapt_with_vmap(jnp.linalg.solve)
        eindet = einx.jax.adapt_with_vmap(jnp.linalg.det)
        eineig = einx.jax.adapt_with_vmap(jnp.linalg.eig)

        A = jnp.asarray(np.random.uniform(size=(10, 3, 3)))
        x = jnp.asarray(np.random.uniform(size=(3)))
        assert_allclose(
            jnp.linalg.solve(A, x),
            einsolve("i [n n], [n] -> i [n]", A, x),
            setup,
        )

        A = jnp.asarray(np.random.uniform(size=(10, 3, 3)))
        x = jnp.asarray(np.random.uniform(size=(10, 3, 2)))
        assert_allclose(
            jnp.linalg.solve(A, x),
            einsolve("i [n n], i [n m] -> i [n m]", A, x),
            setup,
        )
        assert_allclose(
            jnp.linalg.solve(A, x),
            einsolve("i [n n], i [n] m -> i [n] m", A, x),
            setup,
        )

        A = jnp.asarray(np.random.uniform(size=(10, 3, 3)))
        x = jnp.asarray(np.random.uniform(size=(10, 3)))
        assert einsolve("i [n n], j [n] -> [n] (i j)", A, x).shape == (3, 10 * 10)
        assert eindet("i [n n] -> i", A).shape == (10,)
        vals, vecs = eineig("i [n n] -> i [n], [n n] i", A)
        assert vals.shape == (10, 3)
        assert vecs.shape == (3, 3, 10)

        x = jnp.asarray(np.random.uniform(size=(4, 16, 16, 8))).astype("float32")

        def normalize(x, epsilon=1e-5):
            mean = jnp.mean(x)
            var = jnp.var(x)
            return (x - mean) / jax.lax.rsqrt(var + epsilon)

        einnormalize = einx.jax.adapt_with_vmap(normalize, signature="... -> ...")

        assert einnormalize("... [c]", x).shape == x.shape
        assert einnormalize("[...] c", x).shape == x.shape
        assert einnormalize("b [s...] c", x).shape == x.shape
        assert einnormalize("b [s...] (g [c])", x, g=4).shape == x.shape

        x = jnp.asarray(np.random.uniform(size=(4, 16, 16, 8))).astype("float32")

        key = jax.random.PRNGKey(0)
        dropout_rate = 0.1
        dropout_factor = lambda shape: jax.random.bernoulli(key, 1.0 - dropout_rate, shape) / (1.0 - dropout_rate)

        assert einx.multiply("..., ...", x, dropout_factor).shape == x.shape
        assert einx.multiply("b ... c, b c", x, dropout_factor).shape == x.shape
        assert einx.multiply("b ..., b", x, dropout_factor).shape == x.shape

        x = jnp.asarray(np.random.uniform(size=(2, 3))).astype("float32")
        y = jnp.asarray(np.random.uniform(size=(4, 3))).astype("float32")

        def myfunc(x, y):
            return jnp.sum(x) * 2 + jnp.flip(y)

        einmyfunc = einx.jax.adapt_with_vmap(myfunc)
        assert einmyfunc("a [b], c [b] -> c [b] a", x, y).shape == (4, 3, 2)

        x = jnp.asarray(np.random.uniform(size=(2, 3, 4, 5))).astype("float32")

        assert einx.jax.adapt_numpylike_reduce(jnp.linalg.norm)("b [s...] c", x).shape == (2, 5)


if importlib.util.find_spec("scipy"):
    import scipy.linalg

    def test_compare_scipy():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x)

        # scipy.linalg.khatri_rao
        x = np.random.uniform(size=(2, 3))
        y = np.random.uniform(size=(5, 3))
        assert_allclose(
            scipy.linalg.khatri_rao(x, y),
            einx.multiply("a c, b c -> (a b) c", x, y),
            setup,
        )


if importlib.util.find_spec("eindex"):
    import eindex.numpy as EX

    def test_compare_eindex():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x)

        # EX.argfind
        x = np.random.uniform(size=(2, 3, 4, 5))
        assert_allclose(
            EX.argmin(x, "a b c d -> [d] c a b"),
            einx.argmin("a b c [d] -> [1] c a b", x),
            setup,
        )
        assert_allclose(
            EX.argmin(x, "a b c d -> [c, d] a b"),
            einx.argmin("a b [c d] -> [2] a b", x),
            setup,
        )
        assert_allclose(
            EX.argmax(x, "a b c d -> [a, c, d] b"),
            einx.argmax("[a] b [c d] -> [3] b", x),
            setup,
        )
        with pytest.raises(Exception):
            EX.argmin(x, "a b c d -> c [d] a b")  # coordinate axis not first
        einx.argmin("a b c [d] -> c [1] a b", x)

        # EX.gather
        x = np.random.uniform(size=(2, 3, 4, 5))
        coords = (np.random.uniform(size=(6, 4, 5))[None, ...] * np.asarray(x.shape[:2])[:, None, None, None]).astype("int32")
        assert_allclose(
            EX.gather(x, coords, "a b c d, [a, b] e c d -> e c d"),
            einx.get_at("[a b] c d, [2] e c d -> e c d", x, coords),
            setup,
        )
        coords2 = einx.id("i e c d -> e i c d", coords)
        with pytest.raises(Exception):
            EX.gather(x, coords2, "a b c d, e [a, b] c d -> e c d")  # coordinate axis not first
        einx.get_at("[a b] c d, e [2] c d -> e c d", x, coords2)

        x = np.random.uniform(size=(1, 4, 5))
        coords = (np.random.uniform(size=(6, 4, 5))[None, ...] * np.asarray(x.shape[:1])[:, None, None, None]).astype("int32")
        assert_allclose(
            EX.gather(x, coords, "a c d, [a] e c d -> e c d"),
            einx.get_at("[a] c d, [1] e c d -> e c d", x, coords),
            setup,
        )
        einx.get_at("[a] c d, e c d -> e c d", x, coords[0])
        with pytest.raises(Exception):
            EX.gather(x, coords[0], "a c d, e c d -> e c d")

        img = np.random.uniform(size=(17, 3, 4, 5))
        idx = (np.random.uniform(size=(17,))[None, :] * np.asarray(img.shape[1:3])[:, None]).astype("int32")
        assert_allclose(
            EX.gather(img, idx, "b h w c, [h, w] b -> b c"),
            einx.get_at("b [h w] c, [2] b -> b c", img, idx),
            setup,
        )
        with pytest.raises(Exception):
            EX.gather(img, idx, "b h w c, b [h, w] -> b c")

        # EX.scatter
        d = 5
        e = 6
        updates = np.random.uniform(size=(2, 3, 4))
        coords = (np.random.uniform(size=(2, 3, 4))[None, ...] * np.asarray([d, e])[:, None, None, None]).astype("int32")
        assert_allclose(
            EX.scatter(updates, coords, "a b c, [d, e] a b c -> d e b c", d=d, e=e),
            einx.add_at("[d e] b c, [2] a b c, a b c", partial(np.zeros, dtype=updates.dtype), coords, updates, d=d, e=e),
            setup,
        )


if importlib.util.find_spec("einops") and importlib.util.find_spec("jax"):
    import einops
    import jax.numpy as jnp
    import jax

    def test_compare_multiheadattention():
        import einx

        setup = types.SimpleNamespace(to_numpy=lambda x: x)

        q = jnp.asarray(np.random.uniform(size=(2, 16, 64)).astype("float32"))
        k = jnp.asarray(np.random.uniform(size=(2, 16, 64)).astype("float32"))
        v = jnp.asarray(np.random.uniform(size=(2, 16, 64)).astype("float32"))
        heads = 4

        def attn_einx_full(q, k, v):
            A = einx.dot("b q (h [c]), b k (h [c]) -> b q k h", q, k, h=heads)
            A = einx.softmax("b q [k] h", A / jnp.sqrt(q.shape[-1] / heads))
            return einx.dot("b q [k] h, b [k] (h c) -> b q (h c)", A, v)

        def attn_einops(q, k, v):
            q = einops.rearrange(q, "b q (h c) -> b q h c", h=heads)
            k = einops.rearrange(k, "b k (h c) -> b k h c", h=heads)
            v = einops.rearrange(v, "b k (h c) -> b k h c", h=heads)
            A = jnp.einsum("bqhc,bkhc->bqkh", q, k) / jnp.sqrt(q.shape[-1])
            A = jax.nn.softmax(A, axis=-2)
            output = jnp.einsum("bqkh,bkhc->bqhc", A, v)
            return einops.rearrange(output, "b q h c -> b q (h c)")

        def attn(q, k, v):
            A = einx.dot("[c], k [c] -> k", q, k)
            A = einx.softmax("[k]", A / jnp.sqrt(q.shape[-1]))
            return einx.dot("[k], [k] c -> c", A, v)

        einattn = einx.jax.adapt_with_vmap(attn)

        def attn_einx_decomposed(q, k, v, einattn=einattn):
            return einattn("b q (h [c]), b [k] (h [c]), b [k] (h [c]) -> b q (h [c])", q, k, v, h=heads)

        assert_allclose(
            attn_einx_full(q, k, v),
            attn_einops(q, k, v),
            setup,
        )
        assert_allclose(
            attn_einx_full(q, k, v),
            attn_einx_decomposed(q, k, v),
            setup,
        )

        qs, ks = jnp.arange(q.shape[1]), jnp.arange(k.shape[1])
        A = jnp.asarray(np.random.uniform(size=(2, qs.shape[0], ks.shape[0], heads)).astype("float32"))

        mask1 = einx.greater_equal("q, k -> q k", qs, ks)
        A1 = einx.where("q k, b q k h,", mask1, A, -jnp.inf)

        mask2 = qs[:, np.newaxis] >= ks[np.newaxis, :]
        A2 = jnp.where(mask2[np.newaxis, :, :, np.newaxis], A, -jnp.inf)

        assert_allclose(A1, A2, setup)

        mask3 = jnp.tril(jnp.ones((qs.shape[0], ks.shape[0]), dtype=bool))
        assert_allclose(mask1, mask2, setup)
        assert_allclose(mask1, mask3, setup)


if importlib.util.find_spec("flax"):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    def test_compare_flax():
        import einx

        class MyLayer(nn.Module):
            c_out: int = 64

            @nn.compact
            def __call__(self, x):
                weight = lambda shape: self.param("weight", nn.initializers.normal(0.01), shape)
                x = einx.dot("... [c_in], [c_in] c_out -> ... c_out", x, weight, c_out=self.c_out)
                return x

        x = jnp.zeros((4, 8, 8, 3))

        model = MyLayer(c_out=64)
        params = model.init({"params": jax.random.PRNGKey(42)}, x)
        assert params["params"]["weight"].shape == (3, 64)

        assert model.apply(params, x).shape == (4, 8, 8, 64)
