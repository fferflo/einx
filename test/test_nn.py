import einx
import importlib
import pytest
import numpy as np
from functools import partial

norms = [
    ("[b...] c", {}),
    ("b [s...] (g [c])", {"g": 2}),
    ("b [s...] c", {}),
    ("b... [c]", {}),
    ("b [s...] ([g] c)", {"g": 2}),
]

if importlib.util.find_spec("torch"):
    import torch
    import einx.nn.torch

    if "compiler" in dir(torch):
        compiler = torch.compiler
    else:
        import torch._dynamo as compiler

    def test_torch_linear():
        compiler.reset()
        x = torch.zeros((4, 128, 128, 3))

        layer = einx.nn.torch.Linear("b... [c1->c2]", c2=32)
        assert layer.forward(x).shape == (4, 128, 128, 32)
        layer = torch.compile(layer)
        assert layer.forward(x).shape == (4, 128, 128, 32)

    @pytest.mark.parametrize("expr_kwargs", norms)
    @pytest.mark.parametrize("mean", [True, False])
    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("decay_rate", [None, 0.9])
    def test_torch_norm(expr_kwargs, mean, scale, decay_rate):
        compiler.reset()
        expr, kwargs = expr_kwargs
        x = torch.zeros((4, 128, 128, 32))

        layer = einx.nn.torch.Norm(expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs)
        layer.train()
        assert layer.forward(x).shape == (4, 128, 128, 32)
        layer.eval()
        assert layer.forward(x).shape == (4, 128, 128, 32)

        layer = torch.compile(layer, fullgraph=True)
        layer.train()
        assert layer.forward(x).shape == (4, 128, 128, 32)
        layer.eval()
        assert layer.forward(x).shape == (4, 128, 128, 32)

    def test_torch_dropout():
        compiler.reset()
        x = torch.zeros((4, 128, 128, 3))

        layer = einx.nn.torch.Dropout("[b] ... [c]", drop_rate=0.2)
        layer.train()
        assert layer.forward(x).shape == (4, 128, 128, 3)
        layer = torch.compile(layer)
        assert layer.forward(x).shape == (4, 128, 128, 3)

        layer = einx.nn.torch.Dropout("[b] ... [c]", drop_rate=0.2)
        layer.eval()
        assert layer.forward(x).shape == (4, 128, 128, 3)
        layer = torch.compile(layer)
        assert layer.forward(x).shape == (4, 128, 128, 3)


if importlib.util.find_spec("haiku"):
    import haiku as hk
    import jax.numpy as jnp
    import jax
    import einx.nn.haiku

    def test_haiku_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(42)

        def model(x):
            return einx.nn.haiku.Linear("b... [c1->c2]", c2=32)(x)

        model = hk.transform_with_state(model)

        params, state = model.init(rng=rng, x=x)

        y, state = jax.jit(model.apply)(params=params, state=state, x=x, rng=rng)
        assert y.shape == (4, 128, 128, 32)

    @pytest.mark.parametrize("expr_kwargs", norms)
    @pytest.mark.parametrize("mean", [True, False])
    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("decay_rate", [None, 0.9])
    def test_haiku_norm(expr_kwargs, mean, scale, decay_rate):
        expr, kwargs = expr_kwargs
        x = jnp.zeros((4, 128, 128, 32))
        rng = jax.random.PRNGKey(42)

        def model(x, training):
            return einx.nn.haiku.Norm(
                expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs
            )(x, training)

        model = hk.transform_with_state(model)

        params, state = model.init(rng=rng, x=x, training=True)

        y, state = jax.jit(partial(model.apply, training=False))(
            params=params, state=state, x=x, rng=rng
        )
        assert y.shape == (4, 128, 128, 32)
        y, state = jax.jit(partial(model.apply, training=True))(
            params=params, state=state, x=x, rng=rng
        )
        assert y.shape == (4, 128, 128, 32)

    def test_haiku_dropout():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(42)

        def model(x, training):
            return einx.nn.haiku.Dropout("[b] ... [c]", drop_rate=0.2)(x, training=training)

        model = hk.transform_with_state(model)

        params, state = model.init(rng=rng, x=x, training=True)

        y, state = jax.jit(partial(model.apply, training=True))(
            params=params, state=state, x=x, rng=rng
        )
        assert y.shape == (4, 128, 128, 3)
        y, state = jax.jit(partial(model.apply, training=False))(
            params=params, state=state, x=x, rng=rng
        )
        assert y.shape == (4, 128, 128, 3)


if importlib.util.find_spec("flax"):
    import flax.linen as nn
    import jax.numpy as jnp
    import jax
    import flax
    import einx.nn.flax

    def test_flax_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(0)

        model = einx.nn.flax.Linear("b... [c1->c2]", c2=32)

        params = model.init(rng, x)

        y = jax.jit(model.apply)(params, x=x)
        assert y.shape == (4, 128, 128, 32)

    @pytest.mark.parametrize("expr_kwargs", norms)
    @pytest.mark.parametrize("mean", [True, False])
    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("decay_rate", [None, 0.9])
    def test_flax_norm(expr_kwargs, mean, scale, decay_rate):
        expr, kwargs = expr_kwargs
        x = jnp.zeros((4, 128, 128, 32))
        rng = jax.random.PRNGKey(42)

        model = einx.nn.flax.Norm(expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs)

        params = model.init(rng, x, training=True)
        state, params = flax.core.pop(params, "params")

        y, state = jax.jit(partial(model.apply, training=False, mutable=list(state.keys())))(
            {"params": params, **state}, x=x
        )
        assert y.shape == (4, 128, 128, 32)
        y, state = jax.jit(partial(model.apply, training=True, mutable=list(state.keys())))(
            {"params": params, **state}, x=x
        )
        assert y.shape == (4, 128, 128, 32)

    def test_flax_dropout():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(0)

        model = einx.nn.flax.Dropout("[b] ... [c]", drop_rate=0.2)

        params = model.init({"params": rng, "dropout": rng}, x, training=True)

        y = jax.jit(partial(model.apply, training=True))(params, x=x, rngs={"dropout": rng})
        assert y.shape == (4, 128, 128, 3)
        y = jax.jit(partial(model.apply, training=False))(params, x=x, rngs={"dropout": rng})
        assert y.shape == (4, 128, 128, 3)


if importlib.util.find_spec("equinox"):
    import equinox as eqx
    import jax.numpy as jnp
    import einx.nn.equinox
    import jax

    def test_equinox_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(0)

        layer = einx.nn.equinox.Linear("b... [c1->c2]", c2=32)
        assert layer(x, rng=rng).shape == (4, 128, 128, 32)
        assert layer(x).shape == (4, 128, 128, 32)
        layer = eqx.nn.inference_mode(layer)
        assert layer(x).shape == (4, 128, 128, 32)
        assert layer(x).shape == (4, 128, 128, 32)

    @pytest.mark.parametrize("expr_kwargs", norms)
    @pytest.mark.parametrize("mean", [True, False])
    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("decay_rate", [None])
    def test_equinox_norm(expr_kwargs, mean, scale, decay_rate):
        expr, kwargs = expr_kwargs
        x = jnp.zeros((4, 128, 128, 32))
        for expr, kwargs in norms:
            for mean in [True, False]:
                for scale in [True, False]:
                    for decay_rate in [
                        None
                    ]:  # Stateful layers are currently not supported for Equinox
                        layer = einx.nn.equinox.Norm(
                            expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs
                        )
                        assert layer(x).shape == (4, 128, 128, 32)
                        assert layer(x).shape == (4, 128, 128, 32)
                        layer = eqx.nn.inference_mode(layer)
                        assert layer(x).shape == (4, 128, 128, 32)
                        assert layer(x).shape == (4, 128, 128, 32)

    def test_equinox_dropout():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(0)

        layer = einx.nn.equinox.Dropout("[b] ... [c]", drop_rate=0.2)
        assert layer(x, rng=rng).shape == (4, 128, 128, 3)
        assert layer(x, rng=rng).shape == (4, 128, 128, 3)
        layer = eqx.nn.inference_mode(layer)
        assert layer(x, rng=rng).shape == (4, 128, 128, 3)
        assert layer(x, rng=rng).shape == (4, 128, 128, 3)


if importlib.util.find_spec("keras"):
    import keras

    version = tuple(int(i) for i in keras.__version__.split(".")[:2])
    if version >= (3, 0):
        import tensorflow as tf
        import einx.nn.keras

        def test_keras_linear():
            x = tf.zeros((4, 128, 128, 3))

            layer = einx.nn.keras.Linear("b... [c1->c2]", c2=32)
            model = keras.Sequential([layer])
            assert model(x, training=True).shape == (4, 128, 128, 32)
            assert model(x, training=True).shape == (4, 128, 128, 32)
            assert model(x, training=False).shape == (4, 128, 128, 32)
            assert model(x, training=False).shape == (4, 128, 128, 32)

        @pytest.mark.parametrize("expr_kwargs", norms)
        @pytest.mark.parametrize("mean", [True, False])
        @pytest.mark.parametrize("scale", [True, False])
        @pytest.mark.parametrize("decay_rate", [None, 0.9])
        def test_keras_norm(expr_kwargs, mean, scale, decay_rate):
            expr, kwargs = expr_kwargs
            x = tf.zeros((4, 128, 128, 32))

            layer = einx.nn.keras.Norm(
                expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs
            )
            model = keras.Sequential([layer])
            assert model(x, training=True).shape == (4, 128, 128, 32)
            assert model(x, training=True).shape == (4, 128, 128, 32)
            assert model(x, training=False).shape == (4, 128, 128, 32)
            assert model(x, training=False).shape == (4, 128, 128, 32)

        def test_keras_dropout():
            x = tf.zeros((4, 128, 128, 3))

            layer = einx.nn.keras.Dropout("[b] ... [c]", drop_rate=0.2)
            model = keras.Sequential([layer])
            assert model(x, training=True).shape == (4, 128, 128, 3)
            assert model(x, training=True).shape == (4, 128, 128, 3)
            assert model(x, training=False).shape == (4, 128, 128, 3)
            assert model(x, training=False).shape == (4, 128, 128, 3)
