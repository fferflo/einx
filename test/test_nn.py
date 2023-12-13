import einx, importlib
import numpy as np
from functools import partial

norms = [("[b...] c", {}), ("b [s...] (g [c])", {"g": 2}), ("b [s...] c", {}), ("b... [c]", {}), ("b [s...] ([g] c)", {"g": 2})]

if importlib.util.find_spec("torch"):
    import torch, einx.nn.torch

    def test_torch_linear():
        x = torch.zeros((4, 128, 128, 3))

        layer = einx.nn.torch.Linear("b... [c1|c2]", c2=32)
        assert layer.forward(x).shape == (4, 128, 128, 32)
        layer = torch.compile(layer)
        assert layer.forward(x).shape == (4, 128, 128, 32)

    def test_torch_norm():
        x = torch.zeros((4, 128, 128, 32))
        for expr, kwargs in norms:
            for mean in [True, False]:
                for scale in [True, False]:
                    for decay_rate in [None, 0.9]:
                        layer = einx.nn.torch.Norm(expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs)
                        layer.train()
                        assert layer.forward(x).shape == (4, 128, 128, 32)
                        layer.eval()
                        assert layer.forward(x).shape == (4, 128, 128, 32)

                        layer = torch.compile(layer)
                        layer.train()
                        assert layer.forward(x).shape == (4, 128, 128, 32)
                        layer.eval()
                        assert layer.forward(x).shape == (4, 128, 128, 32)

if importlib.util.find_spec("haiku"):
    import haiku as hk
    import jax.numpy as jnp
    import jax, einx.nn.haiku

    def test_haiku_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(42)

        def model(x):
            return einx.nn.haiku.Linear("b... [c1|c2]", c2=32)(x)
        model = hk.transform_with_state(model)

        params, state = model.init(rng=rng, x=x)

        y, state = jax.jit(model.apply)(params=params, state=state, x=x, rng=rng)
        assert y.shape == (4, 128, 128, 32)

    def test_haiku_norm():
        x = jnp.zeros((4, 128, 128, 32))
        rng = jax.random.PRNGKey(42)

        for expr, kwargs in norms:
            for mean in [True, False]:
                for scale in [True, False]:
                    for decay_rate in [None, 0.9]:
                        def model(x, training):
                            return einx.nn.haiku.Norm(expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs)(x, training)
                        model = hk.transform_with_state(model)

                        params, state = model.init(rng=rng, x=x, training=True)

                        y, state = jax.jit(partial(model.apply, training=False))(params=params, state=state, x=x, rng=rng)
                        assert y.shape == (4, 128, 128, 32)
                        y, state = jax.jit(partial(model.apply, training=True))(params=params, state=state, x=x, rng=rng)
                        assert y.shape == (4, 128, 128, 32)

if importlib.util.find_spec("flax"):
    import flax.linen as nn
    import jax.numpy as jnp
    import jax, flax, einx.nn.flax

    def test_flax_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(0)

        model = einx.nn.flax.Linear("b... [c1|c2]", c2=32)

        params = model.init(rng, x)

        y = jax.jit(model.apply)(params, x=x)
        assert y.shape == (4, 128, 128, 32)

    def test_flax_norm():
        x = jnp.zeros((4, 128, 128, 32))
        rng = jax.random.PRNGKey(42)

        for expr, kwargs in norms:
            for mean in [True, False]:
                for scale in [True, False]:
                    for decay_rate in [None, 0.9]:
                        model = einx.nn.flax.Norm(expr, mean=mean, scale=scale, decay_rate=decay_rate, **kwargs)

                        params = model.init(rng, x, training=True)
                        state, params = flax.core.pop(params, "params")

                        y, state = jax.jit(partial(model.apply, training=False, mutable=list(state.keys())))({"params": params, **state}, x=x)
                        assert y.shape == (4, 128, 128, 32)
                        y, state = jax.jit(partial(model.apply, training=True, mutable=list(state.keys())))({"params": params, **state}, x=x)
                        assert y.shape == (4, 128, 128, 32)
