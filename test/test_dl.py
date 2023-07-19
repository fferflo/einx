import einx, importlib
import numpy as np
from functools import partial

if importlib.util.find_spec("torch"):
    import torch

    def test_torch_linear():
        x = torch.zeros((4, 128, 128, 3))

        layer = einx.torch.Linear(("b... [c1|c2]", {"c2": 32}))
        assert layer.forward(x).shape == (4, 128, 128, 32)
        layer = torch.compile(layer)
        assert layer.forward(x).shape == (4, 128, 128, 32)

    def test_torch_meanvar_norm():
        x = torch.zeros((4, 128, 128, 32))
        for mean in [True, False]:
            for decay_rate in [None, 0.9]:
                layer = einx.torch.Norm("b... [c]", mean=mean, decay_rate=decay_rate)
                layer.train()
                assert layer.forward(x).shape == (4, 128, 128, 32)
                layer.eval()
                assert layer.forward(x).shape == (4, 128, 128, 32)
                if decay_rate is None: # TODO: decay_rate currently does not work with torch.compile
                    layer.train()
                    assert torch.compile(layer).forward(x).shape == (4, 128, 128, 32)
                    layer.eval()
                    assert torch.compile(layer).forward(x).shape == (4, 128, 128, 32)

if importlib.util.find_spec("haiku"):
    import haiku as hk
    import jax.numpy as jnp
    import jax

    def test_haiku_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(42)

        def model(x):
            return einx.haiku.Linear(("b... [c1|c2]", {"c2": 32}))(x)
        model = hk.transform_with_state(model)

        params, state = model.init(rng=rng, x=x)

        y, state = jax.jit(model.apply)(params=params, state=state, x=x, rng=rng)
        assert y.shape == (4, 128, 128, 32)

    def test_haiku_meanvar_norm():
        x = jnp.zeros((4, 128, 128, 32))
        rng = jax.random.PRNGKey(42)

        for mean in [True, False]:
            for decay_rate in [None, 0.9]:
                def model(x, is_training):
                    return einx.haiku.Norm("b... [c]", mean=mean, decay_rate=decay_rate)(x, is_training)
                model = hk.transform_with_state(model)

                params, state = model.init(rng=rng, x=x, is_training=True)

                y, state = jax.jit(partial(model.apply, is_training=False))(params=params, state=state, x=x, rng=rng)
                assert y.shape == (4, 128, 128, 32)
                y, state = jax.jit(partial(model.apply, is_training=True))(params=params, state=state, x=x, rng=rng)
                assert y.shape == (4, 128, 128, 32)

if importlib.util.find_spec("flax"):
    import flax.linen as nn
    import jax.numpy as jnp
    import jax, flax

    def test_flax_linear():
        x = jnp.zeros((4, 128, 128, 3))
        rng = jax.random.PRNGKey(0)

        model = einx.flax.Linear(("b... [c1|c2]", {"c2": 32}))
        
        params = model.init(rng, x)

        y = jax.jit(model.apply)(params, x=x)
        assert y.shape == (4, 128, 128, 32)

    def test_flax_meanvar_norm():
        x = jnp.zeros((4, 128, 128, 32))
        rng = jax.random.PRNGKey(42)

        for mean in [True, False]:
            for decay_rate in [None, 0.9]:
                model = einx.flax.Norm("b... [c]", mean=mean, decay_rate=decay_rate)

                params = model.init(rng, x, is_training=True)
                state, params = flax.core.pop(params, "params")

                y, state = jax.jit(partial(model.apply, is_training=False, mutable=list(state.keys())))({"params": params, **state}, x=x)
                assert y.shape == (4, 128, 128, 32)
                y, state = jax.jit(partial(model.apply, is_training=True, mutable=list(state.keys())))({"params": params, **state}, x=x)
                assert y.shape == (4, 128, 128, 32)